import os
os.environ["HYDRA_FULL_ERROR"] = "1"
from collections import defaultdict
import copy
import gym
import hydra
from lightning.pytorch.utilities.types import OptimizerLRScheduler, STEP_OUTPUT
from modeling.sold.do_world import CounterfactualMPCPlanner
from modeling.distributions import Moments
from modeling.autoencoder.base import Autoencoder
import numpy as np
from omegaconf import DictConfig
import torch
import torch.distributions as D
import torch.nn.functional as F
from train_autoencoder import AutoencoderModule
from typing import Any, Dict, List, Tuple
from utils.instantiate import instantiate_trainer
from utils.module import FreezeParameters
from utils.training import set_seed, print_summary, OnlineModule
from utils.visualizations import  visualize_reward_prediction, visualize_output_attention, visualize_reward_predictor_attention, get_attention_weights


class SOLDModule(OnlineModule):
    def __init__(self, autoencoder: Autoencoder, dynamics_predictor, actor, critic, reward_predictor,
                 dynamics_learning_rate: float, dynamics_grad_clip: float,
                 actor_learning_rate: float, actor_grad_clip: float, critic_learning_rate: float,
                 critic_grad_clip: float, reward_learning_rate: float, reward_grad_clip: float,
                 finetune_autoencoder: bool, autoencoder_learning_rate: float, autoencoder_grad_clip: float,
                 num_context: int | Tuple[int, int], imagination_horizon: int, start_imagination_from_every: bool,
                 actor_entropy_loss_weight: float, actor_gradients: str, return_lambda: float, discount_factor: float,
                 critic_ema_decay: float, env: gym.Env, max_steps: int, num_seed: int, update_freq: int,
                 num_updates: int, eval_freq: int, num_eval_episodes: int, batch_size: int, buffer_capacity: int,
                 save_replay_buffer: bool, planning_mode: str = "actor",
                 counterfactual_mpc_modes: Tuple[str, ...] = ("eval",),
                 counterfactual_mpc: Dict[str, Any] | None = None,
                 intervention_loss_weight: float = 1.0,
                 pseudo_intervention_loss_weight: float = 0.1) -> None:

        self.min_num_context, self.max_num_context = (num_context, num_context) if isinstance(num_context, int) else num_context
        if self.min_num_context > self.max_num_context:
            raise ValueError("min_num_context must be less than or equal to max_num_context.")
        sequence_length = imagination_horizon + self.max_num_context

        super().__init__(env, max_steps, num_seed, update_freq, num_updates, eval_freq, num_eval_episodes, batch_size,
                         sequence_length, buffer_capacity, save_replay_buffer)
        self.automatic_optimization = False
        self.save_hyperparameters(logger=False, ignore=['env'])

        regression_infos = {"max_episode_steps": env.max_episode_steps,  "num_slots": autoencoder.num_slots,
                            "slot_dim": autoencoder.slot_dim}
        self.autoencoder = autoencoder
        self.actor = actor(**regression_infos, output_dim=env.action_space.shape[0], lower_bound=env.action_space.low, upper_bound=env.action_space.high)
        self.critic = critic(**regression_infos)
        self.critic_target = copy.deepcopy(self.critic)
        self.reward_predictor = reward_predictor(**regression_infos)
        self.dynamics_predictor = dynamics_predictor(
                num_slots=autoencoder.num_slots, slot_dim=autoencoder.slot_dim, sequence_length=imagination_horizon,
                action_dim=env.action_space.shape[0], input_buffer_size=sequence_length)

        self.dynamics_learning_rate = dynamics_learning_rate
        self.dynamics_grad_clip = dynamics_grad_clip
        self.actor_learning_rate = actor_learning_rate
        self.actor_grad_clip = actor_grad_clip
        self.actor_entropy_loss_weight = actor_entropy_loss_weight
        self.actor_gradients = actor_gradients
        self.critic_learning_rate = critic_learning_rate
        self.critic_grad_clip = critic_grad_clip
        self.reward_learning_rate = reward_learning_rate
        self.reward_grad_clip = reward_grad_clip

        self.finetune_autoencoder = finetune_autoencoder
        self.autoencoder_grad_clip = autoencoder_grad_clip
        self.autoencoder_optimizer = torch.optim.Adam(self.autoencoder.parameters(), lr=autoencoder_learning_rate)
        self.imagination_horizon = imagination_horizon
        self.start_imagination_from_every = start_imagination_from_every
        self.return_lambda = return_lambda
        self.discount_factor = discount_factor
        self.critic_ema_decay = critic_ema_decay
        self.planning_mode = planning_mode
        self.counterfactual_mpc_modes = set(counterfactual_mpc_modes)
        self.intervention_loss_weight = intervention_loss_weight
        self.pseudo_intervention_loss_weight = pseudo_intervention_loss_weight
        self.counterfactual_mpc = None
        if counterfactual_mpc is not None:
            self.counterfactual_mpc = CounterfactualMPCPlanner(
                action_low=env.action_space.low,
                action_high=env.action_space.high,
                **dict(counterfactual_mpc),
            )

        self.return_moments = Moments()
        self.register_buffer("discounts", torch.full((1, self.imagination_horizon), self.discount_factor))
        self.discounts = torch.cumprod(self.discounts, dim=1) / self.discount_factor

        self.current_losses = defaultdict(list)

    def _squeeze_loader_batch(self, value: Any) -> Any:
        if isinstance(value, torch.Tensor) and value.dim() > 0 and value.shape[0] == 1:
            return value.squeeze(0)
        return value

    def _last_step(self, value: torch.Tensor, expected_last_dim: int | None = None) -> torch.Tensor:
        value = self._squeeze_loader_batch(value)
        if expected_last_dim is not None and value.shape[-1] != expected_last_dim:
            raise ValueError(f"Expected last dim {expected_last_dim}, got {value.shape[-1]}.")
        if expected_last_dim == self.env.action_space.shape[0] and value.dim() == 3:
            return value[:, -1]
        if value.dim() >= 4:
            return value[:, -1]
        return value

    def _encode_intervention_images(
        self,
        images: torch.Tensor,
        actions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        images = self._squeeze_loader_batch(images).to(self.device) / 255.
        if images.dim() == 4:
            images = images.unsqueeze(1)
        if actions is None:
            action = torch.full(
                (images.shape[0], images.shape[1], self.env.action_space.shape[0]),
                float("nan"),
                device=images.device,
                dtype=images.dtype,
            )
        else:
            action = self._squeeze_loader_batch(actions).to(images.device)
            if action.dim() == 2:
                action = action.unsqueeze(1).expand(-1, images.shape[1], -1)
        return self.autoencoder.encode(images, action).detach()[:, -1]

    def _intervention_dict_from_batch(self, batch: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        intervention = {}
        key_map = {
            "intervention_object_mask": "object_mask",
            "intervention_relation_mask": "relation_mask",
            "intervention_mechanism_scale": "mechanism_scale",
        }
        for batch_key, intervention_key in key_map.items():
            if batch_key in batch:
                intervention[intervention_key] = self._squeeze_loader_batch(batch[batch_key]).to(self.device)
        return intervention

    def _compute_true_intervention_loss(self, batch: Dict[str, Any]) -> torch.Tensor | None:
        if not hasattr(self.dynamics_predictor, "intervention_consistency_loss"):
            return None

        has_slot_targets = (
            ("intervention_source_slots" in batch or "intervention_slots" in batch)
            and ("intervention_target_slots" in batch or "intervention_next_slots" in batch)
        )
        has_image_targets = "intervention_obs" in batch and "intervention_next_obs" in batch
        if not has_slot_targets and not has_image_targets:
            return None

        if "intervention_action" in batch:
            intervention_actions = self._last_step(batch["intervention_action"], self.env.action_space.shape[0]).to(self.device)
        elif "intervention_actions" in batch:
            intervention_actions = self._last_step(batch["intervention_actions"], self.env.action_space.shape[0]).to(self.device)
        else:
            return None

        if has_slot_targets:
            source_key = "intervention_source_slots" if "intervention_source_slots" in batch else "intervention_slots"
            target_key = "intervention_target_slots" if "intervention_target_slots" in batch else "intervention_next_slots"
            source_slots = self._last_step(batch[source_key], self.autoencoder.slot_dim).to(self.device)
            target_slots = self._last_step(batch[target_key], self.autoencoder.slot_dim).to(self.device)
        else:
            source_slots = self._encode_intervention_images(batch["intervention_obs"], batch.get("intervention_actions"))
            target_slots = self._encode_intervention_images(batch["intervention_next_obs"], batch.get("intervention_actions"))

        intervention = self._intervention_dict_from_batch(batch)
        if not intervention:
            return None
        return self.dynamics_predictor.intervention_consistency_loss(
            source_slots.detach(),
            torch.nan_to_num(intervention_actions.detach()),
            target_slots.detach(),
            intervention,
        )

    def configure_optimizers(self) -> OptimizerLRScheduler:
        return [torch.optim.Adam(self.dynamics_predictor.parameters(), lr=self.dynamics_learning_rate),
                torch.optim.Adam(self.reward_predictor.parameters(), lr=self.reward_learning_rate),
                torch.optim.Adam(self.actor.parameters(), lr=self.actor_learning_rate),
                torch.optim.Adam(self.critic.parameters(), lr=self.critic_learning_rate)]

    def training_step(self, batch, batch_index: int) -> STEP_OUTPUT:
        dynamics_optimizer, reward_optimizer, actor_optimizer, critic_optimizer = self.optimizers()
        images, actions, rewards = batch["obs"].squeeze(0) / 255., batch["action"].squeeze(0), batch["reward"].squeeze(0)

        if self.finetune_autoencoder:
            self.autoencoder_optimizer.zero_grad()
        outputs = AutoencoderModule.compute_reconstruction_loss(self, images, actions)
        if self.finetune_autoencoder:
            outputs["reconstruction_loss"].backward()
            self.clip_gradients(self.autoencoder_optimizer, gradient_clip_val=self.autoencoder_grad_clip,
                                gradient_clip_algorithm="norm")
            self.autoencoder_optimizer.step()

        if self.after_eval:
            self.log("autoencoder_reconstruction", self.autoencoder.visualize_reconstruction(
                {k: v[0] for k, v in outputs.items() if v.dim() > 0}))

        # Detach slots to prevent gradients from flowing back to the autoencoder model.
        slots = outputs["slots"].detach()

        # Learn to predict dynamics in slot-space.
        dynamics_optimizer.zero_grad()
        outputs |= self.compute_dynamics_loss(images, slots, actions, batch=batch)
        self.manual_backward(outputs["dynamics_loss"])
        self.clip_gradients(dynamics_optimizer, gradient_clip_val=self.dynamics_grad_clip, gradient_clip_algorithm="norm")
        dynamics_optimizer.step()

        # Learn to predict rewards from the slot representation.
        reward_optimizer.zero_grad()
        outputs |= self.compute_reward_loss(images, outputs["reconstructions"], slots, rewards)
        self.manual_backward(outputs["reward_loss"])
        self.clip_gradients(reward_optimizer, gradient_clip_val=self.reward_grad_clip, gradient_clip_algorithm="norm")
        reward_optimizer.step()

        # Update the target critic network.
        for critic_param, critic_target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            critic_target_param.data.copy_((1 - self.critic_ema_decay) * critic_param.data + self.critic_ema_decay * critic_target_param.data)

        # Perform latent imagination to train the actor and critic.
        lambda_returns, predicted_values_targ, predicted_values_dist, action_log_probs, action_entropies = self.imagine_ahead(slots, actions)

        # Learn the actor.
        actor_optimizer.zero_grad()
        outputs |= self.compute_actor_loss(lambda_returns, predicted_values_targ, action_log_probs, action_entropies)
        self.manual_backward(outputs["actor_loss"])
        self.clip_gradients(actor_optimizer, gradient_clip_val=self.actor_grad_clip, gradient_clip_algorithm="norm")
        actor_optimizer.step()

        # Learn the critic.
        critic_optimizer.zero_grad()
        outputs |= self.compute_critic_loss(predicted_values_dist, lambda_returns, predicted_values_targ)
        self.manual_backward(outputs["critic_loss"])
        self.clip_gradients(critic_optimizer, gradient_clip_val=self.critic_grad_clip, gradient_clip_algorithm="norm")
        critic_optimizer.step()

        # Log all losses.
        for key, value in outputs.items():
            if key.endswith("_loss"):
                self.log("train/" + key, value)
        self.log_gradients(model_names=("reward_predictor", "actor", "critic"))
        return outputs

    def compute_dynamics_loss(self, images: torch.Tensor, slots: torch.Tensor, actions: torch.Tensor,
                              batch: Dict[str, Any] | None = None) -> Dict[str, Any]:
        self.dynamics_predictor.train()
        batch_size, sequence_length, num_slots, slot_dim = slots.size()
        num_context = torch.randint(self.min_num_context, self.max_num_context + 1, (1,)).item()
        context_slots = slots[:, :num_context].detach()

        future_slots = self.dynamics_predictor.predict_slots(slots, actions[:, 1:].clone().detach(), steps=self.imagination_horizon, num_context=num_context)
        future_outputs = self.autoencoder.decode(future_slots)
        slot_loss = F.mse_loss(future_slots, slots[:, num_context:num_context + self.imagination_horizon])
        image_loss = F.mse_loss(future_outputs["reconstructions"], images[:, num_context:num_context + self.imagination_horizon])
        dynamics_loss = slot_loss + image_loss

        outputs = {"slot_loss": slot_loss, "image_loss": image_loss}
        if hasattr(self.dynamics_predictor, "auxiliary_losses"):
            auxiliary_losses = self.dynamics_predictor.auxiliary_losses()
            dynamics_loss = dynamics_loss + auxiliary_losses.get("do_world_regularization_loss", 0.0)
            outputs.update(auxiliary_losses)

        if (batch is not None
                and hasattr(self.dynamics_predictor, "language_alignment_loss")
                and "language_embedding" in batch
                and "mechanism_label" in batch):
            language_embeddings = batch["language_embedding"].squeeze(0).to(slots.device)
            mechanism_labels = batch["mechanism_label"].squeeze(0).to(slots.device)
            language_alignment_loss = self.dynamics_predictor.language_alignment_loss(
                language_embeddings, mechanism_labels)
            dynamics_loss = dynamics_loss + language_alignment_loss
            outputs["language_alignment_loss"] = language_alignment_loss

        if batch is not None:
            true_intervention_loss = self._compute_true_intervention_loss(batch)
            if true_intervention_loss is not None:
                dynamics_loss = dynamics_loss + self.intervention_loss_weight * true_intervention_loss
                outputs["intervention_consistency_loss"] = true_intervention_loss

        if (self.pseudo_intervention_loss_weight > 0.0
                and hasattr(self.dynamics_predictor, "pseudo_intervention_losses")
                and num_context < sequence_length):
            source_slots = slots[:, num_context - 1].detach()
            target_next_slots = slots[:, num_context].detach()
            source_actions = torch.nan_to_num(actions[:, num_context].detach())
            pseudo_losses = self.dynamics_predictor.pseudo_intervention_losses(
                source_slots, source_actions, target_next_slots)
            dynamics_loss = dynamics_loss + self.pseudo_intervention_loss_weight * pseudo_losses["pseudo_intervention_loss"]
            outputs.update(pseudo_losses)

        if self.after_eval:
            x_ticks = [f'T={0}']
            for t in range(1, num_context + self.imagination_horizon):
                x_ticks.append(f'{t}')

            context_outputs = self.autoencoder.decode(context_slots)
            context_outputs["images"] = images[:, :num_context]
            context_outputs["xticks"] = np.array([x_ticks, ])[:, :num_context]
            context_image = self.autoencoder.visualize_reconstruction({k: v[0] for k, v in context_outputs.items()})
            future_outputs["images"] = images[:, num_context:num_context + self.imagination_horizon]
            future_outputs["xticks"] = np.array([x_ticks, ])[:, num_context:]
            future_image = self.autoencoder.visualize_reconstruction({k: v[0] for k, v in future_outputs.items()})
            dynamics_image = torch.cat(
                [context_image, torch.ones(3, context_image.size(1), 2), future_image], dim=2)
            self.log("dynamics_prediction", dynamics_image)

        outputs["dynamics_loss"] = dynamics_loss
        return outputs

    def compute_reward_loss(self, images: torch.Tensor, reconstructions: torch.Tensor, slots: torch.Tensor, rewards: torch.Tensor) -> Dict[str, Any]:
        is_firsts = torch.isnan(rewards)  # We add NaN as a reward on the first time-step.
        predicted_rewards_dist = self.reward_predictor(slots.detach())
        log_probs = predicted_rewards_dist.log_prob(torch.nan_to_num(rewards).unsqueeze(2))
        masked_log_probs = log_probs[~is_firsts]

        # Log visualizations related to reward prediction.
        if self.after_eval:
            with torch.no_grad():
                # Log prediction vs ground truth reward over the sequence.
                reward_image = visualize_reward_prediction(
                    images[0], reconstructions[0], rewards[0],
                    predicted_rewards_dist.mean.squeeze(2)[0])
                self.log("reward_prediction", reward_image)

                # Log visualization of reward predictor attention to inspect reward-predictive elements.
                outputs = self.autoencoder.decode(slots[0:1])
                if "masks" in outputs:
                    output_weights = get_attention_weights(self.reward_predictor, slots[0:1, ])
                    attention_image = visualize_reward_predictor_attention(images[0], reconstructions[0], rewards[0], predicted_rewards_dist.mean.squeeze(2)[0], output_weights, outputs["rgbs"][0], outputs["masks"][0])
                    self.log("reward_predictor_attention", attention_image)

        return {"reward_loss": -masked_log_probs.mean(),
                "reward_mse_loss": F.mse_loss(predicted_rewards_dist.mean.squeeze(2)[~is_firsts], rewards[~is_firsts]).item()}

    def imagine_ahead(self, slots: torch.Tensor, actions: torch.Tensor) -> List[torch.Tensor]:
        self.dynamics_predictor.eval()
        batch_size, sequence_length, num_slots, slot_dim = slots.size()
        action_log_probs, action_entropies = [], []

        if self.start_imagination_from_every:
            num_context = self.max_num_context
            slots_context = slots.unfold(dimension=1, size=self.max_num_context, step=1).flatten(end_dim=1).permute(0, 3, 1, 2)
            actions_context = actions.unfold(dimension=1, size=self.max_num_context, step=1).flatten(end_dim=1).permute(0, 2, 1)[:, 1:]
        else:
            num_context = torch.randint(self.min_num_context, self.max_num_context + 1, (1,)).item()
            slots_context = slots[:, :num_context].detach()
            actions_context = actions[:, 1:num_context].detach()

        # Actor update
        # Freeze models except action model and imagine next states
        with FreezeParameters([self.reward_predictor, self.critic]):
            for t in range(self.imagination_horizon):
                action_dist = self.actor(slots_context.detach(), start=slots_context.shape[1] - 1)
                selected_action = action_dist.rsample().squeeze(1)
                actions_context = torch.cat([actions_context, selected_action.unsqueeze(1)], dim=1)
                action_log_probs.append(action_dist.log_prob(selected_action.unsqueeze(1)))
                action_entropies.append(action_dist.entropy())

                predicted_slots = self.dynamics_predictor.predict_slots(slots_context, actions_context, steps=1, num_context=slots_context.shape[1])
                slots_context = torch.cat([slots_context, predicted_slots], dim=1)

        with FreezeParameters([self.reward_predictor, self.critic]):
            predicted_rewards = self.reward_predictor(slots_context, start=num_context).mean.squeeze()
            predicted_values = self.critic(slots_context, start=num_context).mean.squeeze()

        lambda_returns = self.compute_lambda_returns(predicted_rewards, predicted_values)

        action_log_probs = torch.stack(action_log_probs, dim=1).squeeze(2)
        action_entropies = torch.stack(action_entropies, dim=1)

        # Value update
        slots_context = slots_context.detach()
        # Predict imagined values
        predicted_values_targ = self.critic_target(slots_context[:, :-1], start=num_context - 1).mean.squeeze()
        predicted_values_dist = self.critic(slots_context[:, :-1], start=num_context - 1)

        if self.after_eval:
            with torch.no_grad():
                # Log visualization of a latent imagination sequence.
                outputs = self.autoencoder.decode(slots_context[0:1])
                x_ticks = [''] * num_context
                x_ticks.append(f'rew={predicted_rewards[0, 0].item():.2f}')
                for t in range(1, self.imagination_horizon):
                    x_ticks.append(f'{predicted_rewards[0, t].item():.2f}')
                outputs["xticks"] = np.array([x_ticks, ])
                context_image = self.autoencoder.visualize_reconstruction(
                    {k: v[0, :num_context] for k, v in outputs.items()})
                future_image = self.autoencoder.visualize_reconstruction(
                    {k: v[0, num_context:] for k, v in outputs.items()})
                latent_imagination_image = torch.cat(
                    [context_image, torch.ones(3, context_image.size(1), 2), future_image], dim=2)
                self.log("latent_imagination", latent_imagination_image)

                # Log visualization of actor attention.
                if "masks" in outputs:
                    output_weights = get_attention_weights(self.actor, slots_context[:1, :num_context + self.imagination_horizon])
                    actor_attention_image = visualize_output_attention(output_weights, outputs["reconstructions"][0], outputs["rgbs"][0], outputs["masks"][0])
                    self.log("actor_attention", actor_attention_image)
        return lambda_returns, predicted_values_targ, predicted_values_dist, action_log_probs, action_entropies

    def compute_actor_loss(self, lambda_returns: torch.Tensor, predicted_values_targ: torch.Tensor,
                           action_log_probs: torch.Tensor, action_entropies: torch.Tensor) -> Dict[str, Any]:
        # Compute advantage estimates.
        offset, invscale = self.return_moments(lambda_returns[:, :-1])
        normed_lambda_returns = (lambda_returns[:, :-1] - offset) / invscale
        normed_base = (predicted_values_targ[:, :-1] - offset) / invscale
        advantage = normed_lambda_returns - normed_base

        if self.actor_gradients == "dynamics":
            actor_return_loss = -torch.mean(self.discounts.detach()[:, :-1] * advantage)
        elif self.actor_gradients == "reinforce":
            actor_return_loss = torch.mean(action_log_probs[:, :-1] * advantage.detach())
        else:
            raise ValueError(f"Invalid actor_gradients: {self.actor_gradients}.")

        actor_entropy_loss = -torch.mean(self.discounts.detach() * action_entropies)
        return {"actor_loss": actor_return_loss + self.actor_entropy_loss_weight * actor_entropy_loss, "actor_return_loss": actor_return_loss,
                "actor_entropy_loss": self.actor_entropy_loss_weight * actor_entropy_loss}

    def compute_critic_loss(self, predicted_values_dist: D.Distribution, lambda_returns: torch.Tensor,
                            predicted_values_targ: torch.Tensor, regularization_loss_weight: float = 0.1) -> Dict[str, Any]:
        return_loss = torch.mean(self.discounts.detach() * (-predicted_values_dist.log_prob(lambda_returns.detach().unsqueeze(2))))
        target_regularization_loss = torch.mean(self.discounts.detach() * (-predicted_values_dist.log_prob(predicted_values_targ.detach().unsqueeze(2))))
        return {"critic_loss": return_loss + regularization_loss_weight * target_regularization_loss,
                "critic_return_loss": return_loss,
                "critic_target_regularization_loss": regularization_loss_weight * target_regularization_loss,
                "return_mse_loss": F.mse_loss(predicted_values_dist.mean.squeeze(2), lambda_returns).item()}

    def compute_lambda_returns(self, rewards, values):
        vals = [values[:, -1:]]
        interm = rewards + self.discount_factor * values * (1 - self.return_lambda)
        for t in reversed(range(self.imagination_horizon)):
            vals.append(interm[:, t].unsqueeze(1) + self.discount_factor * self.return_lambda * vals[-1])
        ret = torch.cat(list(reversed(vals)), dim=1)[:, :-1]
        return ret

    def select_action(self, observation: torch.Tensor, is_first: bool = False, mode: str = "train") -> torch.Tensor:
        observation = observation.unsqueeze(0) / 255.  # Expand batch dimension (1, 3, height, width).

        # Encode image into slots and append to context.
        last_slots = None if is_first else self._slot_history[:, -1]
        slots = self.autoencoder.encode(observation.unsqueeze(1), self.last_action.unsqueeze(0).unsqueeze(1),
                                        prior_slots=last_slots)  # Expand sequence (and batch) dimension.
        self._slot_history = slots if is_first else torch.cat([self._slot_history, slots], dim=1)

        if mode == "random":
            selected_action = torch.from_numpy(self.env.action_space.sample().astype(np.float32))
        elif (self.planning_mode == "counterfactual_mpc"
              and self.counterfactual_mpc is not None
              and mode in self.counterfactual_mpc_modes):
            selected_action = self.counterfactual_mpc.plan(
                self._slot_history,
                self.dynamics_predictor,
                self.reward_predictor,
                actor=self.actor,
            )
        else:
            action_dist = self.actor(self._slot_history, start=self._slot_history.shape[1] - 1)
            if mode == "train":
                selected_action = action_dist.sample().squeeze()
            elif mode == "eval":
                selected_action = action_dist.mode.squeeze()
            else:
                raise ValueError(f"Invalid mode: {mode}")

        low = torch.as_tensor(self.env.action_space.low, device=selected_action.device, dtype=selected_action.dtype)
        high = torch.as_tensor(self.env.action_space.high, device=selected_action.device, dtype=selected_action.dtype)
        return torch.maximum(torch.minimum(selected_action, high), low).detach()


@hydra.main(config_path="../configs", config_name="train_sold", version_base=None)
def train(cfg: DictConfig):
    if cfg.logger.log_to_wandb:
        import wandb
        wandb.init(project="sold", name=cfg.experiment, config=dict(cfg), sync_tensorboard=True)
    print_summary(cfg)
    set_seed(cfg.seed)
    sold = hydra.utils.instantiate(cfg.model)
    trainer = instantiate_trainer(cfg)
    trainer.fit(sold, ckpt_path=os.path.abspath(cfg.checkpoint) if cfg.checkpoint else None)

    if cfg.logger.log_to_wandb:
        wandb.finish()


if __name__ == "__main__":
    train()

from collections import defaultdict
import json
import os
from typing import Any, Dict, List, Optional

import hydra
from omegaconf import DictConfig
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from evaluate_sold import get_checkpoint_files
from train_sold import SOLDModule
from utils.training import set_seed

os.environ["HYDRA_FULL_ERROR"] = "1"


@torch.no_grad()
def play_episode(sold: SOLDModule, mode: str = "eval") -> Dict[str, Any]:
    obs, done, info = sold.env.reset(), False, {}
    episode = defaultdict(list)
    episode["obs"].append(obs.cpu())

    while not done:
        action = sold.select_action(obs.to(sold.device), is_first=len(episode["obs"]) == 1, mode=mode).cpu()
        obs, reward, done, info = sold.env.step(action)
        episode["obs"].append(obs.cpu())
        episode["action"].append(action)
        episode["reward"].append(torch.as_tensor(reward))
        for key, value in info.items():
            if key.startswith("intervention_"):
                episode[key].append(torch.as_tensor(value).cpu())

    if "success" in info:
        episode["success"] = info["success"]
    return episode


def _episode_tensors(episode: Dict[str, Any], action_dim: int, device: torch.device) -> Dict[str, torch.Tensor]:
    observations = torch.stack(episode["obs"]).unsqueeze(0).to(device)
    actions = torch.stack(episode["action"]).unsqueeze(0).to(device)
    first_action = torch.full((1, 1, action_dim), float("nan"), device=device, dtype=actions.dtype)
    actions = torch.cat((first_action, actions), dim=1)
    rewards = torch.stack(episode["reward"]).unsqueeze(0).to(device)
    return {"obs": observations, "action": actions, "reward": rewards}


def _discounted_return(rewards: torch.Tensor, discount_factor: float) -> torch.Tensor:
    discounts = torch.tensor(
        [discount_factor ** step for step in range(rewards.shape[1])],
        device=rewards.device,
        dtype=rewards.dtype,
    )
    return (rewards * discounts.unsqueeze(0)).sum(dim=1)


def _stack_optional_episode_field(episode: Dict[str, Any], key: str, device: torch.device) -> Optional[torch.Tensor]:
    if key not in episode or len(episode[key]) == 0:
        return None
    return torch.stack([torch.as_tensor(value) for value in episode[key]]).to(device)


def _compute_intervention_error(sold: SOLDModule, episode: Dict[str, Any]) -> float:
    if not hasattr(sold.dynamics_predictor, "intervention_consistency_loss"):
        return float("nan")

    source_slots = _stack_optional_episode_field(episode, "intervention_source_slots", sold.device)
    target_slots = _stack_optional_episode_field(episode, "intervention_target_slots", sold.device)
    if target_slots is None:
        target_slots = _stack_optional_episode_field(episode, "intervention_next_slots", sold.device)
    actions = _stack_optional_episode_field(episode, "intervention_action", sold.device)
    if actions is None:
        actions = _stack_optional_episode_field(episode, "intervention_actions", sold.device)

    if source_slots is None or target_slots is None or actions is None:
        return float("nan")

    if source_slots.dim() == 4:
        source_slots = source_slots[:, -1]
    if target_slots.dim() == 4:
        target_slots = target_slots[:, -1]
    if actions.dim() == 3:
        actions = actions[:, -1]

    intervention = {}
    for episode_key, intervention_key in {
        "intervention_object_mask": "object_mask",
        "intervention_relation_mask": "relation_mask",
        "intervention_mechanism_scale": "mechanism_scale",
    }.items():
        value = _stack_optional_episode_field(episode, episode_key, sold.device)
        if value is not None:
            intervention[intervention_key] = value

    if not intervention:
        return float("nan")
    return sold.dynamics_predictor.intervention_consistency_loss(
        source_slots,
        torch.nan_to_num(actions),
        target_slots,
        intervention,
    ).item()


@torch.no_grad()
def compute_do_world_metrics(
    sold: SOLDModule,
    episode: Dict[str, Any],
    horizon: int,
    num_counterfactuals: int,
    relation_threshold: float,
    discount_factor: float,
) -> Dict[str, float]:
    if not hasattr(sold.dynamics_predictor, "rollout_from_slots"):
        raise TypeError("Do-World evaluation requires a dynamics model with rollout_from_slots(...).")

    action_dim = sold.env.action_space.shape[0]
    tensors = _episode_tensors(episode, action_dim, sold.device)
    images = tensors["obs"] / 255.
    actions = tensors["action"]
    slots = sold.autoencoder.encode(images, actions).detach()

    sequence_length = slots.shape[1]
    if sequence_length < 3:
        return {}

    num_context = min(sold.max_num_context, sequence_length - 1)
    rollout_horizon = min(horizon, sequence_length - num_context)
    if rollout_horizon <= 0:
        return {}

    predicted_slots = sold.dynamics_predictor.predict_slots(
        slots, actions[:, 1:], steps=rollout_horizon, num_context=num_context)
    target_slots = slots[:, num_context:num_context + rollout_horizon]

    prediction_error = F.mse_loss(predicted_slots[:, :1], target_slots[:, :1]).item()
    multi_step_error = F.mse_loss(predicted_slots, target_slots).item()

    context_slots = slots[:, :num_context]
    action_sequence = torch.nan_to_num(actions[:, num_context:num_context + rollout_horizon])
    factual_rollout, aux = sold.dynamics_predictor.rollout_from_slots(
        context_slots, action_sequence, return_aux=True)
    factual_future = factual_rollout[:, 1:]
    predicted_rewards = sold.reward_predictor(factual_future, start=0).mean.squeeze(-1)
    factual_return = _discounted_return(predicted_rewards, discount_factor)

    relation_weights = aux.get("relation_weights")
    relation_edge_density = float("nan")
    relation_sparsity = float("nan")
    mechanism_entropy = float("nan")
    mechanism_diversity = float("nan")
    if relation_weights is not None:
        relation_edge_density = (relation_weights > relation_threshold).float().mean().item()
        relation_sparsity = 1.0 - relation_edge_density
    if "mechanism_probs" in aux:
        mechanism_usage = aux["mechanism_probs"].mean(dim=tuple(range(aux["mechanism_probs"].dim() - 1)))
        mechanism_usage = mechanism_usage.clamp_min(1e-8)
        mechanism_entropy_tensor = -(mechanism_usage * mechanism_usage.log()).sum()
        mechanism_entropy = mechanism_entropy_tensor.item()
        mechanism_diversity = mechanism_entropy_tensor.exp().item()

    counterfactual_drop = float("nan")
    factual_counterfactual_gap = float("nan")
    robust_return = float("nan")
    if num_counterfactuals > 0 and hasattr(sold.dynamics_predictor, "generate_counterfactual_interventions"):
        interventions = sold.dynamics_predictor.generate_counterfactual_interventions(
            context_slots[:, -1],
            num_counterfactuals,
            relation_weights=relation_weights[:, 0] if relation_weights is not None and relation_weights.dim() == 4 else None,
        )
        counterfactual_returns = []
        for intervention in interventions:
            counterfactual_future = sold.dynamics_predictor.rollout_from_slots(
                context_slots, action_sequence, intervention=intervention)[:, 1:]
            counterfactual_rewards = sold.reward_predictor(counterfactual_future, start=0).mean.squeeze(-1)
            counterfactual_returns.append(_discounted_return(counterfactual_rewards, discount_factor))
        counterfactual_returns = torch.stack(counterfactual_returns, dim=1)
        counterfactual_drop = (factual_return - counterfactual_returns.mean(dim=1)).mean().item()
        factual_counterfactual_gap = F.relu(factual_return.unsqueeze(1) - counterfactual_returns).mean().item()
        robust_return = counterfactual_returns.min(dim=1).values.mean().item()
    intervention_error = _compute_intervention_error(sold, episode)

    return {
        "prediction_error": prediction_error,
        "multi_step_error": multi_step_error,
        "intervention_error": intervention_error,
        "factual_return": factual_return.mean().item(),
        "robust_return": robust_return,
        "counterfactual_drop": counterfactual_drop,
        "factual_counterfactual_gap": factual_counterfactual_gap,
        "relation_edge_density": relation_edge_density,
        "relation_sparsity": relation_sparsity,
        "mechanism_entropy": mechanism_entropy,
        "mechanism_diversity": mechanism_diversity,
    }


def _mean_metric(records: List[Dict[str, float]], key: str) -> Optional[float]:
    values = [record[key] for record in records if key in record and not np.isnan(record[key])]
    return float(np.mean(values)) if values else None


@hydra.main(config_path="../configs", config_name="evaluate_do_world", version_base=None)
def evaluate(cfg: DictConfig):
    set_seed(cfg.seed)
    output_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    checkpoint_files = get_checkpoint_files(cfg.checkpoint_path)
    metrics_filename = os.path.join(output_dir, "do_world_metrics.jsonl")

    for checkpoint in tqdm(checkpoint_files, disable=len(checkpoint_files) == 1, desc="Evaluating checkpoints"):
        env = hydra.utils.instantiate(cfg.env)
        sold = SOLDModule.load_from_checkpoint(checkpoint, env=env)
        sold.eval()

        episode_returns, successes, model_metric_records = [], [], []
        for _ in range(cfg.eval_episodes):
            episode = play_episode(sold, mode=cfg.mode)
            episode_returns.append(sum(float(reward) for reward in episode["reward"]))
            if "success" in episode:
                successes.append(float(episode["success"]))
            model_metric_records.append(compute_do_world_metrics(
                sold,
                episode,
                horizon=cfg.horizon,
                num_counterfactuals=cfg.num_counterfactuals,
                relation_threshold=cfg.relation_threshold,
                discount_factor=cfg.discount_factor,
            ))

        record = {
            "step": sold.num_steps,
            "checkpoint": checkpoint,
            "episode_return": float(np.mean(episode_returns)),
            "episode_returns": episode_returns,
        }
        if successes:
            record["success_rate"] = float(np.mean(successes))

        metric_keys = sorted({key for metrics in model_metric_records for key in metrics.keys()})
        for key in metric_keys:
            value = _mean_metric(model_metric_records, key)
            if value is not None:
                record[key] = value

        with open(metrics_filename, mode="a") as file:
            file.write(json.dumps(record) + "\n")


if __name__ == "__main__":
    evaluate()

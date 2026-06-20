import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def _make_mlp(input_dim: int, hidden_dim: int, output_dim: int, num_layers: int) -> nn.Sequential:
    if num_layers < 1:
        raise ValueError("num_layers must be at least 1.")

    layers: List[nn.Module] = []
    for layer_index in range(num_layers):
        in_dim = input_dim if layer_index == 0 else hidden_dim
        out_dim = output_dim if layer_index == num_layers - 1 else hidden_dim
        layers.append(nn.Linear(in_dim, out_dim))
        if layer_index < num_layers - 1:
            layers.append(nn.SiLU())
    return nn.Sequential(*layers)


def _as_tensor(value: Any, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value.to(device=device, dtype=dtype)
    return torch.as_tensor(value, device=device, dtype=dtype)


def _expand_to(value: torch.Tensor, shape: Tuple[int, ...]) -> torch.Tensor:
    while value.dim() < len(shape):
        value = value.unsqueeze(0)
    return value.expand(shape)


class ObjectRelationInference(nn.Module):
    """Infers directed object-to-object causal edge weights in slot space."""

    def __init__(self, slot_dim: int, action_dim: int, hidden_dim: int, num_layers: int) -> None:
        super().__init__()
        self.pair_dim = 4 * slot_dim + action_dim
        self.edge_scorer = _make_mlp(self.pair_dim, hidden_dim, 1, num_layers)

    def make_pair_features(self, slots: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        batch_size, num_slots, slot_dim = slots.shape
        senders = slots.unsqueeze(2).expand(batch_size, num_slots, num_slots, slot_dim)
        receivers = slots.unsqueeze(1).expand(batch_size, num_slots, num_slots, slot_dim)
        actions = actions.unsqueeze(1).unsqueeze(2).expand(batch_size, num_slots, num_slots, actions.shape[-1])
        return torch.cat((senders, receivers, receivers - senders, senders * receivers, actions), dim=-1)

    def forward(
        self,
        slots: torch.Tensor,
        actions: torch.Tensor,
        object_mask: Optional[torch.Tensor] = None,
        relation_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size, num_slots, _ = slots.shape
        pair_features = self.make_pair_features(slots, actions)
        logits = self.edge_scorer(pair_features).squeeze(-1)
        weights = torch.sigmoid(logits)

        eye = torch.eye(num_slots, device=slots.device, dtype=torch.bool).unsqueeze(0)
        weights = weights.masked_fill(eye, 0.0)

        if object_mask is not None:
            pair_mask = object_mask.unsqueeze(1) * object_mask.unsqueeze(2)
            weights = weights * pair_mask
        if relation_mask is not None:
            weights = weights * relation_mask

        return weights, logits, pair_features


class ObjectCausalMechanismLibrary(nn.Module):
    """A modular object-level transition model with editable local mechanisms."""

    def __init__(
        self,
        slot_dim: int,
        action_dim: int,
        num_mechanisms: int = 8,
        hidden_dim: int = 256,
        message_dim: int = 128,
        action_embed_dim: int = 128,
        num_mlp_layers: int = 3,
        residual: bool = True,
    ) -> None:
        super().__init__()
        self.slot_dim = slot_dim
        self.action_dim = action_dim
        self.num_mechanisms = num_mechanisms
        self.message_dim = message_dim
        self.residual = residual

        self.relation_inference = ObjectRelationInference(slot_dim, action_dim, hidden_dim, num_mlp_layers)
        self.message_fn = _make_mlp(self.relation_inference.pair_dim, hidden_dim, message_dim, num_mlp_layers)
        self.action_encoder = nn.Linear(action_dim, action_embed_dim)

        mechanism_input_dim = slot_dim + message_dim + action_embed_dim
        self.router = _make_mlp(mechanism_input_dim, hidden_dim, num_mechanisms, num_mlp_layers)
        self.mechanisms = nn.ModuleList([
            _make_mlp(mechanism_input_dim, hidden_dim, slot_dim, num_mlp_layers)
            for _ in range(num_mechanisms)
        ])
        self.mechanism_embeddings = nn.Parameter(torch.randn(num_mechanisms, hidden_dim) / math.sqrt(hidden_dim))

    def _normalize_intervention(
        self,
        intervention: Optional[Dict[str, Any]],
        batch_size: int,
        num_slots: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Dict[str, Optional[torch.Tensor]]:
        if intervention is None:
            intervention = {}

        object_mask = None
        if "object_mask" in intervention:
            object_mask = _expand_to(_as_tensor(intervention["object_mask"], device, dtype), (batch_size, num_slots))
        elif "remove_object" in intervention:
            object_mask = torch.ones(batch_size, num_slots, device=device, dtype=dtype)
            object_mask[:, int(intervention["remove_object"])] = 0.0

        relation_mask = None
        if "relation_mask" in intervention:
            relation_mask = _expand_to(
                _as_tensor(intervention["relation_mask"], device, dtype),
                (batch_size, num_slots, num_slots),
            )
        elif "cut_relations" in intervention:
            relation_mask = torch.ones(batch_size, num_slots, num_slots, device=device, dtype=dtype)
            for sender, receiver in intervention["cut_relations"]:
                relation_mask[:, int(sender), int(receiver)] = 0.0

        mechanism_scale = None
        if "mechanism_scale" in intervention:
            mechanism_scale = _as_tensor(intervention["mechanism_scale"], device, dtype)
        elif "perturb_mechanism" in intervention:
            mechanism_scale = torch.ones(self.num_mechanisms, device=device, dtype=dtype)
            scale = float(intervention.get("mechanism_perturbation", 0.5))
            mechanism_scale[int(intervention["perturb_mechanism"])] = scale

        return {
            "object_mask": object_mask,
            "relation_mask": relation_mask,
            "mechanism_scale": mechanism_scale,
        }

    def forward(
        self,
        slots: torch.Tensor,
        actions: torch.Tensor,
        intervention: Optional[Dict[str, Any]] = None,
        return_aux: bool = False,
    ) -> torch.Tensor | Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        batch_size, num_slots, _ = slots.shape
        actions = torch.nan_to_num(actions)
        normalized = self._normalize_intervention(
            intervention, batch_size, num_slots, slots.device, slots.dtype)
        object_mask = normalized["object_mask"]
        relation_mask = normalized["relation_mask"]

        masked_slots = slots if object_mask is None else slots * object_mask.unsqueeze(-1)
        edge_weights, edge_logits, pair_features = self.relation_inference(
            masked_slots, actions, object_mask=object_mask, relation_mask=relation_mask)

        pair_messages = self.message_fn(pair_features)
        object_messages = (edge_weights.unsqueeze(-1) * pair_messages).sum(dim=1)

        action_embeddings = self.action_encoder(actions).unsqueeze(1).expand(batch_size, num_slots, -1)
        mechanism_inputs = torch.cat((masked_slots, object_messages, action_embeddings), dim=-1)

        mechanism_logits = self.router(mechanism_inputs)
        mechanism_probs = torch.softmax(mechanism_logits, dim=-1)
        mechanism_deltas = torch.stack([mechanism(mechanism_inputs) for mechanism in self.mechanisms], dim=-2)

        mechanism_scale = normalized["mechanism_scale"]
        if mechanism_scale is not None:
            if mechanism_scale.dim() == 1:
                mechanism_scale = mechanism_scale.view(1, 1, self.num_mechanisms, 1)
            elif mechanism_scale.dim() == 2:
                mechanism_scale = mechanism_scale.view(batch_size, 1, self.num_mechanisms, 1)
            elif mechanism_scale.dim() == 3:
                mechanism_scale = mechanism_scale.unsqueeze(-1)
            mechanism_deltas = mechanism_deltas * mechanism_scale

        delta = (mechanism_probs.unsqueeze(-1) * mechanism_deltas).sum(dim=-2)
        next_slots = masked_slots + delta if self.residual else delta
        if object_mask is not None:
            next_slots = next_slots * object_mask.unsqueeze(-1)

        if not return_aux:
            return next_slots

        aux = {
            "relation_weights": edge_weights,
            "relation_logits": edge_logits,
            "mechanism_probs": mechanism_probs,
            "mechanism_logits": mechanism_logits,
            "mechanism_deltas": mechanism_deltas,
        }
        if object_mask is not None:
            aux["object_mask"] = object_mask
        return next_slots, aux


class DoWorldDynamicsModel(nn.Module):
    """Object-level counterfactual world model compatible with SOLD's dynamics interface."""

    def __init__(
        self,
        num_slots: int,
        slot_dim: int,
        sequence_length: int,
        action_dim: int,
        num_mechanisms: int = 8,
        hidden_dim: int = 256,
        message_dim: int = 128,
        action_embed_dim: int = 128,
        num_mlp_layers: int = 3,
        residual: bool = True,
        input_buffer_size: int = 5,
        teacher_forcing: bool = False,
        sparse_loss_weight: float = 1e-3,
        mechanism_entropy_loss_weight: float = 1e-3,
        mechanism_usage_loss_weight: float = 1e-3,
        mechanism_invariance_loss_weight: float = 0.05,
        counterfactual_mechanism_scale: float = 0.5,
        language_embedding_dim: Optional[int] = None,
        language_temperature: float = 0.07,
        language_loss_weight: float = 0.1,
    ) -> None:
        super().__init__()
        self.num_slots = num_slots
        self.slot_dim = slot_dim
        self.sequence_length = sequence_length
        self.action_dim = action_dim
        self.input_buffer_size = input_buffer_size
        self.teacher_forcing = teacher_forcing
        self.batched_processing = True
        self.num_mechanisms = num_mechanisms
        self.counterfactual_mechanism_scale = counterfactual_mechanism_scale

        self.transition = ObjectCausalMechanismLibrary(
            slot_dim=slot_dim,
            action_dim=action_dim,
            num_mechanisms=num_mechanisms,
            hidden_dim=hidden_dim,
            message_dim=message_dim,
            action_embed_dim=action_embed_dim,
            num_mlp_layers=num_mlp_layers,
            residual=residual,
        )

        self.sparse_loss_weight = sparse_loss_weight
        self.mechanism_entropy_loss_weight = mechanism_entropy_loss_weight
        self.mechanism_usage_loss_weight = mechanism_usage_loss_weight
        self.mechanism_invariance_loss_weight = mechanism_invariance_loss_weight
        self.language_temperature = language_temperature
        self.language_loss_weight = language_loss_weight
        self.language_projection = (
            nn.Linear(language_embedding_dim, hidden_dim)
            if language_embedding_dim is not None else None
        )
        self._last_aux: Optional[Dict[str, torch.Tensor]] = None

    def _stack_aux(self, aux_records: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        keys = aux_records[0].keys()
        return {key: torch.stack([record[key] for record in aux_records if key in record], dim=1) for key in keys}

    def step(
        self,
        slots: torch.Tensor,
        actions: torch.Tensor,
        intervention: Optional[Dict[str, Any]] = None,
        return_aux: bool = False,
    ) -> torch.Tensor | Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        return self.transition(slots, actions, intervention=intervention, return_aux=return_aux)

    def forward(
        self,
        slots: torch.Tensor,
        actions: torch.Tensor,
        intervention: Optional[Dict[str, Any]] = None,
        return_aux: bool = False,
    ) -> torch.Tensor | Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        batch_size, sequence_length, _, _ = slots.shape
        if actions.shape[1] < sequence_length:
            pad = actions[:, -1:].expand(batch_size, sequence_length - actions.shape[1], -1)
            actions = torch.cat((actions, pad), dim=1)

        predictions, aux_records = [], []
        for time_index in range(sequence_length):
            predicted_slots, aux = self.step(
                slots[:, time_index],
                actions[:, time_index],
                intervention=intervention,
                return_aux=True,
            )
            predictions.append(predicted_slots)
            aux_records.append(aux)

        predictions = torch.stack(predictions, dim=1)
        self._last_aux = self._stack_aux(aux_records)
        if return_aux:
            return predictions, self._last_aux
        return predictions

    def predict_slots(self, slots: torch.Tensor, actions: torch.Tensor, steps: int, num_context: int) -> torch.Tensor:
        actions = torch.nan_to_num(actions)
        if self.teacher_forcing:
            source_slots = slots[:, num_context - 1:num_context + steps - 1]
            source_actions = actions[:, num_context - 1:num_context + steps - 1]
            return self.forward(source_slots, source_actions)

        current_slots = slots[:, num_context - 1].clone()
        predicted_slots, aux_records = [], []
        for step_index in range(steps):
            action_index = min(num_context - 1 + step_index, actions.shape[1] - 1)
            current_slots, aux = self.step(current_slots, actions[:, action_index], return_aux=True)
            predicted_slots.append(current_slots)
            aux_records.append(aux)
        self._last_aux = self._stack_aux(aux_records)
        return torch.stack(predicted_slots, dim=1)

    def rollout_from_slots(
        self,
        slot_context: torch.Tensor,
        action_sequence: torch.Tensor,
        intervention: Optional[Dict[str, Any]] = None,
        return_aux: bool = False,
    ) -> torch.Tensor | Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        current_slots = slot_context[:, -1].clone()
        rollout_slots, aux_records = [current_slots], []
        for time_index in range(action_sequence.shape[1]):
            current_slots, aux = self.step(
                current_slots,
                action_sequence[:, time_index],
                intervention=intervention,
                return_aux=True,
            )
            rollout_slots.append(current_slots)
            aux_records.append(aux)
        rollout = torch.stack(rollout_slots, dim=1)
        aux = self._stack_aux(aux_records) if aux_records else {}
        if return_aux:
            return rollout, aux
        return rollout

    def auxiliary_losses(self) -> Dict[str, torch.Tensor]:
        zero = next(self.parameters()).sum() * 0.0
        if not self._last_aux:
            return {"do_world_regularization_loss": zero}

        relation_weights = self._last_aux["relation_weights"]
        mechanism_probs = self._last_aux["mechanism_probs"].clamp_min(1e-8)

        relation_sparse_loss = relation_weights.abs().mean()
        mechanism_entropy_loss = -(mechanism_probs * mechanism_probs.log()).sum(dim=-1).mean()

        usage = mechanism_probs.mean(dim=tuple(range(mechanism_probs.dim() - 1))).clamp_min(1e-8)
        uniform_log_prob = math.log(self.num_mechanisms)
        mechanism_usage_loss = (usage * (usage.log() + uniform_log_prob)).sum()

        context_usage = mechanism_probs.mean(dim=2)
        mechanism_invariance_loss = context_usage.var(dim=0, unbiased=False).mean() if context_usage.shape[0] > 1 else zero

        regularization_loss = (
            self.sparse_loss_weight * relation_sparse_loss
            + self.mechanism_entropy_loss_weight * mechanism_entropy_loss
            + self.mechanism_usage_loss_weight * mechanism_usage_loss
            + self.mechanism_invariance_loss_weight * mechanism_invariance_loss
        )

        return {
            "relation_sparse_loss": relation_sparse_loss,
            "mechanism_entropy_loss": mechanism_entropy_loss,
            "mechanism_usage_loss": mechanism_usage_loss,
            "mechanism_invariance_loss": mechanism_invariance_loss,
            "do_world_regularization_loss": regularization_loss,
        }

    def language_alignment_loss(
        self,
        language_embeddings: torch.Tensor,
        mechanism_labels: torch.Tensor,
    ) -> torch.Tensor:
        if self.language_projection is None:
            raise RuntimeError("language_embedding_dim must be set to use language_alignment_loss.")

        language_embeddings = language_embeddings.reshape(-1, language_embeddings.shape[-1])
        mechanism_labels = mechanism_labels.reshape(-1).long()
        projected_language = F.normalize(self.language_projection(language_embeddings), dim=-1)
        mechanism_embeddings = F.normalize(self.transition.mechanism_embeddings, dim=-1)
        logits = projected_language @ mechanism_embeddings.T / self.language_temperature
        return self.language_loss_weight * F.cross_entropy(logits, mechanism_labels)

    def intervention_consistency_loss(
        self,
        source_slots: torch.Tensor,
        actions: torch.Tensor,
        target_next_slots: torch.Tensor,
        intervention: Dict[str, Any],
    ) -> torch.Tensor:
        """Supervise a one-step do-query when true intervened targets are available."""
        predicted_next_slots = self.step(source_slots, actions, intervention=intervention)
        target_next_slots = target_next_slots.to(device=predicted_next_slots.device, dtype=predicted_next_slots.dtype)
        return F.mse_loss(predicted_next_slots, target_next_slots)

    def pseudo_intervention_losses(
        self,
        source_slots: torch.Tensor,
        actions: torch.Tensor,
        target_next_slots: torch.Tensor,
        relation_weights: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Build weak supervision for counterfactual queries from ordinary trajectories.

        Object removal can use a masked factual target: the removed object's slot should stay absent and all other
        slots should follow the observed transition. Weak relation cuts use an invariance target for the least active
        non-diagonal relation, which discourages the model from depending on incidental edges.
        """
        batch_size, num_slots, _ = source_slots.shape
        device, dtype = source_slots.device, source_slots.dtype
        zero = source_slots.sum() * 0.0

        object_mask = torch.ones(batch_size, num_slots, device=device, dtype=dtype)
        slot_activity = target_next_slots.detach().square().mean(dim=-1)
        remove_indices = slot_activity.argmin(dim=1)
        object_mask.scatter_(1, remove_indices.unsqueeze(1), 0.0)
        object_target = target_next_slots * object_mask.unsqueeze(-1)
        object_pred = self.step(source_slots, actions, intervention={"object_mask": object_mask})
        object_removal_loss = F.mse_loss(object_pred, object_target)

        weak_relation_cut_loss = zero
        if num_slots > 1:
            if relation_weights is None:
                _, aux = self.step(source_slots, actions, return_aux=True)
                relation_weights = aux["relation_weights"]
            candidate_weights = relation_weights.detach().clone()
            eye = torch.eye(num_slots, device=device, dtype=torch.bool).unsqueeze(0)
            candidate_weights = candidate_weights.masked_fill(eye, float("inf"))
            weak_relation_indices = candidate_weights.reshape(batch_size, -1).argmin(dim=1)
            relation_mask = torch.ones(batch_size, num_slots, num_slots, device=device, dtype=dtype)
            relation_mask.reshape(batch_size, -1).scatter_(1, weak_relation_indices.unsqueeze(1), 0.0)
            weak_relation_pred = self.step(source_slots, actions, intervention={"relation_mask": relation_mask})
            weak_relation_cut_loss = F.mse_loss(weak_relation_pred, target_next_slots.detach())

        return {
            "pseudo_object_removal_loss": object_removal_loss,
            "pseudo_weak_relation_cut_loss": weak_relation_cut_loss,
            "pseudo_intervention_loss": object_removal_loss + weak_relation_cut_loss,
        }

    @torch.no_grad()
    def generate_counterfactual_interventions(
        self,
        slots: torch.Tensor,
        num_interventions: int,
        relation_weights: Optional[torch.Tensor] = None,
    ) -> List[Dict[str, torch.Tensor]]:
        batch_size, num_slots, _ = slots.shape
        device, dtype = slots.device, slots.dtype
        interventions: List[Dict[str, torch.Tensor]] = []

        if relation_weights is None:
            _, aux = self.step(slots, torch.zeros(batch_size, self.action_dim, device=device, dtype=dtype), return_aux=True)
            relation_weights = aux["relation_weights"]

        for intervention_index in range(num_interventions):
            kind = intervention_index % 3
            if kind == 0:
                object_mask = torch.ones(batch_size, num_slots, device=device, dtype=dtype)
                object_index = num_slots - 1 - ((intervention_index // 3) % num_slots)
                object_mask[:, object_index] = 0.0
                interventions.append({"object_mask": object_mask})
            elif kind == 1:
                relation_mask = torch.ones(batch_size, num_slots, num_slots, device=device, dtype=dtype)
                candidate_weights = relation_weights.clone()
                eye = torch.eye(num_slots, device=device, dtype=torch.bool).unsqueeze(0)
                candidate_weights = candidate_weights.masked_fill(eye, 0.0)
                flat_indices = candidate_weights.reshape(batch_size, -1).argmax(dim=1)
                relation_mask.reshape(batch_size, -1).scatter_(1, flat_indices.unsqueeze(1), 0.0)
                interventions.append({"relation_mask": relation_mask})
            else:
                mechanism_scale = torch.ones(self.num_mechanisms, device=device, dtype=dtype)
                mechanism_index = (intervention_index // 3) % self.num_mechanisms
                mechanism_scale[mechanism_index] = self.counterfactual_mechanism_scale
                interventions.append({"mechanism_scale": mechanism_scale})

        return interventions


class CounterfactualMPCPlanner:
    """Cross-entropy-method MPC that scores factual and counterfactual latent rollouts."""

    def __init__(
        self,
        action_low: Optional[Sequence[float]] = None,
        action_high: Optional[Sequence[float]] = None,
        horizon: int = 15,
        num_candidates: int = 1024,
        num_elites: int = 64,
        num_iterations: int = 5,
        discount_factor: float = 0.99,
        robustness_weight: float = 0.5,
        gap_weight: float = 0.3,
        num_counterfactuals: int = 4,
        robustness_reduction: str = "min",
        init_std: float = 1.0,
        min_std: float = 0.05,
        momentum: float = 0.1,
        use_actor_prior: bool = True,
        discrete_actions: Optional[Sequence[Any]] = None,
    ) -> None:
        if discrete_actions is None and (action_low is None or action_high is None):
            raise ValueError("Either continuous action bounds or discrete_actions must be provided.")

        self.action_low = torch.as_tensor(action_low, dtype=torch.float32) if action_low is not None else None
        self.action_high = torch.as_tensor(action_high, dtype=torch.float32) if action_high is not None else None
        self.discrete_action_table = None
        if discrete_actions is not None:
            discrete_action_table = torch.as_tensor(discrete_actions, dtype=torch.float32)
            if discrete_action_table.dim() == 1:
                discrete_action_table = discrete_action_table.unsqueeze(-1)
            self.discrete_action_table = discrete_action_table
        self.horizon = horizon
        self.num_candidates = num_candidates
        self.num_elites = num_elites
        self.num_iterations = num_iterations
        self.discount_factor = discount_factor
        self.robustness_weight = robustness_weight
        self.gap_weight = gap_weight
        self.num_counterfactuals = num_counterfactuals
        self.robustness_reduction = robustness_reduction
        self.init_std = init_std
        self.min_std = min_std
        self.momentum = momentum
        self.use_actor_prior = use_actor_prior
        self.last_info: Dict[str, torch.Tensor] = {}

    def _action_bounds(self, device: torch.device, dtype: torch.dtype) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.action_low is None or self.action_high is None:
            raise RuntimeError("Continuous action bounds are not configured.")
        low = self.action_low.to(device=device, dtype=dtype)
        high = self.action_high.to(device=device, dtype=dtype)
        return low, high

    def _actor_prior(
        self,
        slot_history: torch.Tensor,
        dynamics_model: DoWorldDynamicsModel,
        actor: Optional[nn.Module],
    ) -> torch.Tensor:
        if actor is None or not self.use_actor_prior:
            action_dim = (
                self.action_low.numel()
                if self.action_low is not None
                else self.discrete_action_table.shape[-1]
            )
            return torch.zeros(self.horizon, action_dim, device=slot_history.device, dtype=slot_history.dtype)

        context = slot_history.clone()
        actions = []
        for _ in range(self.horizon):
            action_dist = actor(context, start=context.shape[1] - 1)
            action = action_dist.mode.squeeze(1)
            actions.append(action.squeeze(0))
            next_slots = dynamics_model.rollout_from_slots(context, action.unsqueeze(1))[:, -1:]
            context = torch.cat((context, next_slots), dim=1)
        return torch.stack(actions, dim=0)

    def _rollout_return(
        self,
        slots: torch.Tensor,
        reward_predictor: nn.Module,
    ) -> torch.Tensor:
        rewards = reward_predictor(slots, start=0).mean.squeeze(-1)
        discounts = torch.tensor(
            [self.discount_factor ** step for step in range(slots.shape[1])],
            device=slots.device,
            dtype=slots.dtype,
        )
        return (rewards * discounts.unsqueeze(0)).sum(dim=1)

    def _score_action_sequences(
        self,
        slot_history: torch.Tensor,
        action_sequences: torch.Tensor,
        dynamics_model: DoWorldDynamicsModel,
        reward_predictor: nn.Module,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        if not hasattr(dynamics_model, "rollout_from_slots"):
            raise TypeError("CounterfactualMPCPlanner requires a dynamics model with rollout_from_slots(...).")

        num_candidates = action_sequences.shape[0]
        context = slot_history.expand(num_candidates, -1, -1, -1)

        factual_rollout = dynamics_model.rollout_from_slots(context, action_sequences)[:, 1:]
        factual_return = self._rollout_return(factual_rollout, reward_predictor)

        if self.num_counterfactuals <= 0 or not hasattr(dynamics_model, "generate_counterfactual_interventions"):
            score = factual_return
            zero = torch.zeros_like(factual_return)
            return score, {
                "factual_return": factual_return,
                "robust_return": factual_return,
                "counterfactual_gap": zero,
            }

        with torch.no_grad():
            _, aux = dynamics_model.step(
                context[:, -1],
                torch.zeros(num_candidates, dynamics_model.action_dim, device=context.device, dtype=context.dtype),
                return_aux=True,
            )
            interventions = dynamics_model.generate_counterfactual_interventions(
                context[:, -1],
                self.num_counterfactuals,
                relation_weights=aux["relation_weights"],
            )

        counterfactual_returns = []
        for intervention in interventions:
            counterfactual_rollout = dynamics_model.rollout_from_slots(
                context, action_sequences, intervention=intervention)[:, 1:]
            counterfactual_returns.append(self._rollout_return(counterfactual_rollout, reward_predictor))

        counterfactual_returns = torch.stack(counterfactual_returns, dim=1)
        if self.robustness_reduction == "mean":
            robust_return = counterfactual_returns.mean(dim=1)
        elif self.robustness_reduction == "min":
            robust_return = counterfactual_returns.min(dim=1).values
        else:
            raise ValueError(f"Invalid robustness_reduction: {self.robustness_reduction}")

        counterfactual_gap = F.relu(factual_return.unsqueeze(1) - counterfactual_returns).mean(dim=1)
        score = factual_return + self.robustness_weight * robust_return - self.gap_weight * counterfactual_gap
        return score, {
            "factual_return": factual_return,
            "robust_return": robust_return,
            "counterfactual_gap": counterfactual_gap,
        }

    @torch.no_grad()
    def plan(
        self,
        slot_history: torch.Tensor,
        dynamics_model: DoWorldDynamicsModel,
        reward_predictor: nn.Module,
        actor: Optional[nn.Module] = None,
    ) -> torch.Tensor:
        if self.discrete_action_table is not None:
            return self._plan_discrete(slot_history, dynamics_model, reward_predictor)

        device, dtype = slot_history.device, slot_history.dtype
        low, high = self._action_bounds(device, dtype)
        action_dim = low.numel()
        num_elites = min(self.num_elites, self.num_candidates)

        mean = self._actor_prior(slot_history, dynamics_model, actor).to(device=device, dtype=dtype)
        std = torch.full((self.horizon, action_dim), self.init_std, device=device, dtype=dtype)

        best_info: Dict[str, torch.Tensor] = {}
        for _ in range(self.num_iterations):
            noise = torch.randn(self.num_candidates, self.horizon, action_dim, device=device, dtype=dtype)
            action_sequences = mean.unsqueeze(0) + std.unsqueeze(0) * noise
            action_sequences = torch.maximum(torch.minimum(action_sequences, high.view(1, 1, -1)), low.view(1, 1, -1))

            scores, info = self._score_action_sequences(slot_history, action_sequences, dynamics_model, reward_predictor)
            elite_indices = torch.topk(scores, k=num_elites).indices
            elite_actions = action_sequences[elite_indices]

            new_mean = elite_actions.mean(dim=0)
            new_std = elite_actions.std(dim=0, unbiased=False).clamp_min(self.min_std)
            mean = self.momentum * mean + (1.0 - self.momentum) * new_mean
            std = self.momentum * std + (1.0 - self.momentum) * new_std
            best_info = {key: value[elite_indices[0]].detach() for key, value in info.items()}
            best_info["score"] = scores[elite_indices[0]].detach()

        self.last_info = best_info
        return torch.maximum(torch.minimum(mean[0], high), low)

    @torch.no_grad()
    def _plan_discrete(
        self,
        slot_history: torch.Tensor,
        dynamics_model: DoWorldDynamicsModel,
        reward_predictor: nn.Module,
    ) -> torch.Tensor:
        if self.discrete_action_table is None:
            raise RuntimeError("Discrete action table is not configured.")

        device, dtype = slot_history.device, slot_history.dtype
        action_table = self.discrete_action_table.to(device=device, dtype=dtype)
        num_actions = action_table.shape[0]
        num_elites = min(self.num_elites, self.num_candidates)
        probs = torch.full((self.horizon, num_actions), 1.0 / num_actions, device=device, dtype=dtype)

        best_info: Dict[str, torch.Tensor] = {}
        for _ in range(self.num_iterations):
            sampled_indices = torch.stack([
                torch.multinomial(probs[time_index], self.num_candidates, replacement=True)
                for time_index in range(self.horizon)
            ], dim=1)
            action_sequences = action_table[sampled_indices]

            scores, info = self._score_action_sequences(slot_history, action_sequences, dynamics_model, reward_predictor)
            elite_indices = torch.topk(scores, k=num_elites).indices
            elite_action_indices = sampled_indices[elite_indices]
            elite_probs = F.one_hot(elite_action_indices, num_actions).to(dtype=dtype).mean(dim=0)
            probs = self.momentum * probs + (1.0 - self.momentum) * elite_probs
            probs = probs.clamp_min(1e-6)
            probs = probs / probs.sum(dim=-1, keepdim=True)

            best_info = {key: value[elite_indices[0]].detach() for key, value in info.items()}
            best_info["score"] = scores[elite_indices[0]].detach()

        self.last_info = best_info
        return action_table[probs[0].argmax()]

    @torch.no_grad()
    def __call__(
        self,
        slot_history: torch.Tensor,
        dynamics_model: DoWorldDynamicsModel,
        reward_predictor: nn.Module,
        actor: Optional[nn.Module] = None,
    ) -> torch.Tensor:
        return self.plan(slot_history, dynamics_model, reward_predictor, actor=actor)


def make_do_world_dynamics_model(
    num_slots: int,
    slot_dim: int,
    sequence_length: int,
    action_dim: int,
    token_dim: Optional[int] = None,
    hidden_dim: int = 256,
    num_layers: Optional[int] = None,
    num_heads: Optional[int] = None,
    residual: bool = True,
    input_buffer_size: int = 5,
    teacher_forcing: bool = False,
    num_mechanisms: int = 8,
    message_dim: int = 128,
    action_embed_dim: int = 128,
    num_mlp_layers: int = 3,
    sparse_loss_weight: float = 1e-3,
    mechanism_entropy_loss_weight: float = 1e-3,
    mechanism_usage_loss_weight: float = 1e-3,
    mechanism_invariance_loss_weight: float = 0.05,
    counterfactual_mechanism_scale: float = 0.5,
    language_embedding_dim: Optional[int] = None,
    language_temperature: float = 0.07,
    language_loss_weight: float = 0.1,
) -> DoWorldDynamicsModel:
    del token_dim, num_layers, num_heads
    return DoWorldDynamicsModel(
        num_slots=num_slots,
        slot_dim=slot_dim,
        sequence_length=sequence_length,
        action_dim=action_dim,
        num_mechanisms=num_mechanisms,
        hidden_dim=hidden_dim,
        message_dim=message_dim,
        action_embed_dim=action_embed_dim,
        num_mlp_layers=num_mlp_layers,
        residual=residual,
        input_buffer_size=input_buffer_size,
        teacher_forcing=teacher_forcing,
        sparse_loss_weight=sparse_loss_weight,
        mechanism_entropy_loss_weight=mechanism_entropy_loss_weight,
        mechanism_usage_loss_weight=mechanism_usage_loss_weight,
        mechanism_invariance_loss_weight=mechanism_invariance_loss_weight,
        counterfactual_mechanism_scale=counterfactual_mechanism_scale,
        language_embedding_dim=language_embedding_dim,
        language_temperature=language_temperature,
        language_loss_weight=language_loss_weight,
    )

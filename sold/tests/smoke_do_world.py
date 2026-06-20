import os
import tempfile
from pathlib import Path

import numpy as np
import torch

from aggregate_do_world_results import aggregate, write_csv, write_markdown
from datasets.do_world import DoWorldNPZDataset
from modeling.sold.do_world import CounterfactualMPCPlanner, make_do_world_dynamics_model
from run_do_world_experiments import build_commands
from utils.language import HashingLanguageEncoder, make_language_encoder


def test_core_do_world_paths() -> None:
    model = make_do_world_dynamics_model(
        num_slots=4,
        slot_dim=16,
        sequence_length=5,
        action_dim=3,
        hidden_dim=32,
        message_dim=16,
        action_embed_dim=8,
        num_mechanisms=3,
        num_mlp_layers=2,
        language_embedding_dim=12,
    )
    slots = torch.randn(2, 5, 4, 16)
    actions = torch.randn(2, 5, 3)
    pred = model.predict_slots(slots, actions, steps=2, num_context=3)
    assert pred.shape == (2, 2, 4, 16)

    losses = model.auxiliary_losses()
    assert "do_world_regularization_loss" in losses

    source = slots[:, 2]
    target = slots[:, 3]
    action = actions[:, 3]
    mask = torch.ones(2, 4)
    mask[:, -1] = 0
    int_loss = model.intervention_consistency_loss(
        source,
        action,
        target * mask.unsqueeze(-1),
        {"object_mask": mask},
    )
    assert int_loss.ndim == 0

    pseudo = model.pseudo_intervention_losses(source, action, target)
    assert "pseudo_intervention_loss" in pseudo
    assert pseudo["pseudo_intervention_loss"].ndim == 0

    encoder = HashingLanguageEncoder(12)
    emb = encoder.encode(("red block pushes blue cube", "remove distractor object"))
    lang_loss = model.language_alignment_loss(emb, torch.tensor((0, 1)))
    assert lang_loss.ndim == 0
    factory_encoder = make_language_encoder("hashing", embedding_dim=12)
    assert factory_encoder.encode(("push",)).shape == (1, 12)

    class Reward(torch.nn.Module):
        def forward(self, slots, start=0):
            class Dist:
                def __init__(self, mean):
                    self.mean = mean

            return Dist(slots.mean(dim=(-1, -2), keepdim=False).unsqueeze(-1))

    reward = Reward()
    continuous = CounterfactualMPCPlanner(
        action_low=(-1, -1, -1),
        action_high=(1, 1, 1),
        horizon=3,
        num_candidates=8,
        num_elites=2,
        num_iterations=2,
        num_counterfactuals=2,
        use_actor_prior=False,
    )
    continuous_action = continuous.plan(slots[:1, :3], model, reward)
    assert continuous_action.shape == (3,)

    discrete = CounterfactualMPCPlanner(
        discrete_actions=((1, 0, 0), (0, 1, 0), (0, 0, 1)),
        horizon=3,
        num_candidates=8,
        num_elites=2,
        num_iterations=2,
        num_counterfactuals=2,
    )
    discrete_action = discrete.plan(slots[:1, :3], model, reward)
    assert discrete_action.shape == (3,)


def test_do_world_npz_dataset() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        episode_dir = os.path.join(tmp, "toy", "train", "0")
        os.makedirs(episode_dir)
        np.savez(
            os.path.join(episode_dir, "episode.npz"),
            images=np.random.randint(0, 255, (6, 8, 8, 3), dtype=np.uint8),
            actions=np.random.randn(6, 2).astype("float32"),
            rewards=np.random.randn(6).astype("float32"),
            language_description=np.array(["red block pushes blue cube"] * 6),
            mechanism_label=np.array([1] * 6),
            intervention_object_mask=np.ones((6, 4), dtype="float32"),
        )
        dataset = DoWorldNPZDataset(tmp, "toy", "train", sequence_length=4, language_embedding_dim=16)
        sample = dataset[0]
        assert sample["obs"].shape == (4, 3, 8, 8)
        assert sample["action"].shape == (4, 2)
        assert sample["language_embedding"].shape == (4, 16)
        assert sample["mechanism_label"].shape == (4,)


def test_experiment_and_result_utilities() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = os.path.join(tmp, "experiments")
        commands = build_commands(
            benchmarks=("causalworld",),
            seeds=(0,),
            include_ablations=True,
            baselines=(),
            output_root=Path(root),
        )
        assert len(commands) == 5

        metrics_dir = os.path.join(root, "causalworld", "do_world", "seed_0")
        os.makedirs(metrics_dir)
        with open(os.path.join(metrics_dir, "do_world_metrics.jsonl"), "w") as file:
            file.write('{"episode_return": 1.0, "success_rate": 0.5, "prediction_error": 0.25}\n')

        rows = aggregate(Path(root), ["episode_return", "success_rate", "prediction_error"])
        assert len(rows) == 1
        assert rows[0]["episode_return_mean"] == 1.0
        write_csv(rows, Path(tmp) / "summary.csv")
        write_markdown(rows, Path(tmp) / "summary.md")
        assert os.path.exists(os.path.join(tmp, "summary.csv"))
        assert os.path.exists(os.path.join(tmp, "summary.md"))


if __name__ == "__main__":
    test_core_do_world_paths()
    test_do_world_npz_dataset()
    test_experiment_and_result_utilities()
    print("do_world smoke tests passed")

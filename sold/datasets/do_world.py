"""Offline Do-World episode datasets.

Expected directory layout:
    root/data_dir/train/0/episode.npz
    root/data_dir/train/1/episode.npz
    root/data_dir/val/0/episode.npz

Required NPZ keys:
    obs or images, action or actions

Optional NPZ keys:
    reward or rewards
    intervention_obs, intervention_next_obs, intervention_action
    intervention_source_slots, intervention_target_slots, intervention_next_slots
    intervention_object_mask, intervention_relation_mask, intervention_mechanism_scale
    language_embedding
    language_description and mechanism_label
"""

import os
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from torch.utils.data import Dataset

from utils.language import HashingLanguageEncoder


class DoWorldNPZDataset(Dataset):
    def __init__(
        self,
        path: str,
        data_dir: str,
        split: str,
        sequence_length: int,
        language_embedding_dim: int = 512,
    ) -> None:
        super().__init__()
        self.path = path
        self.data_dir = data_dir
        self.split = split
        self.sequence_length = sequence_length
        self.split_path = os.path.join(self.path, self.data_dir, split)
        self.language_encoder = HashingLanguageEncoder(language_embedding_dim)

        self.episode_paths: List[str] = []
        for episode_dir in os.listdir(self.split_path):
            episode_path = os.path.join(self.split_path, episode_dir, "episode.npz")
            if os.path.isfile(episode_path):
                self.episode_paths.append(episode_path)
        self.episode_paths.sort()

    def __len__(self) -> int:
        return len(self.episode_paths)

    def _slice(self, value: np.ndarray, start: int, end: int) -> np.ndarray:
        if value.ndim > 0 and value.shape[0] >= end:
            return value[start:end]
        return value

    def _images_to_tensor(self, value: np.ndarray) -> torch.Tensor:
        tensor = torch.from_numpy(value)
        if tensor.dim() == 4 and tensor.shape[-1] in (1, 3, 4):
            tensor = tensor[..., :3].permute(0, 3, 1, 2)
        return tensor

    def _get_first(self, episode: np.lib.npyio.NpzFile, keys: List[str]) -> Optional[np.ndarray]:
        for key in keys:
            if key in episode:
                return episode[key]
        return None

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        episode = np.load(self.episode_paths[index], allow_pickle=True)
        observations = self._get_first(episode, ["obs", "images"])
        actions = self._get_first(episode, ["action", "actions"])
        rewards = self._get_first(episode, ["reward", "rewards"])
        if observations is None or actions is None:
            raise KeyError("DoWorldNPZDataset requires `obs`/`images` and `action`/`actions` keys.")

        max_start = max(1, len(observations) - self.sequence_length)
        start = np.random.randint(0, max_start) if self.split == "train" else 0
        end = start + self.sequence_length

        sample: Dict[str, torch.Tensor] = {
            "obs": self._images_to_tensor(self._slice(observations, start, end)),
            "action": torch.from_numpy(self._slice(actions, start, end)).float(),
        }
        if rewards is None:
            sample["reward"] = torch.full((self.sequence_length,), float("nan"))
        else:
            sample["reward"] = torch.from_numpy(self._slice(rewards, start, end)).float()

        optional_tensor_keys = [
            "intervention_obs",
            "intervention_next_obs",
            "intervention_action",
            "intervention_actions",
            "intervention_source_slots",
            "intervention_slots",
            "intervention_target_slots",
            "intervention_next_slots",
            "intervention_object_mask",
            "intervention_relation_mask",
            "intervention_mechanism_scale",
            "language_embedding",
            "mechanism_label",
        ]
        for key in optional_tensor_keys:
            if key not in episode:
                continue
            value = self._slice(episode[key], start, end)
            tensor = self._images_to_tensor(value) if key.endswith("obs") else torch.from_numpy(value)
            sample[key] = tensor.float() if tensor.dtype.is_floating_point else tensor

        if "language_embedding" not in sample and "language_description" in episode:
            descriptions = self._slice(episode["language_description"], start, end)
            descriptions = [str(description) for description in np.asarray(descriptions).reshape(-1)]
            sample["language_embedding"] = self.language_encoder.encode(descriptions)

        return sample


def load_do_world_npz_dataset(path: str, data_dir: str, split: str, sequence_length: int, **kwargs) -> DoWorldNPZDataset:
    return DoWorldNPZDataset(path, data_dir, split, sequence_length, **kwargs)

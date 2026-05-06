"""
Geopolitical GNN Dataset
========================
PyTorch Dataset that yields sliding windows of temporal graphs.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


class GeopoliticalDataset(Dataset):
    """
    Parameters
    ----------
    data_dir : str
        Directory containing node_features.npy, edge_features.npy,
        edge_labels.npy, valid_mask.npy, metadata.json.
    temporal_window : int
        Number of previous time-steps to use as input (default 12).
    split : {"train","val","test"}
        Which time-ordered split to return.
    split_strategy : {"time", "real_targets_only"}
        "time" uses all available target months chronologically.
        "real_targets_only" restricts target labels to real GDELT months when
        augmented synthetic history is present.
    """

    def __init__(
        self,
        data_dir: str = "./gdelt_processed_data",
        temporal_window: int = 12,
        split: str = "train",
        split_strategy: str = "time",
    ):
        super().__init__()
        self.temporal_window = temporal_window
        self.split = split
        self.split_strategy = split_strategy

        data_dir = Path(data_dir)
        self.node_features = np.load(data_dir / "node_features.npy")
        self.edge_features = np.load(data_dir / "edge_features.npy")
        self.edge_labels = np.load(data_dir / "edge_labels.npy")
        self.valid_mask = np.load(data_dir / "valid_mask.npy")

        with open(data_dir / "metadata.json") as f:
            metadata = json.load(f)
        self.country_indices = metadata["country_indices"]
        self.periods = metadata["periods"]
        self.synthetic_months = int(metadata.get("synthetic_months", 0))

        if split_strategy == "time":
            all_indices = list(range(temporal_window, len(self.periods)))
        elif split_strategy == "real_targets_only":
            all_indices = [
                idx for idx in range(temporal_window, len(self.periods))
                if idx >= self.synthetic_months
            ]
        else:
            raise ValueError(f"Unknown split_strategy: {split_strategy}")

        self.indices = self._split_indices(all_indices, split)
        split_desc = "real-target-only" if split_strategy == "real_targets_only" else "full chronological"
        print(f"[Dataset] {split.title()} periods: {len(self.indices)} ({split_desc})")

    def _split_indices(self, indices: list[int], split: str) -> list[int]:
        n = len(indices)
        n_train = max(1, int(0.6 * n))
        n_val = max(1, int(0.2 * n))
        if n_train + n_val >= n:
            n_train = max(1, n - 2)
            n_val = 1

        if split == "train":
            return indices[:n_train]
        if split == "val":
            return indices[n_train:n_train + n_val]
        if split == "test":
            return indices[n_train + n_val:]
        raise ValueError(f"Unknown split: {split}")

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, ...]:
        t = self.indices[idx]
        node_x = torch.from_numpy(self.node_features[t - self.temporal_window:t])
        edge_x = torch.from_numpy(self.edge_features[t - self.temporal_window:t])
        label = torch.from_numpy(self.edge_labels[t])
        mask = torch.from_numpy(self.valid_mask[t])
        adj = (edge_x[:, :, :, 0] + edge_x[:, :, :, 1]).sum(dim=0) > 0
        return node_x, edge_x, label, mask, adj

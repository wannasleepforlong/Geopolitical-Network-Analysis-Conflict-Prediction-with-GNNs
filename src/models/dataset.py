"""
Geopolitical GNN Dataset
========================
PyTorch Dataset that yields sliding windows of temporal graphs.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Tuple, Optional

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
        Which 70/10/20 split to return.
    """

    def __init__(
        self,
        data_dir: str = "./gdelt_processed_data",
        temporal_window: int = 12,
        split: str = "train",
    ):
        super().__init__()
        self.temporal_window = temporal_window
        self.split = split

        data_dir = Path(data_dir)
        self.node_features = np.load(data_dir / "node_features.npy")      # (T, N, 6)
        self.edge_features = np.load(data_dir / "edge_features.npy")      # (T, N, N, 5)
        self.edge_labels = np.load(data_dir / "edge_labels.npy")          # (T, N, N)
        self.valid_mask = np.load(data_dir / "valid_mask.npy")            # (T, N, N)

        with open(data_dir / "metadata.json") as f:
            metadata = json.load(f)
        self.country_indices = metadata["country_indices"]
        self.periods = metadata["periods"]

        # We can only predict at t when we have history [t-window ... t-1]
        # Labels are at t (shifted relative to features).
        # So valid indices range from temporal_window to T-1.
        self.indices = list(range(temporal_window, len(self.periods)))

        # Train/val/test split on *time* (not random — temporal ordering matters)
        n = len(self.indices)
        n_train = int(0.7 * n)
        n_val = int(0.1 * n)
        if split == "train":
            self.indices = self.indices[:n_train]
            print(f"[Dataset] Train periods: {len(self.indices)}")
        elif split == "val":
            self.indices = self.indices[n_train:n_train + n_val]
            print(f"[Dataset] Val periods: {len(self.indices)}")
        elif split == "test":
            self.indices = self.indices[n_train + n_val:]
            print(f"[Dataset] Test periods: {len(self.indices)}")
        else:
            raise ValueError(f"Unknown split: {split}")

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, ...]:
        t = self.indices[idx]
        # Input window: [t - window, ..., t - 1]
        node_x = torch.from_numpy(self.node_features[t - self.temporal_window : t])   # (W, N, F_n)
        edge_x = torch.from_numpy(self.edge_features[t - self.temporal_window : t])   # (W, N, N, F_e)
        label = torch.from_numpy(self.edge_labels[t])                                   # (N, N)
        mask = torch.from_numpy(self.valid_mask[t])                                   # (N, N)
        # Adjacency for GNN: any interaction in current window
        adj = (edge_x[:, :, :, 0] + edge_x[:, :, :, 1]).sum(dim=0) > 0                # (N, N)
        return node_x, edge_x, label, mask, adj

"""
GNN Data Preprocessor
=====================
Converts temporal networks into numpy tensors for PyTorch training.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Any

import numpy as np


class GNNDataPreprocessor:
    """Feature engineering and label generation for geopolitical networks."""

    def __init__(self, output_dir: str = "./gdelt_processed_data") -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def process_and_save(
        self,
        networks: Dict[str, Any],
        country_indices: Dict[str, int],
    ) -> Dict[str, Any]:
        """
        Create node features, edge features, labels, and valid mask.

        Returns
        -------
        dict with keys: node_features, edge_features, edge_labels, valid_mask, metadata
        """
        periods = sorted(networks.keys())
        num_periods = len(periods)
        num_countries = len(country_indices)

        node_features = np.zeros((num_periods, num_countries, 6), dtype=np.float32)
        edge_features = np.zeros((num_periods, num_countries, num_countries, 5), dtype=np.float32)
        edge_labels = np.zeros((num_periods, num_countries, num_countries), dtype=np.float32)
        valid_mask = np.zeros((num_periods, num_countries, num_countries), dtype=bool)

        goldstein_avgs: list[np.ndarray] = []
        event_presence: list[np.ndarray] = []

        for t, period in enumerate(periods):
            net = networks[period]
            conflict = net["adjacency_conflict"]
            coop = net["adjacency_cooperation"]

            tone_avg = np.divide(
                net["adjacency_tone_sum"],
                net["adjacency_tone_count"],
                out=np.zeros_like(net["adjacency_tone_sum"]),
                where=net["adjacency_tone_count"] > 0,
            )
            gold_avg = np.divide(
                net["adjacency_goldstein_sum"],
                net["adjacency_goldstein_count"],
                out=np.zeros_like(net["adjacency_goldstein_sum"]),
                where=net["adjacency_goldstein_count"] > 0,
            )

            total_out = conflict + coop
            out_counts = total_out.sum(axis=1)
            in_counts = total_out.sum(axis=0)

            # Node features
            node_features[t, :, 0] = conflict.sum(axis=1)
            node_features[t, :, 1] = conflict.sum(axis=0)
            node_features[t, :, 2] = coop.sum(axis=1)
            node_features[t, :, 3] = np.divide(
                net["adjacency_tone_sum"].sum(axis=1),
                np.maximum(1, out_counts),
                out=np.zeros(num_countries, dtype=np.float32),
                where=out_counts > 0,
            )
            node_features[t, :, 4] = np.divide(
                net["adjacency_tone_sum"].sum(axis=0),
                np.maximum(1, in_counts),
                out=np.zeros(num_countries, dtype=np.float32),
                where=in_counts > 0,
            )
            node_features[t, :, 5] = out_counts + in_counts

            # Edge features
            edge_features[t, :, :, 0] = conflict
            edge_features[t, :, :, 1] = coop
            edge_features[t, :, :, 2] = tone_avg
            edge_features[t, :, :, 3] = gold_avg
            edge_features[t, :, :, 4] = np.divide(
                np.abs(conflict - coop),
                np.maximum(1, conflict + coop),
                out=np.zeros_like(conflict, dtype=np.float32),
                where=(conflict + coop) > 0,
            )

            # Valid mask: edges that have at least one event, no self-loops
            valid_mask[t] = total_out > 0
            np.fill_diagonal(valid_mask[t], False)

            goldstein_avgs.append(gold_avg)
            event_presence.append(total_out > 0)

        # Labels: conflict escalation if Goldstein drops >0.5 next period
        for t in range(num_periods - 1):
            cur = goldstein_avgs[t]
            nxt = goldstein_avgs[t + 1]
            mask = event_presence[t] | event_presence[t + 1]
            edge_labels[t] = (
                (nxt < cur - 0.5) & mask
            ).astype(np.float32)
        np.fill_diagonal(edge_labels[-1], 0.0)

        # Save
        np.save(self.output_dir / "node_features.npy", node_features)
        np.save(self.output_dir / "edge_features.npy", edge_features)
        np.save(self.output_dir / "edge_labels.npy", edge_labels)
        np.save(self.output_dir / "valid_mask.npy", valid_mask)

        metadata = {
            "country_indices": country_indices,
            "reverse_indices": {
                str(i): code for i, code in sorted(
                    {v: k for k, v in country_indices.items()}.items()
                )
            },
            "periods": periods,
        }
        with open(self.output_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

        print(f"[Preprocessor] Saved tensors to {self.output_dir}")
        return {
            "node_features": node_features,
            "edge_features": edge_features,
            "edge_labels": edge_labels,
            "valid_mask": valid_mask,
            "metadata": metadata,
        }

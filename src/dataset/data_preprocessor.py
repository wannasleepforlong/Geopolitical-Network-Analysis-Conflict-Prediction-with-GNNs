from pathlib import Path
from typing import Dict, Any
import json
import logging 

import numpy as np


class GNNDataPreprocessor:
    def __init__(self, logger = None, output_dir: str = "./gdelt_processed_data"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logger or logging.getLogger("null")
        self.logger.addHandler(logging.NullHandler())

    def process_and_save(self, networks: Dict[str, Any], country_indices: Dict[str, int]) -> Dict[str, Any]:
        periods = sorted(networks.keys())
        num_periods = len(periods)
        num_countries = len(country_indices)

        node_features = np.zeros((num_periods, num_countries, 6), dtype=np.float32)
        edge_features = np.zeros((num_periods, num_countries, num_countries, 5), dtype=np.float32)
        edge_labels = np.zeros((num_periods, num_countries, num_countries), dtype=np.float32)
        valid_mask = np.zeros((num_periods, num_countries, num_countries), dtype=bool)

        avg_goldstein = []
        event_presence = []

        for t, period in enumerate(periods):
            net = networks[period]
            conflict = net['adjacency_conflict']
            coop = net['adjacency_cooperation']
            tone_avg = np.divide(
                net['adjacency_tone_sum'],
                net['adjacency_tone_count'],
                out=np.zeros_like(net['adjacency_tone_sum']),
                where=net['adjacency_tone_count'] > 0,
            )
            goldstein_avg = np.divide(
                net['adjacency_goldstein_sum'],
                net['adjacency_goldstein_count'],
                out=np.zeros_like(net['adjacency_goldstein_sum']),
                where=net['adjacency_goldstein_count'] > 0,
            )
            total_outgoing = conflict + coop
            outgoing_counts = total_outgoing.sum(axis=1)
            incoming_counts = total_outgoing.sum(axis=0)

            node_features[t, :, 0] = conflict.sum(axis=1)
            node_features[t, :, 1] = conflict.sum(axis=0)
            node_features[t, :, 2] = coop.sum(axis=1)
            node_features[t, :, 3] = np.divide(
                net['adjacency_tone_sum'].sum(axis=1),
                np.maximum(1, outgoing_counts),
                out=np.zeros(num_countries, dtype=np.float32),
                where=outgoing_counts > 0,
            )
            node_features[t, :, 4] = np.divide(
                net['adjacency_tone_sum'].sum(axis=0),
                np.maximum(1, incoming_counts),
                out=np.zeros(num_countries, dtype=np.float32),
                where=incoming_counts > 0,
            )
            node_features[t, :, 5] = outgoing_counts + incoming_counts

            edge_features[t, :, :, 0] = conflict
            edge_features[t, :, :, 1] = coop
            edge_features[t, :, :, 2] = tone_avg
            edge_features[t, :, :, 3] = goldstein_avg
            edge_features[t, :, :, 4] = np.divide(
                np.abs(conflict - coop),
                np.maximum(1, conflict + coop),
                out=np.zeros_like(conflict, dtype=np.float32),
                where=(conflict + coop) > 0,
            )

            valid_mask[t] = (total_outgoing > 0)
            np.fill_diagonal(valid_mask[t], False)

            avg_goldstein.append(goldstein_avg)
            event_presence.append(total_outgoing > 0)

        for t in range(num_periods - 1):
            current_gold = avg_goldstein[t]
            next_gold = avg_goldstein[t + 1]
            label_mask = (event_presence[t] | event_presence[t + 1])
            edge_labels[t] = (
                (next_gold < current_gold - 0.5) & label_mask
            ).astype(np.float32)
        np.fill_diagonal(edge_labels[-1], 0.0)

        np.save(self.output_dir / 'node_features.npy', node_features)
        np.save(self.output_dir / 'edge_features.npy', edge_features)
        np.save(self.output_dir / 'edge_labels.npy', edge_labels)
        np.save(self.output_dir / 'valid_mask.npy', valid_mask)

        metadata = {
            'country_indices': country_indices,
            'reverse_indices': {str(i): code for i, code in sorted({v: k for k, v in country_indices.items()}.items())},
            'periods': periods,
        }
        with open(self.output_dir / 'metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)

        self.logger.info(f"Saved processed data to {self.output_dir}")
        return {
            'node_features': node_features,
            'edge_features': edge_features,
            'edge_labels': edge_labels,
            'valid_mask': valid_mask,
            'metadata': metadata,
        }

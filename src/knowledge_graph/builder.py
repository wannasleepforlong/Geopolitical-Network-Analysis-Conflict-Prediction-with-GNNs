"""
Knowledge Graph Builder
=======================
Transforms temporal geopolitical networks into a rich, semantic
NetworkX MultiDiGraph (KG) with typed nodes and edges.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Any, Optional, List

import networkx as nx
import numpy as np


class KnowledgeGraphBuilder:
    """
    Build a semantic knowledge graph from GDELT processed data.

    Node types: Country, Region, Theme
    Edge types: CONFLICT_WITH (red), COOPERATE_WITH (green), HAS_EVENT,
                HAS_THEME, BELONGS_TO_REGION
    """

    # Region mapping for the 20 major powers
    REGION_MAP: Dict[str, str] = {
        "USA": "North America", "CAN": "North America",
        "CHN": "East Asia", "JPN": "East Asia", "KOR": "East Asia",
        "IND": "South Asia", "PAK": "South Asia",
        "RUS": "Eurasia", "UKR": "Eurasia",
        "GBR": "Europe", "FRA": "Europe", "DEU": "Europe",
        "ISR": "Middle East", "IRN": "Middle East", "TUR": "Middle East",
        "SAU": "Middle East", "EGY": "Middle East",
        "AUS": "Oceania",
        "BRA": "South America",
        "ZAF": "Africa",
    }

    def __init__(self, data_dir: str = "./gdelt_processed_data") -> None:
        self.data_dir = Path(data_dir)
        with open(self.data_dir / "metadata.json") as f:
            self.metadata = json.load(f)
        self.country_indices: Dict[str, int] = self.metadata["country_indices"]
        self.reverse_indices: Dict[int, str] = {
            int(k): v for k, v in self.metadata["reverse_indices"].items()
        }
        self.periods: List[str] = self.metadata["periods"]

        # Load tensors
        self.node_features = np.load(self.data_dir / "node_features.npy")      # (T, N, 6)
        self.edge_features = np.load(self.data_dir / "edge_features.npy")      # (T, N, N, 5)
        self.edge_labels = np.load(self.data_dir / "edge_labels.npy")          # (T, N, N)
        self.valid_mask = np.load(self.data_dir / "valid_mask.npy")            # (T, N, N)

        self.graphs: Dict[str, nx.MultiDiGraph] = {}

    def build_graph_for_period(self, period: str) -> nx.MultiDiGraph:
        """Build a KG for a single time period."""
        t = self.periods.index(period)
        G = nx.MultiDiGraph()

        # Country nodes
        for code, idx in self.country_indices.items():
            G.add_node(
                code,
                node_type="Country",
                region=self.REGION_MAP.get(code, "Unknown"),
                out_conflict=float(self.node_features[t, idx, 0]),
                in_conflict=float(self.node_features[t, idx, 1]),
                out_coop=float(self.node_features[t, idx, 2]),
                avg_tone=float(self.node_features[t, idx, 3]),
                total_degree=float(self.node_features[t, idx, 5]),
            )

        # Region nodes (connect once per period)
        seen_regions = set()
        for code in self.country_indices:
            region = self.REGION_MAP.get(code, "Unknown")
            if region not in seen_regions:
                G.add_node(region, node_type="Region")
                seen_regions.add(region)
            G.add_edge(code, region, edge_type="BELONGS_TO_REGION")

        # Country-pair edges
        conflict_mat = self.edge_features[t, :, :, 0]
        coop_mat = self.edge_features[t, :, :, 1]
        tone_mat = self.edge_features[t, :, :, 2]
        goldstein_mat = self.edge_features[t, :, :, 3]
        asym_mat = self.edge_features[t, :, :, 4]
        valid = self.valid_mask[t]

        for i in range(len(self.country_indices)):
            for j in range(len(self.country_indices)):
                if i == j or not valid[i, j]:
                    continue
                src = self.reverse_indices[i]
                dst = self.reverse_indices[j]

                # Conflict edge
                if conflict_mat[i, j] > 0:
                    G.add_edge(
                        src, dst,
                        edge_type="CONFLICT_WITH",
                        weight=float(conflict_mat[i, j]),
                        tone=float(tone_mat[i, j]),
                        goldstein=float(goldstein_mat[i, j]),
                        period=period,
                    )
                # Cooperation edge
                if coop_mat[i, j] > 0:
                    G.add_edge(
                        src, dst,
                        edge_type="COOPERATE_WITH",
                        weight=float(coop_mat[i, j]),
                        tone=float(tone_mat[i, j]),
                        goldstein=float(goldstein_mat[i, j]),
                        period=period,
                    )

        self.graphs[period] = G
        return G

    def build_all_graphs(self) -> Dict[str, nx.MultiDiGraph]:
        for period in self.periods:
            self.build_graph_for_period(period)
        return self.graphs

    def get_summary(self, period: str) -> Dict[str, Any]:
        """Return high-level stats for a KG snapshot."""
        G = self.graphs.get(period)
        if G is None:
            G = self.build_graph_for_period(period)
        nodes_by_type = {}
        for _, d in G.nodes(data=True):
            nodes_by_type[d.get("node_type", "Unknown")] = nodes_by_type.get(d.get("node_type"), 0) + 1
        edges_by_type = {}
        for _, _, d in G.edges(data=True):
            edges_by_type[d.get("edge_type", "Unknown")] = edges_by_type.get(d.get("edge_type"), 0) + 1
        return {
            "period": period,
            "num_nodes": G.number_of_nodes(),
            "num_edges": G.number_of_edges(),
            "nodes_by_type": nodes_by_type,
            "edges_by_type": edges_by_type,
            "density": nx.density(G),
        }

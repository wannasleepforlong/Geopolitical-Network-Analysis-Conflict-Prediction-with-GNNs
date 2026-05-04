"""
Knowledge Graph Exporter
==========================
Export NetworkX MultiDiGraph to interactive PyVis HTML,
Cytoscape JSON, or static image.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Any, Optional

import networkx as nx
import numpy as np


class KnowledgeGraphExporter:
    """Export KG snapshots to various formats."""

    # Country coordinates (approx centroids) for geospatial layouts
    COORDS: Dict[str, tuple] = {
        "USA": (37.0902, -95.7129), "CAN": (56.1304, -106.3468),
        "CHN": (35.8617, 104.1954), "JPN": (36.2048, 138.2529), "KOR": (35.9078, 127.7669),
        "IND": (20.5937, 78.9629), "PAK": (30.3753, 69.3451),
        "RUS": (61.5240, 105.3188), "UKR": (48.3794, 31.1656),
        "GBR": (55.3781, -3.4360), "FRA": (46.2276, 2.2137), "DEU": (51.1657, 10.4515),
        "ISR": (31.0461, 34.8516), "IRN": (32.4279, 53.6880), "TUR": (38.9637, 35.2433),
        "SAU": (23.8859, 45.0792), "EGY": (26.8206, 30.8025),
        "AUS": (-25.2744, 133.7751),
        "BRA": (-14.2350, -51.9253),
        "ZAF": (-30.5595, 22.9375),
    }

    def __init__(self, output_dir: str = "./gdelt_visualizations"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def to_pyvis_html(
        self,
        graph: nx.MultiDiGraph,
        period: str,
        title: Optional[str] = None,
        height: str = "700px",
    ) -> str:
        """Return PyVis-generated HTML string for embedding."""
        try:
            from pyvis.network import Network
        except ImportError:
            return "<p>PyVis not installed. Run: pip install pyvis</p>"

        net = Network(height=height, directed=True, notebook=False, heading=title or f"KG {period}")
        net.barnes_hut(gravity=-2000, central_gravity=0.3, spring_length=150)

        # Nodes
        for node, data in graph.nodes(data=True):
            ntype = data.get("node_type", "Unknown")
            if ntype == "Country":
                size = 10 + min(30, data.get("total_degree", 0) * 2)
                color = self._country_color(data.get("region", "Unknown"))
                label = f"{node}"
                title_tooltip = (
                    f"{node} — {data.get('region', '')}\n"
                    f"Conflicts out: {data.get('out_conflict', 0):.0f}\n"
                    f"Coop out: {data.get('out_coop', 0):.0f}\n"
                    f"Avg tone: {data.get('avg_tone', 0):.1f}\n"
                    f"Degree: {data.get('total_degree', 0):.0f}"
                )
            else:
                size = 8
                color = "#95a5a6"
                label = node
                title_tooltip = f"{ntype}: {node}"

            net.add_node(
                node,
                label=label,
                size=size,
                color=color,
                title=title_tooltip,
            )

        # Edges (aggregate multi-edges into one visual edge per type)
        for u, v, key, data in graph.edges(data=True, keys=True):
            etype = data.get("edge_type", "UNKNOWN")
            if etype == "BELONGS_TO_REGION":
                continue  # skip region edges for clarity
            weight = data.get("weight", 1)
            if etype == "CONFLICT_WITH":
                color = "#e74c3c"
                width = min(8, 1 + weight)
                dashes = False
            elif etype == "COOPERATE_WITH":
                color = "#2ecc71"
                width = min(6, 1 + weight * 0.5)
                dashes = False
            else:
                color = "#95a5a6"
                width = 1
                dashes = True

            net.add_edge(
                u, v,
                color=color,
                width=width,
                title=f"{etype}\nweight={weight:.1f}\ntone={data.get('tone', 0):.1f}",
                arrows="to",
                dashes=dashes,
            )

        # Save to file and return HTML string
        out_file = self.output_dir / f"kg_{period}.html"
        net.save_graph(str(out_file))
        return out_file.read_text(encoding="utf-8")

    def to_cytoscape_json(self, graph: nx.MultiDiGraph, period: str) -> str:
        """Export Cytoscape.js-compatible JSON."""
        elements: list[dict] = []
        for node, data in graph.nodes(data=True):
            elements.append({
                "data": {"id": node, "label": node, **data},
            })
        for u, v, key, data in graph.edges(data=True, keys=True):
            elements.append({
                "data": {"source": u, "target": v, "id": f"{u}-{v}-{key}", **data},
            })
        out_file = self.output_dir / f"kg_{period}.json"
        out_file.write_text(json.dumps(elements, indent=2), encoding="utf-8")
        return str(out_file)

    @staticmethod
    def _country_color(region: str) -> str:
        palette = {
            "North America": "#3498db",
            "East Asia": "#e67e22",
            "South Asia": "#9b59b6",
            "Eurasia": "#1abc9c",
            "Europe": "#2ecc71",
            "Middle East": "#f1c40f",
            "Oceania": "#e74c3c",
            "South America": "#34495e",
            "Africa": "#16a085",
            "Unknown": "#95a5a6",
        }
        return palette.get(region, "#95a5a6")

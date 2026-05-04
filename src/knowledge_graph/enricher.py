"""
Knowledge Graph Enricher
========================
Enrich KG nodes / edges with LLM-generated summaries and theme tags.
"""
from __future__ import annotations

from typing import Dict, Any, Optional
import networkx as nx


class KnowledgeGraphEnricher:
    """
    Uses an LLM client to enrich the KG with natural-language summaries
    and extracted themes.
    """

    THEME_LIST = [
        "MILITARY", "DIPLOMACY", "TRADE", "SANCTIONS", "TERRITORY",
        "HUMAN_RIGHTS", "ENERGY", "CYBER", "NUCLEAR", "MIGRATION",
        "CLIMATE", "HEALTH", "TERRORISM", "ELECTION",
    ]

    def __init__(self, llm_client: Optional[Any] = None) -> None:
        self.llm = llm_client

    def enrich_pair_summary(
        self,
        graph: nx.MultiDiGraph,
        src: str,
        dst: str,
    ) -> str:
        """
        Summarise the relationship between two countries using
        edge metadata and (optionally) an LLM.
        """
        edges = graph.get_edge_data(src, dst)
        if not edges:
            return f"No recorded interactions between {src} and {dst} in this period."

        conflict_count = sum(1 for _, d in edges.items() if d["edge_type"] == "CONFLICT_WITH")
        coop_count = sum(1 for _, d in edges.items() if d["edge_type"] == "COOPERATE_WITH")
        avg_tone = sum(d.get("tone", 0) for d in edges.values()) / len(edges)
        avg_goldstein = sum(d.get("goldstein", 0) for d in edges.values()) / len(edges)

        summary = (
            f"**{src} ↔ {dst}**: "
            f"{conflict_count} conflict event(s), {coop_count} cooperation event(s). "
            f"Avg tone {avg_tone:.1f}, Goldstein {avg_goldstein:.1f}."
        )

        if self.llm is not None:
            prompt = f"""Given these geopolitical statistics between {src} and {dst}:
- Conflict events: {conflict_count}
- Cooperation events: {coop_count}
- Average media tone: {avg_tone:.1f}
- Average Goldstein conflict score: {avg_goldstein:.1f}
Write a 1-sentence summary of the relationship dynamic."""
            try:
                extra = self.llm.chat_json(
                    [{"role": "user", "content": prompt}],
                    temperature=0.3,
                )
                if isinstance(extra, dict) and "summary" in extra:
                    summary += f"\n\nLLM insight: {extra['summary']}"
            except Exception:
                pass
        return summary

    def tag_country_themes(self, graph: nx.MultiDiGraph, country: str) -> list[str]:
        """Simple heuristic theme extraction from edge types and counts."""
        edges = [
            d for _, _, d in graph.edges(country, data=True)
            if d.get("edge_type") in ("CONFLICT_WITH", "COOPERATE_WITH")
        ]
        if not edges:
            return []
        # Heuristic: very negative Goldstein → MILITARY/TERRITORY, positive → DIPLOMACY/TRADE
        themes = []
        avg_g = sum(d.get("goldstein", 0) for d in edges) / len(edges)
        if avg_g < -5:
            themes.append("MILITARY")
        if avg_g < -3:
            themes.append("TERRITORY")
        if avg_g > 3:
            themes.append("DIPLOMACY")
        if avg_g > 5:
            themes.append("TRADE")
        return themes[:3]

#!/usr/bin/env python3
"""
scripts/setup_demo.py
======================
One-command setup for the Geopolitical Conflict Prediction demo.

What it does:
1. Checks dependencies
2. Fetches clean GDELT data (or uses cache)
3. Builds temporal networks
4. Preprocesses into tensors
5. Pre-builds KG HTML snapshots for the web app
6. Prints launch command

Usage:
    python scripts/setup_demo.py
"""
from __future__ import annotations

import sys
from pathlib import Path

# Add project root to path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import argparse
import json

from src.data import GDELTEventCollector, GeopoliticalNetworkBuilder, GNNDataPreprocessor, COUNTRY_LIST
from src.knowledge_graph import KnowledgeGraphBuilder, KnowledgeGraphExporter


def parse_args():
    p = argparse.ArgumentParser(description="Setup geopolitical demo")
    p.add_argument("--start", default="2023-06-01", help="GDELT start date")
    p.add_argument("--end", default="2024-06-30", help="GDELT end date")
    p.add_argument("--skip_fetch", action="store_true", help="Skip data fetching (use existing)")
    p.add_argument("--device", default="cpu", help="torch device for optional training")
    return p.parse_args()


def check_data_exists() -> bool:
    d = Path("./gdelt_processed_data")
    return all((d / f).exists() for f in ["metadata.json", "node_features.npy", "edge_features.npy"])


def main():
    args = parse_args()
    print("=" * 70)
    print("GEOPOLITICAL CONFLICT PREDICTION — DEMO SETUP")
    print("=" * 70)

    # 1. Data pipeline
    if not check_data_exists() and not args.skip_fetch:
        print("\n[1/3] Fetching GDELT events...")
        collector = GDELTEventCollector()
        events = collector.fetch_events(
            start_date=args.start,
            end_date=args.end,
            countries=COUNTRY_LIST,
            use_cache=True,
        )
        if events.empty:
            print("ERROR: No events fetched. Check connection or date range.")
            sys.exit(1)
        print(f"  → {len(events)} events collected")

        print("\n[2/3] Building temporal networks...")
        builder = GeopoliticalNetworkBuilder()
        networks = builder.build_temporal_networks(events)
        if not networks:
            print("ERROR: No networks built.")
            sys.exit(1)

        print("\n[3/3] Preprocessing into tensors...")
        preprocessor = GNNDataPreprocessor()
        preprocessor.process_and_save(networks, builder.country_indices)
    else:
        print("\n[1-3] Using existing processed data.")

    # 4. Pre-build KG HTML snapshots
    print("\n[4/4] Pre-building Knowledge Graph HTML snapshots...")
    kg_builder = KnowledgeGraphBuilder()
    kg_builder.build_all_graphs()
    exporter = KnowledgeGraphExporter()
    for period in kg_builder.periods[:3] + kg_builder.periods[-3:]:
        exporter.to_pyvis_html(kg_builder.graphs[period], period)
    print(f"  → Saved {len(kg_builder.periods)} KG HTML files to ./gdelt_visualizations/")

    # Done
    print("\n" + "=" * 70)
    print("SETUP COMPLETE ✅")
    print("=" * 70)
    print("\nLaunch the web app with:")
    print("    streamlit run app.py")
    print("\nOptional: train a model with:")
    print("    python train.py --model gat --epochs 30 --device cpu")
    print("=" * 70)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
extend_data.py
==============
Fetch additional historical GDELT data to expand the training set.

Usage:
    python extend_data.py --start_date 2020-01-01 --end_date 2023-12-31
"""
from __future__ import annotations

import argparse
import json
import pickle
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).parent))

from src.data.event_collector import GDELTEventCollector
from src.data.network_builder import GeopoliticalNetworkBuilder
from src.data.data_preprocessor import GNNDataPreprocessor


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--start_date", default="2020-01-01")
    p.add_argument("--end_date", default="2023-12-31")
    p.add_argument("--cache_dir", default="./gdelt_cache")
    p.add_argument("--output_dir", default="./gdelt_processed_data_extended")
    return p.parse_args()


def main():
    args = parse_args()
    print(f"[extend_data] Fetching GDELT events from {args.start_date} to {args.end_date}")
    
    collector = GDELTEventCollector(cache_dir=args.cache_dir)
    
    # Fetch in 6-month chunks to avoid timeouts and show progress
    start = datetime.strptime(args.start_date, "%Y-%m-%d")
    end = datetime.strptime(args.end_date, "%Y-%m-%d")
    
    all_frames = []
    chunk_start = start
    
    while chunk_start < end:
        # 6-month chunk
        chunk_end = min(
            datetime(chunk_start.year + (1 if chunk_start.month > 6 else 0), 
                    ((chunk_start.month - 1 + 6) % 12) + 1, 1),
            end
        )
        if chunk_end <= chunk_start:
            chunk_end = end
        
        chunk_start_s = chunk_start.strftime("%Y-%m-%d")
        chunk_end_s = chunk_end.strftime("%Y-%m-%d")
        
        print(f"\n[extend_data] === Fetching chunk: {chunk_start_s} to {chunk_end_s} ===")
        try:
            df = collector.fetch_events(chunk_start_s, chunk_end_s, use_cache=True)
            if not df.empty:
                print(f"[extend_data] Got {len(df)} events for chunk")
                all_frames.append(df)
            else:
                print(f"[extend_data] No events for chunk")
        except Exception as e:
            print(f"[extend_data] ERROR in chunk {chunk_start_s}-{chunk_end_s}: {e}")
        
        chunk_start = chunk_end
    
    if not all_frames:
        print("[extend_data] No new data fetched. Exiting.")
        return
    
    print(f"\n[extend_data] Concatenating {len(all_frames)} chunks...")
    combined = pd.concat(all_frames, ignore_index=True)
    print(f"[extend_data] Total events: {len(combined)}")
    
    # Save raw combined events
    raw_path = Path(args.cache_dir) / f"events_extended_{args.start_date}_{args.end_date}.pkl"
    combined.to_pickle(raw_path)
    print(f"[extend_data] Saved raw combined events to {raw_path}")
    
    # Build networks
    print("\n[extend_data] Building temporal networks...")
    builder = GeopoliticalNetworkBuilder()
    networks = builder.build_temporal_networks(combined, time_window="M")
    print(f"[extend_data] Built {len(networks)} monthly networks")
    
    # Preprocess
    print("\n[extend_data] Preprocessing into tensors...")
    preprocessor = GNNDataPreprocessor(output_dir=args.output_dir)
    result = preprocessor.process_and_save(networks, builder.country_indices)
    print(f"[extend_data] Saved tensors to {args.output_dir}")
    print(f"[extend_data] Periods: {result['metadata']['periods']}")
    print(f"[extend_data] Shape summary:")
    print(f"  node_features: {result['node_features'].shape}")
    print(f"  edge_features: {result['edge_features'].shape}")
    print(f"  edge_labels: {result['edge_labels'].shape}")


if __name__ == "__main__":
    main()

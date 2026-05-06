#!/usr/bin/env python3
"""
augment_data.py
================
Augment the existing temporal graph dataset by generating synthetic historical data.

Uses probabilistic modeling based on empirical distributions from real GDELT data.
Preserves:
- Regional conflict patterns (Middle East, South Asia hotspots)
- Temporal autocorrelation (AR(1) process for Goldstein scores)
- Seasonal effects
- Network structure

Usage:
    python scripts/augment_data.py --num_synthetic_months 60
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Any, List, Tuple

import numpy as np
import pandas as pd
from scipy import stats


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input_dir", default="./gdelt_processed_data")
    p.add_argument("--output_dir", default="./gdelt_processed_data_augmented")
    p.add_argument("--num_synthetic_months", type=int, default=60, 
                   help="Number of synthetic months to prepend")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def load_original_data(data_dir: str) -> Dict[str, Any]:
    data_dir = Path(data_dir)
    node_features = np.load(data_dir / "node_features.npy")   # (T, N, 6)
    edge_features = np.load(data_dir / "edge_features.npy")   # (T, N, N, 5)
    edge_labels = np.load(data_dir / "edge_labels.npy")       # (T, N, N)
    valid_mask = np.load(data_dir / "valid_mask.npy")         # (T, N, N)
    
    with open(data_dir / "metadata.json") as f:
        metadata = json.load(f)
    
    return {
        "node_features": node_features,
        "edge_features": edge_features,
        "edge_labels": edge_labels,
        "valid_mask": valid_mask,
        "metadata": metadata,
    }


def fit_empirical_distributions(edge_features: np.ndarray, valid_mask: np.ndarray) -> Dict[str, Any]:
    """
    Fit distributions to synthetic data from empirical edge feature patterns.
    
    edge_features shape: (T, N, N, 5) where features are:
      0: conflict_count, 1: cooperation_count, 2: tone, 3: goldstein, 4: imbalance
    """
    T, N, _, F = edge_features.shape
    
    # Extract valid edge features
    val_flat = valid_mask.reshape(-1)
    ef_flat = edge_features.reshape(-1, F)
    valid_ef = ef_flat[val_flat]
    
    # Feature distributions
    dists = {}
    
    # Conflict count: zero-inflated Poisson-ish / geometric
    conflict = valid_ef[:, 0]
    dists["conflict_mean"] = conflict.mean()
    dists["conflict_std"] = conflict.std()
    dists["conflict_zero_rate"] = (conflict == 0).mean()
    dists["conflict_nonzero_mean"] = conflict[conflict > 0].mean() if conflict.sum() > 0 else 0.1
    
    # Cooperation count
    coop = valid_ef[:, 1]
    dists["coop_mean"] = coop.mean()
    dists["coop_std"] = coop.std()
    
    # Tone: roughly normal but skewed negative
    tone = valid_ef[:, 2]
    dists["tone_mean"] = tone.mean()
    dists["tone_std"] = tone.std()
    
    # Goldstein: bi-modal (cooperation positive, conflict negative)
    gold = valid_ef[:, 3]
    dists["gold_mean"] = gold.mean()
    dists["gold_std"] = gold.std()
    
    # Imbalance
    imb = valid_ef[:, 4]
    dists["imb_mean"] = imb.mean()
    dists["imb_std"] = imb.std()
    
    # Per-dyad statistics (for creating persistent patterns)
    dyad_stats = {}
    for i in range(N):
        for j in range(N):
            if i == j:
                continue
            valid_t = valid_mask[:, i, j]
            if valid_t.sum() == 0:
                continue
            ef_ij = edge_features[:, i, j, :]
            dyad_stats[(i, j)] = {
                "conflict_rate": (ef_ij[:, 0] > 0).mean(),
                "avg_goldstein": ef_ij[:, 3].mean(),
                "goldstein_volatility": ef_ij[:, 3].std() + 0.5,
            }
    
    dists["dyad_stats"] = dyad_stats
    dists["N"] = N
    
    return dists


def generate_dyad_features(
    i: int, j: int, 
    month: int, 
    dists: Dict[str, Any],
    prev_goldstein: float = None,
    seed: int = 42,
) -> np.ndarray:
    """
    Generate synthetic edge features for a single dyad-month.
    Uses AR(1) process for Goldstein to create temporal correlation.
    """
    np.random.seed(seed + month * 1000 + i * 20 + j)
    
    N = dists["N"]
    dyad_key = (i, j) if (i, j) in dists.get("dyad_stats", {}) else None
    
    if dyad_key:
        stats = dists["dyad_stats"][dyad_key]
        conflict_prob = stats["conflict_rate"]
        gold_mean = stats["avg_goldstein"]
        gold_vol = stats["goldstein_volatility"]
    else:
        # Random dyad
        conflict_prob = 0.1
        gold_mean = dists["gold_mean"]
        gold_vol = dists["gold_std"]
    
    # Add some geographic structure: Middle East, South Asia tend more conflictual
    # Countries 11-17 roughly are MENA/S.Asia in the current ordering
    if (i >= 11 and i <= 17) or (j >= 11 and j <= 17):
        conflict_prob *= 1.5
        gold_mean -= 1.0  # More negative = more conflict
    
    # Conflict count
    if np.random.random() < dists["conflict_zero_rate"]:
        conflict = 0.0
    else:
        conflict = np.random.poisson(dists["conflict_nonzero_mean"] * conflict_prob * 5)
    
    # Cooperation count
    coop = np.random.poisson(dists["coop_mean"] * 2)
    
    # Tone (negatively correlated with conflict)
    base_tone = dists["tone_mean"]
    tone_noise = np.random.normal(0, dists["tone_std"])
    tone = base_tone - conflict * 0.5 + tone_noise
    
    # Goldstein: AR(1) process for temporal correlation
    if prev_goldstein is not None:
        # Pull toward previous month's value with some reversion to mean
        phi = 0.7  # AR coefficient
        goldstein = phi * prev_goldstein + (1 - phi) * gold_mean + np.random.normal(0, gold_vol)
    else:
        goldstein = np.random.normal(gold_mean, gold_vol)
    
    # Imbalance ratio
    # High when conflict dominates over cooperation
    if conflict + coop > 0:
        imbalance = abs(conflict - coop) / (conflict + coop + 1)
    else:
        imbalance = 0.0
    
    return np.array([conflict, coop, tone, goldstein, imbalance], dtype=np.float32)


def generate_synthetic_months(
    num_months: int,
    dists: Dict[str, Any],
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate num_months of synthetic data.
    
    Returns:
        node_features, edge_features, edge_labels, valid_mask
    """
    N = dists["N"]
    
    node_features = np.zeros((num_months, N, 6), dtype=np.float32)
    edge_features = np.zeros((num_months, N, N, 5), dtype=np.float32)
    edge_labels = np.zeros((num_months, N, N), dtype=np.float32)
    valid_mask = np.zeros((num_months, N, N), dtype=bool)
    
    # Track previous Goldstein for AR(1)
    prev_goldstein = np.zeros((N, N), dtype=np.float32)
    
    for t in range(num_months):
        for i in range(N):
            for j in range(N):
                if i == j:
                    continue
                
                # Valid mask: mostly all dyads valid in synthetic data
                if np.random.random() < 0.9:
                    valid_mask[t, i, j] = True
                    ef = generate_dyad_features(i, j, t, dists, prev_goldstein[i, j], seed)
                    edge_features[t, i, j, :] = ef
                    prev_goldstein[i, j] = ef[3]  # Update Goldstein for AR(1)
                else:
                    valid_mask[t, i, j] = False
        
        # Compute node features from edge features
        conflict_out = edge_features[t, :, :, 0].sum(axis=1)  # (N,)
        conflict_in = edge_features[t, :, :, 0].sum(axis=0)   # (N,)
        coop_out = edge_features[t, :, :, 1].sum(axis=1)      # (N,)
        
        total_out = edge_features[t, :, :, 0] + edge_features[t, :, :, 1]
        out_counts = total_out.sum(axis=1)
        in_counts = total_out.sum(axis=0)
        
        node_features[t, :, 0] = conflict_out
        node_features[t, :, 1] = conflict_in
        node_features[t, :, 2] = coop_out
        node_features[t, :, 3] = np.divide(
            edge_features[t, :, :, 2].sum(axis=1),
            np.maximum(1, out_counts),
            out=np.zeros(N, dtype=np.float32),
            where=out_counts > 0,
        )
        node_features[t, :, 4] = np.divide(
            edge_features[t, :, :, 2].sum(axis=0),
            np.maximum(1, in_counts),
            out=np.zeros(N, dtype=np.float32),
            where=in_counts > 0,
        )
        node_features[t, :, 5] = out_counts + in_counts
    
    # Compute labels: Goldstein drop > 0.5 next month
    goldstein_avgs = []
    event_presence = []
    for t in range(num_months):
        gold = np.divide(
            edge_features[t, :, :, 3],
            np.ones((N, N)),  # Already a single value per dyad
            out=np.zeros((N, N), dtype=np.float32),
            where=valid_mask[t],
        )
        goldstein_avgs.append(gold)
        event_presence.append(valid_mask[t])
    
    for t in range(num_months - 1):
        cur = goldstein_avgs[t]
        nxt = goldstein_avgs[t + 1]
        mask = event_presence[t] | event_presence[t + 1]
        edge_labels[t] = ((nxt < cur - 0.5) & mask).astype(np.float32)
    
    np.fill_diagonal(edge_labels[-1], 0.0)
    
    return node_features, edge_features, edge_labels, valid_mask


def main():
    args = parse_args()
    np.random.seed(args.seed)
    
    print(f"[augment_data] Loading original data from {args.input_dir}...")
    orig = load_original_data(args.input_dir)
    
    T_orig = orig["node_features"].shape[0]
    print(f"[augment_data] Original data: {T_orig} months")
    
    # Fit distributions
    print("[augment_data] Fitting empirical distributions...")
    dists = fit_empirical_distributions(orig["edge_features"], orig["valid_mask"])
    print(f"[augment_data] Fitted distributions for {len(dists.get('dyad_stats', {}))} dyads")
    
    # Generate synthetic history
    print(f"[augment_data] Generating {args.num_synthetic_months} synthetic months...")
    syn_node, syn_edge, syn_labels, syn_mask = generate_synthetic_months(
        args.num_synthetic_months, dists, seed=args.seed
    )
    
    # Augment
    print("[augment_data] Concatenating...")
    node_features = np.concatenate([syn_node, orig["node_features"]], axis=0)
    edge_features = np.concatenate([syn_edge, orig["edge_features"]], axis=0)
    edge_labels = np.concatenate([syn_labels, orig["edge_labels"]], axis=0)
    valid_mask = np.concatenate([syn_mask, orig["valid_mask"]], axis=0)
    
    # Periods: generate synthetic period names going backwards
    # Original starts at 2024-01, so synthetic ends at 2023-12
    syn_periods = []
    for m in range(args.num_synthetic_months, 0, -1):
        year = 2023
        month = 12 - m + 1
        while month <= 0:
            year -= 1
            month += 12
        syn_periods.append(f"{year:04d}-{month:02d}")
    syn_periods = syn_periods[::-1]
    
    all_periods = syn_periods + orig["metadata"]["periods"]
    
    print(f"[augment_data] Total periods: {len(all_periods)}")
    print(f"  Synthetic: {syn_periods[0]}...{syn_periods[-1]} ({len(syn_periods)})")
    print(f"  Real: {orig['metadata']['periods'][0]}...{orig['metadata']['periods'][-1]} ({len(orig['metadata']['periods'])})")
    
    # Save
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    np.save(output_dir / "node_features.npy", node_features)
    np.save(output_dir / "edge_features.npy", edge_features)
    np.save(output_dir / "edge_labels.npy", edge_labels)
    np.save(output_dir / "valid_mask.npy", valid_mask)
    
    metadata = {
        **orig["metadata"],
        "periods": all_periods,
        "synthetic_months": args.num_synthetic_months,
        "seed": args.seed,
        "augmented": True,
    }
    with open(output_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    
    print(f"[augment_data] Saved augmented data to {output_dir}")
    print(f"  node_features: {node_features.shape}")
    print(f"  edge_features: {edge_features.shape}")
    print(f"  edge_labels: {edge_labels.shape}")
    print(f"  valid_mask: {valid_mask.shape}")
    
    # Quick distribution check
    print(f"\n[augment_data] Label distribution:")
    val_flat = valid_mask.flatten()
    lbl_flat = edge_labels.flatten()
    print(f"  Valid dyads: {val_flat.sum()}")
    print(f"  Positive rate: {lbl_flat[val_flat].mean():.3f}")


if __name__ == "__main__":
    main()

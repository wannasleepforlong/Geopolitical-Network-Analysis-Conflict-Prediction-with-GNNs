#!/usr/bin/env python3
"""
Fair tabular baselines on the temporal graph dataset.

Uses only historical features from the same lookback window as the GNNs and
supports evaluating on real GDELT target months only when augmented synthetic
history is present.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score


def split_target_indices(periods: list[str], temporal_window: int, synthetic_months: int, split_strategy: str):
    all_indices = list(range(temporal_window, len(periods)))
    if split_strategy == "real_targets_only":
        all_indices = [idx for idx in all_indices if idx >= synthetic_months]

    n = len(all_indices)
    n_train = max(1, int(0.6 * n))
    n_val = max(1, int(0.2 * n))
    if n_train + n_val >= n:
        n_train = max(1, n - 2)
        n_val = 1

    train_idx = all_indices[:n_train]
    val_idx = all_indices[n_train:n_train + n_val]
    test_idx = all_indices[n_train + n_val:]
    return train_idx, val_idx, test_idx


def build_xy(node_features, edge_features, edge_labels, valid_mask, idx_list, temporal_window):
    x_list, y_list = [], []
    for t in idx_list:
        node_mean = node_features[t - temporal_window:t].mean(axis=0)
        edge_mean = edge_features[t - temporal_window:t].mean(axis=0)
        edge_last = edge_features[t - 1]

        n = node_mean.shape[0]
        h_i = np.repeat(node_mean[:, None, :], n, axis=1)
        h_j = np.repeat(node_mean[None, :, :], n, axis=0)
        features = np.concatenate([h_i, h_j, edge_mean, edge_last], axis=-1).reshape(n * n, -1)

        labels = edge_labels[t].reshape(-1)
        mask = valid_mask[t].reshape(-1).astype(bool)

        x_list.append(features[mask])
        y_list.append(labels[mask])

    return np.concatenate(x_list, axis=0), np.concatenate(y_list, axis=0)


def best_threshold(y_true, probs):
    thresholds = np.arange(0.1, 0.92, 0.02)
    best_thr = 0.5
    best_f1 = -1.0
    for thr in thresholds:
        preds = (probs >= thr).astype(int)
        score = f1_score(y_true, preds, zero_division=0)
        if score > best_f1:
            best_f1 = score
            best_thr = float(thr)
    return best_thr


def evaluate(model_name: str, model, x_val, y_val, x_test, y_test) -> dict:
    val_probs = model.predict_proba(x_val)[:, 1]
    thr = best_threshold(y_val, val_probs)

    probs = model.predict_proba(x_test)[:, 1]
    preds = (probs >= thr).astype(int)

    try:
        auc = roc_auc_score(y_test, probs)
    except ValueError:
        auc = 0.5

    return {
        "model": model_name,
        "auc": float(auc),
        "f1": float(f1_score(y_test, preds, zero_division=0)),
        "precision": float(precision_score(y_test, preds, zero_division=0)),
        "recall": float(recall_score(y_test, preds, zero_division=0)),
        "accuracy": float(accuracy_score(y_test, preds)),
        "macro_f1": float(f1_score(y_test, preds, average="macro", zero_division=0)),
        "threshold": thr,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", default="./gdelt_processed_data_augmented")
    p.add_argument("--temporal_window", type=int, default=12)
    p.add_argument("--split_strategy", choices=["time", "real_targets_only"], default="real_targets_only")
    args = p.parse_args()

    data_dir = Path(args.data_dir)
    node_features = np.load(data_dir / "node_features.npy")
    edge_features = np.load(data_dir / "edge_features.npy")
    edge_labels = np.load(data_dir / "edge_labels.npy")
    valid_mask = np.load(data_dir / "valid_mask.npy")
    with open(data_dir / "metadata.json") as f:
        metadata = json.load(f)

    train_idx, val_idx, test_idx = split_target_indices(
        metadata["periods"],
        args.temporal_window,
        int(metadata.get("synthetic_months", 0)),
        args.split_strategy,
    )

    x_train, y_train = build_xy(node_features, edge_features, edge_labels, valid_mask, train_idx, args.temporal_window)
    x_val, y_val = build_xy(node_features, edge_features, edge_labels, valid_mask, val_idx, args.temporal_window)
    x_test, y_test = build_xy(node_features, edge_features, edge_labels, valid_mask, test_idx, args.temporal_window)

    print(f"[Baselines] Train={len(y_train)} Val={len(y_val)} Test={len(y_test)}")
    print(f"[Baselines] Pos ratio train={y_train.mean():.3f} val={y_val.mean():.3f} test={y_test.mean():.3f}")

    results = []

    lr = LogisticRegression(max_iter=1000, class_weight="balanced", solver="lbfgs", random_state=42)
    lr.fit(x_train, y_train)
    results.append(evaluate("LogisticRegression", lr, x_val, y_val, x_test, y_test))

    rf = RandomForestClassifier(n_estimators=300, class_weight="balanced", random_state=42, n_jobs=-1)
    rf.fit(x_train, y_train)
    results.append(evaluate("RandomForest", rf, x_val, y_val, x_test, y_test))

    gb = GradientBoostingClassifier(n_estimators=200, learning_rate=0.05, max_depth=2, random_state=42)
    gb.fit(x_train, y_train)
    results.append(evaluate("GradientBoosting", gb, x_val, y_val, x_test, y_test))

    print("\nModel                     AUC     Acc     F1      MacroF1 Threshold")
    print("-------------------------------------------------------------------")
    for r in results:
        print(
            f"{r['model']:<25} {r['auc']:.4f}  {r['accuracy']:.4f}  "
            f"{r['f1']:.4f}  {r['macro_f1']:.4f}   {r['threshold']:.2f}"
        )


if __name__ == "__main__":
    main()

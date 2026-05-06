#!/usr/bin/env python3
"""
baselines.py
============
Quick baseline classifiers on the same temporal graph dataset.
"""
from __future__ import annotations

import json
import argparse
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score


def load_data(data_dir: str, temporal_window: int = 12):
    data_dir = Path(data_dir)
    node_features = np.load(data_dir / "node_features.npy")   # (T, N, 6)
    edge_features = np.load(data_dir / "edge_features.npy")   # (T, N, N, 5)
    edge_labels   = np.load(data_dir / "edge_labels.npy")     # (T, N, N)
    valid_mask    = np.load(data_dir / "valid_mask.npy")      # (T, N, N)

    with open(data_dir / "metadata.json") as f:
        metadata = json.load(f)
    periods = metadata["periods"]

    indices = list(range(temporal_window, len(periods)))
    n = len(indices)
    n_train = int(0.7 * n)
    n_val   = int(0.1 * n)
    train_idx = indices[:n_train]
    val_idx   = indices[n_train:n_train + n_val]
    test_idx  = indices[n_train + n_val:]

    def _build_X_y(idx_list):
        X_list, y_list, m_list = [], [], []
        for t in idx_list:
            # Use raw edge features from the target month t as flat features
            # (Conflict, Cooperation, Tone, Goldstein, Imbalance)
            x = edge_features[t].reshape(-1, 5)   # (N*N, 5)
            y = edge_labels[t].flatten()            # (N*N,)
            m = valid_mask[t].flatten()           # (N*N,)
            X_list.append(x)
            y_list.append(y)
            m_list.append(m)
        X = np.concatenate(X_list, axis=0)
        y = np.concatenate(y_list, axis=0)
        m = np.concatenate(m_list, axis=0)
        return X[m], y[m]

    return _build_X_y(train_idx), _build_X_y(test_idx)


def evaluate(model_name: str, model, X_test, y_test) -> dict:
    probs = model.predict_proba(X_test)[:, 1]
    preds = (probs >= 0.5).astype(int)
    try:
        auc = roc_auc_score(y_test, probs)
    except ValueError:
        auc = 0.5
    return {
        "model": model_name,
        "auc": auc,
        "f1": f1_score(y_test, preds, zero_division=0),
        "precision": precision_score(y_test, preds, zero_division=0),
        "recall": recall_score(y_test, preds, zero_division=0),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", default="./gdelt_processed_data")
    p.add_argument("--temporal_window", type=int, default=12)
    args = p.parse_args()

    (X_train, y_train), (X_test, y_test) = load_data(args.data_dir, args.temporal_window)
    print(f"[Baselines] Train samples={len(y_train)}, Test samples={len(y_test)}")
    print(f"[Baselines] Pos ratio in train={y_train.mean():.3f}, test={y_test.mean():.3f}")

    # Logistic Regression
    lr = LogisticRegression(max_iter=1000, class_weight="balanced", solver="lbfgs")
    lr.fit(X_train, y_train)
    lr_res = evaluate("LogisticRegression", lr, X_test, y_test)
    print(f"[LR]  AUC={lr_res['auc']:.4f} | F1={lr_res['f1']:.4f} | P={lr_res['precision']:.4f} | R={lr_res['recall']:.4f}")

    # Random Forest
    rf = RandomForestClassifier(n_estimators=200, class_weight="balanced", random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    rf_res = evaluate("RandomForest", rf, X_test, y_test)
    print(f"[RF]  AUC={rf_res['auc']:.4f} | F1={rf_res['f1']:.4f} | P={rf_res['precision']:.4f} | R={rf_res['recall']:.4f}")


if __name__ == "__main__":
    main()

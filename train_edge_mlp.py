#!/usr/bin/env python3
"""
train_edge_mlp.py
===================
Train the EdgeMLP baseline (no graph convolutions, just MLP on edge + node features).

Usage:
    python train_edge_mlp.py --epochs 50 --hidden_dim 64 --device cpu --seed 42
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.models import EdgeMLP, GeopoliticalDataset, ConflictPredictionTrainer


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train EdgeMLP baseline for geopolitical conflict prediction")
    p.add_argument("--data_dir", default="./gdelt_processed_data", help="Preprocessed data directory")
    p.add_argument("--epochs", type=int, default=50, help="Maximum epochs")
    p.add_argument("--batch_size", type=int, default=4, help="Batch size")
    p.add_argument("--hidden_dim", type=int, default=16, help="Hidden dimension")
    p.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    p.add_argument("--weight_decay", type=float, default=1e-5, help="Weight decay")
    p.add_argument("--temporal_window", type=int, default=12, help="Temporal window size")
    p.add_argument("--device", default="cpu", help="torch device")
    p.add_argument("--patience", type=int, default=10, help="Early stopping patience")
    p.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    p.add_argument("--out_dir", default="./results/edge_mlp", help="Output directory")
    p.add_argument(
        "--split_strategy",
        choices=["time", "real_targets_only"],
        default="real_targets_only",
        help="How to define temporal train/val/test targets",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = torch.device(args.device)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Detect model params from saved data shape
    node_feats = np.load(Path(args.data_dir) / "node_features.npy")
    edge_feats = np.load(Path(args.data_dir) / "edge_features.npy")
    num_node_features = node_feats.shape[-1]
    num_edge_features = edge_feats.shape[-1]
    print(f"[train] Detected node_features={num_node_features}, edge_features={num_edge_features}")

    # Datasets
    train_ds = GeopoliticalDataset(
        args.data_dir, temporal_window=args.temporal_window, split="train", split_strategy=args.split_strategy
    )
    val_ds = GeopoliticalDataset(
        args.data_dir, temporal_window=args.temporal_window, split="val", split_strategy=args.split_strategy
    )
    test_ds = GeopoliticalDataset(
        args.data_dir, temporal_window=args.temporal_window, split="test", split_strategy=args.split_strategy
    )

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=False)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

    # Model
    model = EdgeMLP(
        num_node_features=num_node_features,
        num_edge_features=num_edge_features,
        hidden_dim=args.hidden_dim,
    )
    print(f"[train] Model: EdgeMLP, params={sum(p.numel() for p in model.parameters()):,}")

    # Trainer
    trainer = ConflictPredictionTrainer(
        model=model,
        device=device,
        lr=args.lr,
        weight_decay=args.weight_decay,
        pos_weight=None,
        use_focal_loss=True,
    )

    training_result = trainer.fit(
        train_loader,
        val_loader,
        epochs=args.epochs,
        patience=args.patience,
        checkpoint_dir=str(out_dir / "checkpoints"),
    )

    # Test evaluation
    test_metrics = trainer.evaluate(test_loader, threshold=trainer.best_threshold)
    print(f"[train] Test AUC={test_metrics['auc']:.4f} | F1={test_metrics['f1']:.4f} | Acc={test_metrics['accuracy']:.4f}")

    # Save predictions
    test_preds = trainer.predict(test_loader)
    test_periods = [test_ds.periods[idx] for idx in test_ds.indices]
    np.save(out_dir / "test_predictions.npy", test_preds)
    np.save(out_dir / "test_labels.npy", np.load(Path(args.data_dir) / "edge_labels.npy")[test_ds.indices])

    # Save results JSON
    result = {
        "config": vars(args),
        "training": training_result,
        "test_metrics": test_metrics,
        "test_periods": test_periods,
    }
    with open(out_dir / "results.json", "w") as f:
        json.dump(result, f, indent=2)

    print(f"[train] Done. Results saved to {out_dir}")


if __name__ == "__main__":
    main()

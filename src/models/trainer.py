"""
Conflict Prediction Trainer
===========================
Training loop with early stopping, metrics, and checkpointing.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict, Any, Optional

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score
from torch.utils.data import DataLoader

from .dataset import GeopoliticalDataset


class ConflictPredictionTrainer:
    """
    Parameters
    ----------
    model : nn.Module
    device : torch.device
    lr : float
    weight_decay : float
    pos_weight : float | None
        Class imbalance weight for positive class.
    """

    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        lr: float = 1e-3,
        weight_decay: float = 1e-5,
        pos_weight: Optional[float] = None,
    ):
        self.model = model.to(device)
        self.device = device
        self.criterion = nn.BCELoss(reduction="none")
        self.pos_weight = pos_weight

        self.optimizer = torch.optim.AdamW(
            model.parameters(), lr=lr, weight_decay=weight_decay
        )
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="max", factor=0.5, patience=5, verbose=True
        )

        self.history: Dict[str, list] = {"train_loss": [], "val_auc": [], "val_f1": []}
        self.best_val_auc = 0.0
        self.patience_counter = 0

    def _forward_model(
        self,
        node_x: torch.Tensor,
        edge_x: torch.Tensor,
        adj: torch.Tensor,
    ) -> torch.Tensor:
        """Normalize model outputs so trainer logic works for GCN and GAT."""
        output = self.model(node_x, edge_x, adj)
        if isinstance(output, tuple):
            return output[0]
        return output

    def _compute_metrics(
        self,
        preds: np.ndarray,
        labels: np.ndarray,
        mask: np.ndarray,
    ) -> Dict[str, float]:
        """Compute AUC, F1, Precision, Recall on masked valid positions."""
        valid = mask.flatten().astype(bool)
        y_true = labels.flatten()[valid]
        y_pred = preds.flatten()[valid]

        # Binary predictions at 0.5 threshold
        y_pred_bin = (y_pred >= 0.5).astype(int)

        try:
            auc = roc_auc_score(y_true, y_pred)
        except ValueError:
            auc = 0.5
        try:
            f1 = f1_score(y_true, y_pred_bin, zero_division=0)
        except ValueError:
            f1 = 0.0
        try:
            prec = precision_score(y_true, y_pred_bin, zero_division=0)
        except ValueError:
            prec = 0.0
        try:
            rec = recall_score(y_true, y_pred_bin, zero_division=0)
        except ValueError:
            rec = 0.0

        return {"auc": float(auc), "f1": float(f1), "precision": float(prec), "recall": float(rec)}

    def train_epoch(self, dataloader: DataLoader) -> float:
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        for node_x, edge_x, labels, mask, adj in dataloader:
            node_x = node_x.to(self.device)
            edge_x = edge_x.to(self.device)
            labels = labels.to(self.device)
            mask = mask.to(self.device)
            adj = adj.to(self.device)

            self.optimizer.zero_grad()

            preds = self._forward_model(node_x, edge_x, adj)

            loss = self.criterion(preds, labels)
            if self.pos_weight is not None:
                # Up-weight positive samples
                pos_weight = torch.as_tensor(self.pos_weight, device=self.device, dtype=preds.dtype)
                weight = torch.where(labels > 0.5, pos_weight, torch.ones_like(preds))
                loss = loss * weight
            loss = (loss * mask.float()).sum() / mask.sum().clamp(min=1.0)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        return total_loss / max(num_batches, 1)

    @torch.no_grad()
    def evaluate(self, dataloader: DataLoader) -> Dict[str, Any]:
        self.model.eval()
        all_preds = []
        all_labels = []
        all_masks = []

        for node_x, edge_x, labels, mask, adj in dataloader:
            node_x = node_x.to(self.device)
            edge_x = edge_x.to(self.device)
            adj = adj.to(self.device)

            preds = self._forward_model(node_x, edge_x, adj)

            all_preds.append(preds.cpu().numpy())
            all_labels.append(labels.numpy())
            all_masks.append(mask.numpy())

        if not all_preds:
            return {"auc": 0.5, "f1": 0.0, "precision": 0.0, "recall": 0.0, "loss": 0.0}

        preds = np.concatenate(all_preds, axis=0)
        labels = np.concatenate(all_labels, axis=0)
        masks = np.concatenate(all_masks, axis=0)

        metrics = self._compute_metrics(preds, labels, masks)
        metrics["loss"] = float(
            np.mean(
                self.criterion(
                    torch.from_numpy(preds), torch.from_numpy(labels)
                ).numpy()[masks]
            )
        )
        return metrics

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int = 50,
        patience: int = 10,
        checkpoint_dir: str = "./checkpoints",
    ) -> Dict[str, Any]:
        Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
        best_path = Path(checkpoint_dir) / "best_model.pt"

        print(f"[Trainer] Starting training on {self.device}")
        start = time.time()

        for epoch in range(1, epochs + 1):
            t0 = time.time()
            train_loss = self.train_epoch(train_loader)
            val_metrics = self.evaluate(val_loader)

            self.history["train_loss"].append(train_loss)
            self.history["val_auc"].append(val_metrics["auc"])
            self.history["val_f1"].append(val_metrics["f1"])

            print(
                f"Epoch {epoch:02d}/{epochs} | "
                f"train_loss={train_loss:.4f} | "
                f"val_auc={val_metrics['auc']:.4f} | "
                f"val_f1={val_metrics['f1']:.4f} | "
                f"time={time.time()-t0:.1f}s"
            )

            self.scheduler.step(val_metrics["auc"])

            # Early stopping + checkpoint
            if val_metrics["auc"] > self.best_val_auc:
                self.best_val_auc = val_metrics["auc"]
                self.patience_counter = 0
                torch.save(self.model.state_dict(), best_path)
                print(f"  → Saved best model (val_auc={self.best_val_auc:.4f})")
            else:
                self.patience_counter += 1
                if self.patience_counter >= patience:
                    print(f"[Trainer] Early stopping at epoch {epoch}")
                    break

        total = time.time() - start
        print(f"[Trainer] Training complete in {total:.1f}s")

        # Load best
        self.model.load_state_dict(torch.load(best_path, map_location=self.device))

        return {
            "history": self.history,
            "best_val_auc": self.best_val_auc,
            "epochs_trained": epoch,
            "total_time": total,
        }

    @torch.no_grad()
    def predict(self, dataloader: DataLoader) -> np.ndarray:
        self.model.eval()
        preds = []
        for node_x, edge_x, labels, mask, adj in dataloader:
            node_x = node_x.to(self.device)
            edge_x = edge_x.to(self.device)
            adj = adj.to(self.device)

            out = self._forward_model(node_x, edge_x, adj)
            preds.append(out.cpu().numpy())
        return np.concatenate(preds, axis=0)

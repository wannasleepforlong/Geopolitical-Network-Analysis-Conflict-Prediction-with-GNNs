"""
src.models package
==================
Graph Neural Network architectures for conflict prediction.

Modules:
    temporal_gcn  – LSTM + GCN
    temporal_gat  – LSTM + multi-head Graph Attention
    dataset       – sliding-window PyTorch Dataset
    trainer       – training loop with early stopping
"""
from .dataset import GeopoliticalDataset
from .temporal_gcn import TemporalGCN
from .temporal_gat import TemporalGAT
from .trainer import ConflictPredictionTrainer

__all__ = [
    "TemporalGCN",
    "TemporalGAT",
    "GeopoliticalDataset",
    "ConflictPredictionTrainer",
]

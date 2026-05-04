"""
Temporal GCN Model
==================
LSTM encoder + Graph Convolutional Network for conflict prediction.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class TemporalGCN(nn.Module):
    """
    Parameters
    ----------
    num_node_features : int
    num_edge_features : int
    hidden_dim : int
    num_gcn_layers : int
    dropout : float
    """

    def __init__(
        self,
        num_node_features: int = 6,
        num_edge_features: int = 5,
        hidden_dim: int = 64,
        num_gcn_layers: int = 2,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_edge_features = num_edge_features

        # Temporal encoder
        self.lstm = nn.LSTM(
            num_node_features,
            hidden_dim,
            num_layers=2,
            batch_first=True,
            dropout=dropout if 2 > 1 else 0.0,
        )

        # Graph convolution layers (simplified message passing)
        self.gcn_layers = nn.ModuleList(
            [
                nn.Linear(hidden_dim, hidden_dim)
                for _ in range(num_gcn_layers)
            ]
        )
        self.gcn_edge_proj = nn.ModuleList(
            [
                nn.Linear(num_edge_features, hidden_dim)
                for _ in range(num_gcn_layers)
            ]
        )

        self.norms = nn.ModuleList(
            [nn.LayerNorm(hidden_dim) for _ in range(num_gcn_layers)]
        )
        self.dropout = nn.Dropout(dropout)

        # Edge predictor
        self.edge_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2 + num_edge_features, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(
        self,
        node_features: torch.Tensor,    # (B, W, N, F_n)
        edge_features: torch.Tensor,    # (B, W, N, N, F_e)
        adjacency: torch.Tensor,        # (B, N, N)
    ) -> torch.Tensor:                  # (B, N, N)
        B, W, N, _ = node_features.shape

        # 1. Temporal encoding
        # Flatten batch+node dims for LSTM
        x = node_features.permute(0, 2, 1, 3).reshape(B * N, W, -1)   # (B*N, W, F_n)
        lstm_out, _ = self.lstm(x)                                     # (B*N, W, H)
        h = lstm_out[:, -1, :]                                         # (B*N, H)
        h = h.view(B, N, self.hidden_dim)                              # (B, N, H)

        # 2. Graph convolution with edge-conditioned message passing
        edge_last = edge_features[:, -1, :, :, :]                     # (B, N, N, F_e)
        for i, (gcn, edge_proj, norm) in enumerate(
            zip(self.gcn_layers, self.gcn_edge_proj, self.norms)
        ):
            # Message: aggregate neighbors weighted by adjacency + last-period edge features
            edge_h = edge_proj(edge_last)                               # (B, N, N, H)
            messages = torch.einsum("bij,bijn->bin", adjacency.float(), edge_h)  # (B, N, H)
            updated = gcn(h) + messages
            h = F.relu(norm(updated))
            h = self.dropout(h)

        # 3. Edge prediction head
        h_i = h.unsqueeze(2).expand(B, N, N, self.hidden_dim)          # (B, N, N, H)
        h_j = h.unsqueeze(1).expand(B, N, N, self.hidden_dim)
        edge_input = torch.cat([h_i, h_j, edge_last], dim=-1)            # (B, N, N, 3H)
        logits = self.edge_mlp(edge_input).squeeze(-1)                   # (B, N, N)
        return torch.sigmoid(logits)

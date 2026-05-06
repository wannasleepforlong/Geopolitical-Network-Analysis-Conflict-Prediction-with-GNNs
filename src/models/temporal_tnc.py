"""
Temporal TNC Model
==================
LSTM encoder + Temporal Node-Conditioned Network for conflict prediction.

Optimized for very small datasets: simpler architecture, better regularization,
proper feature normalization, and edge feature aggregation across temporal window.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class TemporalTNC(nn.Module):
    """
    Lightweight Temporal Network for small-sample conflict prediction.
    
    Uses mean-pooled edge features across temporal windows and a smaller,
    more heavily regularized architecture.
    
    Parameters
    ----------
    num_node_features : int
    num_edge_features : int
    hidden_dim : int
    num_layers : int
    dropout : float
    """

    def __init__(
        self,
        num_node_features: int = 6,
        num_edge_features: int = 5,
        hidden_dim: int = 32,
        num_layers: int = 2,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_edge_features = num_edge_features

        # Temporal encoder: smaller, more regularized
        self.lstm = nn.LSTM(
            num_node_features,
            hidden_dim,
            num_layers=1,  # Simpler
            batch_first=True,
        )

        # BatchNorm before LSTM for normalization
        self.node_bn = nn.BatchNorm1d(num_node_features)

        # Edge feature aggregation: project per-step features and pool
        self.edge_proj = nn.Sequential(
            nn.Linear(num_edge_features, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # GNN layers: simplified node-to-node message passing
        self.gnn_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            )
            for _ in range(num_layers)
        ])

        # Node update with adjacency and residual
        self.node_updaters = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.LayerNorm(hidden_dim),
            )
            for _ in range(num_layers)
        ])

        # Final edge predictor
        self.edge_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2 + hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

        self._reset_parameters()

    def _reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.01)  # smaller init for stability
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def _normalize_adj(self, adj: torch.Tensor) -> torch.Tensor:
        """Symmetric normalization with self-loops."""
        B, N, _ = adj.shape
        device = adj.device
        I = torch.eye(N, device=device).unsqueeze(0)
        A = adj.float() + I
        D = A.sum(dim=-1, keepdim=True).clamp(min=1.0)
        D_inv_sqrt = torch.pow(D, -0.5)
        norm_A = D_inv_sqrt * A * D_inv_sqrt.transpose(-2, -1)
        return norm_A

    def forward(
        self,
        node_features: torch.Tensor,    # (B, W, N, F_n)
        edge_features: torch.Tensor,    # (B, W, N, N, F_e)
        adjacency: torch.Tensor,        # (B, N, N)
    ) -> torch.Tensor:                  # (B, N, N)
        B, W, N, F_n = node_features.shape

        # 1. Normalize node features across the batch
        # (B, W, N, F) -> (B*W*N, F) -> BN -> back
        x_flat = node_features.reshape(B * W * N, F_n)
        x_flat = self.node_bn(x_flat)
        node_features = x_flat.view(B, W, N, F_n)

        # 2. Temporal encoding
        x = node_features.permute(0, 2, 1, 3).reshape(B * N, W, -1)
        lstm_out, _ = self.lstm(x)
        h = lstm_out[:, -1, :].view(B, N, self.hidden_dim)

        # 3. Pool edge features across temporal window
        # (B, W, N, N, F_e) -> mean across W -> (B, N, N, F_e)
        edge_pooled = edge_features.mean(dim=1)   # (B, N, N, F_e)
        edge_emb = self.edge_proj(edge_pooled.view(-1, self.num_edge_features))
        edge_emb = edge_emb.view(B, N, N, self.hidden_dim)

        # 4. Normalize adjacency
        norm_adj = self._normalize_adj(adjacency)  # (B, N, N)

        # 5. GNN message passing with edge-augmented messages
        for gnn_layer, node_updater in zip(self.gnn_layers, self.node_updaters):
            # Aggregate neighbor info: h_j weighted by adj
            # (B, N, N, H) messages from edge_emb
            messages = torch.einsum("bij,bijn->bin", norm_adj, edge_emb)  # (B, N, H)
            
            # Combine with node features
            combined = torch.cat([h, messages], dim=-1)  # (B, N, 2H)
            updated = node_updater(combined)  # (B, N, H)
            
            # Residual + ReLU
            h = h + updated
            h = F.relu(h)

        # 6. Edge prediction
        h_i = h.unsqueeze(2).expand(B, N, N, self.hidden_dim)
        h_j = h.unsqueeze(1).expand(B, N, N, self.hidden_dim)
        edge_input = torch.cat([h_i, h_j, edge_emb], dim=-1)  # (B, N, N, 3H)
        logits = self.edge_mlp(edge_input).squeeze(-1)
        return torch.sigmoid(logits)

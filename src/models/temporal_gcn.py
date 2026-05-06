"""
TemporalGCN v3
================
Completely rewritten for correctness and small-sample performance.

Key fixes:
- Proper symmetric normalization with self-loops
- Corrected edge message passing with edge feature projections
- Per-step edge features stacked across window, projected and summed
- Cleaner architecture with explicit variable names
- LSTM hidden state + edge-pooled features for node representations
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
        dropout: float = 0.2,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_edge_features = num_edge_features

        # Temporal encoder
        self.lstm = nn.LSTM(
            num_node_features,
            hidden_dim,
            num_layers=1,  # Single layer for small data stability
            batch_first=True,
        )

        # Aggregate edge features from temporal window into edge embeddings
        self.edge_time_proj = nn.Sequential(
            nn.Linear(num_edge_features, hidden_dim),
            nn.ReLU(),
        )

        # GCN layers: W*agg(h) + edge_messages
        self.gcn_layers = nn.ModuleList()
        self.edge_projs = nn.ModuleList()
        self.norms = nn.ModuleList()
        for _ in range(num_gcn_layers):
            self.gcn_layers.append(nn.Linear(hidden_dim, hidden_dim))
            self.edge_projs.append(nn.Linear(hidden_dim, hidden_dim, bias=False))
            self.norms.append(nn.LayerNorm(hidden_dim))

        self.dropout = nn.Dropout(dropout)

        # Edge predictor
        self.edge_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2 + hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

        self._reset_parameters()

    def _reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain("relu"))
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def _normalize_adj(self, adj: torch.Tensor) -> torch.Tensor:
        """Symmetric normalization D̃^{-1/2} Ã D̃^{-1/2} with self-loops."""
        B, N, _ = adj.shape
        dtype = adj.dtype
        device = adj.device
        I = torch.eye(N, device=device, dtype=dtype).unsqueeze(0)  # (1, N, N)
        A = adj + I
        D = A.sum(dim=-1, keepdim=True).clamp(min=1.0)
        D_inv_sqrt = torch.rsqrt(D)
        norm_A = D_inv_sqrt * A * D_inv_sqrt.transpose(-2, -1)
        return norm_A

    def forward(
        self,
        node_features: torch.Tensor,    # (B, W, N, F_n)
        edge_features: torch.Tensor,    # (B, W, N, N, F_e)
        adjacency: torch.Tensor,        # (B, N, N)
    ) -> torch.Tensor:                  # (B, N, N)
        B, W, N, F_n = node_features.shape
        _, _, _, _, F_e = edge_features.shape

        # 1. Temporal encoding: mean pool LSTM outputs
        x = node_features.permute(0, 2, 1, 3).reshape(B * N, W, F_n)    # (B*N, W, F_n)
        lstm_out, _ = self.lstm(x)                                      # (B*N, W, H)
        h = lstm_out[:, -1, :].view(B, N, self.hidden_dim)                # (B, N, H)

        # 2. Edge feature aggregation: project per-step and sum across window
        # (B, W, N, N, F_e) -> (B*W*N*N, F_e) -> project
        e = edge_features.reshape(B * W * N * N, F_e)
        edge_emb_all = self.edge_time_proj(e)                           # (B*W*N*N, H)
        edge_emb_all = edge_emb_all.view(B, W, N, N, self.hidden_dim)
        edge_emb = edge_emb_all.sum(dim=1)                               # (B, N, N, H)  sum across window

        # 3. Normalize adjacency
        norm_adj = self._normalize_adj(adjacency.float())                # (B, N, N)

        # 4. GCN message passing
        for gcn_layer, edge_proj, norm in zip(
            self.gcn_layers, self.edge_projs, self.norms
        ):
            # Node message: aggregate neighbors
            neigh_h = torch.einsum("bij,bjh->bih", norm_adj, h)         # (B, N, H)

            # Edge-conditioned messages
            edge_messages_raw = edge_proj(edge_emb)                       # (B, N, N, H)
            edge_messages = torch.einsum("bij,bijn->bin", norm_adj, edge_messages_raw)

            # Update
            updated = gcn_layer(neigh_h) + edge_messages                # (B, N, H)
            h_prev = h
            h = h + updated                                              # Residual
            h = F.relu(norm(h))
            h = self.dropout(h)

        # 5. Edge prediction
        h_i = h.unsqueeze(2).expand(B, N, N, self.hidden_dim)           # (B, N, N, H)
        h_j = h.unsqueeze(1).expand(B, N, N, self.hidden_dim)           # (B, N, N, H)
        edge_input = torch.cat([h_i, h_j, edge_emb], dim=-1)              # (B, N, N, 3H)
        logits = self.edge_mlp(edge_input).squeeze(-1)                    # (B, N, N)
        return torch.sigmoid(logits)

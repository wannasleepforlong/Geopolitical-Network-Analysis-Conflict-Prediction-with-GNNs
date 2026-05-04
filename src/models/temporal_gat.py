"""
Temporal GAT Model
==================
LSTM encoder + Graph Attention Network for conflict prediction.
Returns attention weights for interpretability.
"""
from __future__ import annotations

from typing import Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class GraphAttentionLayer(nn.Module):
    """Single multi-head GAT layer."""

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        num_heads: int = 4,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = out_dim // num_heads
        self.out_dim = out_dim

        self.W = nn.Linear(in_dim, out_dim, bias=False)
        self.a_src = nn.Parameter(torch.Tensor(num_heads, self.head_dim, 1))
        self.a_dst = nn.Parameter(torch.Tensor(num_heads, self.head_dim, 1))
        self.leaky_relu = nn.LeakyReLU(0.2)
        self.dropout = nn.Dropout(dropout)

        nn.init.xavier_uniform_(self.a_src)
        nn.init.xavier_uniform_(self.a_dst)

    def forward(
        self,
        h: torch.Tensor,                # (B, N, in_dim)
        adj: torch.Tensor,              # (B, N, N)
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns (h_out, attn_weights) where attn is (B, H, N, N)."""
        B, N, _ = h.shape
        Wh = self.W(h)                                                  # (B, N, out_dim)
        Wh = Wh.view(B, N, self.num_heads, self.head_dim)
        Wh = Wh.permute(0, 2, 1, 3).contiguous()                      # (B, H, N, d)

        # Attention scores
        e_src = torch.matmul(Wh, self.a_src).squeeze(-1)              # (B, H, N)
        e_dst = torch.matmul(Wh, self.a_dst).squeeze(-1)              # (B, H, N)
        e = e_src.unsqueeze(-1) + e_dst.unsqueeze(-2)                  # (B, H, N, N)
        e = self.leaky_relu(e)

        # Mask with adjacency
        mask = adj.unsqueeze(1).expand(B, self.num_heads, N, N).float()
        e = e.masked_fill(mask == 0, float("-inf"))

        alpha = F.softmax(e, dim=-1)                                    # (B, H, N, N)
        alpha = torch.nan_to_num(alpha, 0.0)
        alpha = self.dropout(alpha)

        h_out = torch.matmul(alpha, Wh)                                # (B, H, N, d)
        h_out = h_out.permute(0, 2, 1, 3).contiguous().view(B, N, self.out_dim)
        return h_out, alpha


class TemporalGAT(nn.Module):
    """
    Parameters
    ----------
    num_node_features : int
    num_edge_features : int
    hidden_dim : int
    num_heads : int
    num_gat_layers : int
    dropout : float
    """

    def __init__(
        self,
        num_node_features: int = 6,
        num_edge_features: int = 5,
        hidden_dim: int = 64,
        num_heads: int = 4,
        num_gat_layers: int = 2,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads

        # Temporal encoder
        self.lstm = nn.LSTM(
            num_node_features,
            hidden_dim,
            num_layers=2,
            batch_first=True,
            dropout=dropout if 2 > 1 else 0.0,
        )

        # GAT layers
        self.gat_layers = nn.ModuleList()
        for i in range(num_gat_layers):
            in_dim = hidden_dim if i == 0 else hidden_dim
            self.gat_layers.append(
                GraphAttentionLayer(in_dim, hidden_dim, num_heads, dropout)
            )

        # Edge feature fusion
        self.edge_proj = nn.Linear(num_edge_features, hidden_dim)

        self.norms = nn.ModuleList(
            [nn.LayerNorm(hidden_dim) for _ in range(num_gat_layers)]
        )
        self.dropout = nn.Dropout(dropout)

        # Predictor
        self.edge_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
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
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Returns (predictions, attention_weights) where attention is (B, H, N, N)."""
        B, W, N, _ = node_features.shape

        # Temporal encode
        x = node_features.permute(0, 2, 1, 3).reshape(B * N, W, -1)
        lstm_out, _ = self.lstm(x)
        h = lstm_out[:, -1, :].view(B, N, self.hidden_dim)

        # GAT with attention
        attn_weights = None
        for i, (gat, norm) in enumerate(zip(self.gat_layers, self.norms)):
            h_gat, attn = gat(h, adjacency)
            if i == len(self.gat_layers) - 1:
                attn_weights = attn  # keep last layer attention
            h = F.elu(norm(h + h_gat))
            h = self.dropout(h)

        # Edge-augmented prediction
        edge_last = edge_features[:, -1, :, :, :]                     # (B, N, N, F_e)
        edge_emb = self.edge_proj(edge_last)                           # (B, N, N, H)

        h_i = h.unsqueeze(2).expand(B, N, N, self.hidden_dim)
        h_j = h.unsqueeze(1).expand(B, N, N, self.hidden_dim)
        combined = torch.cat([h_i, h_j, edge_emb], dim=-1)
        logits = self.edge_mlp(combined).squeeze(-1)
        return torch.sigmoid(logits), attn_weights

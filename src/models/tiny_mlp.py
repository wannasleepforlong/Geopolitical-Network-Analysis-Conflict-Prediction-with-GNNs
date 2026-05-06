"""
TinyMLP Model
================
A very small MLP baseline that should work well on tiny datasets.

No LSTM, no GCN, no GAT. Just temporal aggregation + small MLP.
This is meant to prove that the issue is model size, not data quality.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class TinyMLP(nn.Module):
    """
    Ultra-small MLP that averages node/edge features over time window
    and outputs edge predictions.
    
    Total params: ~200 (vs 18,000 for GCN/GAT)
    
    Parameters
    ----------
    num_node_features : int
    num_edge_features : int  
    hidden_dim : int
    """
    def __init__(
        self,
        num_node_features: int = 6,
        num_edge_features: int = 5,
        hidden_dim: int = 8,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        # Tiny MLP: node_i, node_j, edge -> prediction
        self.predictor = nn.Sequential(
            nn.Linear(num_node_features * 2 + num_edge_features, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        
    def forward(
        self,
        node_features: torch.Tensor,    # (B, W, N, F_n)
        edge_features: torch.Tensor,    # (B, W, N, N, F_e)
        adjacency: torch.Tensor,        # (B, N, N) - unused
    ) -> torch.Tensor:
        B, W, N, F_n = node_features.shape
        _, _, _, _, F_e = edge_features.shape
        
        # Average node features across time window
        h_nodes = node_features.mean(dim=1)  # (B, N, F_n)
        
        # Average edge features across time window  
        e = edge_features.mean(dim=1)        # (B, N, N, F_e)
        
        # Expand node features for each dyad
        h_i = h_nodes.unsqueeze(2).expand(B, N, N, F_n)  # (B, N, N, F_n)
        h_j = h_nodes.unsqueeze(1).expand(B, N, N, F_n)  # (B, N, N, F_n)
        
        # Concatenate and predict
        x = torch.cat([h_i, h_j, e], dim=-1)  # (B, N, N, 2*F_n + F_e)
        logits = self.predictor(x).squeeze(-1)  # (B, N, N)
        return torch.sigmoid(logits)

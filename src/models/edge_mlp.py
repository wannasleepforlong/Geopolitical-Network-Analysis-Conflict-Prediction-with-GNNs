"""
Edge MLP Baseline
=================
Simple but strong baseline: use edge features directly through MLP.
No graph convolutions, just feedforward on edge features + node embeddings.

This should perform better than GNNs on tiny datasets because it doesn't
need to learn graph structure.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class EdgeMLP(nn.Module):
    """
    Strong baseline that uses edge features + simple node temporal encoding.
    
    Parameters
    ----------
    num_node_features : int
    num_edge_features : int  
    hidden_dim : int
    dropout : float
    """
    def __init__(
        self,
        num_node_features: int = 6,
        num_edge_features: int = 5,
        hidden_dim: int = 64,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        # Simple node encoder: average node features across window
        self.node_encoder = nn.Sequential(
            nn.Linear(num_node_features, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
        
        # Edge feature encoder
        self.edge_encoder = nn.Sequential(
            nn.Linear(num_edge_features, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        
        # Final predictor
        self.predictor = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim * 2),  # [source, target, edge]
            nn.LayerNorm(hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        
    def forward(
        self,
        node_features: torch.Tensor,    # (B, W, N, F_n)
        edge_features: torch.Tensor,    # (B, W, N, N, F_e)
        adjacency: torch.Tensor,        # (B, N, N) - not used but kept for API compatibility
    ) -> torch.Tensor:
        B, W, N, F_n = node_features.shape
        _, _, _, _, F_e = edge_features.shape
        
        # Encode nodes: mean across window
        h_nodes = node_features.mean(dim=1)  # (B, N, F_n)
        h_nodes = self.node_encoder(h_nodes)    # (B, N, H)
        
        # Encode edges: use LAST time step features directly
        e = edge_features[:, -1, :, :, :]     # (B, N, N, F_e)
        e_emb = self.edge_encoder(e)           # (B, N, N, H)
        
        # Combine source, target, and edge
        h_i = h_nodes.unsqueeze(2).expand(B, N, N, self.hidden_dim)
        h_j = h_nodes.unsqueeze(1).expand(B, N, N, self.hidden_dim)
        
        combined = torch.cat([h_i, h_j, e_emb], dim=-1)  # (B, N, N, 3H)
        logits = self.predictor(combined).squeeze(-1)     # (B, N, N)
        return torch.sigmoid(logits)

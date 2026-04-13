"""
Visualization and Interpretability Analysis
=============================================
Advanced visualizations and analysis for understanding model predictions.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import networkx as nx
from typing import Dict, List, Tuple, Optional
import torch
import json

logger_import = __import__('logging').getLogger(__name__)


class ConflictNetworkVisualizer:
    """Visualize geopolitical networks and conflict predictions."""
    
    def __init__(self, output_dir: str = "./gdelt_visualizations"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
        # Color palette
        self.colors = {
            'conflict': '#e74c3c',
            'cooperation': '#2ecc71',
            'neutral': '#95a5a6',
        }
    
    def visualize_network_snapshot(
        self,
        adjacency_conflict: np.ndarray,
        adjacency_cooperation: np.ndarray,
        country_codes: Dict[int, str],
        title: str = "Geopolitical Network Snapshot",
        figsize: Tuple = (14, 12),
    ):
        """
        Visualize a single network snapshot with both conflict and cooperation edges.
        
        Args:
            adjacency_conflict: (num_countries, num_countries) conflict matrix
            adjacency_cooperation: (num_countries, num_countries) cooperation matrix
            country_codes: {node_idx: country_code}
            title: Plot title
            figsize: Figure size
        """
        
        num_countries = adjacency_conflict.shape[0]
        
        # Create graph
        G = nx.DiGraph()
        
        # Add nodes
        for i in range(num_countries):
            G.add_node(i)
        
        # Add conflict edges
        conflict_edges = []
        for i in range(num_countries):
            for j in range(num_countries):
                if adjacency_conflict[i, j] > 0:
                    conflict_edges.append((i, j, adjacency_conflict[i, j]))
                    G.add_edge(i, j, weight=adjacency_conflict[i, j], type='conflict')
        
        # Add cooperation edges
        cooperation_edges = []
        for i in range(num_countries):
            for j in range(num_countries):
                if adjacency_cooperation[i, j] > 0:
                    cooperation_edges.append((i, j, adjacency_cooperation[i, j]))
                    if not G.has_edge(i, j):
                        G.add_edge(i, j, weight=adjacency_cooperation[i, j], type='cooperation')
        
        # Layout
        pos = nx.spring_layout(G, k=2, iterations=50, seed=42)
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # Draw cooperation edges (green, thinner)
        cooperation_edge_list = [
            (u, v) for u, v, d in G.edges(data=True)
            if d.get('type') == 'cooperation'
        ]
        cooperation_weights = [
            G[u][v].get('weight', 1)
            for u, v in cooperation_edge_list
        ]
        
        nx.draw_networkx_edges(
            G, pos,
            edgelist=cooperation_edge_list,
            width=[w * 2 for w in cooperation_weights],
            edge_color=self.colors['cooperation'],
            edge_cmap=plt.cm.Greens,
            alpha=0.6,
            arrows=True,
            arrowsize=15,
            ax=ax,
        )
        
        # Draw conflict edges (red, thicker)
        conflict_edge_list = [
            (u, v) for u, v, d in G.edges(data=True)
            if d.get('type') == 'conflict'
        ]
        conflict_weights = [
            G[u][v].get('weight', 1)
            for u, v in conflict_edge_list
        ]
        
        nx.draw_networkx_edges(
            G, pos,
            edgelist=conflict_edge_list,
            width=[w * 3 for w in conflict_weights],
            edge_color=self.colors['conflict'],
            edge_cmap=plt.cm.Reds,
            alpha=0.8,
            arrows=True,
            arrowsize=15,
            ax=ax,
        )
        
        # Draw nodes
        node_sizes = []
        for i in range(num_countries):
            # Size proportional to total degree
            deg = adjacency_conflict[i, :].sum() + adjacency_cooperation[i, :].sum()
            node_sizes.append(500 + deg * 100)
        
        nx.draw_networkx_nodes(
            G, pos,
            node_color='#3498db',
            node_size=node_sizes,
            alpha=0.9,
            ax=ax,
        )
        
        # Draw labels
        labels = {i: country_codes.get(i, f"Node{i}") for i in range(num_countries)}
        nx.draw_networkx_labels(
            G, pos,
            labels=labels,
            font_size=10,
            font_weight='bold',
            ax=ax,
        )
        
        # Legend
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], color='w', marker='o', markerfacecolor='#3498db', markersize=10),
            Line2D([0], [0], color=self.colors['conflict'], linewidth=3, label='Conflict'),
            Line2D([0], [0], color=self.colors['cooperation'], linewidth=2, label='Cooperation'),
        ]
        ax.legend(handles=legend_elements, loc='upper left', fontsize=12)
        
        ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
        ax.axis('off')
        
        plt.tight_layout()
        return fig
    
    def visualize_conflict_heatmap(
        self,
        predictions: np.ndarray,
        country_codes: Dict[int, str],
        title: str = "Predicted Conflict Risk",
        figsize: Tuple = (12, 10),
    ):
        """
        Heatmap of conflict predictions between country pairs.
        
        Args:
            predictions: (num_countries, num_countries) prediction matrix
            country_codes: {node_idx: country_code}
            title: Plot title
        """
        
        num_countries = predictions.shape[0]
        country_list = [country_codes.get(i, f"Node{i}") for i in range(num_countries)]
        
        fig, ax = plt.subplots(figsize=figsize)
        
        sns.heatmap(
            predictions,
            annot=True,
            fmt='.3f',
            cmap='RdYlGn_r',
            xticklabels=country_list,
            yticklabels=country_list,
            cbar_kws={'label': 'Conflict Risk'},
            ax=ax,
            vmin=0,
            vmax=1,
        )
        
        ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
        ax.set_xlabel('Target Country', fontsize=12)
        ax.set_ylabel('Source Country', fontsize=12)
        
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        
        return fig
    
    def visualize_temporal_evolution(
        self,
        networks: List[Dict],
        periods: List[str],
        country_codes: Dict[int, str],
        figsize: Tuple = (16, 10),
    ):
        """
        Visualize network evolution over time.
        
        Args:
            networks: List of network dicts
            periods: List of time period strings
            country_codes: Mapping from index to country code
        """
        
        num_periods = min(6, len(networks))  # Show last 6 periods
        indices = np.linspace(0, len(networks) - 1, num_periods, dtype=int)
        
        fig, axes = plt.subplots(2, 3, figsize=figsize)
        axes = axes.flatten()
        
        for plot_idx, net_idx in enumerate(indices):
            network = networks[net_idx]
            period = periods[net_idx]
            
            adj_conflict = network['adjacency_conflict']
            
            # Normalize for visualization
            adj_norm = (adj_conflict - adj_conflict.min()) / (adj_conflict.max() - adj_conflict.min() + 1e-8)
            
            im = axes[plot_idx].imshow(adj_norm, cmap='Reds', aspect='auto')
            axes[plot_idx].set_title(f"Period: {period}", fontweight='bold')
            axes[plot_idx].set_xlabel('Target')
            axes[plot_idx].set_ylabel('Source')
            
            # Add colorbar
            plt.colorbar(im, ax=axes[plot_idx], label='Conflict Intensity')
        
        # Hide unused subplots
        for idx in range(num_periods, len(axes)):
            axes[idx].axis('off')
        
        fig.suptitle('Network Evolution Over Time', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        return fig
    
    def visualize_attention_weights(
        self,
        attention_weights: torch.Tensor,
        country_codes: Dict[int, str],
        top_edges: int = 15,
        figsize: Tuple = (12, 8),
    ):
        """
        Visualize attention weights from GAT model.
        
        Args:
            attention_weights: Attention weight tensor
            country_codes: Mapping from index to country code
            top_edges: Number of top attention edges to show
        """
        
        if isinstance(attention_weights, torch.Tensor):
            attention_weights = attention_weights.cpu().numpy()
        
        # Flatten and get top edges
        flat_attn = attention_weights.flatten()
        top_indices = np.argsort(flat_attn)[-top_edges:][::-1]
        
        # Convert indices back to edge pairs
        num_nodes = attention_weights.shape[0]
        edges = []
        edge_weights = []
        
        for idx in top_indices:
            i, j = divmod(idx, num_nodes)
            if i != j:  # Skip self-loops
                c1 = country_codes.get(i, f"Node{i}")
                c2 = country_codes.get(j, f"Node{j}")
                edges.append(f"{c1}→{c2}")
                edge_weights.append(flat_attn[idx])
        
        fig, ax = plt.subplots(figsize=figsize)
        
        ax.barh(edges[::-1], edge_weights[::-1], color='#3498db')
        ax.set_xlabel('Attention Weight', fontsize=12)
        ax.set_title('Top Attention Edges (GAT Model)', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='x')
        
        plt.tight_layout()
        return fig
    
    def save_figure(self, fig, filename: str):
        """Save figure to disk."""
        path = self.output_dir / filename
        fig.savefig(path, dpi=300, bbox_inches='tight')
        logger_import.info(f"Saved: {path}")
        plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Analysis
# ─────────────────────────────────────────────────────────────────────────────

class ConflictPredictionAnalyzer:
    """Analyze model predictions and extract insights."""
    
    def __init__(self, metadata: Dict):
        self.metadata = metadata
        self.country_indices = metadata['country_indices']
        self.reverse_indices = {v: k for k, v in self.country_indices.items()}
    
    def analyze_predictions(
        self,
        predictions: np.ndarray,
        country_pairs: Optional[List[Tuple[str, str]]] = None,
    ) -> Dict:
        """
        Analyze prediction patterns and generate insights.
        
        Args:
            predictions: (num_countries, num_countries) prediction matrix
            country_pairs: Optional list of specific pairs to analyze
            
        Returns:
            Dictionary with analysis results
        """
        
        analysis = {
            'global_stats': {
                'mean_risk': float(predictions.mean()),
                'max_risk': float(predictions.max()),
                'min_risk': float(predictions.min()),
                'std_risk': float(predictions.std()),
            },
            'risk_distribution': self._analyze_risk_distribution(predictions),
            'high_risk_pairs': self._get_high_risk_pairs(predictions, top_k=20),
        }
        
        if country_pairs:
            analysis['specific_pairs'] = self._analyze_specific_pairs(
                predictions, country_pairs
            )
        
        return analysis
    
    def _analyze_risk_distribution(self, predictions: np.ndarray) -> Dict:
        """Analyze distribution of risk scores."""
        
        flat = predictions.flatten()
        
        return {
            'mean': float(np.mean(flat)),
            'median': float(np.median(flat)),
            'percentile_25': float(np.percentile(flat, 25)),
            'percentile_75': float(np.percentile(flat, 75)),
            'percentile_90': float(np.percentile(flat, 90)),
            'percentile_95': float(np.percentile(flat, 95)),
        }
    
    def _get_high_risk_pairs(
        self,
        predictions: np.ndarray,
        top_k: int = 20,
    ) -> List[Dict]:
        """Get top-k high risk country pairs."""
        
        num_countries = predictions.shape[0]
        
        pairs = []
        for i in range(num_countries):
            for j in range(num_countries):
                if i != j:
                    pairs.append({
                        'source': self.reverse_indices.get(i, f"Node{i}"),
                        'target': self.reverse_indices.get(j, f"Node{j}"),
                        'risk': float(predictions[i, j]),
                    })
        
        pairs.sort(key=lambda x: x['risk'], reverse=True)
        return pairs[:top_k]
    
    def _analyze_specific_pairs(
        self,
        predictions: np.ndarray,
        country_pairs: List[Tuple[str, str]],
    ) -> Dict:
        """Analyze specific country pairs."""
        
        results = {}
        
        for c1, c2 in country_pairs:
            if c1 in self.country_indices and c2 in self.country_indices:
                i = self.country_indices[c1]
                j = self.country_indices[c2]
                
                results[f"{c1}-{c2}"] = {
                    'forward_risk': float(predictions[i, j]),
                    'reverse_risk': float(predictions[j, i]),
                    'asymmetry': float(abs(predictions[i, j] - predictions[j, i])),
                }
        
        return results
    
    def generate_report(self, analysis: Dict) -> str:
        """Generate human-readable analysis report."""
        
        report = []
        report.append("=" * 80)
        report.append("GEOPOLITICAL CONFLICT PREDICTION ANALYSIS REPORT")
        report.append("=" * 80)
        report.append("")
        
        # Global statistics
        report.append("GLOBAL STATISTICS")
        report.append("-" * 80)
        stats = analysis['global_stats']
        report.append(f"Mean Risk Score:        {stats['mean_risk']:.4f}")
        report.append(f"Max Risk Score:         {stats['max_risk']:.4f}")
        report.append(f"Min Risk Score:         {stats['min_risk']:.4f}")
        report.append(f"Std Dev:                {stats['std_risk']:.4f}")
        report.append("")
        
        # Risk distribution
        report.append("RISK DISTRIBUTION")
        report.append("-" * 80)
        dist = analysis['risk_distribution']
        report.append(f"Median:                 {dist['median']:.4f}")
        report.append(f"25th Percentile:        {dist['percentile_25']:.4f}")
        report.append(f"75th Percentile:        {dist['percentile_75']:.4f}")
        report.append(f"95th Percentile:        {dist['percentile_95']:.4f}")
        report.append("")
        
        # High risk pairs
        report.append("TOP 20 HIGH-RISK COUNTRY PAIRS")
        report.append("-" * 80)
        for i, pair in enumerate(analysis['high_risk_pairs'], 1):
            report.append(
                f"{i:2d}. {pair['source']:4s} → {pair['target']:4s}  Risk: {pair['risk']:.4f}"
            )
        report.append("")
        
        return "\n".join(report)


# ─────────────────────────────────────────────────────────────────────────────
# Example usage
# ─────────────────────────────────────────────────────────────────────────────

def main():
    """Example visualization and analysis."""
    
    # Load metadata
    with open('./gdelt_processed_data/metadata.json') as f:
        metadata = json.load(f)
    
    reverse_indices = {v: k for k, v in metadata['country_indices'].items()}
    
    # Create visualizer
    visualizer = ConflictNetworkVisualizer('./gdelt_visualizations')
    
    # Create analyzer
    analyzer = ConflictPredictionAnalyzer(metadata)
    
    # Example: analyze a random prediction matrix
    num_countries = len(metadata['country_indices'])
    example_predictions = np.random.rand(num_countries, num_countries)
    
    # Analyze
    analysis = analyzer.analyze_predictions(example_predictions)
    
    # Generate report
    report = analyzer.generate_report(analysis)
    print(report)
    
    # Save report
    with open('./gdelt_visualizations/analysis_report.txt', 'w') as f:
        f.write(report)


if __name__ == "__main__":
    main()

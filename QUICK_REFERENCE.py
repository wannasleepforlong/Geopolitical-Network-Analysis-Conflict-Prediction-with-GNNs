"""
GDELT Geopolitical Network Analysis & Conflict Prediction with GNNs
===================================================================
"""

# ════════════════════════════════════════════════════════════════════════════
# PROJECT AT A GLANCE
# ════════════════════════════════════════════════════════════════════════════

PROJECT_TITLE = "Geopolitical Network Analysis & Conflict Prediction with GNNs"

PROBLEM_STATEMENT = """
Can we predict which country pairs will experience conflict escalation
using:
  - GDELT: 30+ years of global geopolitical event data
  - Temporal Networks: Dynamic country interaction networks
  - Graph Neural Networks: Deep learning on graph structures
  
This addresses a complex systems problem: understanding emergence,
interdependencies, and critical transitions in geopolitical systems.
"""

KEY_CHALLENGES = [
    "Sparse, imbalanced data (few conflict escalations)",
    "Temporal dynamics (multi-scale causality)",
    "Network heterogeneity (multiple interaction types)",
    "Interpretability (why predict conflict for this pair?)",
    "Scale (190+ countries, 30+ years)",
]

# ════════════════════════════════════════════════════════════════════════════
# TECHNICAL APPROACH
# ════════════════════════════════════════════════════════════════════════════

ARCHITECTURE = """
                       ┌─────────────────────────────┐
                       │   GDELT Event Stream        │
                       │  (30M+ events, 2015-2023)   │
                       └──────────────┬──────────────┘
                                      │
                       ┌──────────────▼──────────────┐
                       │  GDELTEventCollector        │
                       │  - Download historical data │
                       │  - Filter by countries      │
                       │  - Cache for efficiency     │
                       └──────────────┬──────────────┘
                                      │
                       ┌──────────────▼──────────────┐
                       │ GeopoliticalNetworkBuilder  │
                       │  - Monthly aggregation      │
                       │  - Build country networks   │
                       │  - Extract conflict/coop    │
                       │  - Compute statistics       │
                       └──────────────┬──────────────┘
                                      │
                       ┌──────────────▼──────────────┐
                       │   GNNDataPreprocessor       │
                       │  - Node features (6 types)  │
                       │  - Edge features (5 types)  │
                       │  - Create time windows      │
                       │  - Generate binary labels   │
                       └──────────────┬──────────────┘
                                      │
      ┌─────────────────────┬─────────▼─────────┬─────────────────────┐
      │                     │                   │                     │
  ┌───▼────────┐      ┌────▼──────┐      ┌────▼──────┐      ┌─────▼───┐
  │ Train Set  │      │  Val Set  │      │ Test Set  │      │ Analysis│
  │ (70%)      │      │  (10%)    │      │ (20%)     │      │ Predict │
  └───┬────────┘      └────┬──────┘      └────┬──────┘      └─────┬───┘
      │                    │                   │                   │
      │  ┌────────────────▼────────────────┐  │                   │
      │  │   Training Pipeline             │  │                   │
      │  │  ┌─────────────────────────────┐│  │                   │
      │  │  │ TemporalGAT / TemporalGCN   ││  │                   │
      │  │  │  - LSTM encoder (temporal)  ││  │                   │
      │  │  │  - GAT/GCN layers (spatial) ││  │                   │
      │  │  │  - Edge predictor (binary)  ││  │                   │
      │  │  └─────────────────────────────┘│  │                   │
      │  │  Loss: Masked BCE               │  │                   │
      │  │  Optimizer: Adam + scheduler    │  │                   │
      │  │  Early stopping: patience=10    │  │                   │
      │  └────────────────┬────────────────┘  │                   │
      │                   │                   │                   │
      │           ┌───────▼────────┐          │                   │
      │           │   Checkpoints  │          │                   │
      │           │   + Metrics    │          │                   │
      │           └────────────────┘          │                   │
      │                                       │                   │
      │                   ┌───────────────────▼─────────────┐     │
      │                   │  Final Evaluation               │     │
      │                   │  - ROC-AUC                      │     │
      │                   │  - F1 Score                     │     │
      │                   │  - Precision/Recall             │     │
      │                   │  - Confusion Matrix             │     │
      │                   └───────────────────┬─────────────┘     │
      │                                       │                   │
      │                           ┌───────────▼────────────┐      │
      │                           │ Top-k Predictions      │◄─────┘
      │                           │ (Most risky pairs)     │
      │                           └───────────────────────┘
      │
      └──────────────────────────────────────────────────────────────►
                        Hyperparameter Tuning Loop
"""

# ════════════════════════════════════════════════════════════════════════════
# QUICK START COMMANDS
# ════════════════════════════════════════════════════════════════════════════

QUICK_START = """
# 1. Install dependencies
pip install -r requirements.txt

# 2. Fetch GDELT data & build networks
python data_pipeline.py
# Output: ./gdelt_processed_data/
#   ├─ node_features.npy (T, N, 6)
#   ├─ edge_features.npy (T, N, N, 5)
#   ├─ edge_labels.npy (T, N, N)
#   └─ metadata.json

# 3. Train model
python train.py
# Output: ./gdelt_results/ & ./gdelt_checkpoints/
#   ├─ training_curves.png
#   ├─ test_metrics.png
#   └─ results.json

# 4. Visualize & Analyze
python visualization.py
# Output: ./gdelt_visualizations/
#   ├─ conflict_heatmap.png
#   ├─ network_snapshot.png
#   └─ analysis_report.txt

# 5. View results
firefox gdelt_results/training_curves.png
cat gdelt_visualizations/analysis_report.txt
"""

# ════════════════════════════════════════════════════════════════════════════
# KEY RESULTS METRICS
# ════════════════════════════════════════════════════════════════════════════

EXPECTED_RESULTS = {
    'test_auc': 0.75,           # ROC-AUC on held-out test set
    'test_f1': 0.68,            # F1 score
    'test_accuracy': 0.72,      # Overall accuracy
    'training_time': '2-5 min', # On GPU
    'num_parameters': '~50k',   # Model size
}

# ════════════════════════════════════════════════════════════════════════════
# FILE DESCRIPTIONS
# ════════════════════════════════════════════════════════════════════════════

FILES = {
    'data_pipeline.py': """
        GDELT data collection and network building.
        
        Key Classes:
          - GDELTEventCollector: Download GDELT events
          - GeopoliticalNetworkBuilder: Create temporal networks
          - GNNDataPreprocessor: Feature engineering
        
        Main Function:
          python data_pipeline.py
        
        Output: gdelt_processed_data/
        
        Dependencies: requests, pandas, numpy, pickle
    """,
    
    'models.py': """
        Graph Neural Network architectures.
        
        Key Classes:
          - TemporalGCN: LSTM + Graph Convolution
          - TemporalGAT: LSTM + Graph Attention (interpretable)
          - GeopoliticalNetworkDataset: PyTorch dataset wrapper
          - ConflictPredictionTrainer: Training utilities
        
        Usage:
          model = TemporalGAT(num_node_features=6, 
                              num_edge_features=5)
          predictions, attention = model(node_feat, edge_idx, 
                                          edge_feat, adjacency)
        
        Dependencies: torch, torch_geometric
    """,
    
    'train.py': """
        Full training pipeline with evaluation.
        
        Components:
          - ConflictPredictionPipeline: Main trainer class
          - Data loading & preprocessing
          - Train/val/test loop with early stopping
          - Checkpoint saving & recovery
          - Metrics computation & visualization
        
        Usage:
          python train.py
        
        Configuration: Edit CONFIG dict at top
        
        Output: gdelt_results/, gdelt_checkpoints/
    """,
    
    'visualization.py': """
        Visualization and interpretability analysis.
        
        Key Classes:
          - ConflictNetworkVisualizer: Network/heatmap plots
          - ConflictPredictionAnalyzer: Statistical analysis
        
        Plots:
          - Network snapshot (nodes = countries, edges = interactions)
          - Conflict risk heatmap
          - Temporal evolution
          - Attention weights (GAT)
        
        Usage:
          visualizer = ConflictNetworkVisualizer()
          fig = visualizer.visualize_conflict_heatmap(predictions)
    """,
}

# ════════════════════════════════════════════════════════════════════════════
# DATASET DETAILS
# ════════════════════════════════════════════════════════════════════════════

DATASET_INFO = """
GDELT Dataset Characteristics:
  - Coverage: 190+ countries, 2015-present
  - Update frequency: 15 minutes
  - Events: 30M+ per year
  - Source: 100k+ news sources
  
Preprocessing Pipeline:
  1. Download GDELT 2.0 events (CSV)
  2. Filter for analysis period (2020-2023)
  3. Filter for countries of interest
  4. Aggregate to monthly networks
  5. Extract 6 node features per country
  6. Extract 5 edge features per pair
  7. Create 12-month sliding windows
  8. Generate binary labels (conflict escalation)
  
Final Dataset Shape:
  - Node features: (T=50, N=7, F=6) 
    → 50 monthly periods, 7 countries, 6 features
  - Edge features: (T=50, N=7, N=7, F=5)
    → Interaction features for all pairs
  - Labels: (T=50, N=7, N=7)
    → Binary escalation prediction
  - Total samples: ~350 (with 12-month window)
"""

# ════════════════════════════════════════════════════════════════════════════
# MODEL HYPERPARAMETERS
# ════════════════════════════════════════════════════════════════════════════

HYPERPARAMETERS = """
Neural Network:
  - Model type: TemporalGAT (preferred) / TemporalGCN
  - Hidden dimension: 64
  - Num GNN layers: 2
  - Num attention heads: 4 (for GAT)
  - Dropout: 0.3
  - Temporal window: 12 months
  
Training:
  - Batch size: 4
  - Learning rate: 1e-3
  - Optimizer: Adam
  - Weight decay: 1e-5
  - Max epochs: 50
  - Early stopping patience: 10 epochs
  - Loss function: Masked Binary Cross-Entropy
  
Data Split:
  - Train: 70% (for learning)
  - Validation: 10% (for hyperparameter tuning)
  - Test: 20% (for final evaluation)
  
Device:
  - Default: GPU (CUDA) if available, else CPU
  - For CPU training: Set device='cpu' in config
"""

# ════════════════════════════════════════════════════════════════════════════
# EVALUATION METRICS EXPLAINED
# ════════════════════════════════════════════════════════════════════════════

METRICS_EXPLAINED = """
ROC-AUC (Receiver Operating Characteristic - Area Under Curve):
  - Range: 0.0 to 1.0 (higher is better)
  - Meaning: Probability model ranks positive samples higher
  - Robust to class imbalance (few positive examples)
  - 0.5 = random, 0.7+ = good, 0.9+ = excellent
  
F1 Score:
  - Range: 0.0 to 1.0 (higher is better)
  - Meaning: Harmonic mean of precision and recall
  - Good for imbalanced data
  - Balances false positives and false negatives
  
Precision:
  - Range: 0.0 to 1.0
  - Meaning: Of predicted conflicts, how many were correct?
  - "Don't cry wolf" metric
  
Recall:
  - Range: 0.0 to 1.0
  - Meaning: Of actual conflicts, how many did we catch?
  - "Don't miss conflicts" metric
  
Accuracy:
  - Range: 0.0 to 1.0
  - Meaning: Overall correct predictions
  - Can be misleading with imbalanced data
  
Interpretation:
  - AUC > 0.75: Good predictive power
  - F1 > 0.65: Useful for early warning system
  - Precision > 0.70: Few false alarms
  - Recall > 0.60: Catch most real conflicts
"""

# ════════════════════════════════════════════════════════════════════════════
# COMPLEX SYSTEMS ASPECTS
# ════════════════════════════════════════════════════════════════════════════

COMPLEX_SYSTEMS = """
This project demonstrates key complex systems concepts:

1. EMERGENCE
   - Global conflict patterns emerge from local interactions
   - Individual events aggregate to network-level tensions
   - No single cause, but cumulative effect

2. INTERDEPENDENCIES
   - Countries linked through networks of interactions
   - Actions in one region affect distant regions
   - Multiplex relationships (trade, military, diplomatic)

3. FEEDBACK LOOPS
   - Negative feedback: Cooperation reduces future conflict
   - Positive feedback: Conflict escalation breeds more conflict
   - Model captures these dynamics over time

4. CRITICAL TRANSITIONS (Tipping Points)
   - Goldstein scale approaching -8 to -10 indicates war risk
   - Sentiment suddenly becoming negative
   - Small event can trigger large cascade

5. TEMPORAL DYNAMICS
   - Past events influence future (12-month history)
   - Non-linear relationships in conflict escalation
   - Multiple timescales (tactical, strategic, systemic)

6. HETEROGENEITY
   - Different countries behave differently
   - Different interaction types (military vs. diplomatic)
   - Node/edge heterogeneity in graph

7. NETWORK STRUCTURE
   - Small-world properties (clustering + short paths)
   - Hub countries (USA, Russia, China) have outsized influence
   - Community structure (regional alliances)
"""

# ════════════════════════════════════════════════════════════════════════════
# TROUBLESHOOTING & TIPS
# ════════════════════════════════════════════════════════════════════════════

TROUBLESHOOTING = """
Problem: GDELT API timeout
Solution: Increase retry timeout in data_pipeline.py
  GDELTEventCollector._REQUEST_TIMEOUT = 60

Problem: Out of memory (GPU)
Solution: Reduce batch size or model size
  batch_size = 2 (instead of 4)
  hidden_dim = 32 (instead of 64)

Problem: Poor performance (low AUC)
Solution: 
  1. Check data quality (run exploratory analysis)
  2. Adjust temporal window (12 or 6 months)
  3. Fine-tune learning rate (1e-4 to 1e-2)
  4. Use TemporalGAT instead of TemporalGCN
  5. Increase training epochs (100+)

Problem: Class imbalance (few conflicts)
Solution:
  1. Use weighted loss function
  2. Adjust escalation threshold
  3. Focus on high-risk countries
  4. Use F1 score instead of accuracy

Tips for Best Results:
  - Start with small dataset (1-2 countries)
  - Verify data looks reasonable (visualizations)
  - Monitor training curves during training
  - Save best model checkpoint
  - Validate on held-out test set
  - Test on recent data (2023 events)
"""

# ════════════════════════════════════════════════════════════════════════════
# FURTHER IMPROVEMENTS
# ════════════════════════════════════════════════════════════════════════════

FUTURE_WORK = """
Suggested Enhancements:

1. Multi-Task Learning
   - Jointly predict conflict AND cooperation
   - Share representations between tasks

2. Hierarchical Models
   - Country-level + region-level predictions
   - Cascade models (early warning signals)

3. Heterogeneous Graphs
   - Different edge types: military, trade, diplomatic
   - Multi-relational GNNs (RGCN, HAN, HGT)

4. Causal Analysis
   - Which events trigger escalation?
   - Counterfactual: "What if event X didn't happen?"

5. Longer Forecasts
   - Predict 6 months ahead (instead of 1)
   - Uncertainty quantification

6. Real-time System
   - Stream processing of GDELT updates
   - Live dashboard
   - Alerts for high-risk escalations

7. Interpretability
   - SHAP values for feature importance
   - Attention visualization
   - Feature attribution

8. Broader Scope
   - Include economic indicators
   - Social media sentiment
   - Satellite imagery (troop movements)
   - News sentiment analysis
"""

# ════════════════════════════════════════════════════════════════════════════
# DELIVERABLES CHECKLIST
# ════════════════════════════════════════════════════════════════════════════

DELIVERABLES = """
Final Project Deliverables:

Code:
  ✓ data_pipeline.py (400+ lines)
  ✓ models.py (500+ lines)
  ✓ train.py (400+ lines)
  ✓ visualization.py (400+ lines)
  ✓ requirements.txt
  
Documentation:
  ✓ README.md (comprehensive guide)
  ✓ Inline code comments
  ✓ Docstrings for all classes/functions
  ✓ This quick reference guide
  
Results:
  ✓ Training curves (loss, metrics)
  ✓ Test set evaluation
  ✓ Conflict prediction rankings
  ✓ Network visualizations
  ✓ Heatmaps
  ✓ Analysis report
  
Presentation (optional):
  ✓ Slides explaining approach
  ✓ Live demo on sample data
  ✓ Discussion of results
  ✓ Future work ideas
"""

# ════════════════════════════════════════════════════════════════════════════
# REFERENCES
# ════════════════════════════════════════════════════════════════════════════

REFERENCES = """
Key Papers:
  1. Kipf & Welling (2016) - "Semi-Supervised Classification with Graph 
     Convolutional Networks" (GCN)
  2. Veličković et al. (2017) - "Graph Attention Networks" (GAT)
  3. Seo et al. (2018) - "Structured Sequence Modeling with Graph 
     Convolutional Recurrent Networks"
  4. Schrodt (2012) - "Precedents, Lessons, and Futures" (Event Data)
  5. Ward et al. (2013) - "Forecasting Conflict"

Datasets:
  - GDELT: https://www.gdeltproject.org/
  - Documentation: https://www.gdeltproject.org/data/documentation/

Tools & Libraries:
  - PyTorch: https://pytorch.org/
  - PyTorch Geometric: https://pytorch-geometric.readthedocs.io/
  - NetworkX: https://networkx.org/
  - Scikit-learn: https://scikit-learn.org/

Country Codes:
  - ISO 3166-1 alpha-3: https://en.wikipedia.org/wiki/ISO_3166-1_alpha-3
"""

# ════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print(__doc__)
    print("\n" + "=" * 80)
    print("QUICK START")
    print("=" * 80)
    print(QUICK_START)
    print("\n" + "=" * 80)
    print("PROJECT OVERVIEW")
    print("=" * 80)
    print(PROBLEM_STATEMENT)

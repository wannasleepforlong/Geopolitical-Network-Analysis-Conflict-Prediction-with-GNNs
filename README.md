# Geopolitical Network Analysis & Conflict Prediction with GNNs

A comprehensive machine learning system for predicting international conflict escalation using Graph Neural Networks (GNNs) and GDELT geopolitical event data.

## Project Overview

### Problem Statement
Can we predict which country pairs will experience conflict escalation using:
- **Data Source**: GDELT (Global Database of Events, Language, and Tone)
- **Temporal Networks**: Dynamic networks of geopolitical interactions
- **Machine Learning**: Temporal Graph Neural Networks (LSTM-GCN, Temporal-GAT)

### Key Features

✨ **Multi-Level Analysis**
- Event-level: 30+ years of GDELT geopolitical events
- Network-level: Country-pair interactions with multiple types (conflict/cooperation)
- Temporal-level: Evolution of tensions over 12-month windows

✨ **Advanced Models**
- **TemporalGCN**: LSTM encoder + Graph Convolutional Networks
- **TemporalGAT**: Graph Attention Networks for interpretability
- Bidirectional edge prediction (i→j and j→i)

✨ **Production-Ready**
- Modular architecture
- Comprehensive data pipeline with caching
- Full evaluation suite with visualization
- Early stopping and checkpointing
- GDELT API integration with retry/backoff

---

## System Architecture

```
Data Pipeline                          Model Pipeline
─────────────────────────────────────────────────────────

GDELT Events                       Temporal GCN/GAT
├─ Raw events (30M+)          ├─ Node features (temporal)
├─ Filter by countries        ├─ Edge features (interaction)
└─ Aggregate to networks       └─ Adjacency matrices

GeopoliticalNetworkBuilder     Training & Evaluation
├─ Monthly aggregation        ├─ Train/Val/Test split
├─ Conflict/cooperation       ├─ Batch training
├─ Sentiment features         ├─ Early stopping
└─ Network statistics         └─ Metrics computation

Preprocessing                    Interpretation
├─ Node features              ├─ Attention visualization
├─ Edge features              ├─ Network visualization
├─ Time series windows        └─ Risk analysis reports
└─ Label generation
```

---

## Installation

### Requirements
- Python 3.8-3.11
- PyTorch with CUDA support (recommended)
- PyTorch Geometric
- Scientific Python stack (numpy, pandas, scikit-learn)

### Setup

```bash
# Clone repo
git clone https://github.com/wannasleepforlong/Geopolitical-Network-Analysis-Conflict-Prediction-with-GNNs.git
cd Geopolitical-Network-Analysis-Conflict-Prediction-with-GNNs
# Create virtual environment
python -m venv venv
venv\Scripts\activate   # Windows
# Upgrade pip
pip install --upgrade pip
# Install base dependencies
pip install -r requirements.txt
# Install PyTorch Geometric dependencies (IMPORTANT)
pip install torch-scatter torch-sparse torch-cluster torch-spline-conv \
-f https://data.pyg.org/whl/torch-2.1.0+cpu.html
```

### Verify Installation

```bash
python -c "import torch; print(f'PyTorch version: {torch.__version__}')"
python -c "import torch_geometric; print('PyTorch Geometric OK')"
```

---

## Quick Start

### Step 1: Fetch GDELT Data & Build Networks

```bash
python data_pipeline.py
```

This will:
1. Download GDELT events (2020-2023)
2. Filter for selected countries (IND, CHN, PAK, USA, RUS, JPN, KOR)
3. Build monthly temporal networks
4. Create node/edge features
5. Generate conflict escalation labels
6. Cache preprocessed data

**Output**: `./gdelt_processed_data/`
```
gdelt_processed_data/
├─ node_features.npy      # (T, N, 6)
├─ edge_features.npy      # (T, N, N, 5)
├─ edge_labels.npy        # (T, N, N) binary labels
├─ valid_mask.npy         # (T, N, N) valid edges
└─ metadata.json          # country indices, periods
```

### Step 2: Train Models

```bash
python train.py
```

This will:
1. Load preprocessed data
2. Create train/val/test splits (70/10/20)
3. Train model with early stopping
4. Evaluate on test set
5. Generate visualizations

**Configuration** (edit `train.py`):
```python
CONFIG = {
    'model_type': 'gat',        # 'gat' or 'gcn'
    'hidden_dim': 64,           # Hidden dimension
    'epochs': 50,               # Max epochs
    'batch_size': 4,
    'learning_rate': 1e-3,
}
```

**Output**: `./gdelt_results/`
```
gdelt_results/
├─ results.json              # metrics, config
├─ training_curves.png       # loss & validation metrics
└─ test_metrics.png          # final performance

gdelt_checkpoints/
└─ checkpoint_epoch_*.pt     # saved model weights
```

### Step 3: Analyze & Visualize

```python
from visualization import ConflictNetworkVisualizer, ConflictPredictionAnalyzer
import json
import numpy as np

# Load data and results
metadata = json.load(open('./gdelt_processed_data/metadata.json'))
predictions = np.load('./gdelt_results/predictions.npy')

# Visualize
visualizer = ConflictNetworkVisualizer()
fig = visualizer.visualize_conflict_heatmap(predictions, metadata['reverse_indices'])
visualizer.save_figure(fig, 'conflict_heatmap.png')

# Analyze
analyzer = ConflictPredictionAnalyzer(metadata)
analysis = analyzer.analyze_predictions(predictions)
print(analyzer.generate_report(analysis))
```

---

## Data Pipeline Details

### GDELT Events Fetching

The `GDELTEventCollector` downloads historical GDELT events with:

```python
from data_pipeline import GDELTEventCollector

collector = GDELTEventCollector()
events = collector.fetch_events(
    start_date="2020-01-01",
    end_date="2023-12-31",
    countries=['IND', 'CHN', 'USA'],  # Optional filter
    use_cache=True,
)

# events DataFrame columns:
# EventID, EventDate, Actor1Code, Actor2Code,
# EventCode, GoldsteinScale, AvgTone, QuadClass, ...
```

**GDELT Event Codes** (CAMEO taxonomy):
- 010-190: Various event types (statements, cooperation, conflict)
- **QuadClass**:
  - 1: Verbal Cooperation
  - 2: Material Cooperation
  - 3: Verbal Conflict
  - 4: Material Conflict
- **GoldsteinScale**: -10 (most negative/conflict) to +10 (most positive/cooperation)
- **AvgTone**: -100 (negative) to +100 (positive)

### Network Construction

The `GeopoliticalNetworkBuilder` constructs temporal networks with:

```python
builder = GeopoliticalNetworkBuilder()
networks = builder.build_temporal_networks(events, time_window='M')  # Monthly

# Each network contains:
{
    'adjacency_conflict': (N, N),       # conflict event counts
    'adjacency_cooperation': (N, N),    # cooperation event counts
    'avg_tone': (N, N),                 # average sentiment
    'avg_goldstein': (N, N),            # average conflict score
    'edge_count': (N, N),               # total interaction counts
}
```

### Feature Engineering

The `GNNDataPreprocessor` creates:

**Node Features** (per country, per time step):
1. Outgoing conflict count
2. Incoming conflict count
3. Outgoing cooperation count
4. Average outgoing tone
5. Average incoming tone
6. Total degree (connectivity)

**Edge Features** (per country pair, per time step):
1. Conflict count
2. Cooperation count
3. Average tone (sentiment)
4. Average Goldstein scale
5. Asymmetry (reciprocity measure)

**Labels**: Binary (0/1) conflict escalation detection
- Positive if Goldstein scale drops by >0.5 in next period
- Period: Monthly

---

## Model Architectures

### TemporalGCN (LSTM + Graph Convolution)

```
Input: (B, T, N, F_n)  [batch, time, nodes, features]
  ↓
LSTM Encoder (2 layers)
  ↓ Output: (B, N, hidden)
Graph Convolution (2 layers)
  ↓ Output: (B, N, hidden)
Edge Feature Processor
  ↓ Output: (B, N, N, hidden/2)
Concatenate [node_i, node_j, edge_feat]
  ↓
Edge Prediction Head (2 FC layers + sigmoid)
  ↓ Output: (B, N, N) probabilities
```

**Key Components:**
- LSTM captures temporal patterns
- GCN aggregates neighborhood information
- Edge predictor combines node + edge features

### TemporalGAT (LSTM + Graph Attention)

```
Input: (B, T, N, F_n)
  ↓
LSTM Encoder
  ↓
Graph Attention (multi-head, 2 layers)
  ↓ + Attention weights for interpretability
Edge Feature Processor
  ↓
Edge Prediction Head
  ↓ Output: (B, N, N) probabilities
```

**Advantages:**
- Attention weights show influential countries
- Interpretable: visualize which nodes matter
- Multi-head attention captures diverse patterns

---

## Training & Evaluation

### Training Configuration

```python
CONFIG = {
    'batch_size': 4,
    'epochs': 50,
    'learning_rate': 1e-3,
    'weight_decay': 1e-5,
    'temporal_window': 12,  # 12 months history
    'validation_split': 0.2,
    'test_split': 0.1,
}
```

### Loss Function

Binary Cross-Entropy (masked):
```
loss = BCE(predictions, labels) * valid_mask
loss = sum(loss) / count(valid_mask)
```

Where `valid_mask` excludes:
- Self-loops (i=j)
- Pairs with no historical data

### Metrics

- **ROC-AUC**: Area under ROC curve
- **F1 Score**: Harmonic mean of precision/recall
- **Accuracy**: Correct predictions / total
- **Precision & Recall**: For imbalanced data analysis

### Early Stopping

- Monitor validation AUC
- Stop if no improvement for 10 epochs
- Save best checkpoint

---

## Results Interpretation

### Conflict Risk Predictions

Output: (num_countries, num_countries) matrix of risk scores [0, 1]

```
           IND   CHN   PAK   USA   RUS
IND        --    0.73  0.12  0.08  0.15
CHN        0.81  --    0.09  0.34  0.22
PAK        0.45  0.18  --    0.05  0.08
USA        0.12  0.28  0.03  --    0.19
RUS        0.25  0.31  0.04  0.22  --
```

**Interpretation:**
- `[i,j] > 0.7`: High risk of conflict escalation (IND→CHN)
- `[i,j] ∈ [0.3, 0.7]`: Moderate risk
- `[i,j] < 0.3`: Low risk

### Attention Weights (GAT)

For GAT models, visualize which countries influence predictions:
```
Top attention edges:
1. USA attention to CHN: 0.45
2. CHN attention to RUS: 0.38
3. IND attention to PAK: 0.35
```

**Meaning**: USA nodes pay most attention to CHN nodes when predicting edges.

---

## Project Structure

```
gdelt_geopolitical_gnn/
├─ data_pipeline.py          # GDELT fetching & network building
├─ models.py                 # TemporalGCN, TemporalGAT, Dataset
├─ train.py                  # Training pipeline
├─ visualization.py          # Visualization & analysis
├─ requirements.txt          # Dependencies
├─ README.md                 # This file
│
├─ gdelt_cache/              # GDELT downloaded files (cached)
├─ gdelt_processed_data/     # Preprocessed features & labels
├─ gdelt_checkpoints/        # Saved model weights
├─ gdelt_results/            # Training results & plots
└─ gdelt_visualizations/     # Generated visualizations
```

---

## Advanced Usage

### Custom Countries

```python
# In data_pipeline.py, modify:
countries = ['USA', 'RUS', 'CHN', 'KOR', 'IRN', 'SYR']

events = collector.fetch_events(
    start_date="2015-01-01",
    end_date="2023-12-31",
    countries=countries,
)
```

### Custom Time Windows

```python
# Monthly (default)
networks = builder.build_temporal_networks(events, time_window='M')

# Weekly
networks = builder.build_temporal_networks(events, time_window='W')

# Quarterly
networks = builder.build_temporal_networks(events, time_window='Q')
```

### Predict on New Data

```python
from models import TemporalGCN

model = TemporalGCN(num_node_features=6, num_edge_features=5)
model.load_state_dict(torch.load('./gdelt_checkpoints/best_model.pt'))

# Forward pass on new network
predictions, attention = model(node_feat, edge_index, edge_feat, adjacency)
```

### Fine-tune on Different Domain

```python
# Load pretrained weights
model = TemporalGCN(...)
model.load_state_dict(torch.load('./gdelt_checkpoints/best_model.pt'))

# Freeze encoder, train edge predictor only
for name, param in model.named_parameters():
    if 'lstm' in name or 'gcn' in name:
        param.requires_grad = False

# Train on domain-specific data...
```

---

## Troubleshooting

### GDELT API Timeout

```
Error: GDELTEventCollector: GDELT request timed out
```

**Solution**: Increase timeout or retry limit in `data_pipeline.py`:
```python
GDELTFetcher._REQUEST_TIMEOUT = 60  # seconds
GDELTFetcher._MAX_RETRIES = 5
```

### Out of Memory

```
Error: CUDA out of memory
```

**Solutions**:
1. Reduce batch size: `batch_size: 2`
2. Reduce hidden_dim: `hidden_dim: 32`
3. Reduce temporal_window: `temporal_window: 6`
4. Use CPU: `device: 'cpu'`

### Class Imbalance (Few Positive Samples)

```python
# Add class weights in loss
pos_weight = (edge_labels == 0).sum() / (edge_labels == 1).sum()
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
```

---

## References

### Key Papers

1. **Graph Neural Networks**: 
   - Kipf & Welling (2016) - GCN
   - Veličković et al. (2017) - GAT

2. **Temporal GNNs**:
   - Seo et al. (2018) - LSTM + GCN
   - Zhu et al. (2020) - Temporal GNNs

3. **Conflict Prediction**:
   - Schrodt (2012) - Political Event Data
   - Ward et al. (2013) - Forecasting Conflict

### Datasets & Resources

- **GDELT**: https://www.gdeltproject.org/
- **Country Codes**: https://en.wikipedia.org/wiki/ISO_3166-1_alpha-3
- **PyTorch Geometric**: https://pytorch-geometric.readthedocs.io/

---

## Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create feature branch (`git checkout -b feature/xyz`)
3. Commit changes (`git commit -m 'Add xyz'`)
4. Push to branch (`git push origin feature/xyz`)
5. Open Pull Request

---

## License

MIT License - see LICENSE file for details

---

## Citation

If you use this project, please cite:

```bibtex
@software{gdelt_gnn_2024,
  title={Geopolitical Network Analysis & Conflict Prediction with GNNs},
  author={Your Name},
  year={2024},
  url={https://github.com/yourusername/gdelt-gnn}
}
```

---

## Contact & Support

- **Issues**: GitHub Issues page
- **Questions**: Open a Discussion
- **Email**: your.email@example.com

---

**Last Updated**: 2024
**Status**: Active Development 🚀

# Transfer Learning for Graph Classification with GNNs

A comparative study of six Graph Neural Network (GNN) architectures for molecular graph classification, evaluating how well knowledge **transfers** from large benchmark datasets (NCI1, PROTEINS) to a smaller target dataset (MUTAG). Built with [PyTorch Geometric](https://pytorch-geometric.readthedocs.io/).

## Overview

Each model is trained on a **source** dataset, then its compatible weights are transferred into a fresh model that is fine-tuned on the **target** dataset (MUTAG). This isolates one question: *do representations learned on large molecular graph datasets generalize to a small one?*

The experiment forms a grid of **6 architectures × 2 source datasets**, all sharing an identical data pipeline and evaluation protocol for a fair comparison.

## Architectures Compared

| Model | Core idea | Key components |
|-------|-----------|----------------|
| **GAT** | Graph Attention Network | 4× `GATConv` (multi-head attention), optional batch norm |
| **GIN** | Graph Isomorphism Network | Stacked `GINConv` with MLPs, batch norm, dropout |
| **GraphSAGE** | Neighborhood sampling/aggregation | 4× `SAGEConv` + batch norm |
| **RGCN** | Relational GCN (enhanced) | `RGCNConv` + `GATConv` + `LayerNorm`, 5 layers |
| **GRNN** | Graph + Recurrent hybrid | `GCNConv` layers feeding a `GRU` |
| **Hybrid GAT+GIN** | Dual-branch fusion | Parallel GAT & GIN branches with residuals, concatenated and fused |

## Datasets

All three are standard [TUDataset](https://chrsmrrs.github.io/datasets/) graph-classification benchmarks (binary labels, molecular/bioinformatics graphs):

| Dataset | Domain | Role |
|---------|--------|------|
| **NCI1** | Chemical compounds screened for anti-cancer activity | Source |
| **PROTEINS** | Protein structures (enzyme vs. non-enzyme) | Source |
| **MUTAG** | Mutagenic aromatic & nitro compounds | Target (transfer) |

## Method

1. Load source and target datasets with `NormalizeFeatures`.
2. Split 70/30 train/test (the Hybrid notebooks also carve out a validation set).
3. Train the model on the source dataset; evaluate.
4. Copy all shape-compatible weights into a new model for MUTAG (`load_state_dict(..., strict=False)`).
5. Fine-tune on MUTAG; evaluate again.

Graph-level predictions are produced via `global_mean_pool`. Training uses Adam/AdamW/AdamP with schedulers such as `ReduceLROnPlateau`, `CosineAnnealingLR`, `CyclicLR`, or warm restarts, depending on the model. Cross-entropy (sometimes with label smoothing or focal loss) is the objective.

## Results (summary)

Test-set accuracy before transfer (on the source dataset) and after transfer + fine-tuning (on MUTAG):

### Source: NCI1 → MUTAG

| Model | NCI1 Acc | MUTAG Acc (after transfer) |
|-------|:--------:|:--------------------------:|
| **Hybrid GAT+GIN** | **0.811** | **0.895** |
| SAGE | 0.793 | 0.877 |
| GAT | 0.795 | 0.860 |
| RGCN | 0.746 | 0.807 |
| GIN | 0.762 | 0.807 |
| GRNN | 0.702 | 0.807 |

### Source: PROTEINS → MUTAG

| Model | PROTEINS Acc | MUTAG Acc (after transfer) |
|-------|:------------:|:--------------------------:|
| **Hybrid GAT+GIN** | 0.713 | **0.930** |
| GAT | 0.716 | 0.877 |
| GIN | 0.683 | 0.877 |
| GRNN | 0.677 | 0.860 |
| RGCN | 0.710 | 0.842 |
| SAGE | 0.701 | 0.825 |

**Takeaway:** transfer learning improves MUTAG performance across all architectures, and the **Hybrid GAT+GIN model consistently performs best**, achieving the highest accuracy and lowest log loss in both transfer settings.

> Numbers are from a single run with `random_state=42`; exact values vary with hardware, library versions, and random seeds.

## Requirements

- Python 3.8+
- [PyTorch](https://pytorch.org/)
- [PyTorch Geometric](https://pytorch-geometric.readthedocs.io/) (`torch_geometric`)
- scikit-learn, numpy
- [adamp](https://github.com/clovaai/AdamP) (for the Hybrid models)

```bash
pip install torch torch_geometric scikit-learn numpy adamp
```

## Usage

The notebooks were written for Google Colab, which downloads the TUDatasets automatically on first run.

```bash
jupyter notebook notebooks/GAT_NCI1.ipynb
```

Each notebook is self-contained: it installs dependencies, loads the data, trains the source model, performs transfer learning to MUTAG, and prints the full metrics. Run any notebook top to bottom to reproduce its row of the results tables.

## Project Structure

```
gnn-transfer-learning-graph-classification/
├── notebooks/
│   ├── GAT_NCI1.ipynb
│   ├── GAT_Proteins.ipynb
│   ├── GIN_NCI1.ipynb
│   ├── GIN_PROTEINS.ipynb
│   ├── GRNN_NCI1.ipynb
│   ├── GRNN_PROTEINS.ipynb
│   ├── Hybrid_GAT_GIN_NCI1.ipynb
│   ├── Hybrid_GAT_GIN_Proteins.ipynb
│   ├── RGCN_NCI1.ipynb
│   ├── RGCN_Proteins.ipynb
│   ├── SAGE_NCI1.ipynb
│   └── SAGE_Proteins.ipynb
├── README.md
├── requirements.txt
└── LICENSE
```

## Notes

- A GPU speeds up training considerably, especially for the deeper RGCN and Hybrid models.
- The `evaluate_model` function pads/truncates probability columns so metrics stay consistent when the source and target datasets differ in class count.
- A benign scikit-learn warning ("y_pred values do not sum to one") appears because log-softmax outputs are passed where probabilities are expected; it does not affect the reported accuracy/F1.

## License

Released under the [MIT License](LICENSE).

## Acknowledgements

Datasets courtesy of the [TUDataset](https://chrsmrrs.github.io/datasets/) collection. Built with PyTorch Geometric.

# GENIE GSoC 2026 — Deep Graph Anomaly Detection with Contrastive Learning

**ML4SCI Evaluation Submission** | Author: Mudit

---

## Project Overview

This repository develops a deep learning pipeline for **quark/gluon jet classification and anomaly detection**. 

**Task Status:** 
- [x] **Common Task 1:** Convolutional Autoencoder (Baseline)
- [x] **Common Task 2:** GNN Jet Classifier (Implemented)
- [ ] **Specific Task 1:** Contrastive Anomaly Detection (Upcoming)

**Dataset:** QCD quark/gluon jet events — 3-channel 125×125 images (Tracks, ECAL, HCAL).

---

### Setup & Dependencies
```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

### 1. Convolutional Autoencoder (Task 1)
Trains a symmetrical CNN to reconstruct sparse detector images.
```bash
python src/task1_cae.py --epochs 20 --batch-size 64 --learning-rate 0.001
```

### 2. Graph Attention Network (Task 2)
Projects images onto a $k$-NN topological manifold and classifies them using multi-head attention.
```bash
python src/task2_gnn.py --epochs 10 --knn-k 8 --batch-size 64
```

---

## Task Solutions

### Common Task 1 — Convolutional Autoencoder (CAE)

```bash
python src/task1_cae.py
```

Trains a symmetric convolutional autoencoder on raw 3-channel jet images.

**Architecture:**
```
Input (N, 3, 125, 125)
│
├─ Encoder: 3 × [Conv2d → BatchNorm → ReLU → MaxPool2d]
│           Channels: 3 → 16 → 32 → 64   Spatial: 125 → 62 → 31 → 15
│
├─ Bottleneck: (N, 64, 15, 15) latent representation
│
└─ Decoder: 3 × [ConvTranspose2d → BatchNorm → ReLU] + Upsample + Conv2d → Sigmoid
            Restores: (N, 3, 125, 125)
```

- **Loss:** MSE  |  **Optimizer:** Adam + cosine LR schedule  |  **Epochs:** 30
- **Output:** `outputs/task1_reconstructions.png` — 8 events × 6 channels (orig | recon)

---

### Common Task 2 — GNN Jet Classifier (GAT)

```bash
python src/task2_gnn.py
```

Converts jet images to graphs and classifies quark vs. gluon jets using a **Graph Attention Network**.

**Image → Graph pipeline:**
1. Extract non-zero pixels as particle nodes with features `[η_norm, φ_norm, E_track, E_ECAL, E_HCAL]`
2. Build a **k-NN graph (k=8)** in (η, φ) coordinate space

**Why k-NN over radius graphs?**
Jet images are sparse with variable particle multiplicity. Radius graphs produce isolated nodes in low-multiplicity events, causing degenerate global pooling. k-NN guarantees every node has exactly 8 neighbours, maintaining well-conditioned message-passing gradients across all events.

**Why GAT over GCN?**
Graph Attention Networks learn adaptive, data-driven edge weights rather than fixed normalised adjacency weights. Each jet constituent has varying relevance to its neighbours depending on pT and angular separation — attention heads naturally capture this heterogeneity.

**GAT model:**
```
GATConv(5→32, heads=4) → ELU → BN
GATConv(128→32, heads=4) → ELU → BN
GATConv(128→64, heads=1)
global_{mean,max}_pool → cat → 128-dim
Linear(128→64) → ELU → Dropout(0.3) → Linear(64→1)
```

- **Loss:** BCE with pos_weight  |  **Optimizer:** AdamW  |  **Metric:** ROC-AUC
- **Output:** `outputs/task2_roc.png`

---

## Repository Structure

| Task | Mode | Metric | Value |
|------|------|--------|-------|
| Task 1 — CAE | — | Val MSE | 0.00010 |
| Task 2 — GAT | — | — | (Pending) |
| Task 3 — GraphCLR | — | — | (Pending) |

---

## Repository Structure

```
genie_evaluation_mudit/
├── data/              # dataset files
├── outputs/           # generated plots
│   ├── task1_reconstructions.png
│   └── task1_loss_curve.png
├── src/
│   ├── data_utils.py         # shared data loading & point cloud conversion
│   └── task1_cae.py          # Common Task 1 — Convolutional Autoencoder
└── requirements.txt
```

---

## Physics Motivation

Jets are collimated sprays of particles produced in high-energy collisions. Distinguishing quark-initiated from gluon-initiated jets is important for new physics searches. Treating jets as **graphs** preserves the relational structure lost in 2D image representations.

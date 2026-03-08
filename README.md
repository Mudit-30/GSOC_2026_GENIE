# GENIE GSoC 2026 — GSOC_2026_GENIE

**ML4SCI Evaluation Submission** | Author: Mudit

---

## Project Overview

This repository develops a deep learning pipeline for **quark/gluon jet classification and anomaly detection**. 

**Task Status:** 
- [x] **Common Task 1:** Convolutional Autoencoder (Baseline)
- [x] **Common Task 2:** GNN Jet Classifier (Implemented)
- [x] **Specific Task 1:** Contrastive Anomaly Detection (Implemented)

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

### 3. Graph Contrastive Learning (Task 3)
Applies a SimCLR-inspired NT-Xent contrastive framework for unsupervised anomaly detection.
```bash
python src/task3_contrastive.py --pretrain-epochs 10 --finetune-epochs 10 --tau 0.5
```
> **PyTorch Geometric:** install `torch-scatter` and `torch-sparse` matching your CUDA version from https://data.pyg.org/whl/

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

### Specific Task 1 — Contrastive Anomaly Detection (GraphCLR)

```bash
python src/task3_contrastive.py
```

Learns jet representations **without labels** via contrastive learning (SimCLR-style), then evaluates both unsupervised anomaly detection and supervised linear-probe classification.

**Framework:**

**Phase A — Unsupervised Pretraining:**
1. **Augmentations:** Each jet graph gets two stochastically augmented views (Node/Edge drop + Feature Noise).
2. **Encoder:** Same GAT × 3 backbone as Task 2.
3. **Projection head:** 2-layer MLP (`GNN_DIM → 128`), L2-normalised output.
4. **NT-Xent loss** (τ = 0.5).

**Phase B — Supervised Linear Probe:**
5. Freeze encoder weights and train a single linear layer on top.

**Outputs:** `outputs/task3_roc.png` (linear probe) + `outputs/task3_roc_anomaly.png` (unsupervised)

---

## Results

| Task | Mode | Metric | Value |
|------|------|--------|-------|
| Task 1 — CAE | — | Val MSE | 0.00010 |
| Task 2 — GAT | Supervised | Test ROC-AUC | 0.7686 |
| Task 3 — GraphCLR | Unsupervised anomaly | Test ROC-AUC | 0.5016 |
| Task 3 — GraphCLR | Linear probe | Test ROC-AUC | 0.5015 |

---

## Repository Structure

```
genie_evaluation_mudit/
├── data/              # dataset files
├── outputs/           # generated plots
│   ├── task1_reconstructions.png
│   ├── task2_roc.png
│   ├── task3_roc.png
│   └── task3_roc_anomaly.png
├── src/
│   ├── data_utils.py         # shared data loading & point cloud conversion
│   ├── task1_cae.py          # Common Task 1 — Convolutional Autoencoder
│   ├── task2_gnn.py          # Common Task 2 — GAT Classifier
│   └── task3_contrastive.py  # Specific Task 1 — GraphCLR
└── requirements.txt
```

---

## Physics Motivation

Jets are collimated sprays of particles produced in high-energy collisions. Distinguishing quark-initiated from gluon-initiated jets is important for new physics searches. Treating jets as **graphs** preserves the relational structure lost in 2D image representations.

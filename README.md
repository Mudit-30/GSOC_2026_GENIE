# GENIE GSoC 2026 — GSOC_2026_GENIE

**ML4SCI Evaluation Submission** | Author: Mudit

---

## Project Overview

This repository develops a deep learning pipeline for **quark/gluon jet classification and anomaly detection**. 

**Task Status:** 
- [x] **Common Task 1:** Convolutional Autoencoder (Baseline)
- [x] **Common Task 2:** GNN Jet Classifier (Implemented)
- [x] **Specific Task 1:** Contrastive Anomaly Detection (Implemented)
- [x] **Upstream Fixes:** Expert optimizations (3 Branches on Fork)

**Dataset:** QCD quark/gluon jet events — 3-channel 125×125 images (Tracks, ECAL, HCAL).

---

## Upstream Proof-of-Work (Expert Contributions)

Beyond the specific evaluation tasks, I have contributed expert-level bug fixes and performance optimizations to the official [ML4SCI/GENIE](https://github.com/ML4SCI/GENIE) repository. These contributions are available as independent branches on my **[personal fork](https://github.com/Mudit-30/GENIE)**:

1.  **[Windows Path & NoneType Fix](https://github.com/Mudit-30/GENIE/tree/fix-windows-paths):** Resolved cross-platform OS pathing bugs and `NoneType` attribute errors in `datasets.py`.
2.  **[Diffusion AMP Optimization](https://github.com/Mudit-30/GENIE/tree/optimize-diffusion-amp):** Implemented Automatic Mixed Precision (AMP) and CUDA benchmark tuning, achieving a ~2x speedup in training.
3.  **[PINN Numerical Stability](https://github.com/Mudit-30/GENIE/tree/stabilize-pde-solver):** Replaced unstable exponential weights with a robust `LogSumExp` formulation in the Physics-Informed Neural Network solver.

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

## Physics Motivation & Mathematical Deep-Dive

### 1. The Quark-Gluon Discrimination Problem
Jets are collimated sprays of energetic particles (pions, kaons, etc.) produced by the fragmentation of hard-scattered partons (quarks or gluons). Distinguishing between them is a classic high-energy physics (HEP) challenge:
- **Quarks:** Carry color charge 3, typically produce jets with fewer constituents and narrower profiles.
- **Gluons:** Carry color charge 8 (higher Casimir factor $C_A$), leading to higher radiation splitting, higher particle multiplicity, and broader energy distributions.

Standard variables like **Jet Mass** and **Girth** provide baseline separation, but deep learning models can capture non-linear, high-dimensional correlations in the calorimeter clusters and track geometries.

### 2. Why Graphs? (Permutation Invariance)
A jet is essentially an unordered set of $N$ particles. Traditional 2D image representations (Calorimeter Images) suffer from **sparsity** (most pixels are zero) and **quantization noise**. 
Representing a jet as a **Graph** $G = (V, E)$ allows us to:
1. **Respect Geometry:** Map particles to nodes based on their exact $(\eta, \phi)$ distance.
2. **Permutation Invariance:** Using GNNs with symmetric aggregation (Mean/Max pooling) ensures that the model's output doesn't change if we reorder the particles in the input list ($f(\{x_1, ..., x_n\}) = f(\{x_{\pi(1)}, ..., x_{\pi(n)}\})$).

### 3. Metric Evaluation: ROC-AUC
For Tasks 2 and 3, we use the **Area Under the ROC Curve (AUC)**. To evaluate the classifier, we compute:
$$TPR = \frac{TP}{TP + FN}, \quad FPR = \frac{FP}{FP + TN}$$
The AUC represents the probability that the classifier will rank a randomly chosen positive instance (Gluon) higher than a randomly chosen negative one (Quark). An AUC of 1.0 represents perfect separation, while 0.5 represents a random guess.

---

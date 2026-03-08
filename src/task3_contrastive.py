"""
task3_contrastive.py — Specific Task 1
========================================
Graph Contrastive Learning (GraphCLR) for Unsupervised Anomaly Detection
and Linear-Probe Jet Classification.

Mathematical Framework (Phase A — Unsupervised pretraining)
-----------------------------------------------------------
Let G = (V, E, X) be a jet graph. Let T be a distribution of stochastic augmentations.
For each G in a minibatch B of size N, sample t, t' ~ T to produce views G_i = t(G), G_j = t'(G).
The batch now contains 2N augmented views.

Encoder f_θ: extract representation h_i = f_θ(G_i) ∈ ℝ^{GNN_DIM}.
Projection head g_φ: extract metric space feature z_i = g_φ(h_i) ∈ ℝ^{EMBED_DIM}.
The representations are L2-normalized: z_i = z_i / ||z_i||_2.

The NT-Xent loss for a positive pair (i,j) is:
  L_{i,j} = -log( exp(sim(z_i, z_j)/τ) / ∑_{k=1, k≠i}^{2N} exp(sim(z_i, z_k)/τ) )
"""

import os
import sys
import random
import logging
import argparse
from typing import Tuple, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve, accuracy_score
from tqdm import tqdm
from joblib import Parallel, delayed

from torch_geometric.data import Data, Batch
from torch_geometric.nn import GATConv, global_mean_pool, global_max_pool
# Removed KNNGraph

sys.path.insert(0, os.path.dirname(__file__))
from data_utils import load_dataset, image_to_pointcloud

# ─────────────────────────────────────────────────────────────
# Initialization & Reproducibility
# ─────────────────────────────────────────────────────────────

def setup_logging() -> logging.Logger:
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    return logging.getLogger(__name__)

logger = setup_logging()

def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

# ─────────────────────────────────────────────────────────────
# Stochastic Augmentations
# ─────────────────────────────────────────────────────────────

def augment_graph(data: Data, node_drop_p: float = 0.10, edge_drop_p: float = 0.20, feat_noise_std: float = 0.05) -> Data:
    x = data.x.clone()
    edge_index = data.edge_index.clone()
    num_nodes = x.size(0)

    # Topology: Vertex dropping
    keep_mask = torch.rand(num_nodes) > node_drop_p
    if keep_mask.sum() < 2: keep_mask[:2] = True
    keep_nodes = keep_mask.nonzero(as_tuple=True)[0]
    
    remap = torch.full((num_nodes,), -1, dtype=torch.long)
    remap[keep_nodes] = torch.arange(keep_nodes.size(0))
    
    src, dst = edge_index
    valid_edges = keep_mask[src] & keep_mask[dst]
    edge_index = torch.stack([remap[src[valid_edges]], remap[dst[valid_edges]]], dim=0)
    x = x[keep_nodes]

    # Topology: Edge dropping
    if edge_index.size(1) > 0:
        keep_edges = torch.rand(edge_index.size(1)) > edge_drop_p
        if keep_edges.sum() == 0: keep_edges[0] = True
        edge_index = edge_index[:, keep_edges]

    # Feature: Additive Gaussian Noise
    x = x + torch.randn_like(x) * feat_noise_std
    return Data(x=x, edge_index=edge_index)

# ─────────────────────────────────────────────────────────────
# Data Loading & Architectures
# ─────────────────────────────────────────────────────────────

def build_base_graph(img: np.ndarray, knn_k: int = 8, threshold: float = 0.0) -> Data:
    """Projects 2D image signals onto a sparse k-NN representation without torch-cluster."""
    pts = image_to_pointcloud(img, threshold=threshold)
    x = torch.tensor(pts, dtype=torch.float32)
    pos = x[:, :2]
    
    # Manual K-NN construction
    dist = torch.cdist(pos, pos)
    k_plus_1 = min(knn_k + 1, pos.size(0))
    _, indices = dist.topk(k_plus_1, largest=False)
    
    row = torch.arange(pos.size(0)).view(-1, 1).repeat(1, k_plus_1).view(-1)
    col = indices.view(-1)
    mask = row != col
    edge_index = torch.stack([row[mask], col[mask]], dim=0)
    
    return Data(x=x, pos=pos, edge_index=edge_index)

class ContrastiveJetDataset(Dataset):
    """Yields (View_i, View_j, Label) pairs from precomputed topologies."""
    def __init__(self, X: np.ndarray, y: np.ndarray, knn_k: int = 8, tag: str = "Train"):
        self.y = y
        self.knn_k = knn_k
        logger.info("[Optimized] Parallelizing %s base manifold topologies (n=%s)...", tag, f"{len(y):,}")
        self.base_graphs = Parallel(n_jobs=-1)(
            delayed(build_base_graph)(X[i], knn_k) for i in tqdm(range(len(y)), leave=False)
        )

    def __len__(self) -> int: return len(self.y)
    def __getitem__(self, idx: int) -> Tuple[Data, Data, int]:
        d = self.base_graphs[idx]
        return augment_graph(d), augment_graph(d), int(self.y[idx])

def collate_contrastive(batch: List[Tuple[Data, Data, int]]) -> Tuple[Batch, Batch, torch.Tensor]:
    v1, v2, labels = zip(*batch)
    return Batch.from_data_list(v1), Batch.from_data_list(v2), torch.tensor(labels, dtype=torch.long)

class GATEncoder(nn.Module):
    def __init__(self, in_channels: int = 5, hidden: int = 64) -> None:
        super().__init__()
        self.conv1 = GATConv(in_channels, hidden // 2, heads=4, concat=True, dropout=0.1)
        self.bn1   = nn.BatchNorm1d(hidden * 2)
        self.conv2 = GATConv(hidden * 2, hidden // 2, heads=4, concat=True, dropout=0.1)
        self.bn2   = nn.BatchNorm1d(hidden * 2)
        self.conv3 = GATConv(hidden * 2, hidden, heads=1, concat=False, dropout=0.1)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        h = F.elu(self.bn1(self.conv1(x, edge_index)))
        h = F.elu(self.bn2(self.conv2(h, edge_index)))
        h = F.elu(self.conv3(h, edge_index))
        return torch.cat([global_mean_pool(h, batch), global_max_pool(h, batch)], dim=1)

class ProjectionHead(nn.Module):
    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, in_dim), nn.BatchNorm1d(in_dim), nn.ReLU(inplace=True),
            nn.Linear(in_dim, out_dim)
        )
    def forward(self, h: torch.Tensor) -> torch.Tensor: return self.net(h)

class GraphCLR(nn.Module):
    def __init__(self, gnn_dim: int = 128, embed_dim: int = 128) -> None:
        super().__init__()
        self.encoder = GATEncoder()
        self.projector = ProjectionHead(gnn_dim, embed_dim)

    def encode(self, x: torch.Tensor, ei: torch.Tensor, b: torch.Tensor) -> torch.Tensor: return self.encoder(x, ei, b)
    def forward(self, x: torch.Tensor, ei: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        z = self.projector(self.encoder(x, ei, b))
        return F.normalize(z, dim=1)

# ─────────────────────────────────────────────────────────────
# Objective formulation & Execution
# ─────────────────────────────────────────────────────────────
def nt_xent_loss(z_i: torch.Tensor, z_j: torch.Tensor, tau: float) -> torch.Tensor:
    N = z_i.size(0)
    z = torch.cat([z_i, z_j], dim=0)                 
    sim = torch.mm(z, z.T) / tau                     
    mask = torch.eye(2 * N, dtype=torch.bool, device=z.device)
    sim.masked_fill_(mask, float("-inf"))            
    labels = torch.cat([torch.arange(N, 2*N, device=z.device), torch.arange(0, N, device=z.device)])
    return F.cross_entropy(sim, labels)

def main(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.force_cpu else "cpu")
    logger.info("Hardware Accelerator Context: %s", device.type.upper())

    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(root_dir, "data")
    
    X, y = load_dataset(data_dir, max_events=args.max_events)
    idx = np.random.default_rng(args.seed).permutation(len(y))
    tr_len, va_len = int(0.7*len(y)), int(0.15*len(y))
    tr_id, va_id, te_id = idx[:tr_len], idx[tr_len:tr_len+va_len], idx[tr_len+va_len:]

    tr_ldr = DataLoader(ContrastiveJetDataset(X[tr_id], y[tr_id], knn_k=args.knn_k, tag="TRAIN"), args.batch_size, True,  collate_fn=collate_contrastive, num_workers=0)
    te_ldr = DataLoader(ContrastiveJetDataset(X[te_id], y[te_id], knn_k=args.knn_k, tag="TEST"), args.batch_size, False, collate_fn=collate_contrastive, num_workers=0)

    if device.type == 'cuda': torch.backends.cudnn.benchmark = True
        
    model = GraphCLR(gnn_dim=128, embed_dim=args.embed_dim).to(device)
    if hasattr(torch, 'compile') and os.name != 'nt' and device.type == 'cuda':
        try: model = torch.compile(model)
        except Exception: pass
        
    opt = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=1e-5)
    scaler = torch.amp.GradScaler('cuda', enabled=(device.type == 'cuda'))

    logger.info("Phase A: Initiating Unsupervised Graph Contrastive Pretraining (NT-Xent)...")
    for epoch in range(1, args.pretrain_epochs + 1):
        model.train(); tot = 0.0
        for v1, v2, _ in tqdm(tr_ldr, leave=False, desc=f"Ep {epoch}"):
            v1, v2 = v1.to(device, non_blocking=True), v2.to(device, non_blocking=True)
            with torch.amp.autocast('cuda', enabled=(device.type == 'cuda')):
                loss = nt_xent_loss(model(v1.x, v1.edge_index, v1.batch), model(v2.x, v2.edge_index, v2.batch), args.tau)
                
            opt.zero_grad(set_to_none=True); scaler.scale(loss).backward()
            scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt); scaler.update()
            tot += loss.item() * v1.num_graphs
        logger.info("Epoch %03d | Contrastive Loss: %.4f", epoch, tot / len(tr_ldr.dataset))

    model.eval()
    def extract(ldr: DataLoader) -> Tuple[np.ndarray, np.ndarray]:
        H, Y = [], []
        with torch.no_grad():
            for v1, _, bY in ldr:
                v1 = v1.to(device); H.append(model.encode(v1.x, v1.edge_index, v1.batch).cpu().numpy()); Y.append(bY.numpy())
        return np.concatenate(H), np.concatenate(Y)

    tr_H, tr_Y = extract(tr_ldr)
    te_H, te_Y = extract(te_ldr)

    centroid = tr_H.mean(axis=0)
    scores = np.linalg.norm(te_H - centroid, axis=1) # L2 Distance S(G)
    logger.info("Phase B | Unsupervised Anomaly AUC (Centroid Distance): %.4f", roc_auc_score(te_Y, scores))

    logger.info("Phase B | Instantiating Linear Probe Classifer...")
    probe = nn.Linear(128, 2).to(device)
    for p in model.encoder.parameters(): p.requires_grad = False
    opt_pr = torch.optim.Adam(probe.parameters(), lr=1e-3)
    tr_Ht, tr_Yt = torch.tensor(tr_H).to(device), torch.tensor(tr_Y).to(device)
    
    for _ in range(args.finetune_epochs):
        probe.train(); loss = F.cross_entropy(probe(tr_Ht), tr_Yt)
        opt_pr.zero_grad(); loss.backward(); opt_pr.step()
    
    probe.eval()
    with torch.no_grad(): te_prob = F.softmax(probe(torch.tensor(te_H).to(device)), dim=1)[:, 1].cpu().numpy()
    logger.info("Phase B | Supervised Linear Probe Benchmark AUC: %.4f", roc_auc_score(te_Y, te_prob))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Contrastive Graph Learning for Anomaly Detection")
    parser.add_argument("--batch-size", type=int, default=64, help="Bundle size. Generates 2N constraints")
    parser.add_argument("--pretrain-epochs", type=int, default=10, help="NT-Xent optimization bounds")
    parser.add_argument("--finetune-epochs", type=int, default=10, help="Linear probe optimization bounds")
    parser.add_argument("--learning-rate", type=float, default=3e-4, help="Encoder/Projector optimizer scalar")
    parser.add_argument("--knn-k", type=int, default=8, help="Manifold N(i) cardinality")
    parser.add_argument("--tau", type=float, default=0.5, help="SimCLR InfoNCE temperature scalar")
    parser.add_argument("--embed-dim", type=int, default=128, help="Metric space projection dimension")
    parser.add_argument("--max-events", type=int, default=500, help="Dataset truncation logic")
    parser.add_argument("--force-cpu", action="store_true", help="Force GPU bypass flag")
    parser.add_argument("--seed", type=int, default=42, help="Global determinstic execution constraint")
    
    main(parser.parse_args())

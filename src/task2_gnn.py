"""
task2_gnn.py — Common Task 2
==============================
Graph Attention Network (GAT) for quark/gluon jet classification.

Mathematical Formulation
------------------------
Let G = (V, E, X) be a jet represented as a k-NN graph in (η, φ) space.
Node features x_i ∈ ℝ^{5} encode [η_norm, φ_norm, E_track, E_ECAL, E_HCAL].

The Graph Attention mechanism updates node representations via:
  h_i^{(l+1)} = σ( ∑_{j ∈ N(i)} α_{i,j} W h_j^{(l)} )
Where the attention coefficients α_{i,j} are computed as:
  α_{i,j} = softmax_j( LeakyReLU( a^T [W h_i || W h_j] ) )

Graph-level embedding is obtained via generalized pooling:
  H_G = [ 1/|V| ∑_{i∈V} h_i || max_{i∈V} h_i ]
And the final probability P(Gluon | G) = σ(MLP(H_G)).
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

import torch_geometric
from torch_geometric.data import Data, Batch
from torch_geometric.nn import GATConv, global_mean_pool, global_max_pool
# Removed KNNGraph as it requires torch-cluster

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
    """Ensures deterministic graph construction and topological sampling."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

# ─────────────────────────────────────────────────────────────
# Manifold Construction (Image → Graph)
# ─────────────────────────────────────────────────────────────

def image_to_graph(img: np.ndarray, label: int, knn_k: int = 8, threshold: float = 0.0) -> Data:
    """Projects 2D image signals onto a sparse k-NN representation without torch-cluster."""
    points = image_to_pointcloud(img, threshold=threshold)   # [M, 5]
    x = torch.tensor(points, dtype=torch.float32)            # Node features
    pos = x[:, :2]                                           # Angular coordinates (η, φ)
    
    # Manual K-NN construction using Euclidean distance in (eta, phi) space
    dist = torch.cdist(pos, pos)
    # Get indices of k nearest neighbors (including self, we'll handle that)
    # We want k neighbors excluding self, so we'll take k+1 and then filter
    k_plus_1 = min(knn_k + 1, pos.size(0))
    _, indices = dist.topk(k_plus_1, largest=False)
    
    # Construct edge_index [2, E]
    row = torch.arange(pos.size(0)).view(-1, 1).repeat(1, k_plus_1).view(-1)
    col = indices.view(-1)
    
    # Remove self-loops
    mask = row != col
    edge_index = torch.stack([row[mask], col[mask]], dim=0)
    
    # ensure it doesn't exceed knn_k per node after self-loop removal (if any)
    # topk already gave us exactly k+1, removing self-loop leaves k.
    
    data = Data(x=x, pos=pos, y=torch.tensor([label], dtype=torch.long), edge_index=edge_index)
    return data

class JetGraphDataset(Dataset):
    """
    Optimized generator mapping images to PyG graph objects asynchronously.
    Executes a parallelized pre-computation block during initialization to 
    eliminate dynamic CPU bottlenecks during GPU backpropagation.
    """
    def __init__(self, X: np.ndarray, y: np.ndarray, knn_k: int = 8, tag: str = "Train"):
        self.y = y
        logger.info("[Optimized] Parallelizing %s manifold topologies (n=%s)...", tag, f"{len(y):,}")
        self.graphs = Parallel(n_jobs=-1)(
            delayed(image_to_graph)(X[i], int(y[i]), knn_k) for i in tqdm(range(len(y)), leave=False)
        )
    def __len__(self) -> int: return len(self.graphs)
    def __getitem__(self, idx: int) -> Data: return self.graphs[idx]

def collate_graphs(batch: List[Data]) -> Batch:
    return Batch.from_data_list(batch)

# ─────────────────────────────────────────────────────────────
# GAT Classifier Architecture
# ─────────────────────────────────────────────────────────────

class JetGAT(nn.Module):
    """Discriminative Graph Attention Network with anisotropic message passing."""
    def __init__(self, in_channels: int = 5) -> None:
        super().__init__()

        self.conv1 = GATConv(in_channels, 32, heads=4, concat=True, dropout=0.1)
        self.bn1   = nn.BatchNorm1d(128)

        self.conv2 = GATConv(128, 32, heads=4, concat=True, dropout=0.1)
        self.bn2   = nn.BatchNorm1d(128)

        self.conv3 = GATConv(128, 64, heads=1, concat=False, dropout=0.1)

        self.classifier = nn.Sequential(
            nn.Linear(128, 64),
            nn.ELU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        """
        Forward logic.
        Args:
            x: Node continuous signals [N, 5]
            edge_index: Sparse adjacency COO [2, E]
            batch: Partition array [N]
        Returns:
            torch.Tensor: Log-odds [B]
        """
        h = F.elu(self.bn1(self.conv1(x, edge_index)))    
        h = F.elu(self.bn2(self.conv2(h, edge_index)))    
        h = F.elu(self.conv3(h, edge_index))              

        # Invariant concatenation
        h_graph = torch.cat([global_mean_pool(h, batch), global_max_pool(h, batch)], dim=1) 
        return self.classifier(h_graph).squeeze(-1)       


# ─────────────────────────────────────────────────────────────
# Evaluation Loop
# ─────────────────────────────────────────────────────────────
@torch.no_grad()
def eval_epoch(model: nn.Module, loader: DataLoader, criterion: nn.Module, device: torch.device) -> Tuple[float, np.ndarray, np.ndarray]:
    model.eval()
    total_loss, all_logits, all_labels = 0.0, [], []
    for batch in loader:
        batch = batch.to(device, non_blocking=True)
        with torch.amp.autocast('cuda', enabled=(device.type == 'cuda')):
            logits = model(batch.x, batch.edge_index, batch.batch)
            loss = criterion(logits, batch.y.float())
        total_loss += loss.item() * batch.num_graphs
        all_logits.extend(logits.cpu().numpy())
        all_labels.extend(batch.y.cpu().numpy())
    return total_loss / len(loader.dataset), np.array(all_logits), np.array(all_labels)

# ─────────────────────────────────────────────────────────────
# Entrypoint
# ─────────────────────────────────────────────────────────────
def main(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.force_cpu else "cpu")
    logger.info("Hardware Accelerator: %s", device.type.upper())

    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(root_dir, "data")
    
    X, y = load_dataset(data_dir, max_events=args.max_events)
    n_samples = len(y)
    idx = np.random.default_rng(args.seed).permutation(n_samples)
    tr_len, va_len = int(0.7*n_samples), int(0.15*n_samples)
    tr_id, va_id, te_id = idx[:tr_len], idx[tr_len:tr_len+va_len], idx[tr_len+va_len:]

    loaders = {
        split: DataLoader(JetGraphDataset(X[i], y[i], knn_k=args.knn_k, tag=split.upper()), 
                          batch_size=args.batch_size, shuffle=(split=='train'), 
                          collate_fn=collate_graphs, num_workers=0)
        for split, i in zip(['train', 'val', 'test'], [tr_id, va_id, te_id])
    }

    if device.type == 'cuda':
        torch.backends.cudnn.benchmark = True
    
    model = JetGAT().to(device)
    if hasattr(torch, 'compile') and os.name != 'nt' and device.type == 'cuda':
        try: 
            model = torch.compile(model)
            logger.info("JIT compiled Graph Attention Network model.")
        except Exception: pass
        
    logger.info("Learnable generic parameters: %s", f"{sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    pos_weight = torch.tensor([y[tr_id].mean() / (1 - y[tr_id].mean() + 1e-6)], device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    scaler = torch.amp.GradScaler('cuda', enabled=(device.type == 'cuda'))
    
    best_auc = 0.0
    
    logger.info("Initiating Graph Attention Network training protocol...")
    for epoch in range(1, args.epochs + 1):
        model.train()
        for b in tqdm(loaders['train'], leave=False, desc=f"Epoch {epoch}"):
            b = b.to(device, non_blocking=True)
            with torch.amp.autocast('cuda', enabled=(device.type == 'cuda')):
                loss = criterion(model(b.x, b.edge_index, b.batch), b.y.float())
            
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        
        _, va_log, va_y = eval_epoch(model, loaders['val'], criterion, device)
        va_auc = roc_auc_score(va_y, 1/(1+np.exp(-va_log)))
        if va_auc > best_auc: best_auc = va_auc
        logger.info("Epoch %03d | Val AUC: %.4f", epoch, va_auc)

    _, te_log, te_y = eval_epoch(model, loaders['test'], criterion, device)
    te_prob = 1/(1+np.exp(-te_log))
    logger.info("Final Generalization Test AUC: %.4f", roc_auc_score(te_y, te_prob))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Graph Attention Network for Sparse Jet Classification")
    parser.add_argument("--batch-size", type=int, default=32, help="Graph bundle size per forward pass")
    parser.add_argument("--epochs", type=int, default=10, help="Terminal optimization limit")
    parser.add_argument("--learning-rate", type=float, default=1e-3, help="Optimizer initial state scalar")
    parser.add_argument("--knn-k", type=int, default=8, help="Manifold K-Nearest Neighbor cardinality")
    parser.add_argument("--max-events", type=int, default=500, help="Dataset truncation bound")
    parser.add_argument("--force-cpu", action="store_true", help="Force GPU bypass flag")
    parser.add_argument("--seed", type=int, default=42, help="Deterministic topological generator state")
    
    main(parser.parse_args())

"""
task1_cae.py — Common Task 1
==============================
Convolutional Autoencoder for quark/gluon jet reconstruction.

Mathematical Formulation
------------------------
Let X ∈ ℝ^{C×H×W} be the input jet image, where C=3, H=W=125.
The encoder f_θ: ℝ^{C×H×W} → ℝ^{d} maps the image to a continuous latent space.
The decoder g_φ: ℝ^{d} → [0, 1]^{C×H×W} reconstructs the input image.

The objective is to minimize the empirical risk under the Mean Squared Error:
  L(θ, φ) = 1/N ∑_{i=1}^{N} ||X_i - g_φ(f_θ(X_i))||_2^2
"""

import os
import sys
import random
import logging
import argparse
from typing import Tuple, List, Dict, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(__file__))
from data_utils import load_dataset

# ─────────────────────────────────────────────────────────────
# Initialization & Reproducibility
# ─────────────────────────────────────────────────────────────

def setup_logging() -> logging.Logger:
    """Configures structured logging for stdout."""
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    return logging.getLogger(__name__)

logger = setup_logging()

def set_seed(seed: int = 42) -> None:
    """
    Ensures absolute reproducibility across executions by fixing 
    stochasticity in Python, NumPy, and PyTorch.
    
    Args:
        seed (int): The deterministic seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False # Disabled for strict deterministic behavior

# ─────────────────────────────────────────────────────────────
# Dataset wrapper
# ─────────────────────────────────────────────────────────────

class JetImageDataset(Dataset):
    """
    PyTorch Dataset wrapper for memory-aligned jet images.
    
    Attributes:
        X (torch.Tensor): Jet features tensor [N, 3, 125, 125].
        y (torch.Tensor): Classification target tensor [N].
    """
    def __init__(self, X: np.ndarray, y: np.ndarray) -> None:
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self) -> int:
        return self.X.size(0)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.y[idx]

# ─────────────────────────────────────────────────────────────
# Autoencoder Architecture
# ─────────────────────────────────────────────────────────────

class ConvAutoencoder(nn.Module):
    """
    Symmetric Convolutional Autoencoder for sparse jet representation learning.
    
    Design principles:
    - Translational invariance via Strided/MaxPool downsampling.
    - Mitigated internal covariate shift via Batch Normalization.
    - Checkerboard artifact suppression via Bilinear upsampling.
    """

    def __init__(self) -> None:
        super().__init__()

        # Encoder mapping: f_θ
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1), nn.BatchNorm2d(16), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )

        # Decoder mapping: g_φ
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2), nn.BatchNorm2d(16), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(16, 8, kernel_size=2, stride=2),  nn.BatchNorm2d(8),  nn.ReLU(inplace=True),
            nn.Upsample(size=(125, 125), mode="bilinear", align_corners=False),
            nn.Conv2d(8, 3, kernel_size=3, padding=1),
            nn.Sigmoid(),  # Restrict output domain to [0,1]
        )
        
        self._initialize_weights()

    def _initialize_weights(self) -> None:
        """Kaiming normal initialization for asymmetric ReLU activations."""
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward propagation logic.
        
        Args:
            x (torch.Tensor): Input batch [B, 3, 125, 125]
            
        Returns:
            torch.Tensor: Reconstructed batch [B, 3, 125, 125]
        """
        return self.decoder(self.encoder(x))

# ─────────────────────────────────────────────────────────────
# Optimization Routines
# ─────────────────────────────────────────────────────────────

def train_epoch(model: nn.Module, loader: DataLoader, optimizer: torch.optim.Optimizer, 
                criterion: nn.Module, scaler: torch.amp.GradScaler, device: torch.device) -> float:
    """Execute one training iteration computing empirical risk with Automatic Mixed Precision."""
    model.train()
    total_loss = 0.0
    for imgs, _ in tqdm(loader, leave=False, desc="Optimizing"):
        imgs = imgs.to(device, non_blocking=True)
        
        # Mixed Precision Forward (O(1) memory and hardware acceleration)
        with torch.amp.autocast('cuda', enabled=(device.type == 'cuda')):
            recons = model(imgs)
            loss = criterion(recons, imgs)

        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item() * imgs.size(0)
    return total_loss / len(loader.dataset)


@torch.no_grad()
def eval_epoch(model: nn.Module, loader: DataLoader, criterion: nn.Module, device: torch.device) -> float:
    """Evaluate generalization error without gradient tracking."""
    model.eval()
    total_loss = 0.0
    for imgs, _ in loader:
        imgs = imgs.to(device, non_blocking=True)
        with torch.amp.autocast('cuda', enabled=(device.type == 'cuda')):
            loss = criterion(model(imgs), imgs)
        total_loss += loss.item() * imgs.size(0)
    return total_loss / len(loader.dataset)

# ─────────────────────────────────────────────────────────────
# Visualization & Validation
# ─────────────────────────────────────────────────────────────

def plot_reconstructions(model: nn.Module, dataset: Dataset, device: torch.device, out_dir: str, n_show: int = 8) -> None:
    """Generates qualitative visualization of the autoencoder's manifold projection."""
    model.eval()
    CH_NAMES = ["Tracks", "ECAL", "HCAL"]

    imgs_t = torch.stack([dataset[i][0] for i in range(n_show)]).to(device)
    with torch.no_grad():
        recons_t = model(imgs_t).cpu().numpy()
    imgs_np = imgs_t.cpu().numpy()

    fig, axes = plt.subplots(n_show, 6, figsize=(18, n_show * 2.2))
    fig.suptitle("CAE Manifold Reconstruction — Quark/Gluon Jets", fontsize=15, fontweight="bold")
    
    for row in range(n_show):
        for ch in range(3):
            vmax = imgs_np[row, ch].max() + 1e-6
            axes[row, ch].imshow(imgs_np[row, ch], cmap="hot", vmin=0, vmax=vmax)
            axes[row, ch + 3].imshow(recons_t[row, ch], cmap="hot", vmin=0, vmax=vmax)
            
            if row == 0:
                axes[row, ch].set_title(f"Orig {CH_NAMES[ch]}", fontsize=10, fontweight="medium")
                axes[row, ch + 3].set_title(f"Recon {CH_NAMES[ch]}", fontsize=10, fontweight="medium")
            axes[row, ch].axis("off")
            axes[row, ch + 3].axis("off")

    plt.tight_layout()
    save_path = os.path.join(out_dir, "task1_reconstructions.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Saved reconstruction grid → %s", save_path)


def plot_loss_curve(train_losses: List[float], val_losses: List[float], out_dir: str) -> None:
    """Visualizes convergence trajectories."""
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(train_losses, label="Train MSE", color="#2563EB", lw=2.5)
    ax.plot(val_losses, label="Val MSE", color="#DC2626", lw=2.5, linestyle="--")
    ax.set_xlabel("Optimization Epoch", fontsize=11)
    ax.set_ylabel("Mean Squared Error (L2 Risk)", fontsize=11)
    ax.set_title("CAE Optimization Trajectory", fontsize=12, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)

    save_path = os.path.join(out_dir, "task1_loss_curve.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Saved loss curve → %s", save_path)

# ─────────────────────────────────────────────────────────────
# Execution Entrypoint
# ─────────────────────────────────────────────────────────────

def main(args: argparse.Namespace) -> None:
    """Main execution block orchestrating data loading, model instantiation, and fitting."""
    set_seed(args.seed)
    
    device = torch.device("cuda" if torch.cuda.is_available() and not args.force_cpu else "cpu")
    logger.info("Hardware Accelerator: %s", device.type.upper())

    # Environment Setup
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(root_dir, "data")
    out_dir  = os.path.join(root_dir, "outputs")
    os.makedirs(out_dir, exist_ok=True)

    # 1. Dataset Acquisition
    logger.info("Acquiring canonical dataset from %s ...", data_dir)
    try:
        X, y = load_dataset(data_dir, max_events=args.max_events)
    except FileNotFoundError as e:
        logger.error("Dataset resolution failed: %s", e)
        sys.exit(1)

    max_val = np.max(X)
    X_norm = X / (max_val + 1e-8)
    logger.info("Normalized features to Domain: [0, %.2f]", X_norm.max())

    # 2. Stochastic Data Partitioning
    n_samples = len(y)
    idx = np.random.default_rng(args.seed).permutation(n_samples)
    
    n_train, n_val = int(0.70 * n_samples), int(0.15 * n_samples)
    train_idx = idx[:n_train]
    val_idx   = idx[n_train:n_train + n_val]
    test_idx  = idx[n_train + n_val:]

    logger.info("Partitioning | Train: %s | Val: %s | Test: %s", 
                f"{len(train_idx):,}", f"{len(val_idx):,}", f"{len(test_idx):,}")

    train_loader = DataLoader(
        JetImageDataset(X_norm[train_idx], y[train_idx]), 
        batch_size=args.batch_size, shuffle=True,  num_workers=0, pin_memory=(device.type == 'cuda')
    )
    val_loader = DataLoader(
        JetImageDataset(X_norm[val_idx], y[val_idx]),   
        batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=(device.type == 'cuda')
    )

    # 3. Model Engineering & Compilation
    logger.info("Instantiating ConvAutoencoder & Hardware Components ...")
    model = ConvAutoencoder().to(device)
    
    if hasattr(torch, 'compile') and os.name != 'nt' and device.type == 'cuda':
        try:
            model = torch.compile(model)
            logger.info("[Optimized] Graph-mode execution (torch.compile) activated.")
        except Exception as err:
            logger.warning("[Fallback] torch.compile bypassed: %s", err)

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("Topology defined. Learnable parameters: %s", f"{num_params:,}")

    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4) 
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-5)
    scaler    = torch.amp.GradScaler('cuda', enabled=(device.type == 'cuda'))

    # 4. Neural Optimization Strategy
    logger.info("Initiating training protocol for %s epochs ...", args.epochs)
    train_losses, val_losses = [], []
    best_val_loss = float("inf")
    best_state_dict = None

    for epoch in range(1, args.epochs + 1):
        tr_loss = train_epoch(model, train_loader, optimizer, criterion, scaler, device)
        vl_loss = eval_epoch(model, val_loader, criterion, device)
        scheduler.step()

        train_losses.append(tr_loss)
        val_losses.append(vl_loss)

        if vl_loss < best_val_loss:
            best_val_loss = vl_loss
            # Detach explicitly for unreferenced memory garbage collection
            best_state_dict = {k: v.clone().detach() for k, v in model.state_dict().items()}

        if epoch % 5 == 0 or epoch == 1:
            logger.info("Epoch %03d/%03d | Train Risk: %.5f | Val Risk: %.5f | LR: %.1e",
                        epoch, args.epochs, tr_loss, vl_loss, scheduler.get_last_lr()[0])

    # 5. Output Artifact Generation
    logger.info("Restoring optimal manifold parameters (Val MSE: %.5f) & evaluating outputs ...", best_val_loss)
    model.load_state_dict(best_state_dict)
    
    plot_loss_curve(train_losses, val_losses, out_dir)
    plot_reconstructions(model, JetImageDataset(X_norm[val_idx], y[val_idx]), device, out_dir)
    
    logger.info("Run finished successfully.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Unsupervised Jet Manifold Learning via CAE")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch dimensionality configuration")
    parser.add_argument("--epochs", type=int, default=20, help="Terminal optimization limit")
    parser.add_argument("--learning-rate", type=float, default=1e-3, help="Optimizer base step-size parameter")
    parser.add_argument("--max-events", type=int, default=500, help="Dataset truncation bound (local proto vs colab)")
    parser.add_argument("--force-cpu", action="store_true", help="Force PyTorch mapping to CPU backend")
    parser.add_argument("--seed", type=int, default=42, help="Deterministic global seed state")
    
    main(parser.parse_args())

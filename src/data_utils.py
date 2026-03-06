"""
data_utils.py
-------------
Shared utilities for loading the QCD quark/gluon jet image dataset.

Supports two formats automatically:
  1. HDF5  (.hdf5 / .h5)  — preferred, much faster
       X_jets: (N, 125, 125, 3) channels-LAST  → transposed to (N, 3, 125, 125)
  2. Parquet (.parquet)   — fallback
       X_jets: list<list<list<double>>>         → (N, 3, 125, 125)

Label: y=0 quark, y=1 gluon
"""

import os
import sys
import logging
from typing import Tuple, Optional
from dataclasses import dataclass

import numpy as np
import h5py
import pyarrow.parquet as pq

# ─────────────────────────────────────────────────────────────
# Structured Module Setup
# ─────────────────────────────────────────────────────────────
logger = logging.getLogger(__name__)

@dataclass
class DatasetMetrics:
    """Immutible record for loaded data manifolds."""
    n_samples: int
    feature_shape: Tuple[int, ...]
    n_signal: int
    n_background: int

# ─────────────────────────────────────────────────────────────
# Core Loading Subroutines
# ─────────────────────────────────────────────────────────────

import torch
from torch.utils.data import Dataset, DataLoader, random_split


# ─────────────────────────────────────────────────────────────
# Primary loader — auto-detects HDF5 or parquet
# ─────────────────────────────────────────────────────────────

def load_dataset(data_dir: str, max_events: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Robust resolver and loader for the quark/gluon jet dataset.
    Prioritizes locating the official HDF5 archive over individual Parquets.
    
    Args:
        data_dir (str): Primary directory to scan for data partitions.
        max_events (int | None): Truncation limit for rapid prototyping.

    Returns:
        Tuple[np.ndarray, np.ndarray]: The aligned features [N, 3, 125, 125] and labels [N].
        
    Raises:
        FileNotFoundError: If the exact dataset architecture is missing from the workspace.
    """
    # Defensive heuristic: check project root if `data_dir` fails
    parent_dir = os.path.dirname(os.path.abspath(data_dir))
    h5_name = "quark-gluon_data-set_n139306.hdf5"
    h5_paths = [os.path.join(data_dir, h5_name), os.path.join(parent_dir, h5_name)]

    for path in h5_paths:
        if os.path.exists(path):
            logger.info("Resolving official HDF5 manifold archive: %s", path)
            with h5py.File(path, "r") as f:
                limit = min(max_events, f["X_jets"].shape[0]) if max_events else f["X_jets"].shape[0]
                X = f["X_jets"][:limit].astype(np.float32)  # (N,125,125,3)
                y = f["y"][:limit].astype(np.int64)
                
                # HDF5 stores channels-LAST → convert to channels-FIRST
                X = np.transpose(X, (0, 3, 1, 2))   # (N,125,125,3) → (N,3,125,125)

                metrics = DatasetMetrics(
                    n_samples=limit,
                    feature_shape=X.shape[2:],
                    n_signal=int((y == 1).sum()),
                    n_background=int((y == 0).sum())
                )
                logger.debug("Extraction metrics: %s", metrics)
                logger.info(f"  Loaded {len(X):,} events — X shape {X.shape}, "
                            f"quark={np.sum(y==0):,}, gluon={np.sum(y==1):,}")
                return X, y

    # Fallback to decoupled multi-parquet architecture
    logger.warning("HDF5 archive missing. Falling back to decoupled Parquet resolution in %s.", data_dir)
    return _load_from_parquet(data_dir, max_events)


def _load_from_parquet(data_dir: str, max_events: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
    """Sub-routine resolving decoupled Parquet geometries."""
    # ... fallback logging ...
    pq_files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith(".parquet")]
    if not pq_files:
        msg = f"No generic dataset definitions (HDF5 or Parquet) found recursively in {data_dir}"
        logger.critical(msg)
        raise FileNotFoundError(msg)

    X_list, y_list = [], []
    total_loaded = 0

    for pf in pq_files:
        logger.info(f"  Loading {os.path.basename(pf)} …")
        if max_events is not None and total_loaded >= max_events:
            break
        table = pq.read_table(pf)
        df = table.to_pandas()
        for _, row in df.iterrows():
            if max_events is not None and total_loaded >= max_events: break
            
            # Reconstruct (3, 125, 125)
            x_np = np.stack([
                np.vstack(row["X_jets"][0]),
                np.vstack(row["X_jets"][1]),
                np.vstack(row["X_jets"][2])
            ])
            X_list.append(x_np)
            y_list.append(row["y"])
            total_loaded += 1
    
    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.int64)
    logger.info(f"  Loaded {len(X):,} events — X shape {X.shape}")
    return X, y


# ─────────────────────────────────────────────────────────────
# PyTorch Dataset wrappers
# ─────────────────────────────────────────────────────────────

class JetImageDataset(Dataset):
    """Plain (image, label) dataset for the CAE and supervised tasks."""

    def __init__(self, X: np.ndarray, y: np.ndarray):
        # Normalize each channel to [0, 1] using global max
        max_val = X.max()
        self.X = torch.from_numpy(X / (max_val + 1e-8))   # (N, 3, 125, 125)
        self.y = torch.from_numpy(y)                        # (N,)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# ─────────────────────────────────────────────────────────────
# Train / val / test split helper
# ─────────────────────────────────────────────────────────────

def make_splits(dataset: Dataset, train_frac=0.70, val_frac=0.15, seed=42):
    """70 / 15 / 15 deterministic split."""
    n = len(dataset)
    n_train = int(n * train_frac)
    n_val   = int(n * val_frac)

    n_test  = n - n_train - n_val
    generator = torch.Generator().manual_seed(seed)
    return random_split(dataset, [n_train, n_val, n_test], generator=generator)


# ─────────────────────────────────────────────────────────────
# Image → point cloud conversion  (used in Task 2 & 3)
# ─────────────────────────────────────────────────────────────

def image_to_pointcloud(img: np.ndarray, threshold: float = 0.0):
    """
    Convert a single (3, 125, 125) jet image to a point cloud.

    Strategy: treat each pixel position (η_i, φ_j) where at least one
    channel has energy > threshold as a "particle" (graph node).

    Returns
    -------
    points : np.ndarray  shape (M, 5)
        Columns: [η_norm, φ_norm, E_track, E_ECAL, E_HCAL]
        η_norm, φ_norm ∈ [-1, 1] (pixel index normalised to [-1,1])
    """
    assert img.shape == (3, 125, 125), f"Expected (3,125,125), got {img.shape}"
    ch_track, ch_ecal, ch_hcal = img[0], img[1], img[2]

    # Mask: any channel with energy above threshold
    mask = (np.abs(ch_track) + np.abs(ch_ecal) + np.abs(ch_hcal)) > threshold
    eta_idx, phi_idx = np.where(mask)

    if len(eta_idx) == 0:
        # Return a single zero-padding node to avoid empty graphs
        return np.zeros((1, 5), dtype=np.float32)

    # Normalise pixel coordinates to [-1, 1]
    eta_norm = (eta_idx / 124.0) * 2.0 - 1.0   # 124 = 125 - 1
    phi_norm = (phi_idx / 124.0) * 2.0 - 1.0

    e_track = ch_track[eta_idx, phi_idx]
    e_ecal  = ch_ecal [eta_idx, phi_idx]
    e_hcal  = ch_hcal [eta_idx, phi_idx]

    points = np.column_stack([eta_norm, phi_norm, e_track, e_ecal, e_hcal]).astype(np.float32)
    return points

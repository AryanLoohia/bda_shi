"""
utils.py
========
Shared configuration, helpers, and I/O utilities for the BDA project.
"""

import os
import logging
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def setup_logging(level: int = logging.INFO) -> logging.Logger:
    """Configure and return the project logger."""
    logger = logging.getLogger("bda_project")
    if not logger.handlers:
        handler = logging.StreamHandler()
        fmt = logging.Formatter("[%(levelname)s] %(message)s")
        handler.setFormatter(fmt)
        logger.addHandler(handler)
    logger.setLevel(level)
    return logger

log = setup_logging()

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class Config:
    """Central configuration for the entire pipeline."""

    # --- Paths ---
    data_dir: str = "data"
    output_dir: str = "outputs"
    plots_dir: str = "outputs/plots"

    # --- Relation weights for combined score (Step 5) ---
    alpha_follow: float = 0.20
    beta_retweet: float = 0.35
    gamma_mention: float = 0.20
    delta_reply: float = 0.25

    # --- Threshold strategy (Step 6) ---
    # "percentile" or "target_degree"
    threshold_strategy: str = "percentile"
    percentile: float = 5.0          # keep top p% edges
    target_avg_degree: float = 10.0  # alternative strategy

    # --- Numerical safety ---
    epsilon: float = 1e-9

    # --- Feature engineering ---
    betweenness_k: int = 500  # approx betweenness sample size
    large_graph_threshold: int = 50_000  # nodes beyond which we switch to approx algorithms

    # --- Clustering ---
    cluster_range: Tuple[int, ...] = (3, 4, 5, 6, 7, 8)

    # --- Influencer ranking ---
    top_k: int = 20

    # --- Random seed ---
    seed: int = 42

# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def ensure_dir(path: str) -> str:
    """Create directory if it doesn't exist; return the path."""
    os.makedirs(path, exist_ok=True)
    return path


def save_csv(df: pd.DataFrame, path: str, index: bool = False) -> None:
    """Save a DataFrame to CSV, creating parent dirs as needed."""
    ensure_dir(os.path.dirname(path))
    df.to_csv(path, index=index)
    log.info(f"Saved → {path}  ({len(df)} rows)")


def load_csv(path: str) -> pd.DataFrame:
    """Load a CSV file."""
    df = pd.read_csv(path)
    log.info(f"Loaded ← {path}  ({len(df)} rows)")
    return df

# ---------------------------------------------------------------------------
# Node-ID mapping
# ---------------------------------------------------------------------------

def build_node_mapping(
    *dataframes: pd.DataFrame,
) -> Tuple[Dict, Dict, int]:
    """
    Build a unified integer mapping from heterogeneous node IDs.

    Parameters
    ----------
    *dataframes : pd.DataFrame
        Each must have 'source' and 'target' columns.

    Returns
    -------
    id_to_idx : dict  mapping original_id → int index
    idx_to_id : dict  mapping int index → original_id
    n          : int   total number of unique nodes
    """
    all_ids = set()
    for df in dataframes:
        all_ids.update(df["source"].unique())
        all_ids.update(df["target"].unique())

    sorted_ids = sorted(all_ids)
    id_to_idx = {nid: idx for idx, nid in enumerate(sorted_ids)}
    idx_to_id = {idx: nid for nid, idx in id_to_idx.items()}
    n = len(sorted_ids)
    log.info(f"Node mapping built: {n} unique nodes")
    return id_to_idx, idx_to_id, n

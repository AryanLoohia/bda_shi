"""
graph_construction.py
=====================
Steps 1–6: Load data, build null models, compute significance scores,
combine into a single edge score, and construct the sparse filtered graph.
"""

import os
from typing import Dict, Tuple, Optional

import numpy as np
import pandas as pd
from scipy import sparse

from utils import Config, log, load_csv, save_csv, ensure_dir, build_node_mapping

# ===================================================================
# STEP 1 — Load and preprocess (Higgs edgelist format)
# ===================================================================

def _load_edgelist(path: str, default_value: int = 1) -> pd.DataFrame:
    """
    Load a single edgelist file (space/tab separated, no header).
    Handles both 2-column (source target) and 3-column (source target value) formats.
    """
    log.info(f"  Loading {os.path.basename(path)} ...")

    # Try reading as space/tab separated, no header
    df = pd.read_csv(path, sep=r'\s+', header=None, comment='#',
                     engine='python', dtype={0: int, 1: int})

    if df.shape[1] == 2:
        df.columns = ["source", "target"]
        df["value"] = default_value
    elif df.shape[1] >= 3:
        df = df.iloc[:, :3]
        df.columns = ["source", "target", "value"]
    else:
        raise ValueError(f"Unexpected number of columns in {path}: {df.shape[1]}")

    # Remove self-loops
    before = len(df)
    df = df[df["source"] != df["target"]].copy()
    removed = before - len(df)
    if removed > 0:
        log.info(f"    Removed {removed} self-loops")

    df["value"] = pd.to_numeric(df["value"], errors="coerce").fillna(default_value).astype(float)
    log.info(f"    {len(df)} edges loaded")
    return df


def _subsample_nodes(
    df_follow: pd.DataFrame,
    df_retweet: pd.DataFrame,
    df_mention: pd.DataFrame,
    df_reply: pd.DataFrame,
    max_nodes: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Subsample to the top `max_nodes` most active nodes.
    Activity = total times a node appears as source or target across all relations.
    Then filter all DataFrames to only include edges within the selected nodes.
    """
    from collections import Counter

    log.info(f"  Subsampling to {max_nodes} most active nodes ...")

    # Count activity across all relations
    activity = Counter()
    for df in [df_follow, df_retweet, df_mention, df_reply]:
        activity.update(df["source"].values)
        activity.update(df["target"].values)

    total_nodes = len(activity)
    log.info(f"    Total unique nodes across all files: {total_nodes}")

    if total_nodes <= max_nodes:
        log.info(f"    No subsampling needed ({total_nodes} <= {max_nodes})")
        return df_follow, df_retweet, df_mention, df_reply

    # Pick top-k most active nodes
    top_nodes = set(n for n, _ in activity.most_common(max_nodes))
    log.info(f"    Selected top {len(top_nodes)} nodes by activity")

    # Filter all DataFrames
    def _filter(df):
        mask = df["source"].isin(top_nodes) & df["target"].isin(top_nodes)
        return df[mask].copy()

    df_follow  = _filter(df_follow)
    df_retweet = _filter(df_retweet)
    df_mention = _filter(df_mention)
    df_reply   = _filter(df_reply)

    log.info(f"    After filtering — follows: {len(df_follow)}, retweets: {len(df_retweet)}, "
             f"mentions: {len(df_mention)}, replies: {len(df_reply)}")

    return df_follow, df_retweet, df_mention, df_reply


def load_and_preprocess(cfg: Config) -> Tuple[
    sparse.csr_matrix,  # A_follow
    sparse.csr_matrix,  # R_retweet
    sparse.csr_matrix,  # M_mention
    sparse.csr_matrix,  # T_reply
    Dict, Dict, int,    # id_to_idx, idx_to_id, n
    pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame,  # raw dfs
]:
    """
    Load all four Higgs edgelist files, subsample to max_nodes,
    map node IDs, and return sparse matrices.
    """
    log.info("=" * 60)
    log.info("STEP 1: Loading and preprocessing data")
    log.info("=" * 60)

    follows_path  = os.path.join(cfg.data_dir, cfg.follow_file)
    retweets_path = os.path.join(cfg.data_dir, cfg.retweet_file)
    mentions_path = os.path.join(cfg.data_dir, cfg.mention_file)
    replies_path  = os.path.join(cfg.data_dir, cfg.reply_file)

    df_follow  = _load_edgelist(follows_path, default_value=1)
    df_retweet = _load_edgelist(retweets_path)
    df_mention = _load_edgelist(mentions_path)
    df_reply   = _load_edgelist(replies_path)

    # Subsample to max_nodes
    df_follow, df_retweet, df_mention, df_reply = _subsample_nodes(
        df_follow, df_retweet, df_mention, df_reply, cfg.max_nodes
    )

    log.info(f"  Final edge counts:")
    log.info(f"    follows:  {len(df_follow)} edges")
    log.info(f"    retweets: {len(df_retweet)} edges")
    log.info(f"    mentions: {len(df_mention)} edges")
    log.info(f"    replies:  {len(df_reply)} edges")

    # Build unified node mapping
    id_to_idx, idx_to_id, n = build_node_mapping(
        df_follow, df_retweet, df_mention, df_reply
    )

    # Build sparse matrices
    def _to_sparse(df: pd.DataFrame) -> sparse.csr_matrix:
        rows = df["source"].map(id_to_idx).values
        cols = df["target"].map(id_to_idx).values
        vals = df["value"].values
        mat = sparse.coo_matrix((vals, (rows, cols)), shape=(n, n))
        # Sum duplicates
        mat = mat.tocsr()
        mat.eliminate_zeros()
        return mat

    A_follow  = _to_sparse(df_follow)
    R_retweet = _to_sparse(df_retweet)
    M_mention = _to_sparse(df_mention)
    T_reply   = _to_sparse(df_reply)

    log.info(f"  Sparse matrices built: shape=({n}, {n})")
    log.info(f"    A_follow  nnz={A_follow.nnz}")
    log.info(f"    R_retweet nnz={R_retweet.nnz}")
    log.info(f"    M_mention nnz={M_mention.nnz}")
    log.info(f"    T_reply   nnz={T_reply.nnz}")

    return (
        A_follow, R_retweet, M_mention, T_reply,
        id_to_idx, idx_to_id, n,
        df_follow, df_retweet, df_mention, df_reply,
    )


# ===================================================================
# STEP 2 & 3 — Null model expected values + significance scores
# ===================================================================

def _compute_significance_for_matrix(
    W: sparse.csr_matrix,
    rows: np.ndarray,
    cols: np.ndarray,
    epsilon: float,
) -> np.ndarray:
    """
    Compute significance S_ij = (W_ij - E[W_ij]) / sqrt(E[W_ij] + eps)
    only for the given (rows, cols) candidate pairs.

    Parameters
    ----------
    W       : sparse matrix (n x n)
    rows    : array of row indices for candidate edges
    cols    : array of col indices for candidate edges
    epsilon : numerical safety constant

    Returns
    -------
    S : 1-D array of significance scores for each candidate edge
    """
    # Total mass
    m = W.sum()
    if m == 0:
        return np.zeros(len(rows), dtype=np.float64)

    # Marginals
    out_sums = np.asarray(W.sum(axis=1)).ravel()   # out_i = row sums
    in_sums  = np.asarray(W.sum(axis=0)).ravel()   # in_j  = col sums

    # Observed values — use sparse matrix array indexing (vectorized)
    W_csr = W.tocsr()
    W_obs = np.asarray(W_csr[rows, cols]).ravel().astype(np.float64)

    # Expected values
    E_vals = (out_sums[rows] * in_sums[cols]) / m

    # Significance
    S = (W_obs - E_vals) / np.sqrt(E_vals + epsilon)
    return S


def compute_candidate_edges(
    A: sparse.csr_matrix,
    R: sparse.csr_matrix,
    M: sparse.csr_matrix,
    T: sparse.csr_matrix,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute the union candidate edge set U:
    U = {(i, j) : A_ij > 0 or R_ij > 0 or M_ij > 0 or T_ij > 0}

    Returns
    -------
    rows, cols : arrays of row and column indices
    """
    # Binary union
    B = ((A > 0) + (R > 0) + (M > 0) + (T > 0))
    B = B.tocoo()
    rows = B.row
    cols = B.col
    log.info(f"  Union candidate edge set |U| = {len(rows)}")
    return rows, cols


def compute_all_significance_scores(
    A: sparse.csr_matrix,
    R: sparse.csr_matrix,
    M: sparse.csr_matrix,
    T: sparse.csr_matrix,
    rows: np.ndarray,
    cols: np.ndarray,
    epsilon: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Steps 2-3: Compute significance scores for all four relations
    on the candidate edge set.
    """
    log.info("=" * 60)
    log.info("STEPS 2-3: Computing null model significance scores")
    log.info("=" * 60)

    S_follow  = _compute_significance_for_matrix(A, rows, cols, epsilon)
    S_retweet = _compute_significance_for_matrix(R, rows, cols, epsilon)
    S_mention = _compute_significance_for_matrix(M, rows, cols, epsilon)
    S_reply   = _compute_significance_for_matrix(T, rows, cols, epsilon)

    log.info(f"  S_follow  range: [{S_follow.min():.3f}, {S_follow.max():.3f}]")
    log.info(f"  S_retweet range: [{S_retweet.min():.3f}, {S_retweet.max():.3f}]")
    log.info(f"  S_mention range: [{S_mention.min():.3f}, {S_mention.max():.3f}]")
    log.info(f"  S_reply   range: [{S_reply.min():.3f}, {S_reply.max():.3f}]")

    return S_follow, S_retweet, S_mention, S_reply


# ===================================================================
# STEP 4 — Standardize relation scores
# ===================================================================

def standardize_scores(
    S_follow: np.ndarray,
    S_retweet: np.ndarray,
    S_mention: np.ndarray,
    S_reply: np.ndarray,
    epsilon: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Step 4: Z-standardize each relation score array.
    """
    log.info("=" * 60)
    log.info("STEP 4: Standardizing relation scores")
    log.info("=" * 60)

    def _zscore(arr):
        mu = arr.mean()
        sigma = arr.std()
        return (arr - mu) / (sigma + epsilon)

    Z_follow  = _zscore(S_follow)
    Z_retweet = _zscore(S_retweet)
    Z_mention = _zscore(S_mention)
    Z_reply   = _zscore(S_reply)

    log.info("  Standardized all four score arrays")
    return Z_follow, Z_retweet, Z_mention, Z_reply


# ===================================================================
# STEP 5 — Combine into one edge score
# ===================================================================

def combine_scores(
    Z_follow: np.ndarray,
    Z_retweet: np.ndarray,
    Z_mention: np.ndarray,
    Z_reply: np.ndarray,
    cfg: Config,
) -> np.ndarray:
    """
    Step 5: Weighted combination Q_ij = α*Z_f + β*Z_r + γ*Z_m + δ*Z_t
    """
    log.info("=" * 60)
    log.info("STEP 5: Combining relation scores")
    log.info("=" * 60)

    Q = (cfg.alpha_follow  * Z_follow
       + cfg.beta_retweet  * Z_retweet
       + cfg.gamma_mention * Z_mention
       + cfg.delta_reply   * Z_reply)

    log.info(f"  Combined score Q range: [{Q.min():.3f}, {Q.max():.3f}]")
    log.info(f"  Positive edges: {(Q > 0).sum()} / {len(Q)}")
    return Q


# ===================================================================
# STEP 6 — Construct sparse graph
# ===================================================================

def build_sparse_graph(
    Q: np.ndarray,
    rows: np.ndarray,
    cols: np.ndarray,
    n: int,
    cfg: Config,
) -> Tuple[sparse.csr_matrix, sparse.csr_matrix, pd.DataFrame]:
    """
    Step 6: Apply thresholding to construct the final sparse graph.

    Returns
    -------
    W_sparse : weighted adjacency matrix (CSR)
    B_sparse : binary adjacency matrix (CSR)
    edge_df  : DataFrame with source, target, score columns (integer indices)
    """
    log.info("=" * 60)
    log.info("STEP 6: Constructing sparse graph")
    log.info("=" * 60)

    # Only keep positive scores
    pos_mask = Q > 0
    Q_pos = Q[pos_mask]
    rows_pos = rows[pos_mask]
    cols_pos = cols[pos_mask]
    log.info(f"  Positive-score edges: {len(Q_pos)}")

    if len(Q_pos) == 0:
        log.warning("  No positive edges! Returning empty graph.")
        W_sparse = sparse.csr_matrix((n, n))
        B_sparse = sparse.csr_matrix((n, n))
        edge_df = pd.DataFrame(columns=["source", "target", "score"])
        return W_sparse, B_sparse, edge_df

    # Determine threshold
    if cfg.threshold_strategy == "percentile":
        # Keep top p% of positive edges
        threshold = np.percentile(Q_pos, 100 - cfg.percentile)
        log.info(f"  Strategy: percentile (top {cfg.percentile}%)")
        log.info(f"  Threshold = {threshold:.4f}")

    elif cfg.threshold_strategy == "target_degree":
        # Target average degree
        target_edges = int(cfg.target_avg_degree * n / 2)  # approximate
        target_edges = min(target_edges, len(Q_pos))
        if target_edges <= 0:
            threshold = Q_pos.max() + 1
        else:
            sorted_q = np.sort(Q_pos)[::-1]
            threshold = sorted_q[min(target_edges - 1, len(sorted_q) - 1)]
        log.info(f"  Strategy: target_degree (target={cfg.target_avg_degree})")
        log.info(f"  Threshold = {threshold:.4f}")

    else:
        raise ValueError(f"Unknown threshold strategy: {cfg.threshold_strategy}")

    # Apply threshold
    keep_mask = Q_pos >= threshold
    final_rows = rows_pos[keep_mask]
    final_cols = cols_pos[keep_mask]
    final_vals = Q_pos[keep_mask]

    log.info(f"  Edges after thresholding: {len(final_vals)}")

    # Build sparse matrices
    W_sparse = sparse.coo_matrix(
        (final_vals, (final_rows, final_cols)), shape=(n, n)
    ).tocsr()

    ones = np.ones(len(final_vals))
    B_sparse = sparse.coo_matrix(
        (ones, (final_rows, final_cols)), shape=(n, n)
    ).tocsr()

    # Edge DataFrame
    edge_df = pd.DataFrame({
        "source": final_rows,
        "target": final_cols,
        "score": final_vals,
    })

    avg_degree = B_sparse.nnz / n if n > 0 else 0
    log.info(f"  Final graph: {n} nodes, {B_sparse.nnz} edges")
    log.info(f"  Average degree: {avg_degree:.2f}")

    return W_sparse, B_sparse, edge_df


# ===================================================================
# Full pipeline convenience function
# ===================================================================

def run_graph_construction(cfg: Config):
    """
    Execute Steps 1–6 end-to-end.

    Returns
    -------
    dict with keys:
        A_follow, R_retweet, M_mention, T_reply,
        id_to_idx, idx_to_id, n,
        W_sparse, B_sparse, edge_df,
        Q_all, rows_all, cols_all
    """
    # Step 1
    (A, R, M, T, id_to_idx, idx_to_id, n,
     df_f, df_r, df_m, df_t) = load_and_preprocess(cfg)

    # Steps 2-3
    rows, cols = compute_candidate_edges(A, R, M, T)
    S_f, S_r, S_m, S_t = compute_all_significance_scores(
        A, R, M, T, rows, cols, cfg.epsilon
    )

    # Step 4
    Z_f, Z_r, Z_m, Z_t = standardize_scores(S_f, S_r, S_m, S_t, cfg.epsilon)

    # Step 5
    Q = combine_scores(Z_f, Z_r, Z_m, Z_t, cfg)

    # Step 6
    W_sparse, B_sparse, edge_df = build_sparse_graph(Q, rows, cols, n, cfg)

    # Save edges
    ensure_dir(cfg.output_dir)
    save_csv(edge_df, os.path.join(cfg.output_dir, "sparse_edges.csv"))

    return {
        "A_follow": A, "R_retweet": R, "M_mention": M, "T_reply": T,
        "id_to_idx": id_to_idx, "idx_to_id": idx_to_id, "n": n,
        "df_follow": df_f, "df_retweet": df_r, "df_mention": df_m, "df_reply": df_t,
        "W_sparse": W_sparse, "B_sparse": B_sparse, "edge_df": edge_df,
        "Q_all": Q, "rows_all": rows, "cols_all": cols,
    }

"""
graph_validation.py
===================
Step 7: Validate the sparse graph — compute statistics, compare with
the raw follow graph, and produce diagnostic information.
"""

import os
from typing import Dict

import numpy as np
import pandas as pd
import networkx as nx
from scipy import sparse

from utils import Config, log, save_csv


def _sparse_to_nx(W: sparse.csr_matrix, weighted: bool = True) -> nx.DiGraph:
    """Convert a sparse adjacency matrix to a NetworkX DiGraph."""
    G = nx.DiGraph()
    W_coo = W.tocoo()
    if weighted:
        for r, c, v in zip(W_coo.row, W_coo.col, W_coo.data):
            G.add_edge(int(r), int(c), weight=float(v))
    else:
        for r, c in zip(W_coo.row, W_coo.col):
            G.add_edge(int(r), int(c))
    # Ensure all nodes 0..n-1 are present
    n = W.shape[0]
    G.add_nodes_from(range(n))
    return G


def compute_graph_stats(
    W_sparse: sparse.csr_matrix,
    B_sparse: sparse.csr_matrix,
    n: int,
    num_candidate_edges: int,
) -> Dict:
    """
    Compute comprehensive graph statistics.
    """
    log.info("=" * 60)
    log.info("STEP 7: Graph validation")
    log.info("=" * 60)

    G = _sparse_to_nx(W_sparse, weighted=True)
    num_edges = B_sparse.nnz
    density = num_edges / (n * (n - 1)) if n > 1 else 0

    # Degree stats
    in_degrees = np.asarray(B_sparse.sum(axis=0)).ravel()
    out_degrees = np.asarray(B_sparse.sum(axis=1)).ravel()
    avg_in = in_degrees.mean()
    avg_out = out_degrees.mean()

    # Connected components
    num_wcc = nx.number_weakly_connected_components(G)
    largest_wcc = max(nx.weakly_connected_components(G), key=len)
    largest_wcc_size = len(largest_wcc)

    num_scc = nx.number_strongly_connected_components(G)

    # Reciprocity
    recip = nx.reciprocity(G) if num_edges > 0 else 0.0

    stats = {
        "num_nodes": n,
        "num_candidate_edges": num_candidate_edges,
        "num_edges_after_filtering": num_edges,
        "density": density,
        "avg_in_degree": avg_in,
        "avg_out_degree": avg_out,
        "num_weakly_connected_components": num_wcc,
        "largest_wcc_size": largest_wcc_size,
        "largest_wcc_fraction": largest_wcc_size / n if n > 0 else 0,
        "num_strongly_connected_components": num_scc,
        "reciprocity": recip,
        "in_degree_median": float(np.median(in_degrees)),
        "in_degree_max": float(in_degrees.max()),
        "out_degree_median": float(np.median(out_degrees)),
        "out_degree_max": float(out_degrees.max()),
    }

    log.info("  Graph Statistics:")
    for k, v in stats.items():
        if isinstance(v, float):
            log.info(f"    {k}: {v:.6f}")
        else:
            log.info(f"    {k}: {v}")

    return stats


def compare_with_follow_graph(
    A_follow: sparse.csr_matrix,
    W_sparse: sparse.csr_matrix,
    n: int,
    top_k: int = 20,
) -> Dict:
    """
    Compare the sparse filtered graph with the raw follow graph.
    """
    log.info("-" * 40)
    log.info("  Comparing sparse graph with follow-only graph")
    log.info("-" * 40)

    # Follow graph stats
    follow_density = A_follow.nnz / (n * (n - 1)) if n > 1 else 0
    sparse_density = W_sparse.nnz / (n * (n - 1)) if n > 1 else 0

    log.info(f"    Follow graph density: {follow_density:.6f}")
    log.info(f"    Sparse graph density: {sparse_density:.6f}")
    log.info(f"    Density ratio: {sparse_density / follow_density:.4f}" if follow_density > 0 else "    N/A")

    # Top nodes by in-degree in follow graph
    follow_in = np.asarray(A_follow.sum(axis=0)).ravel()
    top_follow = np.argsort(follow_in)[::-1][:top_k]

    # PageRank on sparse graph
    G_sparse = _sparse_to_nx(W_sparse, weighted=True)
    try:
        pr = nx.pagerank(G_sparse, weight="weight", max_iter=200)
    except Exception:
        pr = {i: 1.0 / n for i in range(n)}

    pr_arr = np.array([pr.get(i, 0) for i in range(n)])
    top_pr = np.argsort(pr_arr)[::-1][:top_k]

    # Overlap
    overlap = len(set(top_follow) & set(top_pr))
    log.info(f"    Overlap@{top_k} (follow in-degree vs sparse PageRank): {overlap}/{top_k}")

    comparison = {
        "follow_density": follow_density,
        "sparse_density": sparse_density,
        "top_follow_indeg_nodes": top_follow.tolist(),
        "top_sparse_pagerank_nodes": top_pr.tolist(),
        f"overlap_at_{top_k}": overlap,
    }

    return comparison


def run_graph_validation(
    data: Dict,
    cfg: Config,
) -> Dict:
    """
    Full graph validation pipeline (Step 7).
    """
    stats = compute_graph_stats(
        data["W_sparse"], data["B_sparse"], data["n"],
        num_candidate_edges=len(data["rows_all"]),
    )

    comparison = compare_with_follow_graph(
        data["A_follow"], data["W_sparse"], data["n"],
        top_k=cfg.top_k,
    )

    # Merge and save
    all_stats = {**stats, **{f"comparison_{k}": v for k, v in comparison.items()
                             if not isinstance(v, list)}}

    stats_df = pd.DataFrame([all_stats])
    save_csv(stats_df, os.path.join(cfg.output_dir, "graph_summary.csv"))

    return {"stats": stats, "comparison": comparison}

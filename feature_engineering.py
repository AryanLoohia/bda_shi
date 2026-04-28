"""
feature_engineering.py
======================
Steps 8–9: Compute the 17-dimensional node feature vector for every node,
apply transformations, and save the final feature matrix.
"""

import os
from typing import Dict, Tuple, Optional
from collections import defaultdict

import numpy as np
import pandas as pd
import networkx as nx
from scipy import sparse
from sklearn.preprocessing import StandardScaler

from utils import Config, log, save_csv

# ===================================================================
# Helper: sparse → NetworkX
# ===================================================================

def _build_nx_graphs(
    W_sparse: sparse.csr_matrix,
    B_sparse: sparse.csr_matrix,
    n: int,
) -> Tuple[nx.DiGraph, nx.Graph]:
    """Build directed (weighted) and undirected graphs from sparse matrices."""
    G_dir = nx.DiGraph()
    G_dir.add_nodes_from(range(n))
    W_coo = W_sparse.tocoo()
    for r, c, v in zip(W_coo.row, W_coo.col, W_coo.data):
        G_dir.add_edge(int(r), int(c), weight=float(v))

    G_und = G_dir.to_undirected()
    return G_dir, G_und


# ===================================================================
# A. Intrinsic node features (1–11)
# ===================================================================

def _compute_degree_features(B_sparse: sparse.csr_matrix, W_sparse: sparse.csr_matrix, n: int):
    """Features 1-4: in/out degree and weighted strength."""
    in_deg  = np.asarray(B_sparse.sum(axis=0)).ravel().astype(float)
    out_deg = np.asarray(B_sparse.sum(axis=1)).ravel().astype(float)
    w_in    = np.asarray(W_sparse.sum(axis=0)).ravel().astype(float)
    w_out   = np.asarray(W_sparse.sum(axis=1)).ravel().astype(float)
    return in_deg, out_deg, w_in, w_out


def _compute_pagerank(G: nx.DiGraph, n: int) -> np.ndarray:
    """Feature 5: weighted PageRank."""
    log.info("    Computing PageRank ...")
    try:
        pr = nx.pagerank(G, weight="weight", max_iter=300, tol=1e-8)
    except Exception:
        pr = {i: 1.0 / n for i in range(n)}
    return np.array([pr.get(i, 0) for i in range(n)])


def _compute_hits(G: nx.DiGraph, n: int) -> Tuple[np.ndarray, np.ndarray]:
    """Features 6-7: HITS authority and hub scores."""
    log.info("    Computing HITS ...")
    try:
        hubs, auths = nx.hits(G, max_iter=300, tol=1e-8)
    except Exception:
        hubs = {i: 1.0 / n for i in range(n)}
        auths = {i: 1.0 / n for i in range(n)}
    auth_arr = np.array([auths.get(i, 0) for i in range(n)])
    hub_arr  = np.array([hubs.get(i, 0) for i in range(n)])
    return auth_arr, hub_arr


def _compute_betweenness(G: nx.DiGraph, n: int, k: int) -> np.ndarray:
    """Feature 8: (approximate) betweenness centrality."""
    log.info(f"    Computing betweenness centrality (k={min(k, n)}) ...")
    sample_k = min(k, n)
    try:
        bc = nx.betweenness_centrality(G, k=sample_k, weight="weight")
    except Exception:
        bc = nx.betweenness_centrality(G, k=sample_k)
    return np.array([bc.get(i, 0) for i in range(n)])


def _compute_kcore(G_und: nx.Graph, n: int) -> np.ndarray:
    """Feature 9: k-core number (undirected)."""
    log.info("    Computing k-core numbers ...")
    kc = nx.core_number(G_und)
    return np.array([kc.get(i, 0) for i in range(n)])


def _compute_node_reciprocity(B_sparse: sparse.csr_matrix, n: int) -> np.ndarray:
    """Feature 10: per-node reciprocity."""
    log.info("    Computing node reciprocity ...")
    B_t = B_sparse.T.tocsr()
    # Mutual links: edges that exist in both directions
    mutual = B_sparse.multiply(B_t)  # element-wise AND
    mutual_per_node = (
        np.asarray(mutual.sum(axis=0)).ravel()
        + np.asarray(mutual.sum(axis=1)).ravel()
    )
    total_per_node = (
        np.asarray(B_sparse.sum(axis=0)).ravel()
        + np.asarray(B_sparse.sum(axis=1)).ravel()
    )
    recip = np.divide(
        mutual_per_node, total_per_node,
        out=np.zeros(n, dtype=float),
        where=total_per_node > 0,
    )
    return recip


def _compute_clustering(G_und: nx.Graph, n: int) -> np.ndarray:
    """Feature 11: local clustering coefficient (undirected)."""
    log.info("    Computing local clustering coefficients ...")
    cc = nx.clustering(G_und)
    return np.array([cc.get(i, 0) for i in range(n)])


# ===================================================================
# B. Neighborhood context features (12–14)
# ===================================================================

def _compute_reach(G: nx.DiGraph, n: int) -> Tuple[np.ndarray, np.ndarray]:
    """Features 12-13: 2-hop and 3-hop reach via BFS."""
    log.info("    Computing 2-hop and 3-hop reach (BFS) ...")
    reach2 = np.zeros(n, dtype=float)
    reach3 = np.zeros(n, dtype=float)

    for i in range(n):
        lengths = nx.single_source_shortest_path_length(G, i, cutoff=3)
        # lengths includes the node itself at distance 0
        r2 = sum(1 for d in lengths.values() if 0 < d <= 2)
        r3 = sum(1 for d in lengths.values() if 0 < d <= 3)
        reach2[i] = r2
        reach3[i] = r3

        if (i + 1) % 2000 == 0:
            log.info(f"      ... processed {i + 1}/{n} nodes")

    return reach2, reach3


def _compute_top5_neighborhood_degree(G: nx.DiGraph, n: int) -> np.ndarray:
    """
    Feature 14: For each node, get degrees of all 2-hop neighbors,
    sort descending, return top 5 values. Returns (n, 5) array.
    """
    log.info("    Computing top-5 neighborhood degree profile ...")
    result = np.zeros((n, 5), dtype=float)

    for i in range(n):
        lengths = nx.single_source_shortest_path_length(G, i, cutoff=2)
        neighbors = [node for node, dist in lengths.items() if dist > 0]
        if not neighbors:
            continue
        degs = sorted([G.degree(nb) for nb in neighbors], reverse=True)
        top5 = degs[:5]
        for k, val in enumerate(top5):
            result[i, k] = val

        if (i + 1) % 2000 == 0:
            log.info(f"      ... processed {i + 1}/{n} nodes")

    return result


# ===================================================================
# C. Community features (15–17)
# ===================================================================

def _detect_communities(G_und: nx.Graph, n: int) -> Dict[int, int]:
    """Run community detection; return node → community_id mapping."""
    log.info("    Detecting communities ...")

    # Try python-louvain first
    try:
        import community as community_louvain
        partition = community_louvain.best_partition(G_und, random_state=42)
        log.info(f"    Using Louvain community detection")
        return partition
    except ImportError:
        pass

    # Fallback: greedy modularity
    log.info("    Louvain not available, using greedy modularity")
    communities = nx.community.greedy_modularity_communities(G_und)
    partition = {}
    for cid, comm in enumerate(communities):
        for node in comm:
            partition[node] = cid
    # Ensure all nodes have a community
    for i in range(n):
        if i not in partition:
            partition[i] = -1
    return partition


def _compute_community_features(
    B_sparse: sparse.csr_matrix,
    partition: Dict[int, int],
    n: int,
    epsilon: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Features 15-17: community_size, within_community_zscore,
    participation_coefficient.
    """
    log.info("    Computing community features ...")

    # Community membership
    community_ids = np.array([partition.get(i, -1) for i in range(n)])
    unique_comms = set(community_ids)

    # Community sizes (Feature 15)
    comm_sizes = defaultdict(int)
    for c in community_ids:
        comm_sizes[c] += 1
    community_size = np.array([comm_sizes[community_ids[i]] for i in range(n)], dtype=float)

    # Undirected adjacency for community features
    B_und = B_sparse + B_sparse.T
    B_und = (B_und > 0).astype(float)

    # Within-community z-score (Feature 16)
    # internal_degree_i = number of links to nodes in same community
    internal_deg = np.zeros(n, dtype=float)
    B_coo = B_und.tocoo()
    for r, c in zip(B_coo.row, B_coo.col):
        if community_ids[r] == community_ids[c]:
            internal_deg[r] += 1

    within_z = np.zeros(n, dtype=float)
    for c in unique_comms:
        mask = community_ids == c
        if mask.sum() <= 1:
            continue
        intdeg_c = internal_deg[mask]
        mu = intdeg_c.mean()
        sigma = intdeg_c.std()
        within_z[mask] = (intdeg_c - mu) / (sigma + epsilon)

    # Participation coefficient (Feature 17)
    # P_i = 1 - sum_c (k_ic / k_i)^2
    node_comm_links = defaultdict(lambda: defaultdict(int))
    total_links = np.zeros(n, dtype=float)

    for r, c in zip(B_coo.row, B_coo.col):
        comm_c = community_ids[c]
        node_comm_links[r][comm_c] += 1
        total_links[r] += 1

    participation = np.zeros(n, dtype=float)
    for i in range(n):
        ki = total_links[i]
        if ki == 0:
            continue
        s = sum((cnt / ki) ** 2 for cnt in node_comm_links[i].values())
        participation[i] = 1.0 - s

    return community_size, within_z, participation


# ===================================================================
# Main feature engineering pipeline
# ===================================================================

def compute_node_features(data: Dict, cfg: Config) -> Tuple[pd.DataFrame, Dict[int, int]]:
    """
    Steps 8-9: Compute all 17 node features, transform, standardize,
    and return the feature DataFrame + community partition.
    """
    log.info("=" * 60)
    log.info("STEPS 8-9: Computing node features")
    log.info("=" * 60)

    W_sparse = data["W_sparse"]
    B_sparse = data["B_sparse"]
    n = data["n"]
    idx_to_id = data["idx_to_id"]

    G_dir, G_und = _build_nx_graphs(W_sparse, B_sparse, n)

    # --- A. Intrinsic ---
    in_deg, out_deg, w_in, w_out = _compute_degree_features(B_sparse, W_sparse, n)
    pagerank = _compute_pagerank(G_dir, n)
    authority, hub = _compute_hits(G_dir, n)
    betweenness = _compute_betweenness(G_dir, n, k=cfg.betweenness_k)
    kcore = _compute_kcore(G_und, n)
    reciprocity = _compute_node_reciprocity(B_sparse, n)
    clustering = _compute_clustering(G_und, n)

    # --- B. Neighborhood ---
    reach2, reach3 = _compute_reach(G_dir, n)
    top5_deg = _compute_top5_neighborhood_degree(G_dir, n)

    # --- C. Community ---
    partition = _detect_communities(G_und, n)
    comm_size, within_z, participation = _compute_community_features(
        B_sparse, partition, n, cfg.epsilon
    )

    # Assemble raw feature matrix
    features = pd.DataFrame({
        "original_node_id": [idx_to_id[i] for i in range(n)],
        "mapped_node_index": list(range(n)),
        "in_degree": in_deg,
        "out_degree": out_deg,
        "weighted_in_strength": w_in,
        "weighted_out_strength": w_out,
        "pagerank": pagerank,
        "hits_authority": authority,
        "hits_hub": hub,
        "betweenness_centrality": betweenness,
        "k_core_number": kcore,
        "reciprocity": reciprocity,
        "local_clustering_coefficient": clustering,
        "two_hop_reach": reach2,
        "three_hop_reach": reach3,
        "top_deg_1": top5_deg[:, 0],
        "top_deg_2": top5_deg[:, 1],
        "top_deg_3": top5_deg[:, 2],
        "top_deg_4": top5_deg[:, 3],
        "top_deg_5": top5_deg[:, 4],
        "community_size": comm_size,
        "within_community_zscore": within_z,
        "participation_coefficient": participation,
        "community_id": [partition.get(i, -1) for i in range(n)],
    })

    # Handle NaN / inf
    features.replace([np.inf, -np.inf], np.nan, inplace=True)
    features.fillna(0, inplace=True)

    # Save raw features
    save_csv(features, os.path.join(cfg.output_dir, "node_features_raw.csv"))

    # --- Transformations ---
    log.info("  Applying log1p to heavy-tailed features ...")
    log1p_cols = [
        "in_degree", "out_degree",
        "weighted_in_strength", "weighted_out_strength",
        "pagerank", "betweenness_centrality",
        "two_hop_reach", "three_hop_reach",
        "top_deg_1", "top_deg_2", "top_deg_3", "top_deg_4", "top_deg_5",
        "community_size",
    ]
    for col in log1p_cols:
        features[col] = np.log1p(features[col])

    # Identify feature columns (exclude IDs and community_id)
    feature_cols = [c for c in features.columns
                    if c not in ("original_node_id", "mapped_node_index", "community_id")]

    # StandardScaler
    log.info("  Standardizing features ...")
    scaler = StandardScaler()
    features[feature_cols] = scaler.fit_transform(features[feature_cols])

    # Final NaN cleanup
    features.fillna(0, inplace=True)

    save_csv(features, os.path.join(cfg.output_dir, "node_features.csv"))
    log.info(f"  Feature matrix shape: {features.shape}")

    return features, partition

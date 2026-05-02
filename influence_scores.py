"""
influence_scores.py
===================
Steps 10-14: PCA, clustering, influence scoring, ranking, baseline comparison.
"""
import os
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from scipy import stats
from collections import Counter
from utils import Config, log, save_csv

_ID_COLS = {"original_node_id", "mapped_node_index", "community_id"}

def _get_feature_cols(df):
    return [c for c in df.columns if c not in _ID_COLS]

def run_pca(features, cfg):
    log.info("=" * 60)
    log.info("STEP 10: PCA")
    log.info("=" * 60)
    feat_cols = _get_feature_cols(features)
    X = features[feat_cols].values
    pca = PCA(random_state=cfg.seed)
    X_pca = pca.fit_transform(X)
    evr = pca.explained_variance_ratio_
    cumvar = np.cumsum(evr)
    for i in range(min(5, len(evr))):
        log.info(f"  PC{i+1}: var={evr[i]:.4f} cum={cumvar[i]:.4f}")
    loadings = pca.components_
    interpretations = {}
    for pc in range(min(3, loadings.shape[0])):
        top_idx = np.argsort(np.abs(loadings[pc]))[::-1][:5]
        top_f = [(feat_cols[j], loadings[pc, j]) for j in top_idx]
        interpretations[f"PC{pc+1}"] = top_f
        log.info(f"  PC{pc+1} top: {top_f}")
    return {"pca": pca, "X_pca": X_pca, "explained_variance_ratio": evr,
            "cumulative_variance": cumvar, "loadings": loadings,
            "feature_cols": feat_cols, "interpretations": interpretations}

def run_clustering(features, pca_result, cfg):
    log.info("=" * 60)
    log.info("STEP 11: Clustering")
    log.info("=" * 60)
    feat_cols = _get_feature_cols(features)
    X = features[feat_cols].values
    results = {}
    for k in cfg.cluster_range:
        km = KMeans(n_clusters=k, random_state=cfg.seed, n_init=10, max_iter=300)
        labels = km.fit_predict(X)
        sil = silhouette_score(X, labels) if k > 1 else 0
        results[k] = {"labels": labels, "silhouette": sil, "inertia": km.inertia_}
        log.info(f"  k={k}: sil={sil:.4f} inertia={km.inertia_:.1f}")
    best_k = max(results, key=lambda k: results[k]["silhouette"])
    best_labels = results[best_k]["labels"]
    log.info(f"  Best k={best_k}")
    profiles = pd.DataFrame(index=range(best_k), columns=feat_cols, dtype=float)
    for c in range(best_k):
        profiles.loc[c] = features.loc[best_labels == c, feat_cols].mean()
    role_labels = _interpret_clusters(profiles, feat_cols, best_k)
    log.info(f"  Roles: {role_labels}")
    counts = Counter(best_labels)
    for c in range(best_k):
        log.info(f"  Cluster {c} ({role_labels.get(c, 'Unknown')}): {counts[c]} nodes")
    return {"all_results": results, "best_k": best_k, "labels": best_labels,
            "cluster_profiles": profiles, "role_labels": role_labels}

def _interpret_clusters(profiles, feat_cols, k):
    # Expanded archetypes covering all 7 influencer types + extras
    archetypes = {
        "Global Influencer": ["pagerank", "in_degree", "weighted_in_strength"],
        "Hub / Broadcaster": ["out_degree", "weighted_out_strength", "hits_hub", "three_hop_reach"],
        "Bridge / Broker": ["betweenness_centrality", "participation_coefficient"],
        "Local Community Leader": ["within_community_zscore", "local_clustering_coefficient"],
        "Authority": ["pagerank", "hits_authority", "k_core_number"],
        "Engagement-Driven": ["reciprocity", "weighted_in_strength", "weighted_out_strength"],
        "Hidden / Niche Influencer": ["k_core_number", "within_community_zscore", "participation_coefficient"],
        "Information Spreader": ["two_hop_reach", "three_hop_reach", "out_degree"],
    }

    # Feature → human-readable name for fallback labeling
    feature_labels = {
        "in_degree": "High In-Degree", "out_degree": "High Out-Degree",
        "weighted_in_strength": "High In-Strength", "weighted_out_strength": "High Out-Strength",
        "pagerank": "High PageRank", "hits_authority": "High Authority",
        "hits_hub": "High Hub", "betweenness_centrality": "High Betweenness",
        "k_core_number": "Dense Core", "reciprocity": "High Reciprocity",
        "local_clustering_coefficient": "Tightly Clustered",
        "two_hop_reach": "Wide 2-Hop Reach", "three_hop_reach": "Wide 3-Hop Reach",
        "participation_coefficient": "Cross-Community",
        "within_community_zscore": "Community Leader",
        "community_size": "Large Community",
    }

    role_map = {}
    used = set()

    # Phase 1: greedily assign named archetypes to clusters with strongest match
    cluster_scores = {}
    for c in range(k):
        for role, feats in archetypes.items():
            avail = [f for f in feats if f in feat_cols]
            if not avail:
                continue
            s = profiles.loc[c, avail].mean()
            cluster_scores[(c, role)] = s

    # Sort by score descending, assign greedily
    sorted_pairs = sorted(cluster_scores.items(), key=lambda x: x[1], reverse=True)
    assigned_clusters = set()
    for (c, role), score in sorted_pairs:
        if c in assigned_clusters or role in used:
            continue
        if score > 0:  # only assign if the cluster actually shows this trait
            role_map[c] = role
            assigned_clusters.add(c)
            used.add(role)

    # Phase 2: for unassigned clusters, generate a data-driven label
    # from their most distinctive (highest z-scored) feature
    for c in range(k):
        if c in assigned_clusters:
            continue
        # Find the feature where this cluster stands out most
        row = profiles.loc[c]
        avail = [f for f in feat_cols if f in row.index]
        if avail:
            top_feat = max(avail, key=lambda f: float(row[f]))
            top_val = float(row[top_feat])
            if top_val > 0:
                label = feature_labels.get(top_feat, top_feat.replace("_", " ").title())
                role_map[c] = f"{label} Group"
            else:
                role_map[c] = "Low-Activity / Peripheral"
        else:
            role_map[c] = "Low-Activity / Peripheral"
        assigned_clusters.add(c)

    return role_map

def compute_influence_scores(features, data, cfg):
    log.info("=" * 60)
    log.info("STEP 12: Influence scores")
    log.info("=" * 60)
    df = features.copy()
    z = lambda col: df[col].values if col in df.columns else np.zeros(len(df))
    df["I_popularity"] = z("in_degree") + z("weighted_in_strength") + z("pagerank")
    df["I_authority"] = z("pagerank") + z("hits_authority") + z("k_core_number")
    df["I_hub"] = z("out_degree") + z("weighted_out_strength") + z("hits_hub") + z("three_hop_reach")
    df["I_bridge"] = z("betweenness_centrality") + z("participation_coefficient") - z("local_clustering_coefficient")
    df["I_local"] = z("within_community_zscore") + z("weighted_in_strength") + z("community_size")
    df["I_engagement"] = z("reciprocity") + z("weighted_in_strength") + z("weighted_out_strength")
    A = data["A_follow"]
    raw_fi = np.asarray(A.sum(axis=0)).ravel().astype(float)
    raw_fi_z = (raw_fi - raw_fi.mean()) / (raw_fi.std() + cfg.epsilon)
    struct = df["I_authority"].values + df["I_bridge"].values + df["I_local"].values
    struct_z = (struct - struct.mean()) / (struct.std() + cfg.epsilon)
    df["I_hidden"] = struct_z - raw_fi_z
    for col in ["I_popularity", "I_authority", "I_hub", "I_local", "I_engagement"]:
        if col in df.columns:
            log.info(f"  {col}: mean={df[col].mean():.4f} max={df[col].max():.4f}")
    return df

def rank_influencers(df, cluster_result, cfg):
    log.info("=" * 60)
    log.info("STEP 13: Ranking influencers")
    log.info("=" * 60)
    k = cfg.top_k
    labels = cluster_result["labels"]
    role_labels = cluster_result["role_labels"]
    df["cluster_label"] = labels
    df["role"] = [role_labels.get(l, "Unknown") for l in labels]
    
    categories = {"popularity": "I_popularity", "authority": "I_authority",
                  "hub_broadcaster": "I_hub",
                  "local_leader": "I_local", "engagement": "I_engagement"}
    rel_feats = {"popularity": ["in_degree", "weighted_in_strength", "pagerank"],
                 "authority": ["pagerank", "hits_authority", "k_core_number"],
                 "hub_broadcaster": ["out_degree", "weighted_out_strength", "hits_hub"],
                 "local_leader": ["within_community_zscore", "community_size"],
                 "engagement": ["reciprocity", "weighted_in_strength"]}
    all_rankings = []
    for cat, scol in categories.items():
        top_idx = df[scol].nlargest(k).index
        top = df.loc[top_idx, ["original_node_id", "mapped_node_index",
                               "cluster_label", "role", "community_id", scol]].copy()
        top["category"] = cat
        top["rank"] = range(1, len(top) + 1)
        for f in rel_feats.get(cat, []):
            if f in df.columns:
                top[f] = df.loc[top_idx, f].values
        all_rankings.append(top)
        node0 = df.loc[top_idx[0]]
        fd = ", ".join(f"{f}={node0.get(f,0):.3f}" for f in rel_feats.get(cat,[]) if f in df.columns)
        log.info(f"  [{cat}] #1: Node {node0['original_node_id']} score={node0[scol]:.4f} {fd}")
    rankings_df = pd.concat(all_rankings, ignore_index=True)
    save_csv(rankings_df, os.path.join(cfg.output_dir, "influencer_rankings.csv"))
    return rankings_df

def compare_baselines(df, data, cfg):
    log.info("=" * 60)
    log.info("STEP 14: Baseline comparison")
    log.info("=" * 60)
    import networkx as nx
    n = data["n"]
    A, R, M, T = data["A_follow"], data["R_retweet"], data["M_mention"], data["T_reply"]
    bl_fi = np.asarray(A.sum(axis=0)).ravel()
    bl_ri = np.asarray(R.sum(axis=0)).ravel()
    bl_tot = bl_fi + bl_ri + np.asarray(M.sum(axis=0)).ravel() + np.asarray(T.sum(axis=0)).ravel()
    G_f = nx.DiGraph()
    G_f.add_nodes_from(range(n))
    A_coo = A.tocoo()
    for r, c in zip(A_coo.row, A_coo.col): G_f.add_edge(int(r), int(c))
    try: pr_f = nx.pagerank(G_f, max_iter=200)
    except: pr_f = {i: 1.0/n for i in range(n)}
    bl_pr = np.array([pr_f.get(i, 0) for i in range(n)])
    baselines = {"follow_indeg": bl_fi, "retweet_indeg": bl_ri,
                 "total_interaction": bl_tot, "pr_follow": bl_pr}
    our = {"I_popularity": df["I_popularity"].values, "I_authority": df["I_authority"].values,
           "I_hidden": df["I_hidden"].values}
    comps = {}
    for on, oa in our.items():
        for bn, ba in baselines.items():
            rho, pv = stats.spearmanr(oa, ba)
            for kk in [10, 20]:
                to = set(np.argsort(oa)[::-1][:kk])
                tb = set(np.argsort(ba)[::-1][:kk])
                ov = len(to & tb)
                jac = ov / len(to | tb) if len(to | tb) > 0 else 0
                key = f"{on}_vs_{bn}"
                if key not in comps: comps[key] = {"spearman": rho}
                comps[key][f"overlap@{kk}"] = ov
                comps[key][f"jaccard@{kk}"] = jac
            log.info(f"  {on} vs {bn}: rho={rho:.4f}")
    th = set(np.argsort(df["I_hidden"].values)[::-1][:20])
    tf = set(np.argsort(bl_fi)[::-1][:20])
    log.info(f"  Hidden vs follow overlap@20: {len(th & tf)}/20")
    return comps

def save_node_roles(df, pca_result, cluster_result, cfg):
    X_pca = pca_result["X_pca"]
    roles = df[["original_node_id", "mapped_node_index"]].copy()
    for i in range(min(3, X_pca.shape[1])):
        roles[f"PC{i+1}"] = X_pca[:, i]
    roles["cluster_label"] = cluster_result["labels"]
    roles["role"] = [cluster_result["role_labels"].get(l, "Unknown") for l in cluster_result["labels"]]
    for f in ["pagerank", "betweenness_centrality", "hits_authority", "hits_hub", "participation_coefficient"]:
        if f in df.columns: roles[f] = df[f].values
    for col in df.columns:
        if col.startswith("I_"): roles[col] = df[col].values
    save_csv(roles, os.path.join(cfg.output_dir, "node_roles.csv"))

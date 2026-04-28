"""
visualization.py
================
Step 15: All plots and visualizations for the BDA project.
"""
import os
from typing import Dict, List
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns
from scipy import sparse
from utils import Config, log, ensure_dir

sns.set_theme(style="whitegrid", font_scale=1.1)

def _savefig(fig, path):
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info(f"  Plot saved → {path}")

# ---------------------------------------------------------------
# 1. Edge score distribution
# ---------------------------------------------------------------
def plot_edge_score_distribution(Q, cfg):
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(Q[Q > 0], bins=80, color="#4C72B0", edgecolor="white", alpha=0.85)
    ax.set_xlabel("Combined Edge Score Q")
    ax.set_ylabel("Frequency")
    ax.set_title("Distribution of Positive Edge Scores")
    _savefig(fig, os.path.join(cfg.plots_dir, "edge_score_distribution.png"))

# ---------------------------------------------------------------
# 2. Degree distribution before/after filtering
# ---------------------------------------------------------------
def plot_degree_distribution(A_follow, B_sparse, cfg):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    # Before: follow graph
    deg_f = np.asarray(A_follow.sum(axis=0)).ravel()
    axes[0].hist(deg_f[deg_f > 0], bins=50, color="#DD8452", edgecolor="white", alpha=0.85)
    axes[0].set_xlabel("In-Degree")
    axes[0].set_ylabel("Count")
    axes[0].set_title("Follow Graph In-Degree")
    axes[0].set_yscale("log")
    # After: sparse graph
    deg_s = np.asarray(B_sparse.sum(axis=0)).ravel()
    axes[1].hist(deg_s[deg_s > 0], bins=50, color="#55A868", edgecolor="white", alpha=0.85)
    axes[1].set_xlabel("In-Degree")
    axes[1].set_ylabel("Count")
    axes[1].set_title("Sparse Graph In-Degree")
    axes[1].set_yscale("log")
    fig.suptitle("Degree Distribution: Before vs After Filtering", fontsize=14, y=1.02)
    fig.tight_layout()
    _savefig(fig, os.path.join(cfg.plots_dir, "degree_distribution.png"))

# ---------------------------------------------------------------
# 3. PCA 2D scatter colored by cluster
# ---------------------------------------------------------------
def plot_pca_2d(pca_result, cluster_result, cfg):
    X = pca_result["X_pca"]
    labels = cluster_result["labels"]
    evr = pca_result["explained_variance_ratio"]
    fig, ax = plt.subplots(figsize=(10, 8))
    k = cluster_result["best_k"]
    cmap = cm.get_cmap("tab10", k)
    for c in range(k):
        mask = labels == c
        role = cluster_result["role_labels"].get(c, f"Cluster {c}")
        ax.scatter(X[mask, 0], X[mask, 1], c=[cmap(c)], label=role,
                   alpha=0.6, s=15, edgecolors="none")
    ax.set_xlabel(f"PC1 ({evr[0]*100:.1f}% variance)")
    ax.set_ylabel(f"PC2 ({evr[1]*100:.1f}% variance)")
    ax.set_title("PCA 2D — Nodes Colored by Cluster Role")
    ax.legend(fontsize=9, markerscale=2.5)
    _savefig(fig, os.path.join(cfg.plots_dir, "pca_2d_scatter.png"))

# ---------------------------------------------------------------
# 4. PCA scatter sized by PageRank
# ---------------------------------------------------------------
def plot_pca_pagerank(pca_result, features, cfg):
    X = pca_result["X_pca"]
    pr = features["pagerank"].values
    pr_norm = (pr - pr.min()) / (pr.max() - pr.min() + 1e-9)
    sizes = 5 + pr_norm * 200
    fig, ax = plt.subplots(figsize=(10, 8))
    sc = ax.scatter(X[:, 0], X[:, 1], c=pr, cmap="YlOrRd", s=sizes,
                    alpha=0.6, edgecolors="grey", linewidth=0.3)
    fig.colorbar(sc, ax=ax, label="PageRank (standardized)")
    evr = pca_result["explained_variance_ratio"]
    ax.set_xlabel(f"PC1 ({evr[0]*100:.1f}%)")
    ax.set_ylabel(f"PC2 ({evr[1]*100:.1f}%)")
    ax.set_title("PCA 2D — Sized by PageRank")
    _savefig(fig, os.path.join(cfg.plots_dir, "pca_pagerank.png"))

# ---------------------------------------------------------------
# 5. Explained variance plot
# ---------------------------------------------------------------
def plot_explained_variance(pca_result, cfg):
    evr = pca_result["explained_variance_ratio"]
    cumvar = pca_result["cumulative_variance"]
    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(1, len(evr) + 1)
    ax.bar(x, evr, color="#4C72B0", alpha=0.7, label="Individual")
    ax.plot(x, cumvar, "o-", color="#C44E52", label="Cumulative")
    ax.axhline(0.9, ls="--", color="grey", alpha=0.5, label="90% threshold")
    ax.set_xlabel("Principal Component")
    ax.set_ylabel("Explained Variance Ratio")
    ax.set_title("PCA Explained Variance")
    ax.legend()
    _savefig(fig, os.path.join(cfg.plots_dir, "explained_variance.png"))

# ---------------------------------------------------------------
# 6. Community size distribution
# ---------------------------------------------------------------
def plot_community_sizes(features, cfg):
    if "community_id" not in features.columns:
        return
    sizes = features.groupby("community_id").size()
    fig, ax = plt.subplots(figsize=(10, 5))
    sizes.sort_values(ascending=False).plot(kind="bar", ax=ax, color="#55A868", edgecolor="white")
    ax.set_xlabel("Community ID")
    ax.set_ylabel("Size")
    ax.set_title("Community Size Distribution")
    _savefig(fig, os.path.join(cfg.plots_dir, "community_sizes.png"))

# ---------------------------------------------------------------
# 7. Bar chart of top influencers by category
# ---------------------------------------------------------------
def plot_top_influencers(rankings_df, cfg):
    # Map each category to its correct score column
    cat_to_score = {
        "popularity": "I_popularity",
        "authority": "I_authority",
        "hub_broadcaster": "I_hub",
        "bridge_broker": "I_bridge",
        "local_leader": "I_local",
        "engagement": "I_engagement",
        "hidden": "I_hidden",
    }
    categories = rankings_df["category"].unique()
    n_cats = len(categories)
    fig, axes = plt.subplots(n_cats, 1, figsize=(12, 3.5 * n_cats))
    if n_cats == 1:
        axes = [axes]
    colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3", "#937860", "#DA8BC3"]
    for idx, cat in enumerate(categories):
        sub = rankings_df[rankings_df["category"] == cat].head(10)
        score_col = cat_to_score.get(cat)
        if score_col is None or score_col not in sub.columns:
            # Fallback: find any I_ column that has non-null data
            candidates = [c for c in sub.columns if c.startswith("I_") and sub[c].notna().any()]
            score_col = candidates[0] if candidates else None
        if score_col is None:
            continue
        ax = axes[idx]
        labels = sub["original_node_id"].astype(str).values
        vals = sub[score_col].fillna(0).values
        color = colors[idx % len(colors)]
        ax.barh(labels[::-1], vals[::-1], color=color, edgecolor="white")
        ax.set_xlabel(score_col)
        ax.set_title(f"Top 10 — {cat.replace('_', ' ').title()}")
    fig.tight_layout()
    _savefig(fig, os.path.join(cfg.plots_dir, "top_influencers_bar.png"))

# ---------------------------------------------------------------
# 8. Cluster-average feature heatmap
# ---------------------------------------------------------------
def plot_cluster_heatmap(cluster_result, cfg):
    profiles = cluster_result["cluster_profiles"]
    roles = cluster_result["role_labels"]
    fig, ax = plt.subplots(figsize=(16, 6))
    data = profiles.values.astype(float)
    ylabels = [f"C{c}: {roles.get(c, '?')}" for c in range(len(profiles))]
    sns.heatmap(data, ax=ax, cmap="RdBu_r", center=0, annot=True, fmt=".2f",
                xticklabels=profiles.columns, yticklabels=ylabels, linewidths=0.5)
    ax.set_title("Cluster-Average Feature Heatmap")
    plt.xticks(rotation=45, ha="right", fontsize=8)
    _savefig(fig, os.path.join(cfg.plots_dir, "cluster_heatmap.png"))

# ---------------------------------------------------------------
# 9. Feature correlation matrix
# ---------------------------------------------------------------
def plot_feature_correlation(features, cfg):
    id_cols = {"original_node_id", "mapped_node_index", "community_id",
               "cluster_label", "role"}
    feat_cols = [c for c in features.columns if c not in id_cols and not c.startswith("I_")]
    corr = features[feat_cols].corr()
    fig, ax = plt.subplots(figsize=(14, 12))
    sns.heatmap(corr, ax=ax, cmap="coolwarm", center=0, annot=True, fmt=".2f",
                linewidths=0.3, square=True)
    ax.set_title("Node Feature Correlation Matrix")
    plt.xticks(rotation=45, ha="right", fontsize=8)
    plt.yticks(fontsize=8)
    _savefig(fig, os.path.join(cfg.plots_dir, "feature_correlation.png"))

# ---------------------------------------------------------------
# 10. Elbow + Silhouette plot
# ---------------------------------------------------------------
def plot_clustering_selection(cluster_result, cfg):
    results = cluster_result["all_results"]
    ks = sorted(results.keys())
    sils = [results[k]["silhouette"] for k in ks]
    inertias = [results[k]["inertia"] for k in ks]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    ax1.plot(ks, inertias, "o-", color="#4C72B0")
    ax1.set_xlabel("k")
    ax1.set_ylabel("Inertia")
    ax1.set_title("Elbow Method")
    ax2.plot(ks, sils, "o-", color="#55A868")
    ax2.set_xlabel("k")
    ax2.set_ylabel("Silhouette Score")
    ax2.set_title("Silhouette Score vs k")
    ax2.axvline(cluster_result["best_k"], ls="--", color="red", alpha=0.6)
    fig.tight_layout()
    _savefig(fig, os.path.join(cfg.plots_dir, "clustering_selection.png"))

# ---------------------------------------------------------------
# Master function
# ---------------------------------------------------------------
def generate_all_plots(data, features, pca_result, cluster_result, rankings_df, cfg):
    log.info("=" * 60)
    log.info("STEP 15: Generating visualizations")
    log.info("=" * 60)
    ensure_dir(cfg.plots_dir)
    plot_edge_score_distribution(data["Q_all"], cfg)
    plot_degree_distribution(data["A_follow"], data["B_sparse"], cfg)
    plot_pca_2d(pca_result, cluster_result, cfg)
    plot_pca_pagerank(pca_result, features, cfg)
    plot_explained_variance(pca_result, cfg)
    plot_community_sizes(features, cfg)
    plot_top_influencers(rankings_df, cfg)
    plot_cluster_heatmap(cluster_result, cfg)
    plot_feature_correlation(features, cfg)
    plot_clustering_selection(cluster_result, cfg)
    log.info("  All plots generated.")

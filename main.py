"""
main.py
=======
Orchestrator — runs all steps of the BDA social network analysis pipeline.

Usage:
    python main.py [--data-dir DATA_DIR] [--output-dir OUTPUT_DIR]
                   [--threshold-strategy {percentile,target_degree}]
                   [--percentile P] [--target-degree D]
"""
import argparse
import time
import os
import sys

from utils import Config, log, ensure_dir

def parse_args():
    p = argparse.ArgumentParser(description="BDA Social Network Influencer Analysis")
    p.add_argument("--data-dir", default="data")
    p.add_argument("--output-dir", default="outputs")
    p.add_argument("--threshold-strategy", default="percentile",
                   choices=["percentile", "target_degree"])
    p.add_argument("--percentile", type=float, default=5.0)
    p.add_argument("--target-degree", type=float, default=10.0)
    return p.parse_args()

def main():
    args = parse_args()
    cfg = Config(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        plots_dir=os.path.join(args.output_dir, "plots"),
        threshold_strategy=args.threshold_strategy,
        percentile=args.percentile,
        target_avg_degree=args.target_degree,
    )
    ensure_dir(cfg.output_dir)
    ensure_dir(cfg.plots_dir)

    t0 = time.time()
    log.info("=" * 70)
    log.info("   BDA PROJECT: Social Network Influencer Analysis")
    log.info("=" * 70)

    # ----- Steps 1-6: Graph construction -----
    from graph_construction import run_graph_construction
    data = run_graph_construction(cfg)

    # ----- Step 7: Graph validation -----
    from graph_validation import run_graph_validation
    val_result = run_graph_validation(data, cfg)

    # ----- Steps 8-9: Feature engineering -----
    from feature_engineering import compute_node_features
    features, partition = compute_node_features(data, cfg)

    # ----- Step 10: PCA -----
    from influence_scores import run_pca, run_clustering, compute_influence_scores
    from influence_scores import rank_influencers, compare_baselines, save_node_roles
    pca_result = run_pca(features, cfg)

    # ----- Step 11: Clustering -----
    cluster_result = run_clustering(features, pca_result, cfg)

    # ----- Step 12: Influence scores -----
    features = compute_influence_scores(features, data, cfg)

    # ----- Step 13: Rankings -----
    rankings_df = rank_influencers(features, cluster_result, cfg)

    # ----- Step 14: Baseline comparison -----
    comparisons = compare_baselines(features, data, cfg)

    # ----- Save node roles -----
    save_node_roles(features, pca_result, cluster_result, cfg)

    # ----- Step 15: Visualizations -----
    from visualization import generate_all_plots
    generate_all_plots(data, features, pca_result, cluster_result, rankings_df, cfg)

    elapsed = time.time() - t0
    log.info("=" * 70)
    log.info(f"   PIPELINE COMPLETE — {elapsed:.1f}s")
    log.info(f"   Outputs: {cfg.output_dir}/")
    log.info("=" * 70)

if __name__ == "__main__":
    main()

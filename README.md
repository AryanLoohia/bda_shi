# Social Network Influencer Analysis

**A Big Data Analysis project that constructs a statistically-filtered directed weighted social graph and discovers seven distinct types of influencers using graph mining, PCA, and clustering.**

Built on the [Higgs Twitter Dataset](https://snap.stanford.edu/data/higgs-twitter.html) — activity around the Higgs boson discovery announcement on Twitter (July 2012).

---

## Table of Contents

- [Motivation](#motivation)
- [Pipeline Overview](#pipeline-overview)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Dataset](#dataset)
- [Methodology](#methodology)
  - [Graph Construction (Steps 1–6)](#graph-construction-steps-16)
  - [Graph Validation (Step 7)](#graph-validation-step-7)
  - [Feature Engineering (Steps 8–9)](#feature-engineering-steps-89)
  - [PCA & Clustering (Steps 10–11)](#pca--clustering-steps-1011)
  - [Influence Scoring & Ranking (Steps 12–14)](#influence-scoring--ranking-steps-1214)
  - [Visualizations (Step 15)](#visualizations-step-15)
- [Key Results](#key-results)
- [Output Files](#output-files)
- [Configuration](#configuration)
- [Dependencies](#dependencies)

---

## Motivation

Naive follow-based graphs are **dense and noisy** — a user may follow thousands of accounts without ever interacting with them. Ranking influencers by follower count alone misses structurally important nodes: brokers who bridge communities, niche authorities, or engagement-driven conversationalists.

This project goes beyond raw degree by:

1. **Statistically filtering** edges using a null model — keeping only interactions that are *surprisingly strong* given both parties' activity levels
2. Computing **17 structural node features** (degree, PageRank, HITS, betweenness, community membership, etc.)
3. Discovering **7 distinct types of influence** — not just "who has the most followers"

---

## Pipeline Overview

```
                        ┌──────────────────────────┐
                        │  4 Input Edgelists        │
                        │  (follow/retweet/         │
                        │   mention/reply)          │
                        └──────────┬───────────────┘
                                   ▼
                ┌──────────────────────────────────┐
                │  STEP 1: Load & Subsample (20k)  │
                │  STEP 2: Null Model E[W_ij]      │
                │  STEP 3: Significance Scores     │
                │  STEP 4: Z-Standardize           │
                │  STEP 5: Weighted Combination     │
                │  STEP 6: Threshold → Sparse Graph │
                └──────────┬───────────────────────┘
                           ▼
                ┌──────────────────────────────────┐
                │  STEP 7: Graph Validation        │
                │  STEPS 8–9: 17 Node Features     │
                │  STEP 10: PCA                    │
                │  STEP 11: KMeans Clustering      │
                │  STEPS 12–13: 7 Influence Scores │
                │  STEP 14: Baseline Comparison    │
                │  STEP 15: 10 Visualizations      │
                └──────────┬───────────────────────┘
                           ▼
                ┌──────────────────────────────────┐
                │  Outputs:                        │
                │  • influencer_rankings.csv       │
                │  • node_roles.csv                │
                │  • 10 plots in outputs/plots/    │
                └──────────────────────────────────┘
```

---

## Project Structure

```
BDA_project/
│
├── data/                              # Input edgelists
│   ├── higgs-social_network.edgelist  # Follow/social connections
│   ├── higgs-retweet_network.edgelist # Retweet interactions
│   ├── higgs-mention_network.edgelist # Mention interactions
│   └── higgs-reply_network.edgelist   # Reply interactions
│
├── outputs/                           # All generated results
│   ├── sparse_edges.csv               # Filtered graph edge list
│   ├── graph_summary.csv              # Graph statistics
│   ├── node_features.csv              # 22-column feature matrix (standardized)
│   ├── node_features_raw.csv          # Raw features before transforms
│   ├── node_roles.csv                 # PCA coords + cluster + role + scores
│   ├── influencer_rankings.csv        # Top-20 per 7 influencer categories
│   └── plots/                         # 10 publication-quality plots
│       ├── edge_score_distribution.png
│       ├── degree_distribution.png
│       ├── pca_2d_scatter.png
│       ├── pca_pagerank.png
│       ├── explained_variance.png
│       ├── community_sizes.png
│       ├── top_influencers_bar.png
│       ├── cluster_heatmap.png
│       ├── feature_correlation.png
│       └── clustering_selection.png
│
├── main.py                  # Orchestrator — runs all steps
├── graph_construction.py    # Steps 1–6: Load, null model, scoring, graph build
├── graph_validation.py      # Step 7: Validation & comparison
├── feature_engineering.py   # Steps 8–9: Node feature computation
├── influence_scores.py      # Steps 10–14: PCA, clustering, scoring, ranking
├── visualization.py         # Step 15: All plots
├── utils.py                 # Config, logging, I/O helpers
├── generate_synthetic_data.py  # Generate test data (500 nodes)
├── report_summary.md        # Mathematical methodology report
└── README.md                # This file
```

---

## Installation

```bash
# Clone / navigate to the project
cd BDA_project

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install pandas numpy scipy networkx scikit-learn matplotlib seaborn python-louvain
```

---

## Usage

### Run the full pipeline

```bash
source .venv/bin/activate
python main.py
```

### Command-line options

```bash
# Use target average degree instead of percentile thresholding
python main.py --threshold-strategy target_degree --target-degree 10

# Change percentile (keep top 10% instead of default 5%)
python main.py --percentile 10

# Custom data/output directories
python main.py --data-dir data/ --output-dir outputs/
```

### Run with synthetic test data (no Higgs files needed)

```bash
python generate_synthetic_data.py   # Creates 500-node test data in data/
python main.py                       # Run pipeline
```

---

## Dataset

This project uses the **Higgs Twitter Dataset** from the Stanford SNAP collection.

| File | Edges | Description |
|------|-------|-------------|
| `higgs-social_network.edgelist` | 14.8M | Who follows whom (2 columns: source target) |
| `higgs-retweet_network.edgelist` | 328k | Who retweeted whom (3 columns: source target value) |
| `higgs-mention_network.edgelist` | 151k | Who mentioned whom |
| `higgs-reply_network.edgelist` | 33k | Who replied to whom |

**Node subsampling**: The full dataset has ~456k nodes. We select the **top 20,000 most active nodes** (ranked by total appearances as source or target across all four files) and keep only edges within this subset. This is not random — it preserves the most socially relevant portion of the graph.

---

## Methodology

### Graph Construction (Steps 1–6)

1. **Load** four edgelist files and **subsample** to 20k most active nodes
2. **Null model**: For each relation W, compute expected interaction:
   ```
   E[W_ij] = (out_i × in_j) / m
   ```
   where `out_i` = total outgoing of node i, `in_j` = total incoming of node j, `m` = total sum
3. **Significance score** per edge per relation:
   ```
   S_ij = (W_ij − E[W_ij]) / √(E[W_ij] + ε)
   ```
4. **Z-standardize** each relation's scores independently
5. **Combine** into one edge score:
   ```
   Q_ij = 0.20·Z_follow + 0.35·Z_retweet + 0.20·Z_mention + 0.25·Z_reply
   ```
6. **Threshold**: Keep top 5% of positive-score edges → sparse backbone graph

### Graph Validation (Step 7)

Compute nodes, edges, density, connected components, reciprocity, degree distribution. Compare sparse graph with raw follow graph (density reduction: ~67×).

### Feature Engineering (Steps 8–9)

17 features per node in three groups:

| # | Feature | Category |
|---|---------|----------|
| 1–4 | In/out degree, weighted in/out strength | Intrinsic |
| 5 | PageRank (weighted) | Intrinsic |
| 6–7 | HITS authority, HITS hub | Intrinsic |
| 8 | Betweenness centrality (approx, k=500) | Intrinsic |
| 9 | k-core number | Intrinsic |
| 10 | Node reciprocity | Intrinsic |
| 11 | Local clustering coefficient | Intrinsic |
| 12–13 | 2-hop reach, 3-hop reach | Neighborhood |
| 14 | Top-5 neighborhood degree profile | Neighborhood |
| 15 | Community size (Louvain) | Community |
| 16 | Within-community z-score | Community |
| 17 | Participation coefficient | Community |

Post-processing: `log1p` transform on heavy-tailed features → `StandardScaler` normalization.

### PCA & Clustering (Steps 10–11)

- **PCA**: 9 components needed for 90% explained variance (from 21 features). PC1 (45%) captures spreading capacity; PC2 (13%) captures popularity/incoming strength; PC3 (9%) captures HITS authority/hub structure.
- **KMeans**: Tested k=3..8, selected by **silhouette score**. Clustering runs on **full 21D standardized features** (not on 2D PCA — the 2D scatter is for visualization only).

### Influence Scoring & Ranking (Steps 12–14)

Seven influence scores computed from standardized features:

| Score | Formula | Captures |
|-------|---------|----------|
| **Popularity** | z(in_deg) + z(in_strength) + z(pagerank) | Who is widely followed/retweeted |
| **Authority** | z(pagerank) + z(hits_auth) + z(k_core) | Recognized expert in dense core |
| **Hub/Broadcaster** | z(out_deg) + z(out_strength) + z(hits_hub) + z(3hop_reach) | Amplifies and spreads information |
| **Bridge/Broker** | z(betweenness) + z(participation) − z(clustering) | Connects different communities |
| **Local Leader** | z(within_comm_z) + z(in_strength) + z(comm_size) | Dominant inside own group |
| **Engagement** | z(reciprocity) + z(in_strength) + z(out_strength) | Active in two-way conversations |
| **Hidden** | z(structural_importance) − z(raw_follow_indegree) | Important but not obviously popular |

Top-20 nodes ranked per category. Compared against naive baselines (follow in-degree, PageRank on follow graph) using Spearman correlation and overlap@k.

### Visualizations (Step 15)

10 plots generated in `outputs/plots/`:
- Edge score distribution (validates heavy-tailed filtering)
- Degree distribution before/after filtering
- PCA 2D scatter (colored by cluster)
- PCA scatter sized by PageRank
- Explained variance scree plot
- Community size distribution
- Top-10 influencers bar chart (7 categories)
- Cluster-average feature heatmap
- Feature correlation matrix
- Clustering selection (elbow + silhouette)

---

## Key Results

*On the Higgs dataset (19,998 nodes, 93.5s runtime):*

| Metric | Value |
|--------|-------|
| Candidate edges | 2,185,262 |
| Edges after filtering (top 5%) | 32,206 |
| Density reduction vs follow graph | **67×** |
| Largest weakly connected component | 77% of nodes |
| Reciprocity | 14.6% |
| PCA components for 90% variance | 9 (out of 21) |
| Best k (silhouette) | 3 |
| **Hidden vs follow overlap@20** | **0/20** |

The last row is the key finding: **none of the top-20 hidden influencers appear in the top-20 by follower count**, proving that raw popularity metrics miss structurally important nodes.

---

## Output Files

| File | Description |
|------|-------------|
| `sparse_edges.csv` | Final filtered edge list (source, target, score) |
| `graph_summary.csv` | Graph statistics (nodes, edges, density, components, reciprocity) |
| `node_features.csv` | Standardized 22-column feature matrix for all nodes |
| `node_features_raw.csv` | Raw features before log1p/standardization |
| `node_roles.csv` | PCA coordinates + cluster label + role + all influence scores |
| `influencer_rankings.csv` | Top-20 nodes per 7 influencer categories with feature values |

---

## Configuration

All parameters are centralized in `utils.py → Config`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `max_nodes` | 20,000 | Subsample to top-k most active nodes |
| `alpha_follow` | 0.20 | Weight for follow score |
| `beta_retweet` | 0.35 | Weight for retweet score (highest — strongest signal) |
| `gamma_mention` | 0.20 | Weight for mention score |
| `delta_reply` | 0.25 | Weight for reply score |
| `threshold_strategy` | "percentile" | "percentile" or "target_degree" |
| `percentile` | 5.0 | Keep top p% of edges |
| `betweenness_k` | 500 | Sample size for approximate betweenness |
| `cluster_range` | (3,4,5,6,7,8) | KMeans k values to try |
| `top_k` | 20 | Number of top influencers per category |

---

## Dependencies

```
pandas
numpy
scipy
networkx
scikit-learn
matplotlib
seaborn
python-louvain    # Louvain community detection (optional — falls back to greedy modularity)
```

Python 3.9+ recommended.

---

## References

- De Domenico, M., Lima, A., Mougel, P., & Musolesi, M. (2013). *The Anatomy of a Scientific Rumor.* Scientific Reports, 3, 2980.
- Higgs Twitter Dataset: https://snap.stanford.edu/data/higgs-twitter.html
- Serrano, M. Á., Boguñá, M., & Vespignani, A. (2009). *Extracting the multiscale backbone of complex weighted networks.* PNAS, 106(16), 6483–6488.

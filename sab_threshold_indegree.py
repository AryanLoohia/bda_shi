import collections
import math
import os
import numpy as np

# --- 1. Data Loading & Parsing Functions ---

def parse_node_dict(filename):
    """Parses interaction dictionaries from text files."""
    data = collections.defaultdict(lambda: collections.defaultdict(int))
    total_sum = 0
    try:
        with open(filename, "r") as f:
            for line in f:
                if not line.strip() or ":" not in line:
                    continue
                parts = line.split(":")
                u = int(parts[0].strip())
                interactions = parts[1].strip().split("}")
                for item in interactions:
                    if "{" in item:
                        pair = item.split("{")[1].split(",")
                        v = int(pair[0])
                        weight = int(pair[1])
                        data[u][v] = weight
                        total_sum += weight
    except FileNotFoundError:
        print(f"File {filename} not found.")
    return data, total_sum


def parse_adjlist(filename):
    """Parses the basic follow graph adjacency list."""
    follows = collections.defaultdict(set)
    try:
        with open(filename, "r") as f:
            for line in f:
                if ":" not in line:
                    continue
                parts = line.split(":")
                u = int(parts[0].strip())
                targets = parts[1].strip().split()
                for v in targets:
                    follows[u].add(int(v))
    except FileNotFoundError:
        print(f"File {filename} not found.")
    return follows


# --- 2. Statistical Functions ---

def calculate_stats(matrix):
    """Computes row and column sums for Z-score expectations."""
    row_sums = collections.defaultdict(int)
    col_sums = collections.defaultdict(int)
    for u, targets in matrix.items():
        for v, weight in targets.items():
            row_sums[u] += weight
            col_sums[v] += weight
    return row_sums, col_sums


def get_z_score(u, v, matrix, row_sums, col_sums, total_all, epsilon=1e-6):
    """Calculates the Z-score for an interaction between u and v."""
    actual = matrix[u].get(v, 0)
    expected = (row_sums[u] * col_sums[v]) / total_all if total_all > 0 else 0
    return (actual - expected) / math.sqrt(expected + epsilon)


# --- 3. Main Execution ---

def main():
    output_dir = "network_results"
    os.makedirs(output_dir, exist_ok=True)

    # Load Data
    print("Loading files...")
    follows_graph = parse_adjlist("one.adjlist")
    retweet_dict, r_total = parse_node_dict("one.retweet.node_dict.txt")
    mention_dict, m_total = parse_node_dict("one.metion.node_dict.txt")
    reply_dict, p_total = parse_node_dict("one.reply.node_dict.txt")

    # Precompute Marginals
    print("Computing interaction marginals...")
    r_a, r_b = calculate_stats(retweet_dict)
    m_a, m_b = calculate_stats(mention_dict)
    p_a, p_b = calculate_stats(reply_dict)

    # Pass 1: Collect raw Z-scores
    print("Pass 1: Collecting raw z-scores...")
    raw_scores = {"r": [], "m": [], "p": []}
    edge_coords = []

    for a, targets in follows_graph.items():
        for b in targets:
            zr = get_z_score(a, b, retweet_dict, r_a, r_b, r_total)
            zm = get_z_score(a, b, mention_dict, m_a, m_b, m_total)
            zp = get_z_score(a, b, reply_dict, p_a, p_b, p_total)
            raw_scores["r"].append(zr)
            raw_scores["m"].append(zm)
            raw_scores["p"].append(zp)
            edge_coords.append((a, b, zr, zm, zp))

    # Calculate Normalization Parameters (Mean and StdDev)
    stats = {k: (np.mean(v), np.std(v) + 1e-9) for k, v in raw_scores.items()}

    # Pass 2: Build Normalized S_ab
    print("Pass 2: Building S_ab for all edges...")
    edge_scores = []
    all_s_scores = []
    for a, b, zr, zm, zp in edge_coords:
        # Standardize each component to Mu=0, Sigma=1
        norm_r = (zr - stats["r"][0]) / stats["r"][1]
        norm_m = (zm - stats["m"][0]) / stats["m"][1]
        norm_p = (zp - stats["p"][0]) / stats["p"][1]
        
        # Weighted sum: 4x priority for Replies
        s_ab = (4 * norm_p) + (1 * norm_m) + (1 * norm_r)
        
        edge_scores.append((a, b, s_ab))
        all_s_scores.append(s_ab)

    # Determine Threshold for Top 50%
    # Using 50th percentile (median) guarantees a 50/50 split of the data
    threshold = float(np.percentile(all_s_scores, 50)) if all_s_scores else 0.0
    print(f"Top 50% S_ab threshold (Median): {threshold:.9f}")

    # Filtering and Indegree Calculation
    print("Filtering edges and computing indegree...")
    filtered_edges = []
    indegree = collections.defaultdict(int)
    
    # Track all unique nodes for complete output
    all_nodes = set(follows_graph.keys())
    for targets in follows_graph.values():
        all_nodes.update(targets)

    for a, b, s_ab in edge_scores:
        if s_ab >= threshold:
            filtered_edges.append((a, b, s_ab))
            indegree[b] += 1

    # Initialize nodes with 0 indegree if they weren't targets of filtered edges
    for node in all_nodes:
        indegree[node] += 0

    # Save Results
    threshold_path = os.path.join(output_dir, "sab_median_threshold.txt")
    scores_path = os.path.join(output_dir, "sab_scores_all_edges.txt")
    filtered_path = os.path.join(output_dir, "sab_edges_filtered.txt")
    indegree_path = os.path.join(output_dir, "sab_filtered_indegrees.txt")

    with open(threshold_path, "w") as f:
        f.write(f"{threshold:.12f}\n")

    with open(scores_path, "w") as f:
        f.write("source target sab\n")
        for a, b, s_ab in edge_scores:
            f.write(f"{a} {b} {s_ab:.12f}\n")

    with open(filtered_path, "w") as f:
        f.write("source target sab\n")
        for a, b, s_ab in filtered_edges:
            f.write(f"{a} {b} {s_ab:.12f}\n")

    with open(indegree_path, "w") as f:
        for node in sorted(indegree):
            f.write(f"{node}: {indegree[node]}\n")

    # Summary Output
    print("-" * 30)
    print(f"Wrote results to: {output_dir}")
    print(f"Total edges processed: {len(edge_scores)}")
    print(f"Edges in Top 50%:      {len(filtered_edges)}")
    print(f"Total unique nodes:    {len(indegree)}")
    print("-" * 30)


if __name__ == "__main__":
    main()
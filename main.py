import collections
import math
import random
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def plot_interaction_distributions(raw_z_scores, all_s_scores):
    """
    Plots the distributions of normalized interaction scores and the net Sab.
    """
    sns.set_style("whitegrid")
    plt.figure(figsize=(12, 8))

    # 1. Plot individual Normalized Z-scores
    # These should overlap significantly around 0
    sns.kdeplot(raw_z_scores['r'], label='Normalized Retweets ($n_r$)', fill=True, alpha=0.2)
    sns.kdeplot(raw_z_scores['m'], label='Normalized Mentions ($n_m$)', fill=True, alpha=0.2)
    sns.kdeplot(raw_z_scores['p'], label='Normalized Replies ($n_p$)', fill=True, alpha=0.2)

    # 2. Plot the Net Sab Score
    # This will be much wider because it's a weighted sum (4np + 1nm + 1nr)
    sns.kdeplot(all_s_scores, label='Net $S_{ab}$ Score', color='black', linewidth=3, linestyle='--')

    plt.axvline(0, color='red', linestyle=':', label='Mean (0.0)')
    plt.title("Distribution of Normalized Interaction Scores & Net $S_{ab}$", fontsize=15)
    plt.xlabel("Standard Deviations from Mean ($\sigma$)", fontsize=12)
    plt.ylabel("Density", fontsize=12)
    plt.legend()
    plt.show()

    
# --- 1. Data Loading & Parsing Functions ---

def parse_node_dict(filename):
    data = collections.defaultdict(lambda: collections.defaultdict(int))
    total_sum = 0
    try:
        with open(filename, 'r') as f:
            for line in f:
                if not line.strip() or ':' not in line: continue
                parts = line.split(':')
                u = int(parts[0].strip())
                interactions = parts[1].strip().split('}')
                for item in interactions:
                    if '{' in item:
                        pair = item.split('{')[1].split(',')
                        v = int(pair[0])
                        weight = int(pair[1])
                        data[u][v] = weight
                        total_sum += weight
    except FileNotFoundError:
        print(f"File {filename} not found.")
    return data, total_sum

def parse_adjlist(filename):
    follows = collections.defaultdict(set)
    try:
        with open(filename, 'r') as f:
            for line in f:
                if ':' not in line: continue
                parts = line.split(':')
                u = int(parts[0].strip())
                targets = parts[1].strip().split()
                for v in targets:
                    follows[u].add(int(v))
    except FileNotFoundError:
        print(f"File {filename} not found.")
    return follows

# --- 2. Statistical Functions ---

def calculate_stats(matrix):
    row_sums = collections.defaultdict(int)
    col_sums = collections.defaultdict(int)
    for u, targets in matrix.items():
        for v, weight in targets.items():
            row_sums[u] += weight
            col_sums[v] += weight
    return row_sums, col_sums

def get_z_score(u, v, matrix, row_sums, col_sums, total_all, epsilon=1e-6):
    actual = matrix[u].get(v, 0)
    expected = (row_sums[u] * col_sums[v]) / total_all if total_all > 0 else 0
    return (actual - expected) / math.sqrt(expected + epsilon)

# --- 3. K-Hop BFS Function ---

def get_k_hop_neighbors(start_node, edge_weights, threshold, k_depth):
    visited = {start_node}
    queue = collections.deque([(start_node, 0)]) 
    reachable_nodes = []

    while queue:
        u, dist = queue.popleft()
        if dist >= k_depth:
            continue
        if u in edge_weights:
            for v, s_weight in edge_weights[u].items():
                if s_weight > threshold and v not in visited:
                    visited.add(v)
                    reachable_nodes.append((v, dist + 1, s_weight))
                    queue.append((v, dist + 1))
    return reachable_nodes

# --- 4. Main Execution ---

def main():
    # Parameters
    N_SAMPLES = 1000   
    K_HOPS = 7     
    OUTPUT_DIR = "network_results"
    OUTPUT_FILE = os.path.join(OUTPUT_DIR, "khop_bfs_clusters.txt")

    # Load Data
    print("Loading files...")
    retweet_dict, R_total = parse_node_dict('one.retweet.node_dict.txt')
    mention_dict, M_total = parse_node_dict('one.metion.node_dict.txt')
    reply_dict, P_total = parse_node_dict('one.reply.node_dict.txt')
    follows_graph = parse_adjlist('one.adjlist')

    # Precompute base stats
    R_a, R_b = calculate_stats(retweet_dict)
    M_a, M_b = calculate_stats(mention_dict)
    P_a, P_b = calculate_stats(reply_dict)

    # --- NEW: PASS 1 - COLLECT RAW SCORES FOR NORMALIZATION ---
    print("Pass 1: Collecting raw Z-scores for normalization...")
    raw_z_scores = {'p': [], 'm': [], 'r': []}
    edge_data = [] # To store coords and avoid triple-looping later

    for a, followed_list in follows_graph.items():
        for b in followed_list:
            z_p = get_z_score(a, b, reply_dict, P_a, P_b, P_total)
            z_m = get_z_score(a, b, mention_dict, M_a, M_b, M_total)
            z_r = get_z_score(a, b, retweet_dict, R_a, R_b, R_total)
            
            raw_z_scores['p'].append(z_p)
            raw_z_scores['m'].append(z_m)
            raw_z_scores['r'].append(z_r)
            edge_data.append((a, b, z_p, z_m, z_r))

    # Calculate Global Means and StdDevs
    # Adding a tiny epsilon (1e-9) to std to avoid division by zero
    norm_stats = {
        'p': (np.mean(raw_z_scores['p']), np.std(raw_z_scores['p']) + 1e-9),
        'm': (np.mean(raw_z_scores['m']), np.std(raw_z_scores['m']) + 1e-9),
        'r': (np.mean(raw_z_scores['r']), np.std(raw_z_scores['r']) + 1e-9)
    }

    print(f"Normalization Stats:")
    print(f"  Reply   - μ: {norm_stats['p'][0]:.4f}, σ: {norm_stats['p'][1]:.4f}")
    print(f"  Mention - μ: {norm_stats['m'][0]:.4f}, σ: {norm_stats['m'][1]:.4f}")
    print(f"  Retweet - μ: {norm_stats['r'][0]:.4f}, σ: {norm_stats['r'][1]:.4f}")

    # --- NEW: PASS 2 - CALCULATE NORMALIZED S_ab ---
    print("Pass 2: Calculating normalized S_ab weights and threshold...")
    edge_weights = collections.defaultdict(dict)
    all_s_scores = []

    for a, b, zp, zm, zr in edge_data:
        # Standardize: (Value - Mean) / StdDev
        n_p = (zp - norm_stats['p'][0]) / norm_stats['p'][1]
        n_m = (zm - norm_stats['m'][0]) / norm_stats['m'][1]
        n_r = (zr - norm_stats['r'][0]) / norm_stats['r'][1]
        
        # Apply Weights to Normalized values
        s_ab = (4 * n_p) + (1 * n_m) + (1 * n_r)
        
        edge_weights[a][b] = s_ab
        all_s_scores.append(s_ab)

    # Dynamic Threshold
    # THRESHOLD = np.mean(all_s_scores) if all_s_scores else 0.0
    THRESHOLD = np.percentile(all_s_scores, 75)
    print(f"Calculated Global Normalized Mean Threshold: {THRESHOLD:.4f}")

    plot_interaction_distributions(raw_z_scores, all_s_scores)
    # --- (Rest of the BFS and file writing logic remains the same) ---
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    available_nodes = list(follows_graph.keys())
    sampled_nodes = random.sample(available_nodes, min(N_SAMPLES, len(available_nodes)))

    print(f"Performing {K_HOPS}-hop BFS...")
    with open(OUTPUT_FILE, 'w') as f:
        f.write(f"K-Hop BFS Reachability Report (K={K_HOPS})\n")
        f.write(f"Normalized Threshold S_ab > {THRESHOLD:.9f}\n")
        f.write("=" * 80 + "\n")
        f.write(f"{'Source':<10} | {'Nodes Found':<12}\n")
        f.write("-" * 80 + "\n")

        for node in sampled_nodes:
            cluster = get_k_hop_neighbors(node, edge_weights, THRESHOLD, K_HOPS)
            f.write(f"{node:<10} | {len(cluster):<12}\n")

    print(f"Complete! Results saved in: {OUTPUT_FILE}")



if __name__ == "__main__":
    main()
import collections
import math
import random
import os

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
    N_SAMPLES = 40    
    K_HOPS = 7     
    OUTPUT_DIR = "network_results"
    OUTPUT_FILE = os.path.join(OUTPUT_DIR, "khop_bfs_clusters.txt")

    # Load Data
    print("Loading files...")
    retweet_dict, R_total = parse_node_dict('one.retweet.node_dict.txt')
    mention_dict, M_total = parse_node_dict('one.metion.node_dict.txt')
    reply_dict, P_total = parse_node_dict('one.reply.node_dict.txt')
    follows_graph = parse_adjlist('one.adjlist')

    # Precompute stats
    R_a, R_b = calculate_stats(retweet_dict)
    M_a, M_b = calculate_stats(mention_dict)
    P_a, P_b = calculate_stats(reply_dict)

    # Calculate S_ab for every FOLLOWS edge
    print("Calculating edge weights and threshold...")
    edge_weights = collections.defaultdict(dict)
    all_s_scores = [] # Collector for global mean calculation

    for a, followed_list in follows_graph.items():
        for b in followed_list:
            z_r = get_z_score(a, b, retweet_dict, R_a, R_b, R_total)
            z_m = get_z_score(a, b, mention_dict, M_a, M_b, M_total)
            z_p = get_z_score(a, b, reply_dict, P_a, P_b, P_total)
            
            s_ab = (4 * z_p) + (2 * z_m) + z_r
            edge_weights[a][b] = s_ab
            all_s_scores.append(s_ab)

    # --- DYNAMIC THRESHOLD INTEGRATION ---
    if all_s_scores:
        THRESHOLD = sum(all_s_scores) / len(all_s_scores)
    else:
        THRESHOLD = 0.0 # Fallback
    
    # THRESHOLD =200
    print(f"Calculated Global Mean Threshold: {THRESHOLD:.4f}")

    # Create directory
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    # Select samples
    available_nodes = list(follows_graph.keys())
    sampled_nodes = random.sample(available_nodes, min(N_SAMPLES, len(available_nodes)))

    # Process and Write to file
    print(f"Performing {K_HOPS}-hop BFS for {len(sampled_nodes)} nodes...")
    with open(OUTPUT_FILE, 'w') as f:
        f.write(f"K-Hop BFS Reachability Report (K={K_HOPS})\n")
        f.write(f"Dynamic Threshold (Mean) S_ab > {THRESHOLD:.4f}\n")
        f.write("=" * 80 + "\n")
        f.write(f"{'Source':<10} | {'Nodes Found':<12} | {'Reached Nodes (ID@Hop)'}\n")
        f.write("-" * 80 + "\n")

        for node in sampled_nodes:
            cluster = get_k_hop_neighbors(node, edge_weights, THRESHOLD, K_HOPS)
            cluster_str = ", ".join([f"{n}@{h}" for n, h, w in cluster])
            f.write(f"{node:<10} | {len(cluster):<12} | {cluster_str}\n")

    print(f"Complete! Results saved in: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
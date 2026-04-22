import collections
import math
import random
import os
import numpy as np
from sklearn.decomposition import PCA

# --- 1. Data Loading & Parsing ---

def load_existing_indegrees(filepath):
    """Parses the previously saved indegree file."""
    indegree = {}
    try:
        with open(filepath, 'r') as f:
            for line in f:
                if ':' not in line: continue
                node, val = line.split(':')
                indegree[int(node.strip())] = int(val.strip())
        print(f"Successfully loaded {len(indegree)} nodes from stored indegrees.")
    except Exception as e:
        print(f"Error loading indegrees: {e}")
        return None
    return indegree

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
    """Still needed to get the graph structure for BFS."""
    follows = collections.defaultdict(set)
    try:
        with open(filename, 'r') as f:
            for line in f:
                if ':' not in line: continue
                parts = line.split(':')
                u = int(parts[0].strip())
                targets = [int(v) for v in parts[1].strip().split()]
                follows[u] = set(targets)
    except FileNotFoundError:
        print(f"File {filename} not found.")
    return follows

# --- 2. Statistical Functions (Z-Score Logic) ---

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

# --- 3. K-Hop BFS ---

def get_k_hop_neighbors(start_node, edge_weights, threshold, k_depth):
    visited = {start_node}
    queue = collections.deque([(start_node, 0)]) 
    reachable_nodes = []
    while queue:
        u, dist = queue.popleft()
        if dist >= k_depth: continue
        if u in edge_weights:
            for v, s_weight in edge_weights[u].items():
                if s_weight > threshold and v not in visited:
                    visited.add(v)
                    reachable_nodes.append(v)
                    queue.append((v, dist + 1))
    return reachable_nodes

# --- 4. Main Execution ---

def main():
    N_SAMPLES = 20000   
    K_HOPS = 4
    TOP_K = 20         
    PCA_DIR = "pca_results"
    INDEGREE_FILE = os.path.join(PCA_DIR, "global_indegrees.txt")

    # Check for existing indegrees first
    if os.path.exists(INDEGREE_FILE):
        indegrees = load_existing_indegrees(INDEGREE_FILE)
    else:
        print("Error: Indegree file not found. Please run the full calculation once first.")
        return

    # Load Graph & Interaction Data
    print("Loading graph structure and interactions...")
    follows_graph = parse_adjlist('one.adjlist')
    retweet_dict, R_total = parse_node_dict('one.retweet.node_dict.txt')
    mention_dict, M_total = parse_node_dict('one.metion.node_dict.txt')
    reply_dict, P_total = parse_node_dict('one.reply.node_dict.txt')

    R_a, R_b = calculate_stats(retweet_dict)
    M_a, M_b = calculate_stats(mention_dict)
    P_a, P_b = calculate_stats(reply_dict)

    print("Calculating S_ab weights...")
    edge_weights = collections.defaultdict(dict)
    all_s_scores = [] 

    for a, targets in follows_graph.items():
        for b in targets:
            z_r = get_z_score(a, b, retweet_dict, R_a, R_b, R_total)
            z_m = get_z_score(a, b, mention_dict, M_a, M_b, M_total)
            z_p = get_z_score(a, b, reply_dict, P_a, P_b, P_total)
            
            s_ab = (4 * z_p) + (1 * z_m) + (1 * z_r)
            edge_weights[a][b] = s_ab
            all_s_scores.append(s_ab)

    THRESHOLD = sum(all_s_scores) / len(all_s_scores) if all_s_scores else 0
    
    # Process Matrix
    all_nodes = list(follows_graph.keys())
    sampled_nodes = random.sample(all_nodes, min(N_SAMPLES, len(all_nodes)))
    data_matrix = []
    valid_ids = []

    print("Generating neighbor vectors...")
    for node in sampled_nodes:
        cluster = get_k_hop_neighbors(node, edge_weights, THRESHOLD, K_HOPS)
        if not cluster: continue

        # Pull from the REUSED indegrees dictionary
        ind_vec = sorted([indegrees.get(m, 0) for m in cluster], reverse=True)
        
        final_vec = ind_vec[:TOP_K]
        while len(final_vec) < TOP_K:
            final_vec.append(0)
            
        data_matrix.append(final_vec)
        valid_ids.append(node)

    # --- PCA ---
    if len(data_matrix) > 3:
        X = np.array(data_matrix)
        pca = PCA(n_components=3)
        pca_res = pca.fit_transform(X)
        
        print("\n" + "="*30)
        print("PCA VARIANCE EXPLAINED")
        for i, v in enumerate(pca.explained_variance_ratio_):
            print(f"PC{i+1}: {v*100:.2f}%")
        print("="*30 + "\n")

        with open(os.path.join(PCA_DIR, "pca_3d_results.csv"), 'w') as f:
            f.write("NodeID,PC1,PC2,PC3\n")
            for i, nid in enumerate(valid_ids):
                f.write(f"{nid},{pca_res[i][0]:.6f},{pca_res[i][1]:.6f},{pca_res[i][2]:.6f}\n")
        print("PCA complete.")
    else:
        print("Insufficient clusters found for PCA.")

if __name__ == "__main__":
    main()
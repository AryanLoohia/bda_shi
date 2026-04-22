import collections
import math
import random
import os
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler # Added for variance scaling

# --- 1. Data Loading & Parsing ---

def load_existing_indegrees(filepath):
    indegree = {}
    try:
        with open(filepath, 'r') as f:
            for line in f:
                if ':' not in line: continue
                node, val = line.split(':')
                indegree[int(node.strip())] = int(val.strip())
        print(f"Loaded {len(indegree)} nodes from stored indegrees.")
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
                        v = int(pair[0]); weight = int(pair[1])
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
                targets = [int(v) for v in parts[1].strip().split()]
                follows[u] = set(targets)
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
    print("Process started...")
    N_SAMPLES = 1000   
    K_HOPS = 6
    TOP_K = 20         
    PCA_DIR = "pca_results"
    INDEGREE_FILE = os.path.join(PCA_DIR, "global_indegrees.txt")

    if not os.path.exists(PCA_DIR):
        os.makedirs(PCA_DIR)

    if os.path.exists(INDEGREE_FILE):
        indegrees = load_existing_indegrees(INDEGREE_FILE)
    else:
        print("Error: Indegree file not found.")
        return

    print("Loading data...")
    follows_graph = parse_adjlist('one.adjlist')
    retweet_dict, R_total = parse_node_dict('one.retweet.node_dict.txt')
    mention_dict, M_total = parse_node_dict('one.metion.node_dict.txt')
    reply_dict, P_total = parse_node_dict('one.reply.node_dict.txt')

    R_a, R_b = calculate_stats(retweet_dict)
    M_a, M_b = calculate_stats(mention_dict)
    P_a, P_b = calculate_stats(reply_dict)

    print("Pass 1: Normalizing Z-scores...")
    raw_scores = {'r': [], 'm': [], 'p': []}
    edge_coords = [] 

    for a, targets in follows_graph.items():
        for b in targets:
            zr = get_z_score(a, b, retweet_dict, R_a, R_b, R_total)
            zm = get_z_score(a, b, mention_dict, M_a, M_b, M_total)
            zp = get_z_score(a, b, reply_dict, P_a, P_b, P_total)
            raw_scores['r'].append(zr)
            raw_scores['m'].append(zm)
            raw_scores['p'].append(zp)
            edge_coords.append((a, b, zr, zm, zp))

    stats = {k: (np.mean(v), np.std(v) + 1e-9) for k, v in raw_scores.items()}
    
    print("Pass 2: Building weighted edges...")
    edge_weights = collections.defaultdict(dict)
    all_s_scores = []

    for a, b, zr, zm, zp in edge_coords:
        norm_r = (zr - stats['r'][0]) / stats['r'][1]
        norm_m = (zm - stats['m'][0]) / stats['m'][1]
        norm_p = (zp - stats['p'][0]) / stats['p'][1]
        
        s_ab = (4 * norm_p) + (1 * norm_m) + (1 * norm_r)
        edge_weights[a][b] = s_ab
        all_s_scores.append(s_ab)

    THRESHOLD = np.mean(all_s_scores) if all_s_scores else 0
    
    all_nodes = list(follows_graph.keys())
    sampled_nodes = random.sample(all_nodes, min(N_SAMPLES, len(all_nodes)))
    data_matrix, valid_ids = [], []

    print("Generating log-scaled neighbor vectors...")
    for node in sampled_nodes:
        cluster = get_k_hop_neighbors(node, edge_weights, THRESHOLD, K_HOPS)
        if not cluster: continue

        # LOG-SCALING APPLIED HERE: np.log1p
        ind_vec = sorted([np.log1p(indegrees.get(m, 0)) for m in cluster], reverse=True)
        
        final_vec = ind_vec[:TOP_K]
        while len(final_vec) < TOP_K:
            final_vec.append(0)
            
        data_matrix.append(final_vec)
        valid_ids.append(node)

    if len(data_matrix) > 3:
        X = np.array(data_matrix)
        X_scaled = StandardScaler().fit_transform(X) # Ensure unit variance across components
        
        pca = PCA(n_components=3)
        pca_res = pca.fit_transform(X_scaled)
        
        print("\n" + "="*30)
        print("FINAL PCA VARIANCE EXPLAINED")
        for i, v in enumerate(pca.explained_variance_ratio_):
            print(f"PC{i+1}: {v*100:.2f}%")
        print("="*30 + "\n")

        output_path = os.path.join(PCA_DIR, "pca_3d_results.csv")
        np.savetxt(output_path, np.column_stack((valid_ids, pca_res)), 
                   delimiter=",", header="NodeID,PC1,PC2,PC3", comments='')
        print(f"PCA complete. Results saved to {output_path}")
    else:
        print("Insufficient data for PCA.")

if __name__ == "__main__":
    main()
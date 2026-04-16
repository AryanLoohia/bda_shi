import collections
import math
import random
import os

# --- Data Loading Functions ---
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
        pass
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
        pass
    return follows

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

# --- Parameters ---
N_SAMPLES = 10   # Number of nodes to sample
K_NEIGHBORS = 5 # Max neighbors per cluster
OUTPUT_DIR = "network_results"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "knn_clusters.txt")

# 1. Setup Environment
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# 2. Load Data
retweet_dict, R_total = parse_node_dict('one.retweet.node_dict.txt')
mention_dict, M_total = parse_node_dict('one.metion.node_dict.txt')
reply_dict, P_total = parse_node_dict('one.reply.node_dict.txt')
follows_graph = parse_adjlist('one.adjlist')

R_a, R_b = calculate_stats(retweet_dict)
M_a, M_b = calculate_stats(mention_dict)
P_a, P_b = calculate_stats(reply_dict)

# 3. Calculate all S_ab values and the Average Threshold
all_scores = []
edge_weights = collections.defaultdict(dict)

for a, followed_list in follows_graph.items():
    for b in followed_list:
        z_r = get_z_score(a, b, retweet_dict, R_a, R_b, R_total)
        z_m = get_z_score(a, b, mention_dict, M_a, M_b, M_total)
        z_p = get_z_score(a, b, reply_dict, P_a, P_b, P_total)
        s_ab = (4 * z_p) + (2 * z_m) + z_r
        
        all_scores.append(s_ab)
        edge_weights[a][b] = s_ab

avg_threshold = sum(all_scores) / len(all_scores) if all_scores else 0

# 4. Process Samples and Generate Output
available_nodes = list(edge_weights.keys())
sampled_nodes = random.sample(available_nodes, min(N_SAMPLES, len(available_nodes)))

with open(OUTPUT_FILE, 'w') as f:
    f.write(f"K-Nearest Neighbors Report\n")
    f.write(f"Global Average S_ab Threshold: {avg_threshold:.4f}\n")
    f.write("-" * 60 + "\n")
    f.write(f"{'Source':<10} | {'Size':<6} | {'Top Neighbors (S_ab Weight)'}\n")
    f.write("-" * 60 + "\n")

    for node in sampled_nodes:
        # Get neighbors that are above the average threshold
        valid_neighbors = [
            (neighbor, weight) 
            for neighbor, weight in edge_weights[node].items() 
            if weight > avg_threshold
        ]
        
        # Sort by weight (descending) and take top K
        valid_neighbors.sort(key=lambda x: x[1], reverse=True)
        cluster = valid_neighbors[:K_NEIGHBORS]
        
        cluster_data = ", ".join([f"{n}({w:.2f})" for n, w in cluster])
        f.write(f"{node:<10} | {len(cluster):<6} | {cluster_data}\n")

print(f"Success! Folder '{OUTPUT_DIR}' created and results saved to '{OUTPUT_FILE}'.")
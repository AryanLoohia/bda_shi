import collections
import math

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
    # Expected interaction if it were random based on user activity levels
    expected = (row_sums[u] * col_sums[v]) / total_all if total_all > 0 else 0
    return (actual - expected) / math.sqrt(expected + epsilon)

# --- Execution ---
retweet_dict, R_total = parse_node_dict('one.retweet.node_dict.txt')
mention_dict, M_total = parse_node_dict('one.metion.node_dict.txt')
reply_dict, P_total = parse_node_dict('one.reply.node_dict.txt')
follows_graph = parse_adjlist('one.adjlist')

R_a, R_b = calculate_stats(retweet_dict)
M_a, M_b = calculate_stats(mention_dict)
P_a, P_b = calculate_stats(reply_dict)

tau = 2
final_results = []

for a, followed_list in follows_graph.items():
    connections = []
    for b in followed_list:
        z_r = get_z_score(a, b, retweet_dict, R_a, R_b, R_total)
        z_m = get_z_score(a, b, mention_dict, M_a, M_b, M_total)
        z_p = get_z_score(a, b, reply_dict, P_a, P_b, P_total)
        
        s_ab = (4 * z_p) + (2 * z_m) + z_r
        
        if s_ab > tau:
            connections.append(f"{b}({round(s_ab, 3)})")
    
    if connections:
        final_results.append(f"{a}: {' '.join(connections)}")

# Save to file
with open('social_network_output.txt', 'w') as f:
    for line in final_results:
        f.write(line + '\n')

print("Process complete. Results saved to 'social_network_output.txt'.")
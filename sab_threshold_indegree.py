import collections
import math
import os
import numpy as np


def parse_node_dict(filename):
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


def main():
    output_dir = "network_results"
    os.makedirs(output_dir, exist_ok=True)

    print("Loading files...")
    follows_graph = parse_adjlist("one.adjlist")
    retweet_dict, r_total = parse_node_dict("one.retweet.node_dict.txt")
    mention_dict, m_total = parse_node_dict("one.metion.node_dict.txt")
    reply_dict, p_total = parse_node_dict("one.reply.node_dict.txt")

    print("Computing interaction marginals...")
    r_a, r_b = calculate_stats(retweet_dict)
    m_a, m_b = calculate_stats(mention_dict)
    p_a, p_b = calculate_stats(reply_dict)

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

    stats = {k: (np.mean(v), np.std(v) + 1e-9) for k, v in raw_scores.items()}

    print("Pass 2: Building S_ab for all edges...")
    edge_scores = []
    all_s_scores = []
    for a, b, zr, zm, zp in edge_coords:
        norm_r = (zr - stats["r"][0]) / stats["r"][1]
        norm_m = (zm - stats["m"][0]) / stats["m"][1]
        norm_p = (zp - stats["p"][0]) / stats["p"][1]
        s_ab = (4 * norm_p) + (1 * norm_m) + (1 * norm_r)
        edge_scores.append((a, b, s_ab))
        all_s_scores.append(s_ab)

    threshold = float(np.median(all_s_scores)) if all_s_scores else 0.0
    print(f"Median S_ab threshold: {threshold:.9f}")

    print("Filtering edges and computing indegree...")
    filtered_edges = []
    indegree = collections.defaultdict(int)
    all_nodes = set(follows_graph.keys())
    for _, targets in follows_graph.items():
        all_nodes.update(targets)

    for a, b, s_ab in edge_scores:
        if s_ab > threshold:
            filtered_edges.append((a, b, s_ab))
            indegree[b] += 1

    for node in all_nodes:
        indegree[node] += 0

    threshold_path = os.path.join(output_dir, "sab_mean_threshold.txt")
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

    print(f"Wrote threshold to {threshold_path}")
    print(f"Wrote all edge scores to {scores_path}")
    print(f"Wrote filtered edges to {filtered_path}")
    print(f"Wrote filtered indegrees to {indegree_path}")
    print(f"Total edges: {len(edge_scores)}")
    print(f"Kept edges: {len(filtered_edges)}")
    print(f"Total nodes in indegree output: {len(indegree)}")


if __name__ == "__main__":
    main()

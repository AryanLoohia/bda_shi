#!/usr/bin/env python3
"""
Sample nodes from an adjacency-list graph and compute BFS-k cluster stats.

Adjacency list input format expected per line:
    node: neighbor1 neighbor2 neighbor3 ...
"""

from __future__ import annotations

import argparse
import os
import random
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Set, Tuple


Graph = Dict[int, Set[int]]


def parse_adjlist(path: str) -> Graph:
    """Load adjacency list file into an undirected graph dictionary."""
    graph: Graph = {}

    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line or ":" not in line:
                continue

            node_part, neighbors_part = line.split(":", 1)
            node_part = node_part.strip()
            if not node_part:
                continue

            try:
                node = int(node_part)
            except ValueError:
                continue

            graph.setdefault(node, set())
            neighbors = neighbors_part.strip().split()

            for item in neighbors:
                try:
                    nbr = int(item)
                except ValueError:
                    continue

                graph[node].add(nbr)
                graph.setdefault(nbr, set()).add(node)

    return graph


def sample_nodes(graph: Graph, n: int, seed: int | None = None) -> List[int]:
    """Sample up to n unique nodes from graph."""
    nodes = list(graph.keys())
    if not nodes:
        return []

    if seed is not None:
        random.seed(seed)

    n = min(n, len(nodes))
    return random.sample(nodes, n)


def bfs_k_hop(graph: Graph, start: int, k: int) -> Set[int]:
    """Return all nodes within BFS distance <= k from start."""
    if start not in graph:
        return set()

    if k < 0:
        return set()

    visited: Set[int] = {start}
    q: deque[Tuple[int, int]] = deque([(start, 0)])

    while q:
        node, dist = q.popleft()
        if dist == k:
            continue

        for nbr in graph.get(node, ()):
            if nbr in visited:
                continue
            visited.add(nbr)
            q.append((nbr, dist + 1))

    return visited


def induced_subgraph_edge_count(graph: Graph, nodes: Set[int]) -> int:
    """Count undirected edges in the node-induced subgraph."""
    edges = 0
    for u in nodes:
        for v in graph.get(u, ()):
            if v in nodes and u < v:
                edges += 1
    return edges


def density(nodes_count: int, edges_count: int) -> float:
    """
    Compute 2E / (N*(N-1)).

    Returns 0.0 when N < 2.
    """
    if nodes_count < 2:
        return 0.0
    return (2.0 * edges_count) / (nodes_count * (nodes_count - 1))


def create_run_output_dir(base_output_dir: str = "outputs") -> Path:
    """Create a new run folder for each execution."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(base_output_dir) / f"run_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def write_result_file(
    out_path: Path,
    sample_index: int,
    source_node: int,
    k: int,
    cluster_nodes: Set[int],
    cluster_edges: int,
) -> None:
    """Write one sampled node result to one numbered file."""
    n_nodes = len(cluster_nodes)
    dens = density(n_nodes, cluster_edges)

    lines = [
        f"sample_index: {sample_index}",
        f"source_node: {source_node}",
        f"k_distance: {k}",
        f"cluster_nodes: {n_nodes}",
        f"cluster_edges: {cluster_edges}",
        f"density_2E_over_N_Nminus1: {dens:.8f}",
    ]

    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_pipeline(
    adjlist_path: str,
    n_samples: int,
    k_distance: int,
    output_dir: str,
    seed: int | None,
) -> Path:
    """Main workflow: parse, sample, BFS, compute stats, write files."""
    graph = parse_adjlist(adjlist_path)
    sampled = sample_nodes(graph, n_samples, seed=seed)
    run_dir = create_run_output_dir(output_dir)

    for idx, node in enumerate(sampled, start=1):
        cluster = bfs_k_hop(graph, node, k_distance)
        edge_count = induced_subgraph_edge_count(graph, cluster)

        write_result_file(
            out_path=run_dir / f"{idx}.txt",
            sample_index=idx,
            source_node=node,
            k=k_distance,
            cluster_nodes=cluster,
            cluster_edges=edge_count,
        )

    return run_dir


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Sample n nodes from adjacency list, run BFS upto k, "
            "and save cluster stats per sample."
        )
    )
    parser.add_argument(
        "--adjlist",
        default="one.adjlist",
        help="Path to adjacency list file (default: one.adjlist)",
    )
    parser.add_argument(
        "--n",
        type=int,
        required=True,
        help="Number of nodes to sample",
    )
    parser.add_argument(
        "--k",
        type=int,
        required=True,
        help="BFS max distance",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs",
        help="Base output directory (default: outputs)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional random seed for reproducible sampling",
    )
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    if args.n <= 0:
        raise SystemExit("--n must be > 0")
    if args.k < 0:
        raise SystemExit("--k must be >= 0")

    adj_path = args.adjlist
    if not os.path.isabs(adj_path):
        adj_path = os.path.join(os.getcwd(), adj_path)
    if not os.path.exists(adj_path):
        raise SystemExit(f"Adjacency list file not found: {adj_path}")

    run_dir = run_pipeline(
        adjlist_path=adj_path,
        n_samples=args.n,
        k_distance=args.k,
        output_dir=args.output_dir,
        seed=args.seed,
    )

    print(f"Done. Results written to: {run_dir}")


if __name__ == "__main__":
    main()

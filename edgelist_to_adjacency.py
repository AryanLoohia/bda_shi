#!/usr/bin/env python3
"""
Convert an edge-list file to adjacency-list format.

Input line format:
    a b
Meaning:
    a follows b  => directed edge a -> b

Output line format:
    a: b1 b2 b3
"""

from __future__ import annotations

import argparse
import os
from collections import defaultdict


def convert_edgelist_to_adjacency(
    input_path: str,
    output_path: str,
    deduplicate: bool = True,
    sort_nodes: bool = True,
    sort_neighbors: bool = True,
) -> None:
    adjacency = defaultdict(list)

    with open(input_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 2:
                continue
            try:
                a = int(parts[0])
                b = int(parts[1])
            except ValueError:
                continue
            adjacency[a].append(b)

    node_items = adjacency.items()
    if sort_nodes:
        node_items = sorted(node_items, key=lambda x: x[0])

    with open(output_path, "w", encoding="utf-8") as out:
        for node, neighbors in node_items:
            if deduplicate:
                neighbors = list(set(neighbors))
            if sort_neighbors:
                neighbors = sorted(neighbors)
            neighbor_str = " ".join(str(n) for n in neighbors)
            out.write(f"{node}: {neighbor_str}\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert one.edgelist style data to adjacency list"
    )
    parser.add_argument(
        "--input",
        default="one.edgelist",
        help="Input edge-list file (default: one.edgelist)",
    )
    parser.add_argument(
        "--output",
        default="one.adjlist",
        help="Output adjacency-list file (default: one.adjlist)",
    )
    parser.add_argument(
        "--keep-duplicates",
        action="store_true",
        help="Keep repeated neighbors instead of deduplicating",
    )
    parser.add_argument(
        "--no-sort-nodes",
        action="store_true",
        help="Do not sort source nodes in output",
    )
    parser.add_argument(
        "--no-sort-neighbors",
        action="store_true",
        help="Do not sort neighbor lists in output",
    )
    args = parser.parse_args()

    input_path = args.input
    output_path = args.output

    if not os.path.isabs(input_path):
        input_path = os.path.join(os.getcwd(), input_path)
    if not os.path.isabs(output_path):
        output_path = os.path.join(os.getcwd(), output_path)

    if not os.path.exists(input_path):
        raise SystemExit(f"Input file not found: {input_path}")

    convert_edgelist_to_adjacency(
        input_path=input_path,
        output_path=output_path,
        deduplicate=not args.keep_duplicates,
        sort_nodes=not args.no_sort_nodes,
        sort_neighbors=not args.no_sort_neighbors,
    )
    print(f"Wrote adjacency list to: {output_path}")


if __name__ == "__main__":
    main()


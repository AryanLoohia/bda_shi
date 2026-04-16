#!/usr/bin/env python3
"""
Convert weighted edge-list files to per-node dictionary-style text.

Input format (per line):
    a b x
Meaning:
    a has done the action for b, x times.

Output format (per line):
    a: {b1,x1} {b2,x2} ...
"""

from __future__ import annotations

import argparse
import os
from collections import defaultdict
from typing import DefaultDict, Dict, Iterable, List, Tuple


WeightedAdjacency = Dict[int, Dict[int, int]]


def parse_weighted_edgelist(input_path: str) -> WeightedAdjacency:
    """
    Parse `a b x` lines and aggregate repeated (a, b) entries.
    """
    adjacency: DefaultDict[int, DefaultDict[int, int]] = defaultdict(
        lambda: defaultdict(int)
    )

    with open(input_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 3:
                continue
            try:
                a = int(parts[0])
                b = int(parts[1])
                x = int(parts[2])
            except ValueError:
                continue

            adjacency[a][b] += x

    return {a: dict(neighbors) for a, neighbors in adjacency.items()}


def write_node_dictionary_txt(
    adjacency: WeightedAdjacency,
    output_path: str,
    sort_nodes: bool = True,
    sort_neighbors: bool = True,
) -> None:
    """
    Write lines like:
        a: {b1,x1} {b2,x2}
    """
    node_items: Iterable[Tuple[int, Dict[int, int]]] = adjacency.items()
    if sort_nodes:
        node_items = sorted(node_items, key=lambda item: item[0])

    with open(output_path, "w", encoding="utf-8") as out:
        for node, neighbors in node_items:
            neighbor_items: Iterable[Tuple[int, int]] = neighbors.items()
            if sort_neighbors:
                neighbor_items = sorted(neighbor_items, key=lambda item: item[0])

            pairs = " ".join(f"{{{nbr},{count}}}" for nbr, count in neighbor_items)
            out.write(f"{node}: {pairs}\n")


def default_output_path(input_path: str) -> str:
    """
    Build default output file path for one input file.
    Example:
        one.reply.edgelist -> one.reply.node_dict.txt
    """
    base, _ = os.path.splitext(input_path)
    return f"{base}.node_dict.txt"


def convert_file(
    input_path: str,
    output_path: str | None,
    sort_nodes: bool = True,
    sort_neighbors: bool = True,
) -> str:
    resolved_output = output_path or default_output_path(input_path)
    adjacency = parse_weighted_edgelist(input_path)
    write_node_dictionary_txt(
        adjacency=adjacency,
        output_path=resolved_output,
        sort_nodes=sort_nodes,
        sort_neighbors=sort_neighbors,
    )
    return resolved_output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert weighted a b x edgelists to dictionary-style node text"
    )
    parser.add_argument(
        "--input",
        nargs="+",
        default=["one.reply.edgelist", "one.retweet.edgelist", "one.metion.edgelist"],
        help=(
            "Input weighted edge-list files. "
            "Default: one.reply.edgelist one.retweet.edgelist one.metion.edgelist"
        ),
    )
    parser.add_argument(
        "--output",
        nargs="*",
        default=None,
        help=(
            "Optional output paths, same count/order as --input. "
            "If omitted, <input_basename>.node_dict.txt is used."
        ),
    )
    parser.add_argument(
        "--no-sort-nodes",
        action="store_true",
        help="Do not sort source nodes in output.",
    )
    parser.add_argument(
        "--no-sort-neighbors",
        action="store_true",
        help="Do not sort neighbors per source node in output.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    inputs: List[str] = args.input
    outputs: List[str] | None = args.output
    if outputs is not None and len(outputs) not in (0, len(inputs)):
        raise SystemExit("--output count must be either 0 or equal to --input count.")

    if outputs is not None and len(outputs) == 0:
        outputs = None

    for idx, input_path in enumerate(inputs):
        in_path = input_path
        if not os.path.isabs(in_path):
            in_path = os.path.join(os.getcwd(), in_path)
        if not os.path.exists(in_path):
            raise SystemExit(f"Input file not found: {in_path}")

        out_path = None
        if outputs is not None:
            out_path = outputs[idx]
            if not os.path.isabs(out_path):
                out_path = os.path.join(os.getcwd(), out_path)

        written = convert_file(
            input_path=in_path,
            output_path=out_path,
            sort_nodes=not args.no_sort_nodes,
            sort_neighbors=not args.no_sort_neighbors,
        )
        print(f"Wrote: {written}")


if __name__ == "__main__":
    main()

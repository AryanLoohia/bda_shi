#!/usr/bin/env python3
"""
Visualize an edge-list graph from twitter_combined.txt (A B per line).

This dataset is huge (~millions of edges), so this script builds a *subgraph*
that is actually viewable by first computing node degrees, then keeping either:
  - the top-N highest-degree nodes, and/or
  - nodes with degree >= min_degree

Outputs:
  - interactive HTML (PyVis)
  - optional GEXF for Gephi
"""

from __future__ import annotations

import argparse
import math
import os
import random
from collections import defaultdict
from typing import DefaultDict, Dict, Iterable, Iterator, Optional, Set, Tuple


def iter_edges(path: str) -> Iterator[Tuple[int, int]]:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            try:
                yield int(parts[0]), int(parts[1])
            except ValueError:
                continue


def human(n: int) -> str:
    units = ["", "K", "M", "B", "T"]
    x = float(n)
    for u in units:
        if abs(x) < 1000.0:
            return f"{x:.1f}{u}" if u else f"{int(x)}"
        x /= 1000.0
    return f"{x:.1f}P"


def pick_nodes_by_degree(
    degrees: Dict[int, int],
    *,
    top_nodes: Optional[int],
    min_degree: int,
) -> Set[int]:
    keep: Set[int] = set()

    if min_degree > 0:
        keep |= {n for n, d in degrees.items() if d >= min_degree}

    if top_nodes is not None and top_nodes > 0:
        # nlargest without importing heapq for speed/readability trade-off
        # (heapq is fine too, but sort is okay for typical node counts here)
        top = sorted(degrees.items(), key=lambda kv: kv[1], reverse=True)[:top_nodes]
        keep |= {n for n, _ in top}

    return keep


def main() -> None:
    ap = argparse.ArgumentParser(description="Visualize twitter_combined edge list")
    ap.add_argument(
        "--input",
        default=None,
        help="Path to edge list file (default: one.edgelist if present, else twitter_combined.txt)",
    )
    ap.add_argument(
        "--directed",
        action="store_true",
        help="Treat edges as directed (default: undirected)",
    )
    ap.add_argument(
        "--top-nodes",
        type=int,
        default=500,
        help="Keep top-N nodes by degree (default: 500). Set 0 to disable.",
    )
    ap.add_argument(
        "--min-degree",
        type=int,
        default=0,
        help="Also keep nodes with degree >= this (default: 0)",
    )
    ap.add_argument(
        "--max-edges",
        type=int,
        default=20000,
        help="Maximum edges to include in the visualized subgraph (default: 20000)",
    )
    ap.add_argument(
        "--output-html",
        default=None,
        help="Output HTML file (default: <input_basename>_graph.html)",
    )
    ap.add_argument(
        "--output-gexf",
        default="",
        help="Optional output GEXF path for Gephi (default: disabled)",
    )
    ap.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used when limiting edges (default: 42)",
    )
    args = ap.parse_args()

    random.seed(args.seed)

    if args.input:
        in_path = args.input
    else:
        in_path = "one.edgelist" if os.path.exists("one.edgelist") else "none.txt"
    if not os.path.isabs(in_path):
        in_path = os.path.join(os.getcwd(), in_path)

    if not os.path.exists(in_path):
        raise SystemExit(f"Input not found: {in_path}")

    # Pass 1: compute degrees (streaming)
    degrees: DefaultDict[int, int] = defaultdict(int)
    edge_count = 0
    for u, v in iter_edges(in_path):
        degrees[u] += 1
        degrees[v] += 1
        edge_count += 1

    top_nodes = None if args.top_nodes <= 0 else args.top_nodes
    keep = pick_nodes_by_degree(degrees, top_nodes=top_nodes, min_degree=args.min_degree)
    if not keep:
        raise SystemExit(
            "No nodes selected. Try lowering --min-degree or increasing --top-nodes."
        )

    # Pass 2: build subgraph among kept nodes, limit edges
    import networkx as nx

    G = nx.DiGraph() if args.directed else nx.Graph()
    kept_edge_count = 0

    # Add nodes first with degree info (for sizing)
    for n in keep:
        G.add_node(n, degree=int(degrees.get(n, 0)))

    for u, v in iter_edges(in_path):
        if u not in keep or v not in keep:
            continue
        G.add_edge(u, v)
        kept_edge_count += 1
        if args.max_edges and kept_edge_count >= args.max_edges:
            break

    # Node size/value for PyVis: compress degrees into a nicer range
    if G.number_of_nodes() > 0:
        degs = [G.nodes[n].get("degree", 0) for n in G.nodes]
        d_min, d_max = min(degs), max(degs)
        for n in G.nodes:
            d = G.nodes[n].get("degree", 0)
            if d_max == d_min:
                val = 10
            else:
                # log-scale helps for heavy-tail degree distributions
                val = 5 + 45 * (math.log1p(d) - math.log1p(d_min)) / (
                    math.log1p(d_max) - math.log1p(d_min)
                )
            G.nodes[n]["value"] = float(val)
            G.nodes[n]["title"] = f"node: {n}<br>degree: {d}"

    # Export GEXF for Gephi (optional)
    if args.output_gexf:
        out_gexf = args.output_gexf
        if not os.path.isabs(out_gexf):
            out_gexf = os.path.join(os.getcwd(), out_gexf)
        nx.write_gexf(G, out_gexf)

    # Interactive HTML via PyVis
    from pyvis.network import Network

    net = Network(height="800px", width="100%", bgcolor="#0b1020", font_color="white")
    net.barnes_hut(gravity=-8000, central_gravity=0.2, spring_length=120, spring_strength=0.005)
    net.from_nx(G)
    net.show_buttons(filter_=["physics"])

    if args.output_html:
        out_html = args.output_html
    else:
        base = os.path.splitext(os.path.basename(in_path))[0]
        out_html = f"{base}_graph.html"
    if not os.path.isabs(out_html):
        out_html = os.path.join(os.getcwd(), out_html)
    net.write_html(out_html, open_browser=False, notebook=False)

    print(
        "\n".join(
            [
                "Done.",
                f"Input edges: {human(edge_count)}",
                f"Selected nodes: {human(len(keep))}",
                f"Subgraph: {human(G.number_of_nodes())} nodes, {human(G.number_of_edges())} edges",
                f"HTML: {out_html}",
                f"GEXF: {args.output_gexf if args.output_gexf else '(disabled)'}",
            ]
        )
    )


if __name__ == "__main__":
    main()


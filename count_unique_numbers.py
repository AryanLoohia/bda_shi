#!/usr/bin/env python3
"""
Count unique numbers (node IDs) in an edge list file.

Expected format per line:
    <u> <v>
"""

from __future__ import annotations

import argparse
import os


def count_unique_numbers(path: str) -> int:
    unique = set()
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 2:
                continue
            try:
                unique.add(int(parts[0]))
                unique.add(int(parts[1]))
            except ValueError:
                # Skip malformed/non-numeric rows.
                continue
    return len(unique)


def main() -> None:
    parser = argparse.ArgumentParser(description="Count unique numbers in edge list")
    parser.add_argument(
        "--input",
        default="one.edgelist",
        help="Input edge-list file (default: one.edgelist)",
    )
    args = parser.parse_args()

    input_path = args.input
    if not os.path.isabs(input_path):
        input_path = os.path.join(os.getcwd(), input_path)

    if not os.path.exists(input_path):
        raise SystemExit(f"Input file not found: {input_path}")

    result = count_unique_numbers(input_path)
    print(f"Unique numbers in {input_path}: {result}")


if __name__ == "__main__":
    main()


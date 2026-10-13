from __future__ import annotations

import argparse
import json
from pathlib import Path

from token_sniper.graph import build_ssg, ssg_quality_snapshot, write_ssg_json


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build Static Structural Graph JSON for a repo root."
    )
    parser.add_argument(
        "--repo",
        type=Path,
        required=True,
        help="Repository root (Python files discovered under here)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Path to write SSG JSON (node-link format)",
    )
    parser.add_argument(
        "--glob",
        type=str,
        default="**/*.py",
        help="Glob relative to repo root (default: **/*.py)",
    )
    parser.add_argument(
        "--metrics",
        action="store_true",
        help="Print quality snapshot JSON to stdout after build",
    )
    parser.add_argument(
        "--expensive-metrics",
        action="store_true",
        help="Include diameter on largest WCC in printed snapshot (use with --metrics)",
    )
    args = parser.parse_args()

    graph = build_ssg(args.repo.resolve(), include_glob=args.glob)
    write_ssg_json(graph, args.output.resolve())
    if args.metrics:
        snap = ssg_quality_snapshot(graph, expensive=args.expensive_metrics)
        print(json.dumps(snap, indent=2))


if __name__ == "__main__":
    main()

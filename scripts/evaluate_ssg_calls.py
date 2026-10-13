from __future__ import annotations

import argparse
import json
from pathlib import Path

from token_sniper.graph.calls_evaluation import (
    iter_ssg_call_edges,
    load_call_gold_json,
    precision_recall_f1,
)
from token_sniper.graph.metrics import ssg_quality_snapshot
from token_sniper.graph.ssg import build_ssg


def main() -> None:
    parser = argparse.ArgumentParser(description="Score SSG call edges against gold JSON.")
    parser.add_argument("--repo", type=Path, required=True, help="Repository root to analyze")
    parser.add_argument("--gold", type=Path, required=True, help="JSON file of gold call edges")
    parser.add_argument(
        "--glob",
        type=str,
        default="**/*.py",
        help="Glob under repo for Python files (default: **/*.py)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional path to write full JSON report (metrics + SSG snapshot)",
    )
    parser.add_argument(
        "--metrics-only",
        action="store_true",
        help="Print only call PRF JSON line to stdout",
    )
    args = parser.parse_args()

    repo = args.repo.resolve()
    graph = build_ssg(repo, include_glob=args.glob)
    gold = load_call_gold_json(args.gold.resolve())
    pred = iter_ssg_call_edges(graph)
    prf = precision_recall_f1(pred, gold)
    call_metrics = {
        "repo_root": str(repo),
        "gold_path": str(args.gold.resolve()),
        "include_glob": args.glob,
        **prf,
    }
    if args.metrics_only:
        print(json.dumps(call_metrics, indent=2))
        return

    report = {
        "call_edge_metrics": call_metrics,
        "ssg_snapshot": ssg_quality_snapshot(graph),
    }
    text = json.dumps(report, indent=2)
    print(text)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()

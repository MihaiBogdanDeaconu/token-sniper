from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

from token_sniper.graph.ssg import build_ssg

if TYPE_CHECKING:
    import networkx as nx


def iter_ssg_call_edges(graph: nx.MultiDiGraph[Any]) -> set[tuple[str, str]]:
    out: set[tuple[str, str]] = set()
    for u, v, _k, data in graph.edges(keys=True, data=True):
        if data.get("edge_type") == "calls":
            out.add((str(u), str(v)))
    return out


def precision_recall_f1(
    predicted: set[tuple[str, str]],
    gold: set[tuple[str, str]],
) -> dict[str, float | int]:
    if not gold and not predicted:
        return {
            "precision": 1.0,
            "recall": 1.0,
            "f1": 1.0,
            "true_positives": 0,
            "false_positives": 0,
            "false_negatives": 0,
            "num_predicted": 0,
            "num_gold": 0,
        }

    tp = len(predicted & gold)
    fp = len(predicted - gold)
    fn = len(gold - predicted)
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
    return {
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "num_predicted": len(predicted),
        "num_gold": len(gold),
    }


def load_call_gold_json(path: Path | str) -> set[tuple[str, str]]:
    p = Path(path)
    raw = json.loads(p.read_text(encoding="utf-8"))
    pairs = raw if isinstance(raw, list) else raw.get("call_edges", [])
    return {(str(a), str(b)) for a, b in pairs}


def evaluate_ssg_calls_against_gold(
    repo_root: Path | str,
    gold_path: Path | str,
    *,
    include_glob: str = "**/*.py",
) -> dict[str, Any]:
    root = Path(repo_root).resolve()
    graph = build_ssg(root, include_glob=include_glob)
    pred = iter_ssg_call_edges(graph)
    gold = load_call_gold_json(gold_path)
    metrics = precision_recall_f1(pred, gold)
    return {
        "repo_root": str(root),
        "gold_path": str(Path(gold_path).resolve()),
        "include_glob": include_glob,
        **metrics,
    }

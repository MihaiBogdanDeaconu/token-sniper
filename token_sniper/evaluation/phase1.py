from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from pathlib import Path

logger = logging.getLogger(__name__)


class TaskRetrievalMetrics(BaseModel):
    instance_id: str
    condition: str
    repo: str
    difficulty: str = ""
    gold_files: list[str] = Field(default_factory=list)
    retrieved_files: list[str] = Field(default_factory=list)
    entry_point_fqns: list[str] = Field(default_factory=list)
    entry_point_method: str = ""
    latency_ms: float = 0.0
    precision_at_k: dict[int, float] = Field(default_factory=dict)
    recall_at_k: dict[int, float] = Field(default_factory=dict)

    model_config = {"frozen": True}


class ConditionSummary(BaseModel):
    condition: str
    task_count: int
    mean_precision_at_k: dict[int, float] = Field(default_factory=dict)
    mean_recall_at_k: dict[int, float] = Field(default_factory=dict)
    per_difficulty: dict[str, dict[str, float]] = Field(default_factory=dict)
    mean_latency_ms: float = 0.0
    tasks_with_no_entry_points: int = 0

    model_config = {"frozen": True}


class Phase1Summary(BaseModel):
    phase: str = "phase1"
    run_id: str
    task_count: int
    k_values: list[int]
    conditions: dict[str, ConditionSummary] = Field(default_factory=dict)
    task_results: list[TaskRetrievalMetrics] = Field(default_factory=list)
    kill_criterion_met: bool = False
    kill_criterion_detail: str = ""

    model_config = {"frozen": True}


def compute_file_level_metrics(
    *,
    retrieved_files: list[str],
    gold_files: list[str],
    k_values: list[int],
) -> tuple[dict[int, float], dict[int, float]]:
    from token_sniper.evaluation.metrics import precision_at_k, recall_at_k

    prec: dict[int, float] = {}
    rec: dict[int, float] = {}
    for k in k_values:
        prec[k] = precision_at_k(retrieved_files, gold_files, k=k)
        rec[k] = recall_at_k(retrieved_files, gold_files, k=k)
    return prec, rec


def aggregate_condition_metrics(
    task_metrics: list[TaskRetrievalMetrics],
    *,
    condition: str,
    k_values: list[int],
) -> ConditionSummary:
    if not task_metrics:
        return ConditionSummary(condition=condition, task_count=0)

    n = len(task_metrics)
    mean_prec: dict[int, float] = {}
    mean_rec: dict[int, float] = {}
    for k in k_values:
        mean_prec[k] = sum(t.precision_at_k.get(k, 0.0) for t in task_metrics) / n
        mean_rec[k] = sum(t.recall_at_k.get(k, 0.0) for t in task_metrics) / n

    by_difficulty: dict[str, list[TaskRetrievalMetrics]] = {}
    for t in task_metrics:
        diff = t.difficulty or "unknown"
        by_difficulty.setdefault(diff, []).append(t)

    per_difficulty: dict[str, dict[str, float]] = {}
    for diff, tasks in by_difficulty.items():
        nd = len(tasks)
        diff_metrics: dict[str, float] = {"count": float(nd)}
        for k in k_values:
            diff_metrics[f"mean_precision@{k}"] = sum(
                t.precision_at_k.get(k, 0.0) for t in tasks
            ) / nd
            diff_metrics[f"mean_recall@{k}"] = sum(
                t.recall_at_k.get(k, 0.0) for t in tasks
            ) / nd
        per_difficulty[diff] = diff_metrics

    mean_latency = sum(t.latency_ms for t in task_metrics) / n
    no_ep = sum(1 for t in task_metrics if not t.entry_point_fqns)

    return ConditionSummary(
        condition=condition,
        task_count=n,
        mean_precision_at_k=mean_prec,
        mean_recall_at_k=mean_rec,
        per_difficulty=per_difficulty,
        mean_latency_ms=round(mean_latency, 1),
        tasks_with_no_entry_points=no_ep,
    )


def evaluate_kill_criterion(
    summaries: dict[str, ConditionSummary],
    k_values: list[int],
) -> tuple[bool, str]:
    emb = summaries.get("embedding_knn")
    bfs = summaries.get("graph_bfs")
    ppr = summaries.get("graph_ppr")

    if emb is None:
        return False, "embedding_knn condition not found; cannot evaluate kill criterion"

    details: list[str] = []
    graph_wins: list[str] = []

    for k in k_values:
        emb_prec = emb.mean_precision_at_k.get(k, 0.0)
        if bfs:
            bfs_prec = bfs.mean_precision_at_k.get(k, 0.0)
            diff = bfs_prec - emb_prec
            details.append(f"  P@{k}: BFS={bfs_prec:.4f} vs kNN={emb_prec:.4f} (Δ={diff:+.4f})")
            if bfs_prec > emb_prec:
                graph_wins.append(f"graph_bfs beats embedding_knn at P@{k}")
        if ppr:
            ppr_prec = ppr.mean_precision_at_k.get(k, 0.0)
            diff = ppr_prec - emb_prec
            details.append(f"  P@{k}: PPR={ppr_prec:.4f} vs kNN={emb_prec:.4f} (Δ={diff:+.4f})")
            if ppr_prec > emb_prec:
                graph_wins.append(f"graph_ppr beats embedding_knn at P@{k}")

    detail_text = "\n".join(details)

    if graph_wins:
        wins_text = "; ".join(graph_wins)
        return False, f"Graph wins detected ({wins_text}):\n{detail_text}"

    return True, f"Neither graph strategy beats embedding_knn at any k:\n{detail_text}"


def build_summary_table(summaries: dict[str, ConditionSummary], k_values: list[int]) -> str:
    conditions = sorted(summaries.keys())
    lines = ["# Phase 1: Static Graph Baseline — Retrieval Comparison", ""]

    lines.append("## Precision@k")
    header = "| Condition | " + " | ".join(f"P@{k}" for k in k_values) + " | Latency (ms) |"
    sep = "|" + "---|" * (len(k_values) + 2)
    lines.extend([header, sep])
    for cond in conditions:
        s = summaries[cond]
        precs = " | ".join(f"{s.mean_precision_at_k.get(k, 0.0):.4f}" for k in k_values)
        lines.append(f"| {cond} | {precs} | {s.mean_latency_ms:.0f} |")
    lines.append("")

    lines.append("## Recall@k")
    header = "| Condition | " + " | ".join(f"R@{k}" for k in k_values) + " |"
    sep = "|" + "---|" * (len(k_values) + 1)
    lines.extend([header, sep])
    for cond in conditions:
        s = summaries[cond]
        recs = " | ".join(f"{s.mean_recall_at_k.get(k, 0.0):.4f}" for k in k_values)
        lines.append(f"| {cond} | {recs} |")
    lines.append("")

    return "\n".join(lines)


def write_phase1_decision(
    *,
    output_dir: Path,
    summary: Phase1Summary,
) -> Path:
    decision_path = output_dir / "DECISION.md"
    output_dir.mkdir(parents=True, exist_ok=True)

    k_values = summary.k_values
    summaries = summary.conditions
    table = build_summary_table(summaries, k_values)

    lines = [table, ""]

    lines.append("## Kill Criterion Evaluation")
    lines.append("")
    lines.append(
        '> "If neither graph_bfs nor graph_ppr beats embedding_knn on '
        'Precision@k for any k, the graph structure is not adding value."'
    )
    lines.append("")

    if summary.kill_criterion_met:
        lines.extend([
            "### DECISION: KILL — Graph structure is NOT adding value",
            "",
            "Neither graph-based strategy outperformed the dense embedding baseline "
            "at any value of k. The static structural graph in its current form does not "
            "provide retrieval signal beyond what embedding cosine similarity already captures.",
            "",
            "**Recommended remediation:**",
            "- Richer edge types (cross-file data flow, type inference)",
            "- Dynamic call graph information",
            "- Graph neural network encodings (Phase 2 VGIB)",
            "- Better entry-point identification",
        ])
    else:
        lines.extend([
            "### DECISION: GO — Proceed to Phase 2",
            "",
            "At least one graph-based strategy outperformed the dense embedding baseline, "
            "indicating that the structural graph carries useful retrieval signal. "
            "Proceed to Phase 2 (VGIB pruning) to learn optimal subgraph selection.",
        ])

    lines.extend(["", "## Detailed Comparison", "", summary.kill_criterion_detail, ""])

    lines.append("## Per-Difficulty Breakdown")
    lines.append("")
    for cond_name, cond_summary in sorted(summaries.items()):
        lines.append(f"### {cond_name}")
        for diff, metrics in sorted(cond_summary.per_difficulty.items()):
            count = int(metrics.get("count", 0))
            metric_parts = [f"n={count}"]
            for k in k_values:
                p = metrics.get(f"mean_precision@{k}", 0.0)
                r = metrics.get(f"mean_recall@{k}", 0.0)
                metric_parts.append(f"P@{k}={p:.4f}")
                metric_parts.append(f"R@{k}={r:.4f}")
            lines.append(f"- **{diff}**: {', '.join(metric_parts)}")
        lines.append("")

    lines.extend([
        "## Note: Shell Baseline (4th Condition)",
        "",
        "The **shell** condition (Phase 0 blind agent using grep/find/cat) has no retrieval step",
        "and therefore no Precision@k / Recall@k metrics. It is compared against the other three",
        "conditions at the **downstream Pass@1** level only, which requires a separate evaluation",
        "run with the full SWE-bench harness (`scripts/run_phase1_full.py`).",
        "",
        "This retrieval comparison covers 3 of 4 conditions: `graph_bfs`, `graph_ppr`, `embedding_knn`.",
        "",
    ])

    decision_path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("DECISION.md written to %s", decision_path)
    return decision_path


def write_phase1_summary_json(
    *,
    output_dir: Path,
    summary: Phase1Summary,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "summary.json"
    summary_path.write_text(
        summary.model_dump_json(indent=2),
        encoding="utf-8",
    )
    logger.info("Summary JSON written to %s", summary_path)
    return summary_path

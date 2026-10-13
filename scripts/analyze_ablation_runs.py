#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import statistics
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

DEFAULT_CONDITIONS = [
    "dense_knn",
    "dense_reranker",
    "dense_reranker_vgib",
    "dense_reranker_vgib_rl",
]


@dataclass
class TaskMetrics:
    instance_id: str
    resolved: bool
    has_patch: bool
    has_session: bool
    wall_time_s: float | None = None
    turns: int | None = None
    tool_calls: int | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    patch_attempts: int | None = None
    valid_patch_attempts: int | None = None
    oracle_file_count: int = 0
    gold_files: int = 0
    gold_in_oracle: int = 0
    agent_status: str | None = None
    failure_stage: str | None = None
    tool_counts: dict[str, int] = field(default_factory=dict)


def _stats(values: list[float]) -> dict[str, float | int]:
    if not values:
        return {"n": 0}
    values_sorted = sorted(values)
    n = len(values)
    out: dict[str, float | int] = {
        "n": n,
        "mean": statistics.mean(values),
        "median": statistics.median(values),
        "min": values_sorted[0],
        "max": values_sorted[-1],
    }
    if n >= 2:
        out["stdev"] = statistics.stdev(values)
    if n >= 10:
        out["p90"] = values_sorted[int(0.9 * (n - 1))]
    return out


def _find_run_dir(condition_dir: Path, run_id: str | None) -> Path | None:
    if run_id:
        candidate = condition_dir / run_id
        if (candidate / "agent_outputs").is_dir():
            return candidate
    candidates = [
        p for p in condition_dir.iterdir()
        if p.is_dir() and (p / "agent_outputs").is_dir()
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def _load_session(session_path: Path) -> dict[str, Any] | None:
    if not session_path.is_file():
        return None
    try:
        return json.loads(session_path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _tool_counts_from_session(session: dict[str, Any]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for turn in session.get("turns", []):
        for tc in turn.get("tool_calls", []):
            name = tc.get("name") or tc.get("tool_name") or "unknown"
            counts[name] = counts.get(name, 0) + 1
    return counts


def _load_mlflow_metrics(condition_dir: Path, run_id: str | None) -> dict[str, Any]:
    nested = condition_dir.parent / "ablation" / condition_dir.name
    if not nested.is_dir():
        return {}
    patterns: list[Path] = []
    if run_id:
        patterns.append(nested / f"ablation_{condition_dir.name}_{run_id}" / "metrics.json")
    patterns.extend(sorted(nested.glob("*/metrics.json"), key=lambda p: p.stat().st_mtime, reverse=True))
    for path in patterns:
        if path.is_file():
            try:
                return json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                continue
    return {}


def analyze_condition(condition_dir: Path) -> dict[str, Any]:
    summary_path = condition_dir / "summary.json"
    if not summary_path.is_file():
        return {"condition": condition_dir.name, "error": f"missing {summary_path}"}

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    run_id = summary.get("run_id")
    run_dir = _find_run_dir(condition_dir, run_id)
    if run_dir is None:
        return {"condition": condition_dir.name, "error": "no run dir with agent_outputs/"}

    per_task_summary = {row["instance_id"]: row for row in summary.get("per_task", [])}
    tasks: list[TaskMetrics] = []

    agent_root = run_dir / "agent_outputs"
    for task_dir in sorted(agent_root.iterdir()) if agent_root.is_dir() else []:
        if not task_dir.is_dir():
            continue
        iid = task_dir.name
        patch_file = task_dir / "patch.diff"
        has_patch = patch_file.is_file() and bool(patch_file.read_text(encoding="utf-8").strip())
        session = _load_session(task_dir / "session.json")

        row = per_task_summary.get(iid, {})
        gold_files = row.get("gold_files") or []
        gold_in_oracle = row.get("gold_in_oracle", 0)
        tm = TaskMetrics(
            instance_id=iid,
            resolved=bool(row.get("resolved", False)),
            has_patch=has_patch,
            has_session=session is not None,
            oracle_file_count=int(row.get("oracle_file_count") or 0),
            gold_files=len(gold_files),
            gold_in_oracle=int(gold_in_oracle),
        )

        task_json = run_dir / "tasks" / f"{iid}.json"
        if task_json.is_file():
            try:
                tdata = json.loads(task_json.read_text(encoding="utf-8"))
                meta = tdata.get("metadata") or {}
                tm.agent_status = meta.get("agent_status")
                tm.failure_stage = meta.get("failure_stage")
            except Exception:
                pass

        if session:
            tm.wall_time_s = float(session.get("total_wall_time_seconds") or 0)
            tm.turns = len(session.get("turns") or [])
            tm.prompt_tokens = int(session.get("total_prompt_tokens") or 0)
            tm.completion_tokens = int(session.get("total_completion_tokens") or 0)
            tm.patch_attempts = int(session.get("patch_attempt_count") or 0)
            tm.valid_patch_attempts = int(session.get("valid_patch_attempt_count") or 0)
            tm.tool_counts = _tool_counts_from_session(session)
            tm.tool_calls = sum(tm.tool_counts.values())

        tasks.append(tm)

    for iid, row in per_task_summary.items():
        if any(t.instance_id == iid for t in tasks):
            continue
        gold_files = row.get("gold_files") or []
        tasks.append(
            TaskMetrics(
                instance_id=iid,
                resolved=bool(row.get("resolved", False)),
                has_patch=False,
                has_session=False,
                oracle_file_count=int(row.get("oracle_file_count") or 0),
                gold_files=len(gold_files),
                gold_in_oracle=int(row.get("gold_in_oracle") or 0),
            )
        )

    total_tasks = int(summary.get("total_tasks") or len(tasks))
    resolved = [t for t in tasks if t.resolved]
    patched = [t for t in tasks if t.has_patch]
    with_session = [t for t in tasks if t.has_session]
    docker_fail = [t for t in tasks if t.failure_stage == "docker_start"]

    def wall_subset(subset: list[TaskMetrics]) -> dict[str, Any]:
        vals = [t.wall_time_s for t in subset if t.wall_time_s is not None]
        return _stats(vals)

    def int_subset(subset: list[TaskMetrics], attr: str) -> dict[str, Any]:
        vals = [float(getattr(t, attr)) for t in subset if getattr(t, attr) is not None]
        return _stats(vals)

    gold_ratios = [
        t.gold_in_oracle / t.gold_files
        for t in tasks
        if t.gold_files > 0
    ]

    tool_totals: dict[str, int] = {}
    for t in with_session:
        for name, count in t.tool_counts.items():
            tool_totals[name] = tool_totals.get(name, 0) + count

    mlflow_metrics = _load_mlflow_metrics(condition_dir, run_id)

    return {
        "condition": condition_dir.name,
        "run_id": run_dir.name,
        "run_dir": str(run_dir),
        "summary_path": str(summary_path),
        "headline": {
            "pass_at_1": summary.get("pass_at_1"),
            "total_tasks": total_tasks,
            "total_evaluated": summary.get("total_evaluated"),
            "total_resolved": summary.get("total_resolved"),
            "patch_rate": len(patched) / total_tasks if total_tasks else 0.0,
            "session_rate": len(with_session) / total_tasks if total_tasks else 0.0,
            "docker_start_failures": len(docker_fail),
        },
        "timing_seconds": {
            "all_tasks": wall_subset(with_session),
            "resolved": wall_subset([t for t in resolved if t.has_session]),
            "unresolved": wall_subset(
                [t for t in with_session if not t.resolved]
            ),
        },
        "turns": {
            "all_tasks": int_subset(with_session, "turns"),
            "resolved": int_subset([t for t in resolved if t.has_session], "turns"),
        },
        "tool_calls_per_task": int_subset(with_session, "tool_calls"),
        "tokens_per_task": {
            "prompt": int_subset(with_session, "prompt_tokens"),
            "completion": int_subset(with_session, "completion_tokens"),
        },
        "retrieval_proxy": {
            "oracle_file_count": int_subset(tasks, "oracle_file_count"),
            "gold_in_oracle_count": int_subset(
                [t for t in tasks if t.gold_files > 0], "gold_in_oracle"
            ),
            "gold_in_oracle_ratio": _stats(gold_ratios),
        },
        "tool_totals": dict(sorted(tool_totals.items(), key=lambda kv: -kv[1])),
        "mlflow_metrics": {
            k: mlflow_metrics[k]
            for k in (
                "mean_fail_to_pass_rate",
                "mean_pass_to_pass_rate",
                "total_tokens",
                "navigation_tokens",
                "reasoning_tokens",
                "tokens_per_resolved_task_total",
                "token_efficiency_total",
            )
            if k in mlflow_metrics
        },
        "tasks_missing_session": [
            t.instance_id for t in tasks if not t.has_session
        ],
        "docker_start_failures": [t.instance_id for t in docker_fail],
    }


def _print_table(reports: list[dict[str, Any]]) -> None:
    cols = [
        ("condition", lambda r: r.get("condition", "?")),
        ("pass@1", lambda r: f"{100 * (r.get('headline') or {}).get('pass_at_1', 0):.1f}%"),
        ("patch%", lambda r: f"{100 * (r.get('headline') or {}).get('patch_rate', 0):.1f}%"),
        ("eval", lambda r: str((r.get("headline") or {}).get("total_evaluated", "?"))),
        ("med wall(s)", lambda r: _fmt_stat(r, "timing_seconds", "all_tasks", "median")),
        ("med wall res", lambda r: _fmt_stat(r, "timing_seconds", "resolved", "median")),
        ("med turns", lambda r: _fmt_stat(r, "turns", "all_tasks", "median")),
        ("gold/oracle", lambda r: _fmt_stat(r, "retrieval_proxy", "gold_in_oracle_ratio", "mean")),
        ("docker fail", lambda r: str((r.get("headline") or {}).get("docker_start_failures", 0))),
    ]

    widths = {name: max(len(name), *(len(fn(rep)) for rep in reports if "error" not in rep)) for name, fn in cols}
    header = "  ".join(name.ljust(widths[name]) for name, _ in cols)
    print(header)
    print("-" * len(header))
    for rep in reports:
        if "error" in rep:
            print(f"{rep['condition']}: ERROR — {rep['error']}")
            continue
        print("  ".join(fn(rep).ljust(widths[name]) for name, fn in cols))


def _fmt_stat(rep: dict[str, Any], *keys: str) -> str:
    cur: Any = rep
    for key in keys:
        cur = cur.get(key, {}) if isinstance(cur, dict) else {}
    if isinstance(cur, (int, float)):
        return f"{cur:.1f}"
    return "-"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=Path("experiments/ablation"),
        help="Root containing dense_knn/, dense_reranker/, ...",
    )
    parser.add_argument(
        "--conditions",
        nargs="*",
        default=DEFAULT_CONDITIONS,
        help="Conditions to analyze",
    )
    parser.add_argument(
        "--json-out",
        type=Path,
        default=None,
        help="Write full report JSON to this path",
    )
    parser.add_argument(
        "--per-task-out",
        type=Path,
        default=None,
        help="Write per-task CSV-like JSON (not implemented as CSV yet)",
    )
    args = parser.parse_args()

    reports: list[dict[str, Any]] = []
    for cond in args.conditions:
        reports.append(analyze_condition(args.base_dir / cond))

    print()
    _print_table(reports)
    print()

    for rep in reports:
        if "error" in rep:
            continue
        print(f"=== {rep['condition']} ({rep['run_id']}) ===")
        h = rep["headline"]
        print(
            f"  pass@1={h['pass_at_1']:.1%}  patches={h['patch_rate']:.1%}  "
            f"evaluated={h['total_evaluated']}/{h['total_tasks']}  "
            f"docker_fail={h['docker_start_failures']}"
        )
        if rep.get("mlflow_metrics"):
            print(f"  mlflow: {json.dumps(rep['mlflow_metrics'])}")
        if rep.get("tool_totals"):
            top = list(rep["tool_totals"].items())[:6]
            print(f"  top tools: {top}")
        if rep.get("tasks_missing_session"):
            print(f"  missing session ({len(rep['tasks_missing_session'])}): "
                  f"{rep['tasks_missing_session'][:5]}{'...' if len(rep['tasks_missing_session']) > 5 else ''}")
        print()

    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(reports, indent=2), encoding="utf-8")
        print(f"Wrote {args.json_out}")

    if any("error" in r for r in reports):
        sys.exit(1)


if __name__ == "__main__":
    main()
from __future__ import annotations

import argparse
import json
import logging
import statistics
from pathlib import Path
from typing import Any

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-7s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def load_grading_results(grade_summary_path: Path) -> dict[str, bool]:
    data = json.loads(grade_summary_path.read_text(encoding="utf-8"))
    resolved = {}
    for tid in data.get("resolved_ids", []):
        resolved[tid] = True
    for tid in data.get("failed_ids", []):
        resolved[tid] = False
    return resolved


def load_per_instance_reports(
    reports_dir: Path,
    regression_weight: float = 0.3,
) -> dict[str, dict[str, Any]]:
    results: dict[str, dict[str, Any]] = {}
    if not reports_dir.exists():
        return results

    for report_path in reports_dir.rglob("report.json"):
        try:
            report = json.loads(report_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue

        for task_id, info in report.items():
            ts = info.get("tests_status", {})
            f2p = ts.get("FAIL_TO_PASS", {})
            p2p = ts.get("PASS_TO_PASS", {})

            f2p_ok = len(f2p.get("success", []))
            f2p_no = len(f2p.get("failure", []))
            p2p_ok = len(p2p.get("success", []))
            p2p_no = len(p2p.get("failure", []))

            f2p_total = f2p_ok + f2p_no
            p2p_total = p2p_ok + p2p_no

            f2p_pass_rate = f2p_ok / f2p_total if f2p_total > 0 else 0.0
            p2p_reg_rate = p2p_no / p2p_total if p2p_total > 0 else 0.0

            soft_quality = (
                (1 - regression_weight) * f2p_pass_rate
                + regression_weight * (1 - p2p_reg_rate)
            )

            results[task_id] = {
                "resolved": info.get("resolved", False),
                "f2p_pass_rate": f2p_pass_rate,
                "p2p_regression_rate": p2p_reg_rate,
                "soft_quality": soft_quality,
                "soft_failure_label": 1.0 - soft_quality,
                "f2p_total": f2p_total,
                "p2p_total": p2p_total,
                "patch_applied": info.get("patch_successfully_applied", False),
            }

    return results


def load_session(session_path: Path) -> dict[str, Any]:
    return json.loads(session_path.read_text(encoding="utf-8"))


def extract_turn_logprobs(session: dict[str, Any]) -> list[dict[str, Any]]:
    results = []
    for turn in session.get("turns", []):
        tool_calls = turn.get("tool_calls", [])
        if not tool_calls:
            continue

        tool_names = [tc.get("name", "?") for tc in tool_calls]
        has_edit = any(
            tc.get("name") == "str_replace_editor"
            and tc.get("arguments", {}).get("command") in ("str_replace", "create", "insert")
            for tc in tool_calls
        )

        results.append({
            "turn_index": turn.get("turn_index", 0),
            "has_edit": has_edit,
            "tool_names": tool_names,
            "halt_score": turn.get("halt_score"),
        })
    return results


def compute_auroc(labels: list[int], scores: list[float]) -> float:
    try:
        from sklearn.metrics import roc_auc_score
        return float(roc_auc_score(labels, scores))
    except (ValueError, ImportError):
        return 0.0


def compute_baseline_mean_logprob(session: dict[str, Any]) -> float | None:
    scores = [
        t.get("halt_score")
        for t in session.get("turns", [])
        if t.get("halt_score") is not None
    ]
    if not scores:
        return None
    return statistics.mean(scores)


def analyze_tasks(
    results_dir: Path,
    grading: dict[str, bool],
    soft_labels: dict[str, dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    soft_labels = soft_labels or {}
    records = []
    for task_dir in sorted(results_dir.iterdir()):
        if not task_dir.is_dir():
            continue
        session_path = task_dir / "agent_output" / "session.json"
        if not session_path.exists():
            continue

        task_id = task_dir.name
        resolved = grading.get(task_id)
        if resolved is None:
            continue

        session = load_session(session_path)
        status = session.get("status", "?")
        halt_submit = session.get("halt_submit_score")
        halt_mean = session.get("halt_mean_score")
        halt_max = session.get("halt_max_score")

        turn_data = extract_turn_logprobs(session)
        edit_turns = [t for t in turn_data if t["has_edit"]]
        edit_halt_scores = [t["halt_score"] for t in edit_turns if t["halt_score"] is not None]

        soft = soft_labels.get(task_id, {})

        records.append({
            "task_id": task_id,
            "resolved": resolved,
            "status": status,
            "n_turns": len(session.get("turns", [])),
            "n_edit_turns": len(edit_turns),
            "halt_submit_score": halt_submit,
            "halt_mean_score": halt_mean,
            "halt_max_score": halt_max,
            "edit_halt_scores": edit_halt_scores,
            "edit_halt_mean": statistics.mean(edit_halt_scores) if edit_halt_scores else None,
            "edit_halt_max": max(edit_halt_scores) if edit_halt_scores else None,
            "edit_halt_min": min(edit_halt_scores) if edit_halt_scores else None,
            "f2p_pass_rate": soft.get("f2p_pass_rate"),
            "p2p_regression_rate": soft.get("p2p_regression_rate"),
            "soft_quality": soft.get("soft_quality"),
            "soft_failure_label": soft.get("soft_failure_label"),
        })

    return records


def run_ablation(records: list[dict[str, Any]]) -> dict[str, Any]:
    strategies = {
        "submit_score_only": [],
        "mean_all_turns": [],
        "edit_turns_mean": [],
        "edit_turns_max": [],
        "edit_turns_min": [],
    }
    labels: list[int] = []

    for r in records:
        label = 1 if r["resolved"] else 0

        submit = r.get("halt_submit_score")
        mean_all = r.get("halt_mean_score")
        edit_mean = r.get("edit_halt_mean")
        edit_max = r.get("edit_halt_max")
        edit_min = r.get("edit_halt_min")

        if submit is None and edit_mean is None:
            continue

        labels.append(label)
        strategies["submit_score_only"].append(submit if submit is not None else 0.5)
        strategies["mean_all_turns"].append(mean_all if mean_all is not None else 0.5)
        strategies["edit_turns_mean"].append(edit_mean if edit_mean is not None else 0.5)
        strategies["edit_turns_max"].append(edit_max if edit_max is not None else 0.5)
        strategies["edit_turns_min"].append(edit_min if edit_min is not None else 0.5)

    results = {"n_tasks": len(labels), "n_resolved": sum(labels)}
    for name, scores in strategies.items():
        auroc = compute_auroc(labels, scores)
        score_vals = [s for s in scores if s is not None]
        results[name] = {
            "auroc": auroc,
            "mean": statistics.mean(score_vals) if score_vals else None,
            "stdev": statistics.stdev(score_vals) if len(score_vals) > 1 else None,
            "min": min(score_vals) if score_vals else None,
            "max": max(score_vals) if score_vals else None,
        }

    mean_logprob_scores = []
    mean_logprob_labels = []
    for r in records:
        edit_scores = r.get("edit_halt_scores", [])
        if not edit_scores:
            continue
        mean_logprob_scores.append(1.0 - statistics.mean(edit_scores))
        mean_logprob_labels.append(1 if r["resolved"] else 0)

    if mean_logprob_labels:
        results["inverted_edit_mean_baseline"] = {
            "auroc": compute_auroc(mean_logprob_labels, mean_logprob_scores),
            "note": "1 - halt_score (lower hallucination prob = higher quality)",
        }

    return results


def compute_mean_logprob_baseline(
    records: list[dict[str, Any]],
    results_dir: Path,
) -> dict[str, Any]:
    labels = []
    scores = []
    for r in records:
        task_id = r["task_id"]
        resolved = r["resolved"]
        logprobs_path = results_dir / task_id / "agent_output" / "edit_logprobs.jsonl"
        if not logprobs_path.exists():
            continue

        turn_means = []
        for line in logprobs_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            entry = json.loads(line)
            lps = entry.get("logprobs", [])
            if lps:
                top1_mean = sum(tok[0] for tok in lps if tok) / len(lps)
                turn_means.append(top1_mean)

        if not turn_means:
            continue

        task_mean_logprob = sum(turn_means) / len(turn_means)
        labels.append(1 if resolved else 0)
        scores.append(task_mean_logprob)

    if len(set(labels)) < 2:
        return {"auroc": 0.0, "n": len(labels), "note": "single class"}

    auroc = compute_auroc(labels, scores)
    return {
        "auroc": auroc,
        "n": len(labels),
        "mean_score_resolved": (
            sum(s for s, l in zip(scores, labels) if l == 1) /
            max(sum(labels), 1)
        ),
        "mean_score_failed": (
            sum(s for s, l in zip(scores, labels) if l == 0) /
            max(len(labels) - sum(labels), 1)
        ),
        "note": "higher logprob = more confident = more likely correct",
    }


def extract_training_data(
    records: list[dict[str, Any]],
    results_dir: Path,
) -> list[dict[str, Any]]:
    training_records = []
    for r in records:
        task_id = r["task_id"]
        resolved = r["resolved"]
        soft_label = r.get("soft_failure_label")
        if soft_label is None:
            soft_label = 0.0 if resolved else 1.0
        binary_label = 0 if resolved else 1

        logprobs_path = results_dir / task_id / "agent_output" / "edit_logprobs.jsonl"
        if logprobs_path.exists():
            for line in logprobs_path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                entry = json.loads(line)
                training_records.append({
                    "task_id": task_id,
                    "turn_index": entry["turn_index"],
                    "label": soft_label,
                    "binary_label": binary_label,
                    "logprobs": entry["logprobs"],
                    "n_tokens": entry["n_tokens"],
                })
            continue

        session_path = results_dir / task_id / "agent_output" / "session.json"
        if not session_path.exists():
            continue

        session = load_session(session_path)
        for turn in session.get("turns", []):
            tool_calls = turn.get("tool_calls", [])
            has_edit = any(
                tc.get("name") == "str_replace_editor"
                and tc.get("arguments", {}).get("command") in ("str_replace", "create", "insert")
                for tc in tool_calls
            )
            if not has_edit:
                continue

            training_records.append({
                "task_id": task_id,
                "turn_index": turn.get("turn_index", 0),
                "label": soft_label,
                "binary_label": binary_label,
                "halt_score": turn.get("halt_score"),
                "n_tool_calls": len(tool_calls),
                "tool_names": [tc.get("name", "?") for tc in tool_calls],
            })

    return training_records


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze HALT on SWE-bench traces")
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--grade-summary", type=Path, required=True)
    parser.add_argument(
        "--reports-dir", type=Path, default=None,
        help="Per-instance swebench report.json dir (for soft labels)",
    )
    parser.add_argument(
        "--regression-weight", type=float, default=0.3,
        help="Weight for P2P regressions vs F2P pass rate in soft label",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("experiments/halt_analysis"))
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading grading results from %s", args.grade_summary)
    grading = load_grading_results(args.grade_summary)
    logger.info("Grading: %d resolved, %d failed", sum(grading.values()), sum(1 for v in grading.values() if not v))

    soft_labels: dict[str, dict[str, Any]] = {}
    if args.reports_dir is not None:
        logger.info("Loading per-instance reports from %s", args.reports_dir)
        soft_labels = load_per_instance_reports(args.reports_dir, args.regression_weight)
        logger.info("Loaded soft labels for %d tasks", len(soft_labels))

    logger.info("Analyzing tasks in %s", args.results_dir)
    records = analyze_tasks(args.results_dir, grading, soft_labels)
    logger.info("Analyzed %d tasks with grading labels", len(records))

    print("\n" + "=" * 70)
    print("HALT Score Distribution by Outcome")
    print("=" * 70)
    for outcome, label in [("RESOLVED", True), ("FAILED", False)]:
        subset = [r for r in records if r["resolved"] == label]
        submit_scores = [r["halt_submit_score"] for r in subset if r["halt_submit_score"] is not None]
        edit_scores = [s for r in subset for s in r.get("edit_halt_scores", [])]
        print(f"\n  {outcome} ({len(subset)} tasks):")
        if submit_scores:
            print(f"    Submit score: mean={statistics.mean(submit_scores):.4f}, "
                  f"stdev={statistics.stdev(submit_scores) if len(submit_scores) > 1 else 0:.4f}, "
                  f"range=[{min(submit_scores):.4f}, {max(submit_scores):.4f}]")
        if edit_scores:
            print(f"    Edit scores:  mean={statistics.mean(edit_scores):.4f}, "
                  f"stdev={statistics.stdev(edit_scores) if len(edit_scores) > 1 else 0:.4f}, "
                  f"range=[{min(edit_scores):.4f}, {max(edit_scores):.4f}], n={len(edit_scores)}")
        else:
            print("    Edit scores:  NONE (edit turns not scored in this run)")

    print("\n" + "=" * 70)
    print("Ablation: AUROC by Scoring Strategy")
    print("=" * 70)
    ablation = run_ablation(records)
    print(f"\n  Tasks: {ablation['n_tasks']} ({ablation['n_resolved']} resolved)")
    for key in ["submit_score_only", "mean_all_turns", "edit_turns_mean", "edit_turns_max", "edit_turns_min"]:
        if key in ablation:
            info = ablation[key]
            print(f"  {key:25s}: AUROC={info['auroc']:.4f}  "
                  f"(mean={info['mean']:.4f}, stdev={info['stdev'] or 0:.4f})")
    if "inverted_edit_mean_baseline" in ablation:
        info = ablation["inverted_edit_mean_baseline"]
        print(f"  {'1-halt (baseline)':25s}: AUROC={info['auroc']:.4f}")

    soft_records = [r for r in records if r.get("soft_failure_label") is not None]
    if soft_records:
        print("\n" + "=" * 70)
        print("Soft Label Distribution (F2P pass rate + P2P regression penalty)")
        print("=" * 70)
        for outcome, label in [("RESOLVED", True), ("FAILED", False)]:
            subset = [r for r in soft_records if r["resolved"] == label]
            f2p_rates = [r["f2p_pass_rate"] for r in subset]
            soft_lbls = [r["soft_failure_label"] for r in subset]
            print(f"\n  {outcome} ({len(subset)} tasks):")
            if f2p_rates:
                print(f"    F2P pass rate: mean={statistics.mean(f2p_rates):.4f}, "
                      f"min={min(f2p_rates):.4f}, max={max(f2p_rates):.4f}")
                print(f"    Soft label:    mean={statistics.mean(soft_lbls):.4f}, "
                      f"min={min(soft_lbls):.4f}, max={max(soft_lbls):.4f}")

    print("\n" + "=" * 70)
    print("Mean Logprob Baseline (no HALT model)")
    print("=" * 70)
    logprob_baseline = compute_mean_logprob_baseline(records, args.results_dir)
    if logprob_baseline["n"] > 0:
        print(f"  AUROC: {logprob_baseline['auroc']:.4f} (n={logprob_baseline['n']})")
        if "mean_score_resolved" in logprob_baseline:
            print(f"  Mean logprob (resolved): {logprob_baseline['mean_score_resolved']:.4f}")
            print(f"  Mean logprob (failed):   {logprob_baseline['mean_score_failed']:.4f}")
    else:
        print("  No edit_logprobs.jsonl files found -- needs eval with fixed runner")

    (args.output_dir / "logprob_baseline.json").write_text(
        json.dumps(logprob_baseline, indent=2, default=str), encoding="utf-8",
    )

    logger.info("Extracting training data for domain-adaptive fine-tuning")
    training_data = extract_training_data(records, args.results_dir)
    n_pos = sum(1 for t in training_data if t["label"] == 0)
    n_neg = sum(1 for t in training_data if t["label"] == 1)
    logger.info("Training data: %d records (%d correct / %d hallucinated)", len(training_data), n_pos, n_neg)

    (args.output_dir / "task_analysis.json").write_text(
        json.dumps(records, indent=2), encoding="utf-8",
    )
    (args.output_dir / "ablation.json").write_text(
        json.dumps(ablation, indent=2, default=str), encoding="utf-8",
    )
    (args.output_dir / "training_data.jsonl").write_text(
        "\n".join(json.dumps(r) for r in training_data) + "\n",
        encoding="utf-8",
    )

    print(f"\nResults saved to {args.output_dir}/")
    print(f"  task_analysis.json  - per-task HALT scores + grading labels")
    print(f"  ablation.json       - AUROC by scoring strategy")
    print(f"  training_data.jsonl - {len(training_data)} labeled records for fine-tuning")


if __name__ == "__main__":
    main()

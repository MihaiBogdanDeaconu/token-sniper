from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Grade SWE-bench patches")
    parser.add_argument(
        "--results-dir",
        default="/workspace/repo/experiments/halt_swebench",
    )
    parser.add_argument(
        "--dataset",
        default="princeton-nlp/SWE-bench_Verified",
    )
    parser.add_argument("--split", default="test")
    parser.add_argument("--max-workers", type=int, default=4)
    parser.add_argument("--run-id", default="grade-run")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)

    task_patches: dict[str, str] = {}
    for d in sorted(os.listdir(results_dir)):
        patch_path = results_dir / d / "agent_output" / "patch.diff"
        if patch_path.is_file() and patch_path.stat().st_size > 0:
            task_patches[d] = patch_path.read_text()

    print(f"Found {len(task_patches)} patches to grade")
    if not task_patches:
        print("Nothing to grade.")
        return

    predictions_path = results_dir / "predictions.jsonl"
    with open(predictions_path, "w") as f:
        for tid, patch_text in task_patches.items():
            f.write(json.dumps({
                "instance_id": tid,
                "model_patch": patch_text,
                "model_name_or_path": "token-sniper-agent",
            }) + "\n")
    print(f"Wrote predictions to {predictions_path}")

    instance_ids = list(task_patches.keys())

    print("\n=== Running swebench evaluation ===")
    cmd = [
        sys.executable, "-m", "swebench.harness.run_evaluation",
        "--dataset_name", args.dataset,
        "--split", args.split,
        "--predictions_path", str(predictions_path),
        "--run_id", args.run_id,
        "--max_workers", str(args.max_workers),
        "--timeout", "300",
        "--instance_ids", *instance_ids,
    ]
    print(f"Command: {' '.join(cmd)}")
    result = subprocess.run(cmd, check=False)
    print(f"swebench exit code: {result.returncode}")

    print("\n" + "=" * 60)
    print("GRADING RESULTS")
    print("=" * 60)

    resolved = []
    failed = []
    no_report = []

    consolidated = Path(f"token-sniper-agent.{args.run_id}.json")
    log_dir = Path(f"logs/run_evaluation/{args.run_id}")

    resolved_set: set[str] = set()
    unresolved_set: set[str] = set()

    if consolidated.is_file():
        print(f"Reading consolidated report: {consolidated}")
        report = json.loads(consolidated.read_text())
        resolved_set = set(report.get("resolved_ids", []))
        unresolved_set = set(report.get("unresolved_ids", []))
        print(f"  resolved_instances={report.get('resolved_instances', '?')}")
        print(f"  unresolved_instances={report.get('unresolved_instances', '?')}")
        print(f"  error_instances={report.get('error_instances', '?')}")
    else:
        print(f"No consolidated report, checking per-instance logs in {log_dir}")
        for tid in sorted(task_patches.keys()):
            for report_path in log_dir.rglob(f"{tid}/report.json"):
                per = json.loads(report_path.read_text())
                if per.get(tid, {}).get("resolved", False):
                    resolved_set.add(tid)
                else:
                    unresolved_set.add(tid)
                break

    for tid in sorted(task_patches.keys()):
        if tid in resolved_set:
            resolved.append(tid)
        elif tid in unresolved_set:
            failed.append(tid)
        else:
            no_report.append(tid)

    for tid in resolved:
        print(f"  RESOLVED: {tid}")
    for tid in failed:
        print(f"  FAILED:   {tid}")
    for tid in no_report:
        print(f"  NO REPORT: {tid}")

    total_graded = len(resolved) + len(failed)
    print()
    print(f"Total patches:   {len(task_patches)}")
    print(f"Total graded:    {total_graded}")
    print(f"No report:       {len(no_report)}")
    print(f"Resolved:        {len(resolved)}")
    if total_graded:
        print(f"Resolution rate: {100*len(resolved)/total_graded:.1f}% (of graded)")
    print(f"Overall rate:    {100*len(resolved)/48:.1f}% (of 48 tasks)")

    grade_summary = {
        "total_tasks": 48,
        "total_patches": len(task_patches),
        "total_graded": total_graded,
        "no_report": len(no_report),
        "resolved": len(resolved),
        "resolved_ids": resolved,
        "failed_ids": failed,
        "no_report_ids": no_report,
    }
    summary_path = results_dir / "grade_summary.json"
    summary_path.write_text(json.dumps(grade_summary, indent=2))
    print(f"\nSummary saved to {summary_path}")


if __name__ == "__main__":
    main()

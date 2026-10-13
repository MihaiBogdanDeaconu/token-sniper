from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np


def load_inputs(path: str) -> list[dict]:
    with open(path) as f:
        return json.load(f)


def load_issue_texts(manifest_path: str) -> dict[str, str]:
    try:
        with open(manifest_path) as f:
            manifest = json.load(f)
        texts = {}
        for task in manifest:
            iid = task.get("instance_id", "")
            ps = task.get("problem_statement", "")
            if iid and ps:
                texts[iid] = ps
        return texts
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def file_name_mentioned(file_path: str, text: str) -> bool:
    stem = Path(file_path).stem
    if len(stem) < 3 or stem in ("__init__", "utils", "base", "core", "main", "test", "tests", "conf", "setup"):
        return False
    return stem.lower() in text.lower()


def extract_traceback_files(text: str) -> set[str]:
    patterns = [
        r'File "([^"]+\.py)"',
        r"File '([^']+\.py)'",
        r'(\S+\.py):\d+',
    ]
    files = set()
    for pat in patterns:
        for m in re.findall(pat, text):
            files.add(Path(m).name)
    return files


def analyze(inputs: list[dict], issue_texts: dict[str, str]) -> dict:
    k_values = [5, 10, 20, 50, 100, 200]
    recalls = {k: [] for k in k_values}

    total_gt_files = 0
    total_gt_found_in_200 = 0
    tasks_with_zero_gt = 0
    missing_files: list[dict] = []

    name_mention_total = 0
    name_mention_hits = 0
    name_could_rescue = 0
    tb_total = 0
    tb_hits = 0

    gt_per_task: list[int] = []
    candidates_per_task: list[int] = []

    for inp in inputs:
        gold = set(inp["gold_files"])
        candidates = inp["candidate_files"]
        iid = inp["instance_id"]

        if not gold:
            continue

        gt_per_task.append(len(gold))
        candidates_per_task.append(len(candidates))

        cand_set = set(candidates)
        found = gold & cand_set
        miss = gold - cand_set

        total_gt_files += len(gold)
        total_gt_found_in_200 += len(found)

        if not found:
            tasks_with_zero_gt += 1

        for f in miss:
            missing_files.append({"task": iid, "file": f})

        for k in k_values:
            top_k = set(candidates[:k])
            r = len(gold & top_k) / len(gold)
            recalls[k].append(r)

        issue = issue_texts.get(iid, "")
        if issue:
            for f in gold:
                name_mention_total += 1
                if file_name_mentioned(f, issue):
                    name_mention_hits += 1
                    if f in miss:
                        name_could_rescue += 1

            tb_files = extract_traceback_files(issue)
            if tb_files:
                for f in gold:
                    tb_total += 1
                    if Path(f).name in tb_files:
                        tb_hits += 1

    mean_recalls = {k: float(np.mean(v)) for k, v in recalls.items()}
    n_missing = total_gt_files - total_gt_found_in_200

    return {
        "recall_at_k": mean_recalls,
        "n_tasks": len(inputs),
        "mean_gt_files_per_task": float(np.mean(gt_per_task)),
        "mean_candidates_per_task": float(np.mean(candidates_per_task)),
        "total_gt_files": total_gt_files,
        "gt_in_top200": total_gt_found_in_200,
        "gt_missing_from_top200": n_missing,
        "pct_missing": round(100 * n_missing / max(total_gt_files, 1), 1),
        "tasks_with_zero_gt_in_candidates": tasks_with_zero_gt,
        "text_signal": {
            "file_name_mentioned_in_issue": name_mention_hits,
            "total_gt_files_checked": name_mention_total,
            "pct_mentioned": round(100 * name_mention_hits / max(name_mention_total, 1), 1),
            "missing_files_rescuable_by_name": name_could_rescue,
        },
        "traceback_signal": {
            "gt_files_in_traceback": tb_hits,
            "total_checked": tb_total,
            "pct": round(100 * tb_hits / max(tb_total, 1), 1),
        },
        "sample_missing": missing_files[:30],
    }


def main():
    inp_path = sys.argv[1] if len(sys.argv) > 1 else "experiments/reranker/val_inputs.json"
    manifest_path = "configs/phases/phase0_tasks.json"

    inputs = load_inputs(inp_path)
    issue_texts = load_issue_texts(manifest_path)
    print(f"Loaded {len(inputs)} tasks, {len(issue_texts)} issue texts")

    results = analyze(inputs, issue_texts)

    print("\n========== RECALL CEILING ==========")
    for k, r in sorted(results["recall_at_k"].items()):
        filled = int(r * 40)
        bar = "#" * filled + "." * (40 - filled)
        print(f"  R@{k:>3d}: {r:.4f}  [{bar}]")

    print(f"\n========== GROUND TRUTH COVERAGE ==========")
    print(f"  Tasks: {results['n_tasks']}")
    print(f"  Mean GT files/task: {results['mean_gt_files_per_task']:.1f}")
    print(f"  Mean candidates/task: {results['mean_candidates_per_task']:.0f}")
    print(f"  GT files total: {results['total_gt_files']}")
    print(f"  GT in top-200: {results['gt_in_top200']} ({100 - results['pct_missing']:.1f}%)")
    print(f"  GT MISSING from top-200: {results['gt_missing_from_top200']} ({results['pct_missing']}%)")
    print(f"  Tasks with ZERO GT in candidates: {results['tasks_with_zero_gt_in_candidates']}")

    ts = results["text_signal"]
    print(f"\n========== TEXT FEATURE SIGNAL ==========")
    print(f"  GT file name mentioned in issue: {ts['file_name_mentioned_in_issue']}/{ts['total_gt_files_checked']} ({ts['pct_mentioned']}%)")
    print(f"  Missing files rescuable by name match: {ts['missing_files_rescuable_by_name']}")

    tb = results["traceback_signal"]
    print(f"  GT in traceback: {tb['gt_files_in_traceback']}/{tb['total_checked']} ({tb['pct']}%)")

    if results["sample_missing"]:
        print(f"\n========== SAMPLE MISSING FILES ==========")
        for m in results["sample_missing"][:15]:
            print(f"  {m['task']}: {m['file']}")

    out = Path(inp_path).parent / "recall_ceiling_analysis.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {out}")


if __name__ == "__main__":
    main()

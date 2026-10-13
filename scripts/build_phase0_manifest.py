from __future__ import annotations

import argparse
import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Any

from token_sniper.evaluation.phase0 import Phase0Task
from token_sniper.evaluation.swebench import _load_dataset_rows

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DEFAULT_DATASET = "SWE-bench-Live/SWE-bench-Live"
DEFAULT_SPLIT = "verified"

BUCKET_TARGETS = {
    "single_file": 20,
    "multi_file_2_3": 20,
    "multi_file_4_plus": 10,
}
BUCKET_ORDER = ("single_file", "multi_file_2_3", "multi_file_4_plus")
MIN_TASKS_PER_REPO = 3


def extract_patched_files(patch_text: str) -> list[str]:
    files = set()
    for line in patch_text.splitlines():
        if line.startswith("+++ b/"):
            files.add(line[6:].strip())
    return sorted(files)


def assign_bucket(file_count: int) -> str:
    if file_count <= 1:
        return "single_file"
    if 2 <= file_count <= 3:
        return "multi_file_2_3"
    return "multi_file_4_plus"


def build_phase0_manifest_tasks(
    dataset_rows: dict[str, dict[str, Any]],
    *,
    targets: dict[str, int] | None = None,
    min_tasks_per_repo: int = MIN_TASKS_PER_REPO,
    max_tasks: int = 0,
) -> list[Phase0Task]:
    bucket_targets = dict(targets or BUCKET_TARGETS)
    instances = sorted(dataset_rows.values(), key=lambda row: str(row["instance_id"]))

    buckets: dict[str, list[dict[str, Any]]] = {bucket: [] for bucket in BUCKET_ORDER}
    for task in instances:
        patch_text = str(task.get("patch") or "")
        files = extract_patched_files(patch_text)
        if not files:
            continue

        bucket = assign_bucket(len(files))
        buckets[bucket].append(
            {
                "instance_id": str(task["instance_id"]),
                "repo": str(task["repo"]),
                "base_commit": str(task["base_commit"]),
                "difficulty": str(task.get("difficulty") or "") or None,
                "gold_patch_files": files,
                "gold_patch_file_count": len(files),
                "bucket": bucket,
            }
        )

    if max_tasks <= 0:
        all_records: list[dict[str, Any]] = []
        for bucket_name in BUCKET_ORDER:
            all_records.extend(buckets.get(bucket_name, []))
        all_records.sort(key=lambda r: str(r["instance_id"]))
        return [Phase0Task(**record) for record in all_records]

    selected_records = _select_tasks_with_repo_minimum(
        buckets=buckets,
        targets=bucket_targets,
        min_tasks_per_repo=min_tasks_per_repo,
    )
    return [Phase0Task(**record) for record in selected_records]


def _select_tasks_with_repo_minimum(
    *,
    buckets: dict[str, list[dict[str, Any]]],
    targets: dict[str, int],
    min_tasks_per_repo: int,
) -> list[dict[str, Any]]:
    repo_tasks: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for bucket_name in BUCKET_ORDER:
        for task in buckets.get(bucket_name, []):
            repo_tasks[task["repo"]].append(task)

    bucket_remaining = dict(targets)
    selected_ids: set[str] = set()
    selected_tasks: list[dict[str, Any]] = []
    repo_selection_counts: dict[str, int] = defaultdict(int)

    eligible_repos = sorted(
        repo_name for repo_name, tasks in repo_tasks.items() if len(tasks) >= min_tasks_per_repo
    )

    while True:
        progress = False
        for repo_name in eligible_repos:
            if repo_selection_counts[repo_name] >= min_tasks_per_repo:
                continue
            next_task = _take_next_repo_task(
                repo_tasks[repo_name],
                selected_ids=selected_ids,
                bucket_remaining=bucket_remaining,
            )
            if next_task is None:
                continue
            _record_selection(
                next_task,
                selected_tasks=selected_tasks,
                selected_ids=selected_ids,
                bucket_remaining=bucket_remaining,
                repo_selection_counts=repo_selection_counts,
            )
            progress = True
        if not progress:
            break

    for bucket_name in BUCKET_ORDER:
        for task in buckets.get(bucket_name, []):
            if bucket_remaining[bucket_name] <= 0:
                break
            if task["instance_id"] in selected_ids:
                continue
            _record_selection(
                task,
                selected_tasks=selected_tasks,
                selected_ids=selected_ids,
                bucket_remaining=bucket_remaining,
                repo_selection_counts=repo_selection_counts,
            )

    return sorted(selected_tasks, key=lambda task: str(task["instance_id"]))


def _take_next_repo_task(
    repo_tasks: list[dict[str, Any]],
    *,
    selected_ids: set[str],
    bucket_remaining: dict[str, int],
) -> dict[str, Any] | None:
    for task in repo_tasks:
        bucket_name = str(task["bucket"])
        if task["instance_id"] in selected_ids:
            continue
        if bucket_remaining.get(bucket_name, 0) <= 0:
            continue
        return task
    return None


def _record_selection(
    task: dict[str, Any],
    *,
    selected_tasks: list[dict[str, Any]],
    selected_ids: set[str],
    bucket_remaining: dict[str, int],
    repo_selection_counts: dict[str, int],
) -> None:
    bucket_name = str(task["bucket"])
    selected_tasks.append(task)
    selected_ids.add(str(task["instance_id"]))
    bucket_remaining[bucket_name] -= 1
    repo_selection_counts[str(task["repo"])] += 1


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a task manifest from SWE-bench")
    parser.add_argument("--output", type=Path, default=Path("configs/phases/phase0_tasks.json"))
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--split", default=DEFAULT_SPLIT)
    parser.add_argument(
        "--max-tasks",
        type=int,
        default=0,
        help="Max tasks to select (0 = include all tasks with valid gold patches)",
    )
    args = parser.parse_args()

    logger.info("Loading dataset %s split %s", args.dataset, args.split)
    dataset = _load_dataset_rows(args.dataset, args.split)
    manifest_tasks = build_phase0_manifest_tasks(dataset, max_tasks=args.max_tasks)

    expected = args.max_tasks if args.max_tasks > 0 else len(manifest_tasks)
    if len(manifest_tasks) != expected and args.max_tasks > 0:
        logger.warning(
            "Could only select %s tasks instead of the requested %s.",
            len(manifest_tasks),
            expected,
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as file_handle:
        json.dump([task.model_dump() for task in manifest_tasks], file_handle, indent=2)
        file_handle.write("\n")

    logger.info("Wrote %s tasks to %s", len(manifest_tasks), args.output)


if __name__ == "__main__":
    main()

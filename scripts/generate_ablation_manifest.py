from __future__ import annotations

import argparse
import json
import math
import random
from collections import defaultdict
from pathlib import Path

from token_sniper.evaluation.phase0 import Phase0Task, load_phase0_manifest

SEED = 42
TRAIN_RATIO = 0.8
TARGET_SIZE = 50
DEFAULT_MANIFEST_PATH = Path("configs/phases/phase0_tasks.json")
DEFAULT_VAL_INPUTS_PATH = Path("experiments/reranker/val_inputs.json")
DEFAULT_OUTPUT_50 = Path("configs/phases/ablation_eval_50.json")
DEFAULT_OUTPUT_VAL = Path("configs/phases/ablation_eval_val.json")


def stratified_split(
    tasks: list[Phase0Task], *, train_ratio: float, seed: int,
) -> tuple[list[Phase0Task], list[Phase0Task]]:
    rng = random.Random(seed)
    by_repo: dict[str, list[Phase0Task]] = defaultdict(list)
    for t in tasks:
        by_repo[t.repo].append(t)

    repos = sorted(by_repo.keys())
    rng.shuffle(repos)

    target_val = int(len(tasks) * (1 - train_ratio))
    train_tasks: list[Phase0Task] = []
    val_tasks: list[Phase0Task] = []

    for repo in repos:
        if len(val_tasks) < target_val:
            val_tasks.extend(by_repo[repo])
        else:
            train_tasks.extend(by_repo[repo])

    rng.shuffle(train_tasks)
    rng.shuffle(val_tasks)
    return train_tasks, val_tasks


def stratified_sample(
    tasks: list[Phase0Task], *, target: int, seed: int,
) -> list[Phase0Task]:
    rng = random.Random(seed)

    by_bucket: dict[str, list[Phase0Task]] = defaultdict(list)
    for t in tasks:
        by_bucket[t.bucket].append(t)

    total = len(tasks)
    samples: list[Phase0Task] = []
    remaining = target

    allocations: dict[str, int] = {}
    for bucket, bucket_tasks in sorted(by_bucket.items()):
        raw = (len(bucket_tasks) / total) * target
        allocations[bucket] = max(1, math.floor(raw))

    allocated = sum(allocations.values())
    if allocated < target:
        deficit = target - allocated
        buckets_by_size = sorted(by_bucket.keys(), key=lambda b: len(by_bucket[b]), reverse=True)
        for i in range(deficit):
            allocations[buckets_by_size[i % len(buckets_by_size)]] += 1
    elif allocated > target:
        excess = allocated - target
        buckets_by_size = sorted(by_bucket.keys(), key=lambda b: len(by_bucket[b]))
        for i in range(excess):
            b = buckets_by_size[i % len(buckets_by_size)]
            if allocations[b] > 1:
                allocations[b] -= 1

    for bucket, count in sorted(allocations.items()):
        pool = by_bucket[bucket]
        rng.shuffle(pool)
        count = min(count, len(pool))
        samples.extend(pool[:count])

    rng.shuffle(samples)
    return samples


def load_val_input_ids(path: Path) -> list[str]:
    records = json.loads(path.read_text(encoding="utf-8"))
    return [record["instance_id"] for record in records]


def tasks_from_val_inputs(
    *,
    manifest_path: Path,
    val_inputs_path: Path,
) -> tuple[list[Phase0Task], list[str]]:
    val_input_ids = load_val_input_ids(val_inputs_path)
    manifest = load_phase0_manifest(manifest_path)
    by_id = {task.instance_id: task for task in manifest.tasks}

    selected: list[Phase0Task] = []
    missing: list[str] = []
    for instance_id in val_input_ids:
        task = by_id.get(instance_id)
        if task is None:
            missing.append(instance_id)
            continue
        selected.append(task)

    if missing:
        raise SystemExit(
            f"{len(missing)} val_inputs tasks missing from {manifest_path}: "
            + ", ".join(missing[:5])
            + (" ..." if len(missing) > 5 else ""),
        )
    return selected, missing


def print_bucket_distribution(tasks: list[Phase0Task], *, label: str) -> None:
    by_bucket: dict[str, list[Phase0Task]] = defaultdict(list)
    for task in tasks:
        by_bucket[task.bucket].append(task)
    print(f"{label} bucket distribution:")
    for bucket, bucket_tasks in sorted(by_bucket.items()):
        print(f"  {bucket}: {len(bucket_tasks)}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--full-val",
        action="store_true",
        help="Emit all tasks from val_inputs.json instead of a 50-task subset",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=DEFAULT_MANIFEST_PATH,
        help="Phase 0 task metadata source",
    )
    parser.add_argument(
        "--val-inputs",
        type=Path,
        default=DEFAULT_VAL_INPUTS_PATH,
        help="Precomputed reranker val_inputs.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output manifest path (default: ablation_eval_val.json or ablation_eval_50.json)",
    )
    args = parser.parse_args()

    output_path = args.output or (
        DEFAULT_OUTPUT_VAL if args.full_val else DEFAULT_OUTPUT_50
    )

    if not args.val_inputs.is_file():
        raise SystemExit(f"val_inputs not found: {args.val_inputs}")

    if args.full_val:
        selected, _ = tasks_from_val_inputs(
            manifest_path=args.manifest,
            val_inputs_path=args.val_inputs,
        )
        print(f"Full val manifest from {args.val_inputs}: {len(selected)} tasks")
        print_bucket_distribution(selected, label="Val")
        print(f"Repos covered: {len({task.repo for task in selected})}")
    else:
        manifest = load_phase0_manifest(args.manifest)
        print(f"Full manifest: {len(manifest.tasks)} tasks")

        _, val_tasks = stratified_split(manifest.tasks, train_ratio=TRAIN_RATIO, seed=SEED)
        print(f"Val split: {len(val_tasks)} tasks")

        val_input_ids = set(load_val_input_ids(args.val_inputs))
        before = len(val_tasks)
        val_tasks = [task for task in val_tasks if task.instance_id in val_input_ids]
        print(f"Filtered to tasks with precomputed features: {before} -> {len(val_tasks)}")

        print_bucket_distribution(val_tasks, label="Val")
        selected = stratified_sample(val_tasks, target=TARGET_SIZE, seed=SEED)
        print(f"\nSelected {len(selected)} tasks")
        print_bucket_distribution(selected, label="Subset")
        print(f"Repos covered: {len({task.repo for task in selected})}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps([task.model_dump() for task in selected], indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"\nManifest written to {output_path}")


if __name__ == "__main__":
    main()

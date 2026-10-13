from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from scripts.eval_retention import retention_header
from token_sniper.evaluation.eval_stages import (
    KEEP_THRESHOLD,
    candidates_path,
    rank_within_files,
    read_record,
    retention_path,
    selection,
    sha256_file,
    summarize_selection,
)
from token_sniper.evaluation.locagent_gold import build_gold_index
from token_sniper.evaluation.locagent_scoring import (
    LITE_K_GROUPS,
    LOCBENCH_K_GROUPS,
    score_predictions,
)
from token_sniper.evaluation.phase0 import load_phase0_manifest

logger = logging.getLogger(__name__)

K_GROUPS = {"lite_274": LITE_K_GROUPS, "locbench_560": LOCBENCH_K_GROUPS}


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-8s %(name)s — %(message)s",
                        datefmt="%H:%M:%S")
    parser = argparse.ArgumentParser(description="Eval stage C: LocAgent scoring and selection metrics")
    parser.add_argument("--dataset-key", choices=sorted(K_GROUPS), required=True,
                        help="Only the LocAgent benchmarks have edit_functions gold to score against")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--cache-name", default=None,
                        help="Cache namespace under --cache-root (default: the dataset key)")
    parser.add_argument("--reranker-model", type=Path, required=True)
    parser.add_argument("--checkpoint", action="append", required=True, metavar="NAME=PATH")
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    cache = args.cache_root / (args.cache_name or args.dataset_key)
    out = args.output_dir
    (out / "predictions").mkdir(parents=True, exist_ok=True)
    tasks = load_phase0_manifest(args.manifest).tasks
    gold = build_gold_index(args.dataset_key)
    reranker_sha = sha256_file(args.reranker_model)
    checkpoints = {name: sha256_file(Path(path)) for name, path in (c.split("=", 1) for c in args.checkpoint)}

    writers = {name: (out / "predictions" / f"{name}.jsonl").open("w", encoding="utf-8")
               for name in ["reranker", *checkpoints]}
    selection_rows: dict[str, list[dict]] = {name: [] for name in checkpoints}
    indexed: list[str] = []
    excluded: dict[str, str] = {}

    for task in tasks:
        cpath = candidates_path(cache, task.instance_id)
        _, cand = read_record(cpath)
        if "excluded" in cand:
            excluded[task.instance_id] = cand["excluded"]
            continue
        indexed.append(task.instance_id)
        candidates_sha = sha256_file(cpath)

        ranked_files = None
        for name, ckpt_sha in checkpoints.items():
            _, rec = read_record(
                retention_path(cache, ckpt_sha, task.instance_id),
                retention_header(ckpt_sha, reranker_sha, candidates_sha, args.top_k),
            )
            ranked_files = [fp for fp, _ in rec["ranked_files"]]
            modules, entities = rank_within_files(ranked_files[:args.top_k], rec["nodes"])
            writers[name].write(json.dumps({
                "instance_id": task.instance_id,
                "found_files": ranked_files,
                "found_modules": modules,
                "found_entities": entities,
            }) + "\n")
            row = selection(rec["nodes"], gold[task.instance_id], rec["pool_tokens"])
            selection_rows[name].append({"instance_id": task.instance_id, **row})
        writers["reranker"].write(json.dumps({
            "instance_id": task.instance_id, "found_files": ranked_files, "found_modules": [], "found_entities": [],
        }) + "\n")

    for handle in writers.values():
        handle.close()

    scores = {}
    for name in writers:
        path = out / "predictions" / f"{name}.jsonl"
        scores[name] = {
            "all_instances": score_predictions(
                path, dataset=args.dataset_key, k_values_list=K_GROUPS[args.dataset_key], python=Path(sys.executable),
            ),
            "indexed_only": score_predictions(
                path, dataset=args.dataset_key, k_values_list=K_GROUPS[args.dataset_key], selected_list=indexed,
                python=Path(sys.executable),
            ),
        }
        acc = {level: {k: v for k, v in metrics.items() if k.startswith("Acc@")}
               for level, metrics in scores[name]["all_instances"].items()}
        logger.info("%s (all instances): %s", name, json.dumps(acc))

    summary = {name: summarize_selection(rows) for name, rows in selection_rows.items()}
    logger.info("Selection (p_v >= %.1f): %s", KEEP_THRESHOLD, json.dumps(summary, indent=2))

    (out / "scores.json").write_text(json.dumps(scores, indent=2))
    (out / "selection.json").write_text(json.dumps({"keep_threshold": KEEP_THRESHOLD, **summary}, indent=2))
    with (out / "selection_instances.jsonl").open("w", encoding="utf-8") as handle:
        for name, rows in selection_rows.items():
            for row in rows:
                handle.write(json.dumps({"checkpoint": name, **row}) + "\n")
    (out / "excluded.json").write_text(json.dumps(excluded, indent=2))
    (out / "inputs.json").write_text(json.dumps({
        "manifest_instances": len(tasks), "indexed_instances": len(indexed), "excluded_instances": len(excluded),
        "top_k": args.top_k, "reranker_sha256": reranker_sha, "checkpoints_sha256": checkpoints,
        "cache": str(cache),
    }, indent=2))


if __name__ == "__main__":
    main()

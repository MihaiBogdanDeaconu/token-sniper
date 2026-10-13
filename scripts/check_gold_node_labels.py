from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from scripts.eval_candidates import ssg_path
from token_sniper.evaluation.locagent_gold import dataset_rows, extract_gold_labels
from token_sniper.evaluation.locagent_ranking import entity_keys
from token_sniper.evaluation.phase0 import load_phase0_manifest
from token_sniper.graph.io import read_ssg_json
from token_sniper.graph.schema import is_file_node_id
from token_sniper.vgib_selector.gold_nodes import gold_nodes, parse_patch

logger = logging.getLogger(__name__)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-8s %(name)s — %(message)s",
                        datefmt="%H:%M:%S")
    parser = argparse.ArgumentParser(description="Compare line-owner gold labels with LocAgent edit_functions")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--ssg-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    rows = {row["instance_id"]: row for row in dataset_rows("lite_274")}
    tasks = load_phase0_manifest(args.manifest).tasks
    args.output_dir.mkdir(parents=True, exist_ok=True)

    records = []
    with (args.output_dir / "instances.jsonl").open("w", encoding="utf-8") as handle:
        for n, task in enumerate(tasks, 1):
            row = rows[task.instance_id]
            path = ssg_path(args.ssg_dir, task)
            if not path.exists():
                record = {"instance_id": task.instance_id, "no_ssg": True}
            else:
                graph = read_ssg_json(path)
                labels = gold_nodes(graph, parse_patch(row["patch"]))
                keys = [entity_keys(nid, graph.nodes[nid]) for nid in labels.nodes]
                ours = {k[1] for k in keys if k is not None}
                theirs = set(extract_gold_labels(task.instance_id, row["edit_functions"]).gold_entities)
                record = {
                    "instance_id": task.instance_id,
                    "no_ssg": False,
                    "exact": ours == theirs,
                    "covers": theirs <= ours,
                    "theirs": sorted(theirs),
                    "ours": sorted(ours),
                    "missed": sorted(theirs - ours),
                    "extra": sorted(ours - theirs),
                    "file_node_labels": sorted(nid for nid in labels.nodes if is_file_node_id(nid)),
                    "missing_files": list(labels.missing_files),
                    "added_functions": row["added_functions"],
                }
            handle.write(json.dumps(record) + "\n")
            records.append(record)
            logger.info("[%d/%d] %s", n, len(tasks), task.instance_id)

    checked = [r for r in records if not r["no_ssg"]]
    ours_total = sum(len(r["ours"]) for r in checked)
    theirs_total = sum(len(r["theirs"]) for r in checked)
    hit_total = sum(len(set(r["ours"]) & set(r["theirs"])) for r in checked)
    summary = {
        "instances": len(records),
        "without_ssg": len(records) - len(checked),
        "exact_rate": round(sum(r["exact"] for r in checked) / len(checked), 4),
        "covers_rate": round(sum(r["covers"] for r in checked) / len(checked), 4),
        "micro_precision": round(hit_total / ours_total, 4),
        "micro_recall": round(hit_total / theirs_total, 4),
        "with_extra_labels": sum(bool(r["extra"]) for r in checked),
        "with_file_node_labels": sum(bool(r["file_node_labels"]) for r in checked),
        "with_missing_files": sum(bool(r["missing_files"]) for r in checked),
        "with_added_functions": sum(bool(r["added_functions"]) for r in checked),
        "extra_when_added_functions": sum(bool(r["extra"]) and bool(r["added_functions"]) for r in checked),
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    logger.info("Summary: %s", json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

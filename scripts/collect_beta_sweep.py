from __future__ import annotations

import argparse
import csv
import io
import json
import math
import statistics
import sys
from pathlib import Path
from typing import Any

import torch

from token_sniper.evaluation.eval_stages import sha256_file
from token_sniper.evaluation.telemetry import AgentRunRecord, read_records

CHECKPOINT = "vgib"
MISSING = "-"
COLUMNS = [
    "beta", "job_name", "best_epoch",
    "val_token_ratio", "val_func_at_10", "val_gold_percentile", "val_loss",
    "dataset", "instances",
    "token_retention_topk", "token_retention_pool",
    "gold_files_kept_rate", "gold_functions_kept_rate", "gold_functions_in_topk_rate",
    "function_recall", "func_acc_at_10",
    "resolve_rate", "resolve_n",
]


def _read_json(path: Path, run: dict[str, Any]) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"beta={run['beta']} ({run['job_name']}): {path} not found")
    return json.loads(path.read_text(encoding="utf-8"))


def _field(mapping: dict[str, Any], key: str, where: str) -> Any:
    if key not in mapping:
        raise KeyError(f"{where}: no {key!r} (has {sorted(mapping)})")
    return mapping[key]


def resolve_rate(records: list[AgentRunRecord], checkpoint_sha: str, beta: float) -> tuple[Any, Any]:
    mine = [r for r in records if r.selector_checkpoint_sha256 == checkpoint_sha]
    if not mine:
        return MISSING, MISSING
    where = f"telemetry for beta={beta} (checkpoint {checkpoint_sha[:12]})"
    arms = {(r.dataset, r.arm, r.model) for r in mine}
    if len(arms) != 1:
        raise ValueError(f"{where}: records from several (dataset, arm, model): {sorted(arms)}; pass one set")
    betas = {r.beta for r in mine}
    if betas != {beta}:
        raise ValueError(f"{where}: records carry beta {sorted(betas, key=str)}")
    keys = [(r.instance_id, r.seed) for r in mine]
    if len(set(keys)) != len(keys):
        raise ValueError(f"{where}: duplicate (instance_id, seed) records")
    return round(statistics.fmean(r.resolved for r in mine), 4), len(mine)


def collect_row(run: dict[str, Any], results_root: Path, dataset: str,
                records: list[AgentRunRecord]) -> tuple[dict[str, Any], dict[str, Any]]:
    beta = run["beta"]
    eval_dir = results_root / run["eval_job_name"]
    stage_c = eval_dir / dataset
    inputs = _read_json(stage_c / "inputs.json", run)
    selection = _read_json(stage_c / "selection.json", run)
    scores = _read_json(stage_c / "scores.json", run)

    ckpt_path = eval_dir / "models" / f"{CHECKPOINT}_best.pt"
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"beta={beta} ({run['job_name']}): {ckpt_path} not found")
    sha = sha256_file(ckpt_path)
    scored_sha = _field(inputs["checkpoints_sha256"], CHECKPOINT, f"{stage_c}/inputs.json")
    if sha != scored_sha:
        raise ValueError(f"beta={beta}: {ckpt_path} sha256 {sha} is not the scored checkpoint {scored_sha}")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if not math.isclose(_field(ckpt, "beta", str(ckpt_path)), beta):
        raise ValueError(f"beta={beta}: {ckpt_path} was trained with beta={ckpt['beta']}")

    sel = _field(selection, CHECKPOINT, f"{stage_c}/selection.json")
    function_acc = _field(scores, CHECKPOINT, f"{stage_c}/scores.json")["all_instances"]["function"]
    resolved, resolved_n = resolve_rate(records, sha, beta)
    row = {
        "beta": beta,
        "job_name": run["job_name"],
        "best_epoch": _field(ckpt, "epoch", str(ckpt_path)),
        "val_token_ratio": _field(ckpt, "val_token_ratio", str(ckpt_path)),
        "val_func_at_10": _field(ckpt, "best_val_func_at_10", str(ckpt_path)),
        "val_gold_percentile": _field(ckpt, "best_val_gold_percentile", str(ckpt_path)),
        "val_loss": _field(ckpt, "best_val_loss", str(ckpt_path)),
        "dataset": dataset,
        "instances": sel["instances"],
        "token_retention_topk": sel["mean_kept_ratio_of_topk"],
        "token_retention_pool": sel["mean_kept_ratio_of_pool"],
        "gold_files_kept_rate": sel["gold_files_kept_rate"],
        "gold_functions_kept_rate": sel["gold_functions_kept_rate"],
        "gold_functions_in_topk_rate": sel["gold_functions_in_topk_rate"],
        "function_recall": sel["mean_function_recall"],
        "func_acc_at_10": _field(function_acc, "Acc@10", f"{stage_c}/scores.json function"),
        "resolve_rate": resolved,
        "resolve_n": resolved_n,
    }
    comparable = {k: inputs[k] for k in ("reranker_sha256", "top_k", "manifest_instances", "cache")}
    return row, comparable


def collect(manifest: dict[str, Any], results_root: Path, dataset: str,
            records: list[AgentRunRecord]) -> list[dict[str, Any]]:
    if not manifest["runs"]:
        raise ValueError(f"sweep {manifest['sweep_id']} has no runs")
    rows, setups = [], []
    for run in manifest["runs"]:
        row, setup = collect_row(run, results_root, dataset, records)
        rows.append(row)
        setups.append(setup)
    for run, setup in zip(manifest["runs"], setups, strict=True):
        if setup != setups[0]:
            raise ValueError(
                f"beta={run['beta']} was scored with {setup}, beta={manifest['runs'][0]['beta']} with "
                f"{setups[0]}; the F3 points must share reranker, top-k and eval set"
            )
    return sorted(rows, key=lambda r: r["beta"])


def to_csv(rows: list[dict[str, Any]]) -> str:
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=COLUMNS, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue()


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Collect the beta-sweep F3 series as CSV")
    parser.add_argument("--manifest", type=Path, required=True, help="Sweep manifest from submit_beta_sweep.py")
    parser.add_argument("--results-root", type=Path, required=True,
                        help="Directory holding the eval jobs' result dirs (/eval/results on the eval PVC)")
    parser.add_argument("--dataset", default=None, help="Default: the manifest's eval_dataset")
    parser.add_argument("--telemetry", type=Path, nargs="+", default=[],
                        help="Agent run telemetry.jsonl files for the resolve-rate column")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    records = [r for path in args.telemetry for r in read_records(path)]
    rows = collect(manifest, args.results_root, args.dataset or manifest["eval_dataset"], records)
    text = to_csv(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(text, encoding="utf-8")
    sys.stdout.write(text)
    print(f"Wrote {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()

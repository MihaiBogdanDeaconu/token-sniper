from __future__ import annotations

import argparse
import json
import logging
import statistics
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import xgboost as xgb

from scripts.eval_candidates import ssg_path
from scripts.eval_retention import retention_header
from scripts.setup_phase1 import INDEX_TIMING, SSG_TIMING
from token_sniper.evaluation.eval_stages import (
    candidates_path,
    read_record,
    retention_path,
    sha256_file,
)
from token_sniper.evaluation.phase0 import load_phase0_manifest
from token_sniper.rl.trainer import load_warm_start_policy

if TYPE_CHECKING:
    from token_sniper.evaluation.phase0 import Phase0Task

logger = logging.getLogger(__name__)

LOCAL_EMBEDDING_PROVIDERS = {"sentence-transformers", "nomic-local"}
STAGE_A_FIELDS = ("stage_a_latency_ms", "stage_a_components_ms", "stage_a_device")
STAGE_B_FIELDS = ("selector_latency_ms", "vgib_input_ms", "selector_device")
EXCLUDED_ONE_OFF = (
    "bare repo clones, token-count sidecars, VGIB feature caches and failed builds are one-off costs "
    "not measured and excluded"
)


def count_params(checkpoint: Path) -> int:
    policy = load_warm_start_policy(checkpoint, device="cpu")
    return sum(p.numel() for p in policy.parameters())


def count_trees(reranker_model: Path) -> int:
    booster = xgb.Booster()
    booster.load_model(str(reranker_model))
    return len(booster.get_dump())


def query_latencies(
    cache: Path, tasks: list[Phase0Task], checkpoint_sha: str, reranker_sha: str, top_k: int,
) -> tuple[list[dict[str, float]], set[str], list[Phase0Task]]:
    rows: list[dict[str, float]] = []
    devices: set[str] = set()
    indexed: list[Phase0Task] = []
    missing: list[str] = []
    for task in tasks:
        cpath = candidates_path(cache, task.instance_id)
        _, cand = read_record(cpath)
        if "excluded" in cand:
            continue
        rpath = retention_path(cache, checkpoint_sha, task.instance_id)
        _, rec = read_record(rpath, retention_header(checkpoint_sha, reranker_sha, sha256_file(cpath), top_k))
        if not all(f in cand for f in STAGE_A_FIELDS) or not all(f in rec for f in STAGE_B_FIELDS):
            missing.append(task.instance_id)
            continue
        rows.append({
            **cand["stage_a_components_ms"],
            "stage_a": cand["stage_a_latency_ms"],
            "vgib_input": rec["vgib_input_ms"],
            "selector": rec["selector_latency_ms"],
        })
        devices.update((cand["stage_a_device"], rec["selector_device"]))
        indexed.append(task)
    if missing:
        raise ValueError(
            f"{len(missing)} instances in {cache} have stage-A or stage-B records without timings "
            f"(e.g. {missing[:3]}): they predate the timing or were imported; re-run "
            "scripts/eval_candidates.py and scripts/eval_retention.py into a fresh --cache-root"
        )
    if not indexed:
        raise ValueError(f"no indexed instances with stage-B records in {cache}")
    return rows, devices, indexed


def build_seconds(ssg_dir: Path, tasks: list[Phase0Task]) -> tuple[float, float, int]:
    commit_dirs = {(t.repo, t.base_commit): ssg_path(ssg_dir, t).parent for t in tasks}
    ssg_total = index_total = 0.0
    missing: list[str] = []
    for (repo, commit), commit_dir in sorted(commit_dirs.items()):
        for name in (SSG_TIMING, INDEX_TIMING):
            path = commit_dir / name
            if not path.exists():
                missing.append(str(path))
                continue
            timing = json.loads(path.read_text(encoding="utf-8"))
            if (timing["repo"], timing["commit"]) != (repo, commit):
                raise ValueError(f"{path} times {timing['repo']}@{timing['commit']}, expected {repo}@{commit}")
            if name == SSG_TIMING:
                ssg_total += timing["seconds"]
            else:
                if timing["provider"] not in LOCAL_EMBEDDING_PROVIDERS:
                    raise ValueError(f"{path}: provider {timing['provider']} is an API, not priceable per GPU hour")
                index_total += timing["seconds"]
    if missing:
        raise FileNotFoundError(
            f"{len(missing)} build timings missing under {ssg_dir} (e.g. {missing[:3]}): "
            "time the builds with scripts/setup_phase1.py"
        )
    return ssg_total, index_total, len(commit_dirs)


def cost_summary(
    *, selector_params: int, reranker_trees: int, latency_rows: list[dict[str, float]], devices: set[str],
    ssg_seconds: float, index_seconds: float, commits: int, gpu_hour_usd: float, cpu_hour_usd: float,
    source: str,
) -> dict[str, Any]:
    if devices == {"cpu"}:
        query_rate = cpu_hour_usd
    elif all(d.startswith("cuda") for d in devices):
        query_rate = gpu_hour_usd
    else:
        raise ValueError(f"stage A and B ran on {sorted(devices)}: one device per measurement")
    instances = len(latency_rows)
    latencies = [r["stage_a"] + r["vgib_input"] + r["selector"] for r in latency_rows]
    mean_ms = statistics.fmean(latencies)
    means = {k: statistics.fmean(r[k] for r in latency_rows) for k in latency_rows[0]}
    stage_a_parts = ", ".join(f"{k} {v:.1f}" for k, v in means.items()
                              if k not in {"stage_a", "vgib_input", "selector"})
    build = ssg_seconds + index_seconds
    return {
        "selector_params": selector_params,
        "reranker_trees": reranker_trees,
        "latency_ms_per_instance": {
            "mean": mean_ms,
            "p50": float(np.percentile(latencies, 50)),
            "p95": float(np.percentile(latencies, 95)),
        },
        "amortized_build_seconds_per_instance": build / instances,
        "gpu_hour_usd": gpu_hour_usd,
        "cpu_hour_usd": cpu_hour_usd,
        "usd_per_instance": mean_ms / 3.6e6 * query_rate + build / 3600 * gpu_hour_usd / instances,
        "instances": instances,
        "source": (
            f"{source}; latency per instance = stage A (mean ms: {stage_a_parts}; total {means['stage_a']:.1f}) "
            f"+ VGIB input: node features + issue embedding ({means['vgib_input']:.1f}) + selector: reranker "
            f"predict + VGIB forward ({means['selector']:.1f}), on {'/'.join(sorted(devices))}, priced at "
            f"{query_rate} USD/h; build = {commits} repo@commit: SSG {ssg_seconds:.0f}s + index "
            f"{index_seconds:.0f}s, all at the GPU rate (the build pod holds a GPU throughout), over "
            f"{instances} instances; {EXCLUDED_ONE_OFF}"
        ),
    }


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-8s %(name)s — %(message)s",
                        datefmt="%H:%M:%S")
    parser = argparse.ArgumentParser(description="T3 selector cost per instance, with the amortised build")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True, help="Stage-A/B cache namespace, e.g. /eval/cache/lite_274")
    parser.add_argument("--reranker-model", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True, help="The VGIB checkpoint the records were built with")
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--ssg-dir", type=Path, required=True, help="SSG dir holding the build timings")
    parser.add_argument("--gpu-hour-usd", type=float, required=True)
    parser.add_argument("--cpu-hour-usd", type=float, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    checkpoint_sha = sha256_file(args.checkpoint)
    reranker_sha = sha256_file(args.reranker_model)
    tasks = load_phase0_manifest(args.manifest).tasks
    rows, devices, indexed = query_latencies(args.cache, tasks, checkpoint_sha, reranker_sha, args.top_k)
    ssg_seconds, index_seconds, commits = build_seconds(args.ssg_dir, indexed)
    cost = cost_summary(
        selector_params=count_params(args.checkpoint), reranker_trees=count_trees(args.reranker_model),
        latency_rows=rows, devices=devices, ssg_seconds=ssg_seconds, index_seconds=index_seconds,
        commits=commits, gpu_hour_usd=args.gpu_hour_usd, cpu_hour_usd=args.cpu_hour_usd,
        source=(f"{args.manifest} ({len(tasks)} tasks, {len(indexed)} indexed); records "
                f"{args.cache} (stage B {checkpoint_sha[:12]}); checkpoint {args.checkpoint} "
                f"(sha256 {checkpoint_sha[:12]}); reranker {args.reranker_model} (sha256 {reranker_sha[:12]}); "
                f"build timings {args.ssg_dir}"),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(cost, indent=2) + "\n", encoding="utf-8")
    logger.info("Wrote %s: %s", args.output, json.dumps(cost, indent=2))


if __name__ == "__main__":
    main()

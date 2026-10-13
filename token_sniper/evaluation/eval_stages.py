from __future__ import annotations

import hashlib
import json
import statistics
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pathlib import Path

    from token_sniper.evaluation.locagent_gold import LocAgentGoldLabels

KEEP_THRESHOLD = 0.5


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def candidates_path(cache: Path, instance_id: str) -> Path:
    return cache / "candidates" / f"{instance_id}.json"


def retention_path(cache: Path, checkpoint_sha256: str, instance_id: str) -> Path:
    return cache / "retention" / checkpoint_sha256[:12] / f"{instance_id}.json"


def write_record(path: Path, header: dict[str, Any], body: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps({"header": header, **body}), encoding="utf-8")
    tmp.replace(path)


def read_record(path: Path, expected_header: dict[str, Any] | None = None) -> tuple[dict[str, Any], dict[str, Any]]:
    body = json.loads(path.read_text(encoding="utf-8"))
    header = body.pop("header")
    if expected_header is not None and header != expected_header:
        raise ValueError(f"{path} was built from {header}, expected {expected_header}")
    return header, body


def check_ssg_matches(path: Path, header: dict[str, Any], ssg_path: Path) -> None:
    actual = sha256_file(ssg_path)
    if header.get("ssg_sha256") != actual:
        raise ValueError(f"{path} was built from SSG sha256 {header.get('ssg_sha256')}, the SSG is now {actual}")


def check_expand_matches(path: Path, body: dict[str, Any], expand_depth: int) -> None:
    if "excluded" in body:
        return
    cached = body.get("expand_depth", 0)
    if cached != expand_depth:
        raise ValueError(
            f"{path} was built with expand_depth={cached}, this run uses {expand_depth}: "
            "use a different --cache-name for the expanded arm"
        )


def rank_within_files(files: list[str], nodes: list[dict[str, Any]]) -> tuple[list[str], list[str]]:
    by_file: dict[str, list[dict[str, Any]]] = {}
    for node in nodes:
        if node["entity_key"] is not None:
            by_file.setdefault(node["file"], []).append(node)

    modules: list[str] = []
    entities: list[str] = []
    for file_path in files:
        for node in sorted(by_file.get(file_path, []), key=lambda n: -n["p_v"]):
            if node["module_key"] not in modules:
                modules.append(node["module_key"])
            if node["entity_key"] not in entities:
                entities.append(node["entity_key"])
    return modules, entities


def selection(nodes: list[dict[str, Any]], gold: LocAgentGoldLabels, pool_tokens: int) -> dict[str, Any]:
    kept = [n for n in nodes if n["p_v"] >= KEEP_THRESHOLD]
    kept_files = {n["file"] for n in kept}
    kept_entities = {n["entity_key"] for n in kept}
    present_entities = {n["entity_key"] for n in nodes}
    return {
        "gold_files_kept": set(gold.gold_files) <= kept_files,
        "gold_functions_in_topk": set(gold.gold_entities) <= present_entities,
        "gold_functions_kept": set(gold.gold_entities) <= kept_entities,
        "function_recall": len(set(gold.gold_entities) & kept_entities) / len(gold.gold_entities),
        "kept_tokens": sum(n["tokens"] for n in kept),
        "topk_tokens": sum(n["tokens"] for n in nodes),
        "pool_tokens": pool_tokens,
    }


def summarize_selection(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "instances": len(rows),
        "gold_files_kept_rate": round(statistics.fmean(r["gold_files_kept"] for r in rows), 4),
        "gold_functions_in_topk_rate": round(statistics.fmean(r["gold_functions_in_topk"] for r in rows), 4),
        "gold_functions_kept_rate": round(statistics.fmean(r["gold_functions_kept"] for r in rows), 4),
        "mean_function_recall": round(statistics.fmean(r["function_recall"] for r in rows), 4),
        "mean_kept_ratio_of_topk": round(statistics.fmean(r["kept_tokens"] / r["topk_tokens"] for r in rows), 4),
        "mean_kept_ratio_of_pool": round(statistics.fmean(r["kept_tokens"] / r["pool_tokens"] for r in rows), 4),
        "median_kept_tokens": statistics.median(r["kept_tokens"] for r in rows),
        "median_topk_tokens": statistics.median(r["topk_tokens"] for r in rows),
        "median_pool_tokens": statistics.median(r["pool_tokens"] for r in rows),
    }

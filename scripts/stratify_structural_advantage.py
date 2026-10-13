from __future__ import annotations

import argparse
import json
import logging
import statistics
import sys
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from token_sniper.evaluation.eval_stages import read_record, sha256_file, write_record
from token_sniper.evaluation.locagent_gold import build_gold_index
from token_sniper.evaluation.locagent_ranking import entity_keys
from token_sniper.evaluation.locagent_scoring import (
    LITE_K_GROUPS,
    LOCBENCH_K_GROUPS,
    score_predictions,
)
from token_sniper.evaluation.statistics import bootstrap_ci
from token_sniper.evaluation.telemetry import read_records

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable

    import networkx as nx

    from token_sniper.evaluation.telemetry import AgentRunRecord

logger = logging.getLogger(__name__)

DATASETS = {
    "lite_274": ("czlll/SWE-bench_Lite", "test"),
    "locbench_560": ("czlll/Loc-Bench_V1", "test"),
}
K_GROUPS = {"lite_274": LITE_K_GROUPS, "locbench_560": LOCBENCH_K_GROUPS}
EMBEDDING_MODEL = "jinaai/jina-code-embeddings-0.5b"
EMBEDDING_DIM = 256
N_SEEDS = 5
STRUCTURAL_EDGE_TYPES = frozenset({"calls", "inherits"})
MIN_HOPS = 2
LOW_SIMILARITY_QUANTILE = 2 / 3
STRATA = ("structural", "rest")
AGGREGATE_ROUNDING = 5e-5 + 1e-9
"""LocAgent rounds each metric to 4 decimals."""


def gold_node_ids(graph: nx.MultiDiGraph, gold_entities: Iterable[str]) -> dict[str, list[str]]:
    wanted = {key: [] for key in gold_entities}
    for nid, data in graph.nodes(data=True):
        keys = entity_keys(nid, data)
        if keys is not None and keys[1] in wanted:
            wanted[keys[1]].append(nid)
    return wanted


def hop_distances(graph: nx.MultiDiGraph, seeds: Iterable[str]) -> dict[str, int]:
    neighbours: dict[str, set[str]] = defaultdict(set)
    for u, v, data in graph.edges(data=True):
        if data["edge_type"] in STRUCTURAL_EDGE_TYPES:
            neighbours[u].add(v)
            neighbours[v].add(u)
    distance = dict.fromkeys(seeds, 0)
    queue = deque(distance)
    while queue:
        node = queue.popleft()
        for nxt in neighbours[node]:
            if nxt not in distance:
                distance[nxt] = distance[node] + 1
                queue.append(nxt)
    return distance


def entity_similarities(issue_embedding: np.ndarray, chunks: dict[str, np.ndarray]) -> dict[str, float]:
    query = issue_embedding / np.linalg.norm(issue_embedding)
    sims = {}
    for fqn, vectors in chunks.items():
        cosines = (vectors @ query) / np.linalg.norm(vectors, axis=1)
        sims[fqn] = float(cosines.max())
    return sims


def similarity_rank_fraction(sims: dict[str, float], gold_similarity: float) -> float:
    return sum(s > gold_similarity for s in sims.values()) / len(sims)


def label_instance(
    *,
    graph: nx.MultiDiGraph,
    gold_entities: list[str],
    seeds: list[str],
    sims: dict[str, float],
) -> dict[str, Any]:
    if not gold_entities:
        return {"unresolved": "no gold entities"}
    if not seeds:
        return {"unresolved": "no seed entity is an SSG node"}
    gold_nodes = gold_node_ids(graph, gold_entities)
    unmapped = sorted(key for key, nodes in gold_nodes.items() if not nodes)
    if unmapped:
        return {"unresolved": f"gold entities without an SSG node: {unmapped}"}
    unembedded = sorted(key for key, nodes in gold_nodes.items() if not any(nid in sims for nid in nodes))
    if unembedded:
        return {"unresolved": f"gold entities without a ChromaDB embedding: {unembedded}"}
    nodes = sorted({nid for ids in gold_nodes.values() for nid in ids})

    distance = hop_distances(graph, seeds)
    reached = [distance[nid] for nid in nodes if nid in distance]
    min_hops = min(reached) if reached else None
    gold_similarity = max(sims[nid] for nid in nodes if nid in sims)
    return {
        "gold_nodes": nodes,
        "min_hops": min_hops,
        "far": min_hops is None or min_hops >= MIN_HOPS,
        "gold_similarity": gold_similarity,
        "similarity_rank_fraction": similarity_rank_fraction(sims, gold_similarity),
    }


def assign_strata(labels: list[dict[str, Any]]) -> float:
    resolved = [label for label in labels if "unresolved" not in label]
    if not resolved:
        raise ValueError("no instance could be labeled")
    threshold = float(np.quantile([r["similarity_rank_fraction"] for r in resolved], LOW_SIMILARITY_QUANTILE))
    for label in labels:
        if "unresolved" in label:
            label["stratum"] = "unresolved"
            continue
        label["low_similarity"] = label["similarity_rank_fraction"] >= threshold
        label["stratum"] = "structural" if label["far"] and label["low_similarity"] else "rest"
    return threshold


def strata_ids(labels: list[dict[str, Any]]) -> dict[str, list[str]]:
    return {s: [label["instance_id"] for label in labels if label["stratum"] == s] for s in STRATA}


def cached_instance_scores(
    path: Path,
    instance_ids: list[str],
    *,
    cache_dir: Path,
    header: dict[str, Any],
    score_one: Callable[[Path, str], dict[str, Any]],
    workers: int = 1,
) -> dict[str, dict[str, Any]]:
    sha = sha256_file(path)
    full_header = {"predictions_sha256": sha, **header}
    records = {iid: cache_dir / sha[:12] / f"{iid}.json" for iid in instance_ids}
    todo = [iid for iid, record in records.items() if not record.exists()]

    def build(iid: str) -> None:
        write_record(records[iid], full_header, {"scores": score_one(path, iid)})

    if todo:
        logger.info("%s: scoring %d instances one by one (%d cached)", path, len(todo), len(records) - len(todo))
        with ThreadPoolExecutor(max_workers=workers) as pool:
            list(pool.map(build, todo))
    return {iid: read_record(record, full_header)[1]["scores"] for iid, record in records.items()}


def localization_by_stratum(
    ids: dict[str, list[str]],
    predictions: dict[str, Path],
    score: Callable[[Path, list[str]], dict[str, Any]],
    instance_scores: Callable[[Path, list[str]], dict[str, dict[str, Any]]],
) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for name, path in predictions.items():
        predicted = {json.loads(line)["instance_id"] for line in path.read_text(encoding="utf-8").splitlines()
                     if line.strip()}
        out[name] = {}
        for stratum, members in ids.items():
            if not members:
                raise ValueError(f"stratum {stratum!r} is empty; LocAgent would score every instance")
            aggregate = score(path, members)
            per_instance = instance_scores(path, members)
            acc: dict[str, dict[str, Any]] = {}
            for level, metrics in aggregate.items():
                for metric, value in metrics.items():
                    if not metric.startswith("Acc@"):
                        continue
                    hits = [per_instance[iid][level][metric] for iid in members]
                    bad = sorted({h for h in hits if h not in (0.0, 1.0)})
                    if bad:
                        raise ValueError(f"{name} {level} {metric}: single-instance values {bad} are not 0/1")
                    rate = statistics.fmean(hits)
                    if abs(rate - value) > AGGREGATE_ROUNDING:
                        raise ValueError(f"{name} {stratum} {level} {metric}: per-instance hit rate {rate:.6f} "
                                         f"!= LocAgent aggregate {value}")
                    acc.setdefault(level, {})[metric] = {
                        "rate": rate, "hits": int(sum(hits)), "bootstrap_ci_95": bootstrap_ci(hits),
                    }
            out[name][stratum] = {
                "instances": len(members),
                "without_prediction": sum(iid not in predicted for iid in members),
                "scores": aggregate,
                "acc": acc,
            }
    return out


def resolve_by_stratum(
    ids: dict[str, list[str]], records: list[AgentRunRecord],
) -> dict[str, Any]:
    stratum_of = {iid: s for s, members in ids.items() for iid in members}
    seen: set[tuple[str, int, str]] = set()
    by_arm: dict[str, dict[str, dict[int, dict[str, bool]]]] = defaultdict(
        lambda: {s: defaultdict(dict) for s in STRATA})
    outside = 0
    for rec in records:
        key = (rec.arm, rec.seed, rec.instance_id)
        if key in seen:
            raise ValueError(f"duplicate telemetry record for arm={rec.arm} seed={rec.seed} {rec.instance_id}")
        seen.add(key)
        stratum = stratum_of.get(rec.instance_id)
        if stratum is None:
            outside += 1
            continue
        by_arm[rec.arm][stratum][rec.seed][rec.instance_id] = rec.resolved

    arms: dict[str, Any] = {}
    for arm, strata in sorted(by_arm.items()):
        arms[arm] = {}
        for stratum, seeds in strata.items():
            if not seeds:
                arms[arm][stratum] = {"instances": 0}
                continue
            per_seed = {seed: {"n": len(runs), "resolved": sum(runs.values()),
                               "rate": sum(runs.values()) / len(runs)}
                        for seed, runs in sorted(seeds.items())}
            per_instance: dict[str, list[bool]] = defaultdict(list)
            for runs in seeds.values():
                for iid, resolved in runs.items():
                    per_instance[iid].append(resolved)
            rates = [p["rate"] for p in per_seed.values()]
            arms[arm][stratum] = {
                "instances": len(per_instance),
                "per_seed": per_seed,
                "mean_over_seeds": statistics.fmean(rates),
                "std_over_seeds": statistics.stdev(rates) if len(rates) > 1 else None,
                "bootstrap_ci_95": bootstrap_ci([statistics.fmean(v) for v in per_instance.values()]),
            }
    return {"records_outside_labeled_instances": outside, "arms": arms}


def compute_labels(args: argparse.Namespace) -> tuple[list[dict[str, Any]], float]:
    from token_sniper.evaluation.phase0 import load_phase0_manifest
    from token_sniper.evaluation.swebench import load_swebench_task_metadata
    from token_sniper.graph.io import read_ssg_json
    from token_sniper.retrieval.config import EmbeddingConfig
    from token_sniper.retrieval.embeddings import create_embedder
    from token_sniper.retrieval.entry_points import identify_entry_points_by_embedding
    from token_sniper.retrieval.vectorstore import EntityVectorStore

    dataset_name, split = DATASETS[args.dataset_key]
    tasks = load_phase0_manifest(args.manifest).tasks
    gold = build_gold_index(args.dataset_key)
    store = EntityVectorStore(persist_directory=str(args.chromadb_dir), collection_prefix="token_sniper")
    embedder = create_embedder(EmbeddingConfig(
        provider="sentence-transformers", name=EMBEDDING_MODEL, dimension=EMBEDDING_DIM, device=args.device,
    ))

    labels = []
    for n, task in enumerate(tasks, 1):
        record: dict[str, Any] = {"instance_id": task.instance_id}
        ssg = args.ssg_dir / task.repo.replace("/", "__") / task.base_commit[:12] / "ssg.json"
        collection_repo = f"{task.repo}@{task.base_commit[:12]}"
        if not ssg.exists():
            record["unresolved"] = "no SSG"
        elif task.instance_id not in gold:
            record["unresolved"] = f"not in {dataset_name}"
        else:
            graph = read_ssg_json(ssg)
            issue = load_swebench_task_metadata(task.instance_id, dataset_name=dataset_name,
                                                split=split).problem_statement
            entry = identify_entry_points_by_embedding(
                issue_text=issue, repo=collection_repo, embedder=embedder, vector_store=store,
                graph=graph, n_candidates=N_SEEDS,
            )
            if entry.error is not None:
                raise RuntimeError(f"{task.instance_id}: seed retrieval failed: {entry.error}")
            chunks = collection_chunks(store, collection_repo)
            if not chunks:
                record["unresolved"] = f"empty ChromaDB collection {collection_repo}"
            else:
                sims = entity_similarities(np.asarray(embedder.embed_query(issue), dtype=float), chunks)
                record["seeds"] = entry.matched_node_ids
                record.update(label_instance(
                    graph=graph, gold_entities=gold[task.instance_id].gold_entities,
                    seeds=entry.matched_node_ids, sims=sims,
                ))
        labels.append(record)
        logger.info("[%d/%d] %s: %s", n, len(tasks), task.instance_id,
                    record.get("unresolved") or f"hops={record['min_hops']} "
                                                 f"rank={record['similarity_rank_fraction']:.3f}")
    return labels, assign_strata(labels)


def collection_chunks(store: Any, collection_repo: str) -> dict[str, np.ndarray]:
    collection = store._get_or_create_collection(store.repo_slug(collection_repo))
    grouped: dict[str, list[list[float]]] = defaultdict(list)
    count = collection.count()
    for offset in range(0, count, 5000):
        batch = collection.get(include=["embeddings", "metadatas"], limit=5000, offset=offset)
        for meta, emb in zip(batch["metadatas"], batch["embeddings"], strict=True):
            grouped[meta["fully_qualified_name"]].append(emb)
    if sum(len(v) for v in grouped.values()) != count:
        raise RuntimeError(f"read {sum(len(v) for v in grouped.values())} of {count} embeddings of {collection_repo}")
    return {fqn: np.asarray(vectors, dtype=float) for fqn, vectors in grouped.items()}


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-8s %(name)s — %(message)s",
                        datefmt="%H:%M:%S")
    parser = argparse.ArgumentParser(description="F4: structural-advantage strata and per-stratum results")
    parser.add_argument("--dataset-key", choices=sorted(DATASETS), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--labels", type=Path, default=None, help="Reuse this labels.jsonl instead of labeling")
    parser.add_argument("--manifest", type=Path, help="Required unless --labels")
    parser.add_argument("--ssg-dir", type=Path, help="Required unless --labels")
    parser.add_argument("--chromadb-dir", type=Path, help="Required unless --labels")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--predictions", action="append", default=[], metavar="NAME=PATH",
                        help="found_files/modules/entities JSONL to score per stratum (repeatable)")
    parser.add_argument("--telemetry", action="append", default=[], type=Path, metavar="PATH",
                        help="telemetry.jsonl of downstream agent runs (repeatable)")
    parser.add_argument("--score-cache", type=Path, default=None,
                        help="Per-instance LocAgent score cache (default: <output-dir>/instance_scores); "
                             "keyed by dataset and prediction file sha256, so it can be shared across runs")
    parser.add_argument("--score-workers", type=int, default=8,
                        help="Concurrent single-instance LocAgent scoring subprocesses")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.labels is not None:
        labels = [json.loads(line) for line in args.labels.read_text(encoding="utf-8").splitlines() if line.strip()]
        threshold = None
    else:
        missing = [flag for flag in ("manifest", "ssg_dir", "chromadb_dir") if getattr(args, flag) is None]
        if missing:
            parser.error(f"labeling needs --{', --'.join(m.replace('_', '-') for m in missing)} (or pass --labels)")
        labels, threshold = compute_labels(args)
        with (args.output_dir / "labels.jsonl").open("w", encoding="utf-8") as handle:
            for label in labels:
                handle.write(json.dumps(label) + "\n")

    ids = strata_ids(labels)
    predictions = {name: Path(path) for name, path in (item.split("=", 1) for item in args.predictions)}
    records = [rec for path in args.telemetry for rec in read_records(path)]

    def score(path: Path, members: list[str]) -> dict[str, Any]:
        return score_predictions(path, dataset=args.dataset_key, k_values_list=K_GROUPS[args.dataset_key],
                                 selected_list=members, python=Path(sys.executable))

    score_cache = (args.score_cache or args.output_dir / "instance_scores") / args.dataset_key

    def instance_scores(path: Path, members: list[str]) -> dict[str, dict[str, Any]]:
        return cached_instance_scores(
            path, members, cache_dir=score_cache,
            header={"dataset_key": args.dataset_key, "k_values_list": K_GROUPS[args.dataset_key]},
            score_one=lambda p, iid: score(p, [iid]), workers=args.score_workers,
        )

    summary = {
        "dataset_key": args.dataset_key,
        "labels": str(args.labels or args.output_dir / "labels.jsonl"),
        "definition": {
            "embedding_model": EMBEDDING_MODEL, "embedding_dim": EMBEDDING_DIM, "n_seeds": N_SEEDS,
            "edge_types": sorted(STRUCTURAL_EDGE_TYPES), "min_hops": MIN_HOPS,
            "low_similarity": f"similarity_rank_fraction >= its {LOW_SIMILARITY_QUANTILE:.4f} quantile",
            "low_similarity_threshold": threshold,
        },
        "counts": {s: sum(label["stratum"] == s for label in labels) for s in (*STRATA, "unresolved")},
        "far_only": sum(label.get("far", False) and not label["low_similarity"] for label in labels
                        if label["stratum"] != "unresolved"),
        "low_similarity_only": sum(label["low_similarity"] and not label["far"] for label in labels
                                   if label["stratum"] != "unresolved"),
        "localization": localization_by_stratum(ids, predictions, score, instance_scores),
        "resolve": resolve_by_stratum(ids, records) if records else None,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    logger.info("Summary: %s", json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

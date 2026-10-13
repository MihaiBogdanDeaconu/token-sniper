from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

from scripts.train_reranker import build_reranker_inputs
from token_sniper.evaluation.eval_stages import (
    candidates_path,
    check_expand_matches,
    check_ssg_matches,
    read_record,
    sha256_file,
    write_record,
)
from token_sniper.evaluation.phase0 import Phase0Task, load_phase0_manifest
from token_sniper.evaluation.swebench import _load_dataset_rows, load_swebench_task_metadata
from token_sniper.retrieval.config import EmbeddingConfig
from token_sniper.retrieval.embeddings import create_embedder
from token_sniper.retrieval.retriever import BM25Retriever, DenseRetriever
from token_sniper.retrieval.vectorstore import EntityVectorStore

logger = logging.getLogger(__name__)

DATASETS = {
    "lite_274": ("czlll/SWE-bench_Lite", "test"),
    "locbench_560": ("czlll/Loc-Bench_V1", "test"),
    "verified_500": ("princeton-nlp/SWE-bench_Verified", "test"),
    "live_50": ("SWE-bench-Live/SWE-bench-Live", "verified"),
}
EMBEDDING_MODEL = "jinaai/jina-code-embeddings-0.5b"
EMBEDDING_DIM = 256


def ssg_path(ssg_dir: Path, task: Phase0Task) -> Path:
    return ssg_dir / task.repo.replace("/", "__") / task.base_commit[:12] / "ssg.json"


def collection_repo(task: Phase0Task) -> str:
    return f"{task.repo}@{task.base_commit[:12]}"


def candidates_header(ssg_dir: Path, store: EntityVectorStore, task: Phase0Task) -> dict[str, object]:
    return {
        "ssg_sha256": sha256_file(ssg_path(ssg_dir, task)),
        "collection_count": store.get_collection_stats(collection_repo(task))["entity_count"],
    }


class _TimedRetriever:
    def __init__(self, retriever: DenseRetriever | BM25Retriever) -> None:
        self._retriever = retriever
        self.ms = 0.0

    def retrieve(self, **kwargs):
        t0 = time.perf_counter()
        try:
            return self._retriever.retrieve(**kwargs)
        finally:
            self.ms += (time.perf_counter() - t0) * 1000


def _exclusion_reason(task: Phase0Task, ssg_dir: Path, dataset_name: str, split: str) -> str:
    ssg = ssg_path(ssg_dir, task)
    if not ssg.exists():
        skip = ssg.with_name(".skip")
        return f"no SSG ({skip.read_text().strip()})" if skip.exists() else "no SSG"
    try:
        load_swebench_task_metadata(task.instance_id, dataset_name=dataset_name, split=split)
    except KeyError:
        return f"not in {dataset_name}"
    return "dense retrieval returned no entities (empty embedding collection)"


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-8s %(name)s — %(message)s",
                        datefmt="%H:%M:%S")
    parser = argparse.ArgumentParser(description="Eval stage A: cached retrieval candidates")
    parser.add_argument("--dataset-key", choices=sorted(DATASETS), required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--ssg-dir", type=Path, required=True)
    parser.add_argument("--chromadb-dir", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--cache-name", default=None,
                        help="Cache namespace under --cache-root (default: the dataset key). "
                             "Give the expanded arm its own, e.g. lite_274_expand3.")
    parser.add_argument("--import-reranker-inputs", type=Path, default=None)
    parser.add_argument("--expand-depth", type=int, default=0, metavar="HOPS",
                        help="Add the graph neighbourhood of the top retrieved entities within this "
                             "many hops (0 = off, as in every number reported so far)")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.expand_depth and args.import_reranker_inputs is not None:
        parser.error("--import-reranker-inputs holds unexpanded records; do not mix it with --expand-depth")

    dataset_name, split = DATASETS[args.dataset_key]
    cache = args.cache_root / (args.cache_name or args.dataset_key)
    logger.info("Cache: %s | expand depth: %d", cache, args.expand_depth)
    tasks = load_phase0_manifest(args.manifest).tasks
    by_id = {t.instance_id: t for t in tasks}
    store = EntityVectorStore(persist_directory=str(args.chromadb_dir), collection_prefix="token_sniper")

    if args.import_reranker_inputs is not None:
        records = json.loads(args.import_reranker_inputs.read_text(encoding="utf-8"))
        for rec in records:
            path = candidates_path(cache, rec["instance_id"])
            if path.exists():
                continue
            task = by_id[rec["instance_id"]]
            write_record(path, candidates_header(args.ssg_dir, store, task), {
                "candidate_files": rec["candidate_files"],
                "feature_matrix": rec["feature_matrix"],
                "entities": [],
                "expand_depth": 0,
            })
        logger.info("Imported %d records from %s", len(records), args.import_reranker_inputs)

    embedder = create_embedder(EmbeddingConfig(
        provider="sentence-transformers", name=EMBEDDING_MODEL, dimension=EMBEDDING_DIM, device=args.device,
    ))
    dense = _TimedRetriever(DenseRetriever(embedder=embedder, vector_store=store))
    bm25 = _TimedRetriever(BM25Retriever(vector_store=store))

    built = excluded = cached = 0
    t_start = time.monotonic()
    for n, task in enumerate(tasks, 1):
        path = candidates_path(cache, task.instance_id)
        if path.exists():
            header, body = read_record(path)
            check_expand_matches(path, body, args.expand_depth)
            if "excluded" not in body:
                check_ssg_matches(path, header, ssg_path(args.ssg_dir, task))
            cached += 1
            continue

        _load_dataset_rows(dataset_name, split)
        dense.ms = bm25.ms = 0.0
        t_build = time.perf_counter()
        inputs = build_reranker_inputs(
            tasks=[task], ssg_dir=args.ssg_dir, embedder=embedder, vector_store=store,
            retriever=dense, bm25_retriever=bm25, dataset_name=dataset_name, split=split,
            label=args.dataset_key, expand_depth=args.expand_depth,
        )
        build_ms = (time.perf_counter() - t_build) * 1000
        if inputs:
            inp = inputs[0]
            write_record(path, candidates_header(args.ssg_dir, store, task), {
                "candidate_files": [f.file_path for f in inp.candidate_features],
                "feature_matrix": inp.to_feature_matrix().tolist(),
                "entities": [
                    {"entity_id": e.entity_id, "fully_qualified_name": e.fully_qualified_name,
                     "file_path": e.file_path, "score": e.score, "rank": e.rank}
                    for e in inp.retrieved_entities
                ],
                "expand_depth": args.expand_depth,
                "stage_a_latency_ms": build_ms,
                "stage_a_components_ms": {
                    "dense_retrieval": dense.ms,
                    "bm25_retrieval": bm25.ms,
                    "ssg_load_entry_points_features": build_ms - dense.ms - bm25.ms,
                },
                "stage_a_device": args.device,
            })
            built += 1
        else:
            reason = _exclusion_reason(task, args.ssg_dir, dataset_name, split)
            write_record(path, {}, {"excluded": reason})
            excluded += 1
            logger.warning("%s excluded: %s", task.instance_id, reason)
        logger.info("[%d/%d] %s done, %.0fs elapsed", n, len(tasks), task.instance_id, time.monotonic() - t_start)

    logger.info("%s stage A: %d built, %d excluded, %d already cached", args.dataset_key, built, excluded, cached)


if __name__ == "__main__":
    main()

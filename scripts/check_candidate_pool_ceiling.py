from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Literal

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

KNN_CANDIDATES = 200
RERANK_TOP_K = 20


def _run(
    *,
    manifest_path: Path,
    dataset_key: Literal["lite_274", "locbench_560"],
    ssg_dir: Path,
    persist_dir: str,
    provider: str,
    reranker_model: Path | None,
) -> None:
    from token_sniper.evaluation.locagent_gold import build_gold_index
    from token_sniper.evaluation.phase0 import load_phase0_manifest
    from token_sniper.evaluation.swebench import load_swebench_task_metadata
    from token_sniper.graph.io import read_ssg_json
    from token_sniper.retrieval.config import EmbeddingConfig
    from token_sniper.retrieval.embeddings import create_embedder
    from token_sniper.retrieval.entry_points import identify_entry_points_by_embedding
    from token_sniper.retrieval.reranker import GraphReranker, extract_candidate_features
    from token_sniper.retrieval.retriever import BM25Retriever, DenseRetriever, merge_retrieval_results
    from token_sniper.retrieval.vectorstore import EntityVectorStore

    dataset_name, split = {
        "lite_274": ("czlll/SWE-bench_Lite", "test"),
        "locbench_560": ("czlll/Loc-Bench_V1", "test"),
    }[dataset_key]

    tasks = load_phase0_manifest(manifest_path).tasks
    gold_index = build_gold_index(dataset_key)

    embedder = create_embedder(EmbeddingConfig(provider=provider, name="jinaai/jina-code-embeddings-0.5b"))
    store = EntityVectorStore(persist_directory=persist_dir, collection_prefix="token_sniper")
    dense = DenseRetriever(embedder=embedder, vector_store=store)
    bm25 = BM25Retriever(vector_store=store)
    reranker = GraphReranker(model_path=reranker_model)

    pre_hits = post_hits = scored = 0
    pre_misses: list[tuple[str, list[str]]] = []
    post_misses: list[tuple[str, list[str]]] = []

    for i, task in enumerate(tasks):
        gold = gold_index.get(task.instance_id)
        if gold is None or not gold.gold_files:
            continue

        commit_short = task.base_commit[:12]
        ssg_path = ssg_dir / task.repo.replace("/", "__") / commit_short / "ssg.json"
        if not ssg_path.exists():
            logger.warning("[%d/%d] %s: no SSG at %s, skipping", i + 1, len(tasks), task.instance_id, ssg_path)
            continue

        try:
            metadata = load_swebench_task_metadata(task.instance_id, dataset_name=dataset_name, split=split)
        except KeyError:
            logger.warning("[%d/%d] %s: not in %s, skipping", i + 1, len(tasks), task.instance_id, dataset_name)
            continue

        issue_text = metadata.problem_statement
        collection_repo = f"{task.repo}@{commit_short}"
        graph = read_ssg_json(ssg_path)

        dense_result = dense.retrieve(issue_text=issue_text, repo=collection_repo, k=KNN_CANDIDATES)
        if not dense_result.entities:
            logger.warning("[%d/%d] %s: 0 dense entities, skipping", i + 1, len(tasks), task.instance_id)
            continue
        try:
            bm25_result = bm25.retrieve(issue_text=issue_text, repo=collection_repo, k=KNN_CANDIDATES)
            merged = merge_retrieval_results(dense_result, bm25_result)
        except Exception as exc:
            logger.warning("[%d/%d] %s: BM25 failed (%s), dense only", i + 1, len(tasks), task.instance_id, exc)
            merged = dense_result

        pre_pool_files = {e.file_path for e in merged.entities}
        gold_set = set(gold.gold_files)
        pre_missing = sorted(gold_set - pre_pool_files)
        if not pre_missing:
            pre_hits += 1
        else:
            pre_misses.append((task.instance_id, pre_missing))

        entry_result = identify_entry_points_by_embedding(
            issue_text=issue_text, repo=collection_repo, embedder=embedder,
            vector_store=store, graph=graph, n_candidates=5,
        )
        features = extract_candidate_features(
            knn_result=merged, graph=graph,
            entry_point_node_ids=entry_result.matched_node_ids, issue_text=issue_text,
        )
        ranked = reranker.rerank(features, top_k=RERANK_TOP_K)
        post_pool_files = {fp for fp, _ in ranked}
        post_missing = sorted(gold_set - post_pool_files)
        if not post_missing:
            post_hits += 1
        else:
            post_misses.append((task.instance_id, post_missing))

        scored += 1
        if scored % 25 == 0:
            logger.info(
                "[%d/%d] running: pre-rerank ceiling=%.1f%% post-rerank ceiling=%.1f%% (n=%d)",
                i + 1, len(tasks), 100 * pre_hits / scored, 100 * post_hits / scored, scored,
            )

    if scored == 0:
        logger.error("No instances scored -- check --ssg-dir/--persist-dir point at the eval index.")
        return

    logger.info("=" * 60)
    logger.info("%s: %d/%d instances scored", dataset_key, scored, len(tasks))
    logger.info(
        "Pre-rerank pool (dense+BM25 merge, k=%d each): %d/%d = %.1f%% have all gold files present",
        KNN_CANDIDATES, pre_hits, scored, 100 * pre_hits / scored,
    )
    logger.info(
        "Post-rerank pool (top-%d files, what VGIB sees): %d/%d = %.1f%% have all gold files present",
        RERANK_TOP_K, post_hits, scored, 100 * post_hits / scored,
    )
    if 100 * pre_hits / scored < 90:
        logger.warning("Pre-rerank ceiling below 90%% -- widen dense/BM25 k before trusting T1/T2 numbers.")
        for iid, missing in pre_misses[:10]:
            logger.warning("  %s missing at pre-rerank: %s", iid, missing)
    if 100 * post_hits / scored < 90:
        logger.warning("Post-rerank ceiling below 90%% -- widen reranker top_k before trusting T1/T2 numbers.")
        for iid, missing in post_misses[:10]:
            logger.warning("  %s missing at post-rerank (dropped by reranker): %s", iid, missing)


def main() -> int:
    parser = argparse.ArgumentParser(description="Candidate-pool recall ceiling (step G)")
    parser.add_argument("--manifest", type=Path, required=True, help="e.g. /eval/manifests/lite_274_tasks.json")
    parser.add_argument("--dataset-key", choices=["lite_274", "locbench_560"], required=True)
    parser.add_argument("--ssg-dir", type=Path, default=Path("/eval/ssg"))
    parser.add_argument("--persist-dir", type=str, default="/eval/chromadb")
    parser.add_argument("--provider", default="sentence-transformers")
    parser.add_argument("--reranker-model", type=Path, default=None, help="Optional trained reranker checkpoint")
    args = parser.parse_args()

    _run(
        manifest_path=args.manifest,
        dataset_key=args.dataset_key,
        ssg_dir=args.ssg_dir,
        persist_dir=args.persist_dir,
        provider=args.provider,
        reranker_model=args.reranker_model,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())

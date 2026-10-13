from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

from token_sniper.evaluation.metrics import (
    average_precision,
    precision_at_k,
    recall_at_k,
)
from token_sniper.evaluation.phase0 import load_phase0_manifest
from token_sniper.evaluation.swebench import load_swebench_task_metadata
from token_sniper.retrieval.config import EmbeddingConfig
from token_sniper.retrieval.embeddings import create_embedder
from token_sniper.retrieval.retriever import DenseRetriever
from token_sniper.retrieval.vectorstore import EntityVectorStore

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

K_VALUES = [5, 10, 20, 50]
MAX_K = 200


def run_knn_baseline(
    *,
    manifest_path: Path,
    chromadb_dir: Path,
    output_path: Path,
    dataset_name: str,
    split: str,
) -> dict[str, Any]:
    manifest = load_phase0_manifest(manifest_path)
    embedder = create_embedder(EmbeddingConfig(provider="voyageai", name="voyage-code-3"))
    vector_store = EntityVectorStore(
        persist_directory=str(chromadb_dir),
        collection_prefix="token_sniper",
    )
    retriever = DenseRetriever(embedder=embedder, vector_store=vector_store)

    per_task: list[dict[str, Any]] = []
    agg: dict[str, list[float]] = {f"P@{k}": [] for k in K_VALUES}
    agg.update({f"R@{k}": [] for k in K_VALUES})
    agg["MAP"] = []
    agg["MRR"] = []

    for i, task in enumerate(manifest.tasks):
        logger.info("[%d/%d] %s", i + 1, len(manifest.tasks), task.instance_id)
        commit_short = task.base_commit[:12]
        collection_repo = f"{task.repo}@{commit_short}"

        try:
            metadata = load_swebench_task_metadata(
                task.instance_id, dataset_name=dataset_name, split=split,
            )
            issue_text = metadata.problem_statement
        except KeyError:
            logger.warning("Task %s not found in dataset, skipping", task.instance_id)
            continue

        result = retriever.retrieve(issue_text=issue_text, repo=collection_repo, k=MAX_K)
        retrieved_files = result.file_paths
        gold_files = task.gold_patch_files

        task_metrics: dict[str, float] = {}
        for k in K_VALUES:
            p = precision_at_k(retrieved_files, gold_files, k=k)
            r = recall_at_k(retrieved_files, gold_files, k=k)
            task_metrics[f"P@{k}"] = p
            task_metrics[f"R@{k}"] = r
            agg[f"P@{k}"].append(p)
            agg[f"R@{k}"].append(r)

        ap = average_precision(retrieved_files, gold_files)
        task_metrics["MAP"] = ap
        agg["MAP"].append(ap)

        rr = _reciprocal_rank(retrieved_files, gold_files)
        task_metrics["MRR"] = rr
        agg["MRR"].append(rr)

        per_task.append({
            "instance_id": task.instance_id,
            "repo": task.repo,
            "gold_files": gold_files,
            "retrieved_files": retrieved_files[:20],
            "metrics": task_metrics,
        })

    means = {k: (sum(v) / len(v) if v else 0.0) for k, v in agg.items()}

    summary = {
        "task_count": len(per_task),
        "k_values": K_VALUES,
        "aggregate_metrics": means,
        "per_task": per_task,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    logger.info("KNN baseline results written to %s", output_path)
    logger.info("Aggregate: %s", json.dumps(means, indent=2))
    return summary


def _reciprocal_rank(retrieved: list[str], gold: list[str]) -> float:
    gold_set = set(gold)
    for i, item in enumerate(retrieved, start=1):
        if item in gold_set:
            return 1.0 / i
    return 0.0


def main() -> None:
    parser = argparse.ArgumentParser(description="KNN baseline retrieval evaluation")
    parser.add_argument("--manifest", type=Path, default=Path("configs/phases/phase0_tasks.json"))
    parser.add_argument("--chromadb-dir", type=Path, default=Path("data/chromadb"))
    parser.add_argument("--output", type=Path, default=Path("experiments/knn_baseline/metrics.json"))
    parser.add_argument("--dataset", default="SWE-bench-Live/SWE-bench-Live")
    parser.add_argument("--split", default="verified")
    args = parser.parse_args()

    run_knn_baseline(
        manifest_path=args.manifest,
        chromadb_dir=args.chromadb_dir,
        output_path=args.output,
        dataset_name=args.dataset,
        split=args.split,
    )


if __name__ == "__main__":
    main()

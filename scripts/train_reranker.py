from __future__ import annotations

import argparse
import json
import logging
import random
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from token_sniper.evaluation.metrics import (
    average_precision,
    precision_at_k,
    recall_at_k,
)
from token_sniper.evaluation.phase0 import Phase0Task, load_phase0_manifest
from token_sniper.evaluation.swebench import load_swebench_task_metadata
from token_sniper.graph.io import read_ssg_json
from token_sniper.retrieval.config import EmbeddingConfig
from token_sniper.retrieval.embeddings import create_embedder
from token_sniper.retrieval.entry_points import identify_entry_points_by_embedding
from token_sniper.retrieval.reranker import (
    GraphReranker,
    NeuralReranker,
    RerankerInput,
    extract_candidate_features,
)
from token_sniper.retrieval.retriever import (
    BM25Retriever,
    DenseRetriever,
    expand_with_graph_neighbors,
    merge_retrieval_results,
)
from token_sniper.retrieval.vectorstore import EntityVectorStore

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

K_VALUES = [5, 10, 20, 50]
KNN_CANDIDATES = 200
TRAIN_RATIO = 0.8
SEED = 42


def _ssg_path(ssg_dir: Path, task: Phase0Task) -> Path:
    repo_slug = task.repo.replace("/", "__")
    commit_short = task.base_commit[:12]
    return ssg_dir / repo_slug / commit_short / "ssg.json"


def _reciprocal_rank(retrieved: list[str], gold: list[str]) -> float:
    gold_set = set(gold)
    for i, item in enumerate(retrieved, start=1):
        if item in gold_set:
            return 1.0 / i
    return 0.0


def _compute_metrics(
    ranked_files: list[str], gold_files: list[str],
) -> dict[str, float]:
    metrics: dict[str, float] = {}
    for k in K_VALUES:
        metrics[f"P@{k}"] = precision_at_k(ranked_files, gold_files, k=k)
        metrics[f"R@{k}"] = recall_at_k(ranked_files, gold_files, k=k)
    metrics["MAP"] = average_precision(ranked_files, gold_files)
    metrics["MRR"] = _reciprocal_rank(ranked_files, gold_files)
    return metrics


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


def load_explicit_split(
    tasks: list[Phase0Task], split_file: Path,
) -> tuple[list[Phase0Task], list[Phase0Task]]:
    split = json.loads(split_file.read_text())
    train_ids, val_ids = set(split["train"]), set(split["val"])
    by_id = {t.instance_id: t for t in tasks}

    train_tasks = [by_id[i] for i in split["train"] if i in by_id]
    val_tasks = [by_id[i] for i in split["val"] if i in by_id]

    missing = (train_ids | val_ids) - by_id.keys()
    if missing:
        raise ValueError(
            f"{len(missing)} instance_ids in {split_file} not found in the manifest: "
            f"{sorted(missing)[:10]}"
        )
    return train_tasks, val_tasks


def build_reranker_inputs(
    *,
    tasks: list[Phase0Task],
    ssg_dir: Path,
    embedder: Any,
    vector_store: EntityVectorStore,
    retriever: DenseRetriever,
    bm25_retriever: BM25Retriever,
    dataset_name: str,
    split: str,
    label: str = "",
    expand_depth: int = 0,
    expand_seeds: int = 5,
    expand_max_neighbors: int = 200,
) -> list[RerankerInput]:
    inputs: list[RerankerInput] = []
    skipped = 0

    for i, task in enumerate(tasks):
        logger.info("[%s %d/%d] %s", label, i + 1, len(tasks), task.instance_id)

        ssg = _ssg_path(ssg_dir, task)
        if not ssg.exists():
            logger.warning("  SSG not found: %s — skipping", ssg)
            skipped += 1
            continue

        try:
            metadata = load_swebench_task_metadata(
                task.instance_id, dataset_name=dataset_name, split=split,
            )
        except KeyError:
            logger.warning("  Task not in dataset — skipping")
            skipped += 1
            continue

        issue_text = metadata.problem_statement
        commit_short = task.base_commit[:12]
        collection_repo = f"{task.repo}@{commit_short}"

        graph = read_ssg_json(ssg)

        dense_result = retriever.retrieve(
            issue_text=issue_text, repo=collection_repo, k=KNN_CANDIDATES,
        )
        if not dense_result.entities:
            logger.warning("  KNN returned 0 entities — skipping")
            skipped += 1
            continue

        try:
            bm25_result = bm25_retriever.retrieve(
                issue_text=issue_text, repo=collection_repo, k=KNN_CANDIDATES,
            )
            knn_result = merge_retrieval_results(dense_result, bm25_result)
            logger.info(
                "  Hybrid: %d dense + %d BM25 → %d merged candidates",
                len(dense_result.entities), len(bm25_result.entities), len(knn_result.entities),
            )
        except Exception as exc:
            logger.warning("  BM25 failed (%s), using dense only", exc)
            knn_result = dense_result

        if expand_depth:
            before = len(knn_result.file_paths)
            knn_result = expand_with_graph_neighbors(
                knn_result, graph, n_seeds=expand_seeds,
                max_depth=expand_depth, max_neighbors=expand_max_neighbors,
            )
            logger.info(
                "  Graph expansion (%d seeds, %d hops): %d → %d files",
                expand_seeds, expand_depth, before, len(knn_result.file_paths),
            )

        entry_result = identify_entry_points_by_embedding(
            issue_text=issue_text,
            repo=collection_repo,
            embedder=embedder,
            vector_store=vector_store,
            graph=graph,
            n_candidates=5,
        )
        entry_node_ids = entry_result.matched_node_ids

        features = extract_candidate_features(
            knn_result=knn_result,
            graph=graph,
            entry_point_node_ids=entry_node_ids,
            issue_text=issue_text,
        )

        inp = RerankerInput(
            instance_id=task.instance_id,
            candidate_features=features,
            gold_files=task.gold_patch_files,
            retrieved_entities=list(knn_result.entities),
        )

        n_gold_in_candidates = sum(1 for f in features if f.file_path in set(task.gold_patch_files))
        logger.info(
            "  → %d candidates, %d gold, %d gold-in-candidates, %d entry points",
            len(features), len(task.gold_patch_files), n_gold_in_candidates,
            len(entry_node_ids),
        )
        inputs.append(inp)

    logger.info(
        "%s feature extraction done: %d tasks OK, %d skipped",
        label, len(inputs), skipped,
    )
    return inputs


def evaluate_model(
    *,
    model_name: str,
    inputs: list[RerankerInput],
    score_fn: Any,
    top_k: int = 50,
) -> dict[str, Any]:
    agg: dict[str, list[float]] = defaultdict(list)
    per_task: list[dict[str, Any]] = []

    for inp in inputs:
        if not inp.candidate_features:
            continue

        ranked = score_fn(inp.candidate_features, top_k=top_k)
        ranked_files = [fp for fp, _ in ranked]

        metrics = _compute_metrics(ranked_files, inp.gold_files)
        for k, v in metrics.items():
            agg[k].append(v)

        per_task.append({
            "instance_id": inp.instance_id,
            "metrics": metrics,
        })

    means = {k: sum(v) / len(v) if v else 0.0 for k, v in agg.items()}
    return {
        "model": model_name,
        "n_tasks": len(per_task),
        "aggregate": means,
        "per_task": per_task,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train graph-aware reranker")
    parser.add_argument("--manifest", type=Path, default=Path("configs/phases/phase0_tasks.json"))
    parser.add_argument(
        "--split-file", type=Path, default=None,
        help=(
            "Optional {'train': [...instance_ids], 'val': [...instance_ids]} JSON "
            "(e.g. configs/phases/reranker_split.json) giving the exact train/val "
            "split to use. When set, skips stratified_split()'s own re-derived "
            "grouping so the run matches a previously recorded/validated split "
            "(e.g. one already checked for train/eval contamination)."
        ),
    )
    parser.add_argument("--ssg-dir", type=Path, default=Path("data/ssg"))
    parser.add_argument("--chromadb-dir", type=Path, default=Path("data/chromadb"))
    parser.add_argument("--output-dir", type=Path, default=Path("experiments/reranker"))
    parser.add_argument("--dataset", default="SWE-bench-Live/SWE-bench-Live")
    parser.add_argument("--split", default="verified")
    parser.add_argument("--embedding-provider", default="sentence-transformers")
    parser.add_argument("--embedding-model", default="jinaai/jina-code-embeddings-0.5b")
    parser.add_argument("--n-estimators", type=int, default=300)
    parser.add_argument("--max-depth", type=int, default=6)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--neural-epochs", type=int, default=80)
    parser.add_argument("--neural-lr", type=float, default=5e-4)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    np.random.seed(args.seed)
    random.seed(args.seed)

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest = load_phase0_manifest(args.manifest)
    logger.info("Loaded manifest: %d tasks", len(manifest.tasks))

    embed_cfg = EmbeddingConfig(
        provider=args.embedding_provider,
        name=args.embedding_model,
    )
    embedder = create_embedder(embed_cfg)
    vector_store = EntityVectorStore(
        persist_directory=str(args.chromadb_dir),
        collection_prefix="token_sniper",
    )
    retriever = DenseRetriever(embedder=embedder, vector_store=vector_store)
    bm25_retriever = BM25Retriever(vector_store=vector_store)

    if args.split_file is not None:
        train_tasks, val_tasks = load_explicit_split(manifest.tasks, args.split_file)
        logger.info(
            "Split (from %s): %d train, %d val", args.split_file, len(train_tasks), len(val_tasks),
        )
    else:
        train_tasks, val_tasks = stratified_split(
            manifest.tasks, train_ratio=TRAIN_RATIO, seed=args.seed,
        )
        logger.info("Split (re-derived, seed=%d): %d train, %d val", args.seed, len(train_tasks), len(val_tasks))

    t0 = time.monotonic()
    train_inputs = build_reranker_inputs(
        tasks=train_tasks,
        ssg_dir=args.ssg_dir,
        embedder=embedder,
        vector_store=vector_store,
        retriever=retriever,
        bm25_retriever=bm25_retriever,
        dataset_name=args.dataset,
        split=args.split,
        label="TRAIN",
    )
    val_inputs = build_reranker_inputs(
        tasks=val_tasks,
        ssg_dir=args.ssg_dir,
        embedder=embedder,
        vector_store=vector_store,
        retriever=retriever,
        bm25_retriever=bm25_retriever,
        dataset_name=args.dataset,
        split=args.split,
        label="VAL",
    )
    feat_time = time.monotonic() - t0
    logger.info("Feature extraction: %.1fs", feat_time)

    _save_inputs(train_inputs, output_dir / "train_inputs.json")
    _save_inputs(val_inputs, output_dir / "val_inputs.json")

    knn_baseline = evaluate_model(
        model_name="knn_baseline",
        inputs=val_inputs,
        score_fn=lambda feats, top_k: [
            (f.file_path, f.knn_score_max) for f in
            sorted(feats, key=lambda x: -x.knn_score_max)[:top_k]
        ],
    )
    logger.info("KNN baseline (val): %s", json.dumps(knn_baseline["aggregate"], indent=2))

    logger.info("=== Training XGBoost reranker ===")
    xgb_reranker = GraphReranker()
    t0 = time.monotonic()
    xgb_result = xgb_reranker.fit(
        train_inputs,
        val_inputs=val_inputs,
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        learning_rate=args.learning_rate,
    )
    xgb_train_time = time.monotonic() - t0
    logger.info("XGBoost training: %.1fs", xgb_train_time)

    xgb_reranker.save(output_dir / "xgb_reranker.json")

    xgb_eval = evaluate_model(
        model_name="xgb_reranker",
        inputs=val_inputs,
        score_fn=lambda feats, top_k: xgb_reranker.rerank(feats, top_k=top_k),
    )
    logger.info("XGBoost reranker (val): %s", json.dumps(xgb_eval["aggregate"], indent=2))

    logger.info("=== Training Neural reranker ===")
    neural_reranker = NeuralReranker()
    t0 = time.monotonic()
    neural_result = neural_reranker.fit(
        train_inputs,
        val_inputs=val_inputs,
        epochs=args.neural_epochs,
        lr=args.neural_lr,
    )
    neural_train_time = time.monotonic() - t0
    logger.info("Neural training: %.1fs", neural_train_time)

    neural_eval = evaluate_model(
        model_name="neural_reranker",
        inputs=val_inputs,
        score_fn=lambda feats, top_k: [
            (f.file_path, s) for f, s in
            sorted(
                zip(feats, neural_reranker.score(feats)),
                key=lambda x: -x[1],
            )[:top_k]
        ],
    )
    logger.info("Neural reranker (val): %s", json.dumps(neural_eval["aggregate"], indent=2))

    summary = {
        "seed": args.seed,
        "n_train_tasks": len(train_inputs),
        "n_val_tasks": len(val_inputs),
        "feature_extraction_time_s": round(feat_time, 1),
        "xgb": {
            "train_time_s": round(xgb_train_time, 1),
            "n_estimators": args.n_estimators,
            "max_depth": args.max_depth,
            "learning_rate": args.learning_rate,
            "training_result": _json_safe(xgb_result),
            "val_metrics": xgb_eval["aggregate"],
        },
        "neural": {
            "train_time_s": round(neural_train_time, 1),
            "epochs": args.neural_epochs,
            "lr": args.neural_lr,
            "training_result": _json_safe(neural_result),
            "val_metrics": neural_eval["aggregate"],
        },
        "knn_baseline_metrics": knn_baseline["aggregate"],
        "lift_over_knn": {
            metric: {
                "knn": knn_baseline["aggregate"].get(metric, 0.0),
                "xgb": xgb_eval["aggregate"].get(metric, 0.0),
                "neural": neural_eval["aggregate"].get(metric, 0.0),
                "xgb_lift": (
                    xgb_eval["aggregate"].get(metric, 0.0)
                    - knn_baseline["aggregate"].get(metric, 0.0)
                ),
                "neural_lift": (
                    neural_eval["aggregate"].get(metric, 0.0)
                    - knn_baseline["aggregate"].get(metric, 0.0)
                ),
            }
            for metric in ["R@5", "R@10", "R@20", "MAP", "MRR"]
        },
    }

    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8",
    )
    (output_dir / "knn_baseline_val.json").write_text(
        json.dumps(knn_baseline, indent=2), encoding="utf-8",
    )
    (output_dir / "xgb_eval_val.json").write_text(
        json.dumps(xgb_eval, indent=2), encoding="utf-8",
    )
    (output_dir / "neural_eval_val.json").write_text(
        json.dumps(neural_eval, indent=2), encoding="utf-8",
    )

    logger.info("=== Training Complete ===")
    logger.info("Results in %s", output_dir)
    _print_comparison_table(summary)


def _print_comparison_table(summary: dict[str, Any]) -> None:
    lift = summary.get("lift_over_knn", {})
    print("\n" + "=" * 70)
    print(f"{'Metric':<10} {'KNN':>10} {'XGBoost':>10} {'Neural':>10} {'XGB Lift':>10} {'NN Lift':>10}")
    print("-" * 70)
    for metric in ["R@5", "R@10", "R@20", "MAP", "MRR"]:
        m = lift.get(metric, {})
        print(
            f"{metric:<10} "
            f"{m.get('knn', 0):.4f}     "
            f"{m.get('xgb', 0):.4f}     "
            f"{m.get('neural', 0):.4f}     "
            f"{m.get('xgb_lift', 0):+.4f}     "
            f"{m.get('neural_lift', 0):+.4f}"
        )
    print("=" * 70 + "\n")


def _save_inputs(inputs: list[RerankerInput], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    records = []
    for inp in inputs:
        records.append({
            "instance_id": inp.instance_id,
            "gold_files": inp.gold_files,
            "n_candidates": len(inp.candidate_features),
            "candidate_files": [f.file_path for f in inp.candidate_features],
            "feature_matrix": inp.to_feature_matrix().tolist(),
            "relevance_labels": inp.relevance_labels().tolist(),
        })
    path.write_text(json.dumps(records, indent=2), encoding="utf-8")
    logger.info("Saved %d inputs to %s", len(records), path)


def _json_safe(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


if __name__ == "__main__":
    main()

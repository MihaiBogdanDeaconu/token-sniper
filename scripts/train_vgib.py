from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import torch

from token_sniper.evaluation.phase0 import Phase0Task, load_phase0_manifest
from token_sniper.evaluation.swebench import load_swebench_task_metadata
from token_sniper.graph.io import read_ssg_json
from token_sniper.retrieval.config import EmbeddingConfig
from token_sniper.retrieval.embeddings import create_embedder
from token_sniper.retrieval.entry_points import identify_entry_points_by_embedding
from token_sniper.retrieval.retriever import (
    BM25Retriever,
    DenseRetriever,
    expand_with_graph_neighbors,
    merge_retrieval_results,
)
from token_sniper.retrieval.vectorstore import EntityVectorStore
from token_sniper.vgib_selector.dataset import (
    RerankedVGIBTask,
    load_reranked_vgib_task,
    structural_prior_inputs,
)
from token_sniper.vgib_selector.features import ChromaSourceEncoder
from token_sniper.vgib_selector.gold_nodes import gold_nodes, parse_patch
from token_sniper.vgib_selector.token_counts import read_token_sidecar
from token_sniper.vgib_selector.train import VGIBTrainConfig, train_vgib_compression

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


KNN_CANDIDATES = 500


def _ssg_path_for_task(ssg_dir: Path, repo: str, base_commit: str) -> Path:
    repo_slug = repo.replace("/", "__")
    commit_short = base_commit[:12]
    return ssg_dir / repo_slug / commit_short / "ssg.json"


def load_split_records(split_path: Path, records: list[dict]) -> tuple[list[dict], list[dict]]:
    split = json.loads(split_path.read_text(encoding="utf-8"))
    train_ids, val_ids = split["train"], split["val"]
    split_ids = set(train_ids) | set(val_ids)
    if len(split_ids) != len(train_ids) + len(val_ids):
        raise ValueError(f"{split_path}: duplicate instance_ids or instances in both train and val")

    by_id = {r["instance_id"]: r for r in records}
    if len(by_id) != len(records):
        raise ValueError("candidate records contain duplicate instance_ids")
    missing = sorted(split_ids - by_id.keys())
    extra = sorted(by_id.keys() - split_ids)
    if missing or extra:
        raise ValueError(
            f"{split_path} does not match the candidate records: "
            f"{len(missing)} split instances have no record {missing[:10]}, "
            f"{len(extra)} records are not in the split {extra[:10]}"
        )
    return [by_id[i] for i in train_ids], [by_id[i] for i in val_ids]


def check_loaded_tasks(
    records: list[dict],
    tasks: list[RerankedVGIBTask],
    dropped: dict[str, str],
    label: str,
) -> None:
    lost = sorted({r["instance_id"] for r in records} - {t.instance_id for t in tasks} - dropped.keys())
    if lost:
        raise RuntimeError(f"{label}: {len(lost)} split instances failed to load: {lost[:10]}")
    if dropped:
        logger.warning("%s: dropped %d instances with no gold node in the subgraph: %s",
                       label, len(dropped), sorted(dropped)[:10])
    uncounted = [t.instance_id for t in tasks if t.token_counts is None]
    if uncounted:
        raise RuntimeError(f"{label}: {len(uncounted)} tasks have no token counts: {uncounted[:10]}")


def generate_candidates_via_retrieval(
    *,
    manifest_tasks: list[Phase0Task],
    ssg_dir: Path,
    embedder: object,
    vector_store: EntityVectorStore,
    dataset_name: str,
    split: str,
    k: int = KNN_CANDIDATES,
    use_graph_expansion: bool = True,
) -> list[dict]:
    dense_retriever = DenseRetriever(embedder=embedder, vector_store=vector_store)
    bm25_retriever = BM25Retriever(vector_store=vector_store)
    records: list[dict] = []

    for i, task in enumerate(manifest_tasks):
        ssg_path = _ssg_path_for_task(ssg_dir, task.repo, task.base_commit)
        if not ssg_path.exists():
            continue

        try:
            metadata = load_swebench_task_metadata(
                task.instance_id, dataset_name=dataset_name, split=split,
            )
        except KeyError:
            continue

        commit_short = task.base_commit[:12]
        collection_repo = f"{task.repo}@{commit_short}"
        issue_text = metadata.problem_statement

        try:
            dense_result = dense_retriever.retrieve(
                issue_text=issue_text, repo=collection_repo, k=k,
            )
            if not dense_result.entities:
                continue

            try:
                bm25_result = bm25_retriever.retrieve(
                    issue_text=issue_text, repo=collection_repo, k=k,
                )
                merged = merge_retrieval_results(dense_result, bm25_result)
            except Exception:
                merged = dense_result

            if use_graph_expansion:
                try:
                    graph = read_ssg_json(ssg_path)
                    merged = expand_with_graph_neighbors(
                        merged, graph,
                        n_seeds=5, max_depth=3, max_neighbors=200,
                    )
                except Exception:
                    pass

            candidate_files = sorted(set(e.file_path for e in merged.entities if e.file_path))
            records.append({
                "instance_id": task.instance_id,
                "gold_files": task.gold_patch_files,
                "candidate_files": candidate_files,
            })

            if (i + 1) % 50 == 0:
                n_gold_found = sum(
                    1 for r in records
                    if any(g in r["candidate_files"] for g in r["gold_files"])
                )
                logger.info(
                    "Retrieval progress: %d/%d tasks, %d records, %d with gold (%.1f%%)",
                    i + 1, len(manifest_tasks), len(records),
                    n_gold_found, 100 * n_gold_found / max(len(records), 1),
                )

        except Exception:
            logger.warning("[%d/%d] %s — retrieval failed", i + 1, len(manifest_tasks), task.instance_id, exc_info=True)

    logger.info("Generated %d candidate records via retrieval", len(records))
    return records


def load_reranked_tasks(
    *,
    reranker_inputs: list[dict],
    task_lookup: dict[str, dict],
    ssg_dir: Path,
    encoder: ChromaSourceEncoder,
    embedder: object,
    feature_cache_dir: Path,
    issue_cache_dir: Path,
    dataset_name: str,
    split: str,
    label: str = "",
    prior_vector_store: EntityVectorStore | None = None,
) -> tuple[list[RerankedVGIBTask], dict[str, str]]:
    tasks: list[RerankedVGIBTask] = []
    dropped: dict[str, str] = {}
    skipped = 0

    for i, record in enumerate(reranker_inputs):
        instance_id = record["instance_id"]
        candidate_files = record["candidate_files"]
        gold_files = record["gold_files"]

        if instance_id not in task_lookup:
            logger.warning("[%s %d/%d] %s not in manifest — skipping", label, i + 1, len(reranker_inputs), instance_id)
            skipped += 1
            continue

        task_info = task_lookup[instance_id]
        repo = task_info["repo"]
        base_commit = task_info["base_commit"]

        ssg = _ssg_path_for_task(ssg_dir, repo, base_commit)
        if not ssg.exists():
            logger.warning("[%s %d/%d] SSG not found: %s — skipping", label, i + 1, len(reranker_inputs), ssg)
            skipped += 1
            continue

        try:
            metadata = load_swebench_task_metadata(
                instance_id, dataset_name=dataset_name, split=split,
            )
        except KeyError:
            logger.warning("[%s %d/%d] %s not in dataset — skipping", label, i + 1, len(reranker_inputs), instance_id)
            skipped += 1
            continue

        commit_short = base_commit[:12]
        collection_repo = f"{repo}@{commit_short}"
        node_tokens = read_token_sidecar(ssg)

        try:
            graph = read_ssg_json(ssg)
            labels = gold_nodes(graph, parse_patch(metadata.gold_patch))
            vgib_task = load_reranked_vgib_task(
                instance_id=instance_id,
                repo=repo,
                issue_text=metadata.problem_statement,
                candidate_files=candidate_files,
                gold_files=gold_files,
                graph=graph,
                encoder=encoder,
                embedder=embedder,
                feature_cache_dir=feature_cache_dir,
                issue_cache_dir=issue_cache_dir,
                collection_repo=collection_repo,
                node_tokens=node_tokens,
                gold_node_ids=set(labels.nodes),
            )

            if prior_vector_store is not None:
                entry = identify_entry_points_by_embedding(
                    issue_text=metadata.problem_statement, repo=collection_repo, embedder=embedder,
                    vector_store=prior_vector_store, graph=graph, n_candidates=5,
                )
                if entry.error:
                    raise RuntimeError(f"seed entities for the structural prior: {entry.error}")
                vgib_task.prior_ppr, vgib_task.prior_seed_mask = structural_prior_inputs(
                    graph, entry.matched_node_ids, vgib_task.features.node_ids,
                )

            n_gold_nodes = int(vgib_task.gold_node_mask.sum().item())
            if n_gold_nodes == 0:
                dropped[instance_id] = (
                    f"none of its {len(labels.nodes)} gold nodes is in the candidate subgraph"
                )
                logger.warning("[%s %d/%d] %s — %s", label, i + 1, len(reranker_inputs),
                               instance_id, dropped[instance_id])
                continue

            logger.info(
                "[%s %d/%d] %s — %d nodes, %d edges, %d candidate files, "
                "%d gold files, %d gold nodes",
                label, i + 1, len(reranker_inputs), instance_id,
                len(vgib_task.features.node_ids),
                vgib_task.edge_index.shape[1] if vgib_task.edge_index.numel() else 0,
                len(candidate_files),
                int(vgib_task.gold_file_mask.sum().item()),
                n_gold_nodes,
            )
            tasks.append(vgib_task)
        except Exception:
            logger.warning(
                "[%s %d/%d] %s — failed to load",
                label, i + 1, len(reranker_inputs), instance_id,
                exc_info=True,
            )
            skipped += 1

        if (i + 1) % 50 == 0:
            logger.info("[%s] Progress: %d/%d loaded, %d skipped", label, len(tasks), i + 1, skipped)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    logger.info("[%s] Loaded %d tasks (%d skipped, %d without a gold node)",
                label, len(tasks), skipped, len(dropped))
    return tasks, dropped


def main() -> None:
    parser = argparse.ArgumentParser(description="Train VGIB context compression selector")
    parser.add_argument("--manifest", type=Path, default=Path("configs/phases/phase0_tasks.json"))
    parser.add_argument("--split-json", type=Path, required=True,
                        help="Train/val split shared with the reranker and GRPO (configs/phases/reranker_split.json)")
    parser.add_argument("--reranker-dir", type=Path, default=Path("experiments/reranker"),
                        help="Directory with train_inputs.json and val_inputs.json from reranker training")
    parser.add_argument("--ssg-dir", type=Path, default=Path("data/ssg"))
    parser.add_argument("--chromadb-dir", type=Path, default=Path("data/chromadb"))
    parser.add_argument("--output-dir", type=Path, default=Path("experiments/vgib"))
    parser.add_argument("--feature-cache-dir", type=Path, default=None,
                        help="Node feature cache, keyed by content. Keep it outside --output-dir "
                             "so it survives a re-clone (default: <output-dir>/feature_cache)")
    parser.add_argument("--issue-cache-dir", type=Path, default=None,
                        help="Issue embedding cache (default: <output-dir>/issue_cache)")
    parser.add_argument("--dataset", default="SWE-bench-Live/SWE-bench-Live")
    parser.add_argument("--split", default="verified")
    parser.add_argument("--embedding-provider", default="sentence-transformers")
    parser.add_argument("--embedding-model", default="jinaai/jina-code-embeddings-0.5b")
    parser.add_argument("--embedding-dim", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--beta", type=float, default=0.01, help="Token ratio loss weight")
    parser.add_argument("--gamma", type=float, default=0.001, help="MI loss weight")
    parser.add_argument("--hidden-dim", type=int, default=384)
    parser.add_argument("--out-dim", type=int, default=256)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default=None)
    ablation = parser.add_argument_group("ablations (Task 13, model side)")
    ablation.add_argument("--no-graph-edges", action="store_true",
                          help="Message passing sees no edges (edge_index = empty); node features unchanged")
    ablation.add_argument("--pooling", choices=("cross_attention", "mean"), default="cross_attention",
                          help="Graph pooling. The compression loss never reads the pooled prediction, "
                               "so this cannot change a compression run")
    ablation.add_argument("--no-lexical-features", action="store_true",
                          help="Drop the issue-conditioned lexical node features")
    prior = parser.add_argument_group("structural prior (Task 11)")
    prior.add_argument("--prior-weight", type=float, default=0.0,
                       help="Weight of mean_i KL(Bern(p_i) || Bern(pi_i)); 0 = no prior")
    prior.add_argument("--prior-alpha", type=float, default=1.0,
                       help="alpha: weight of PPR(i | issue), normalised to the subgraph's max")
    prior.add_argument("--prior-gamma-s", type=float, default=1.0,
                       help="gamma_s: weight of 1[i in N(seed)]")
    prior.add_argument("--prior-base-rate", type=float, default=0.5,
                       help="pi of a node with no PPR outside N(seed); 0.5 is the plain formula")
    args = parser.parse_args()

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest = load_phase0_manifest(args.manifest)
    task_lookup: dict[str, dict] = {}
    for t in manifest.tasks:
        task_lookup[t.instance_id] = {
            "repo": t.repo,
            "base_commit": t.base_commit,
            "gold_patch_files": t.gold_patch_files,
        }
    logger.info("Manifest: %d tasks", len(task_lookup))

    embed_cfg = EmbeddingConfig(
        provider=args.embedding_provider,
        name=args.embedding_model,
        dimension=args.embedding_dim,
    )
    embedder = create_embedder(embed_cfg)
    logger.info("Embedder: %s (dim=%d)", args.embedding_model, embed_cfg.dimension)

    vector_store = EntityVectorStore(
        persist_directory=str(args.chromadb_dir),
        collection_prefix="token_sniper",
    )

    train_inputs_path = args.reranker_dir / "train_inputs.json"
    val_inputs_path = args.reranker_dir / "val_inputs.json"

    if train_inputs_path.exists() and val_inputs_path.exists():
        train_records: list[dict] = json.loads(train_inputs_path.read_text(encoding="utf-8"))
        val_records: list[dict] = json.loads(val_inputs_path.read_text(encoding="utf-8"))
        all_records = train_records + val_records
        logger.info(
            "Reranker outputs found: %d train + %d val = %d total tasks",
            len(train_records), len(val_records), len(all_records),
        )
    else:
        logger.info(
            "Reranker outputs not found at %s — generating candidates via retrieval",
            args.reranker_dir,
        )
        all_records = generate_candidates_via_retrieval(
            manifest_tasks=manifest.tasks,
            ssg_dir=args.ssg_dir,
            embedder=embedder,
            vector_store=vector_store,
            dataset_name=args.dataset,
            split=args.split,
        )
        logger.info("Generated %d candidate records via retrieval", len(all_records))

        cached_records_path = output_dir / "retrieval_candidates.json"
        cached_records_path.write_text(json.dumps(all_records, indent=2), encoding="utf-8")
        logger.info("Cached retrieval candidates to %s", cached_records_path)

    encoder = ChromaSourceEncoder(
        chromadb_dir=args.chromadb_dir,
        collection_prefix="token_sniper",
        model_name=args.embedding_model,
        embedding_dim=args.embedding_dim,
    )

    feature_cache_dir = args.feature_cache_dir or output_dir / "feature_cache"
    feature_cache_dir.mkdir(parents=True, exist_ok=True)
    issue_cache_dir = args.issue_cache_dir or output_dir / "issue_cache"
    issue_cache_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Feature cache: %s | issue cache: %s", feature_cache_dir, issue_cache_dir)

    train_records, val_records = load_split_records(args.split_json, all_records)
    logger.info("Split from %s: %d train, %d val", args.split_json, len(train_records), len(val_records))
    prior_vector_store = vector_store if args.prior_weight > 0 else None

    logger.info("=== Loading reranked VGIB tasks ===")
    t0 = time.monotonic()
    train_tasks, train_dropped = load_reranked_tasks(
        reranker_inputs=train_records,
        task_lookup=task_lookup,
        ssg_dir=args.ssg_dir,
        encoder=encoder,
        embedder=embedder,
        feature_cache_dir=feature_cache_dir,
        issue_cache_dir=issue_cache_dir,
        dataset_name=args.dataset,
        split=args.split,
        label="TRAIN",
        prior_vector_store=prior_vector_store,
    )
    val_tasks, val_dropped = load_reranked_tasks(
        reranker_inputs=val_records,
        task_lookup=task_lookup,
        ssg_dir=args.ssg_dir,
        encoder=encoder,
        embedder=embedder,
        feature_cache_dir=feature_cache_dir,
        issue_cache_dir=issue_cache_dir,
        dataset_name=args.dataset,
        split=args.split,
        label="VAL",
        prior_vector_store=prior_vector_store,
    )
    check_loaded_tasks(train_records, train_tasks, train_dropped, "TRAIN")
    check_loaded_tasks(val_records, val_tasks, val_dropped, "VAL")
    tasks = train_tasks + val_tasks
    load_time = time.monotonic() - t0
    logger.info("Task loading: %.1fs for %d train + %d val tasks", load_time, len(train_tasks), len(val_tasks))

    if not tasks:
        logger.error("No tasks loaded — aborting")
        return

    total_nodes = sum(len(t.features.node_ids) for t in tasks)
    total_gold = sum(int(t.gold_file_mask.sum().item()) for t in tasks)
    total_gold_nodes = sum(int(t.gold_node_mask.sum().item()) for t in tasks)
    avg_nodes = total_nodes / len(tasks)
    logger.info(
        "Dataset: %d tasks, %d total nodes (avg %.0f/task), %d gold files, %d gold nodes",
        len(tasks), total_nodes, avg_nodes, total_gold, total_gold_nodes,
    )

    train_cfg = VGIBTrainConfig(
        output_dir=output_dir,
        beta=args.beta,
        gamma=args.gamma,
        alpha=0.0,
        target_retention=0.0,
        epochs=args.epochs,
        lr=args.lr,
        hidden_dim=args.hidden_dim,
        out_dim=args.out_dim,
        patience=args.patience,
        seed=args.seed,
        device=args.device,
        mlflow_experiment="phase_c_vgib_compression",
        lexical_features=not args.no_lexical_features,
        use_edges=not args.no_graph_edges,
        pooling=args.pooling,
        prior_weight=args.prior_weight,
        prior_alpha=args.prior_alpha,
        prior_gamma_s=args.prior_gamma_s,
        prior_base_rate=args.prior_base_rate,
    )

    logger.info("=== Training VGIB (compression mode) ===")
    t0 = time.monotonic()
    best_path = train_vgib_compression(train_tasks, train_cfg, val_tasks=val_tasks)
    train_time = time.monotonic() - t0
    logger.info("Training complete in %.1fs — best checkpoint: %s", train_time, best_path)

    summary: dict = {
        "mode": "compression",
        "split_json": str(args.split_json),
        "n_train_tasks": len(train_tasks),
        "n_val_tasks": len(val_tasks),
        "n_tasks": len(tasks),
        "dropped_without_gold_node": {"train": train_dropped, "val": val_dropped},
        "total_nodes": total_nodes,
        "avg_nodes_per_task": round(avg_nodes, 1),
        "total_gold_files": total_gold,
        "total_gold_nodes": total_gold_nodes,
        "embedding_model": args.embedding_model,
        "embedding_dim": args.embedding_dim,
        "load_time_s": round(load_time, 1),
        "train_time_s": round(train_time, 1),
        "config": {
            "beta": args.beta,
            "gamma": args.gamma,
            "epochs": args.epochs,
            "lr": args.lr,
            "hidden_dim": args.hidden_dim,
            "out_dim": args.out_dim,
            "patience": args.patience,
            "seed": args.seed,
            "lexical_features": train_cfg.lexical_features,
            "use_edges": train_cfg.use_edges,
            "pooling": train_cfg.pooling,
            "prior_weight": train_cfg.prior_weight,
            "prior_alpha": train_cfg.prior_alpha,
            "prior_gamma_s": train_cfg.prior_gamma_s,
            "prior_base_rate": train_cfg.prior_base_rate,
        },
        "best_checkpoint": str(best_path),
    }

    if best_path.exists():
        ckpt = torch.load(best_path, map_location="cpu", weights_only=False)
        summary["best_epoch"] = ckpt.get("epoch")
        summary["best_val_loss"] = ckpt.get("best_val_loss")
        summary["best_val_gold_percentile"] = ckpt.get("best_val_gold_percentile")
        summary["best_val_func_at_10"] = ckpt.get("best_val_func_at_10")
        summary["val_token_ratio"] = ckpt.get("val_token_ratio")
        summary["first_epoch_within_budget"] = ckpt.get("first_epoch_within_budget")
        summary["feature_versions"] = ckpt.get("feature_versions")
    else:
        logger.error("No checkpoint was saved: no epoch reached val token ratio <= %.2f",
                     train_cfg.max_val_token_ratio)

    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8",
    )

    logger.info("=== VGIB Training Complete ===")
    logger.info("Summary: %s", json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

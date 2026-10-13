from __future__ import annotations

import argparse
import gc
import json
import logging
import sys
from concurrent.futures import BrokenExecutor, ProcessPoolExecutor, as_completed
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)


_worker_encoder = None


def _init_worker(encoder_type: str, chromadb_dir: str | None) -> None:
    global _worker_encoder
    from token_sniper.vgib_selector.features import ChromaSourceEncoder, SentenceTransformerSourceEncoder

    if encoder_type == "chromadb":
        _worker_encoder = ChromaSourceEncoder(chromadb_dir=chromadb_dir)
    else:
        _worker_encoder = SentenceTransformerSourceEncoder()


def _create_encoder(encoder_type: str, chromadb_dir: str | None):
    from token_sniper.vgib_selector.features import ChromaSourceEncoder, SentenceTransformerSourceEncoder

    if encoder_type == "chromadb":
        return ChromaSourceEncoder(chromadb_dir=chromadb_dir)
    return SentenceTransformerSourceEncoder()


def _build_one_task(
    task_dict: dict,
    ssg_path: str,
    trace_path: str,
    feature_cache: str,
    candidate_cache: str,
) -> "TaskGraph":  # noqa: F821
    from token_sniper.edge_pred.dataset import load_task_graph

    return load_task_graph(
        instance_id=task_dict["instance_id"],
        repo=task_dict["repo"],
        ssg_path=Path(ssg_path),
        trace_path=Path(trace_path),
        encoder=_worker_encoder,
        feature_cache_dir=Path(feature_cache),
        candidate_cache_dir=Path(candidate_cache),
    )


def _resolve_paths(task: dict, ssg_dir: Path, traces_dir: Path) -> tuple[Path, Path] | None:
    repo_slug = task["repo"].replace("/", "__")
    commit_short = task["base_commit"][:12]
    ssg_path = ssg_dir / repo_slug / commit_short / "ssg.json"
    trace_path = traces_dir / f"{task['instance_id']}.json"
    if not ssg_path.exists() or not trace_path.exists():
        return None
    return ssg_path, trace_path


_PARALLEL_BATCH_SIZE = 50


def _load_all_task_graphs(
    all_available: list[dict],
    *,
    args: argparse.Namespace,
    chromadb_dir_str: str | None,
    feature_cache_str: str,
    candidate_cache_str: str,
) -> list:
    from token_sniper.edge_pred.dataset import load_task_graph

    all_tasks: list = []

    if args.loader_workers <= 1:
        encoder = _create_encoder(args.encoder, chromadb_dir_str)
        prev_repo: str | None = None
        for i, t in enumerate(all_available):
            resolved = _resolve_paths(t, args.ssg_dir, args.traces_dir)
            if resolved is None:
                continue
            ssg_path, trace_path = resolved
            try:
                tg = load_task_graph(
                    instance_id=t["instance_id"],
                    repo=t["repo"],
                    ssg_path=ssg_path,
                    trace_path=trace_path,
                    encoder=encoder,
                    feature_cache_dir=args.feature_cache,
                    candidate_cache_dir=args.candidate_cache,
                )
                all_tasks.append(tg)
            except Exception as exc:
                logger.warning("[%s] failed to build TaskGraph: %s", t["instance_id"], exc)

            if prev_repo is not None and t["repo"] != prev_repo:
                if hasattr(encoder, "evict_collection"):
                    encoder.evict_collection(prev_repo)
            prev_repo = t["repo"]

            if (i + 1) % 25 == 0 or (i + 1) == len(all_available):
                gc.collect()
                logger.info(
                    "Loaded %d/%d task graphs (%d succeeded)",
                    i + 1, len(all_available), len(all_tasks),
                )
    else:
        for batch_start in range(0, len(all_available), _PARALLEL_BATCH_SIZE):
            batch = all_available[batch_start : batch_start + _PARALLEL_BATCH_SIZE]
            batch_label = f"[batch {batch_start}-{batch_start + len(batch)}]"
            try:
                with ProcessPoolExecutor(
                    max_workers=args.loader_workers,
                    initializer=_init_worker,
                    initargs=(args.encoder, chromadb_dir_str),
                ) as pool:
                    futures = {}
                    for t in batch:
                        resolved = _resolve_paths(t, args.ssg_dir, args.traces_dir)
                        if resolved is None:
                            continue
                        ssg_path, trace_path = resolved
                        fut = pool.submit(
                            _build_one_task, t,
                            str(ssg_path), str(trace_path),
                            feature_cache_str, candidate_cache_str,
                        )
                        futures[fut] = t

                    for fut in as_completed(futures):
                        t = futures[fut]
                        try:
                            all_tasks.append(fut.result())
                        except Exception as exc:
                            logger.warning(
                                "%s [%s] failed: %s", batch_label, t["instance_id"], exc,
                            )
            except BrokenExecutor:
                logger.error(
                    "%s ProcessPool crashed (likely OOM) — skipping remaining tasks in batch",
                    batch_label,
                )

            gc.collect()
            logger.info(
                "%s complete — %d total task graphs loaded so far",
                batch_label, len(all_tasks),
            )

    return all_tasks


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Phase 2 edge predictor")
    parser.add_argument("--manifest", type=Path, default=Path("configs/phases/phase0_tasks.json"))
    parser.add_argument("--ssg-dir", type=Path, default=Path("data/ssg"))
    parser.add_argument("--traces-dir", type=Path, default=Path("data/traces"))
    parser.add_argument("--split-file", type=Path, default=Path("data/edge_pred/splits.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/edge_pred/checkpoints"))
    parser.add_argument("--feature-cache", type=Path, default=Path("data/edge_pred/features"))
    parser.add_argument("--candidate-cache", type=Path, default=Path("data/edge_pred/candidates"))
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--hidden-dim", type=int, default=384)
    parser.add_argument("--out-dim", type=int, default=256)
    parser.add_argument("--negative-ratio", type=int, default=5)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--mlflow-uri", type=str, default="file:./mlruns")
    parser.add_argument("--train-ratio", type=float, default=0.8,
                        help="Fraction of eligible tasks for training (rest = test). Repo-grouped.")
    parser.add_argument("--encoder", choices=["chromadb", "sentence-transformer"], default="chromadb",
                        help="Embedding backend: chromadb (offline lookup, default) or sentence-transformer (local GPU)")
    parser.add_argument("--chromadb-dir", type=Path, default=Path("data/chromadb"),
                        help="Path to ChromaDB directory with pre-computed Jina embeddings")
    parser.add_argument("--use-focal-loss", action="store_true", default=True,
                        help="Use focal loss instead of BCE (better for imbalanced data)")
    parser.add_argument("--no-focal-loss", dest="use_focal_loss", action="store_false")
    parser.add_argument("--loo-threshold", type=int, default=8,
                        help="Use leave-one-out CV when fewer than this many tasks have positives")
    parser.add_argument("--loader-workers", type=int, default=1,
                        help="Worker processes for building task graphs (0 or 1 = sequential, >1 = batched parallel)")
    parser.add_argument("--exclude-repos", type=str,
                        default="aws-cloudformation/cfn-lint,pylint-dev/pylint",
                        help="Comma-separated repos to skip (low-signal / high-load)")
    args = parser.parse_args()

    from token_sniper.edge_pred.dataset import stratified_split
    from token_sniper.edge_pred.train import TrainConfig, train_edge_predictor

    try:
        import mlflow
        mlflow.set_tracking_uri(args.mlflow_uri)
    except ImportError:
        logger.warning("mlflow not installed; training will proceed without tracking")

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))

    excluded = {r.strip() for r in args.exclude_repos.split(",") if r.strip()}
    if excluded:
        before = len(manifest)
        manifest = [t for t in manifest if t["repo"] not in excluded]
        logger.info(
            "Excluded %d/%d tasks from repos: %s",
            before - len(manifest), before, sorted(excluded),
        )

    by_id = {t["instance_id"]: t for t in manifest}

    args.feature_cache.mkdir(parents=True, exist_ok=True)
    args.candidate_cache.mkdir(parents=True, exist_ok=True)

    all_available = [t for t in manifest if _resolve_paths(t, args.ssg_dir, args.traces_dir) is not None]
    logger.info(
        "Building task graphs for %d available tasks with %d worker processes...",
        len(all_available), args.loader_workers,
    )

    chromadb_dir_str = str(args.chromadb_dir) if args.encoder == "chromadb" else None
    feature_cache_str = str(args.feature_cache)
    candidate_cache_str = str(args.candidate_cache)

    all_tasks = _load_all_task_graphs(
        all_available,
        args=args,
        chromadb_dir_str=chromadb_dir_str,
        feature_cache_str=feature_cache_str,
        candidate_cache_str=candidate_cache_str,
    )

    positive_tasks = [tg for tg in all_tasks if tg.positive_pairs.numel() > 0]
    if not positive_tasks:
        logger.error("No trainable tasks. Ensure Phase 2 Issue 1 (traces) completed successfully.")
        sys.exit(1)

    logger.info(
        "%d/%d tasks have positive edges (total positives: %d)",
        len(positive_tasks), len(all_tasks),
        sum(tg.positive_pairs.shape[0] for tg in positive_tasks),
    )

    base_cfg = TrainConfig(
        output_dir=args.output_dir,
        epochs=args.epochs,
        lr=args.lr,
        hidden_dim=args.hidden_dim,
        out_dim=args.out_dim,
        negative_ratio=args.negative_ratio,
        patience=args.patience,
        seed=args.seed,
        device=args.device,
        use_focal_loss=args.use_focal_loss,
    )

    use_loo = len(positive_tasks) < args.loo_threshold
    if use_loo:
        logger.info(
            "Only %d tasks with positives (< %d threshold), using leave-one-out CV",
            len(positive_tasks), args.loo_threshold,
        )
        _run_leave_one_out(positive_tasks, base_cfg, args)
    else:
        _run_standard_split(manifest, positive_tasks, base_cfg, args)


def _run_standard_split(
    manifest: list[dict], positive_tasks: list, cfg: "TrainConfig", args: argparse.Namespace
) -> None:
    from token_sniper.edge_pred.dataset import stratified_split
    from token_sniper.edge_pred.train import train_edge_predictor

    if args.split_file.exists():
        payload = json.loads(args.split_file.read_text(encoding="utf-8"))
        train_ids = set(payload["train"])
        logger.info("Loaded existing split: %d train tasks", len(train_ids))
    else:
        train_ids_list, test_ids = stratified_split(
            manifest,
            train_ratio=args.train_ratio,
            seed=args.seed,
            traces_dir=args.traces_dir,
            ssg_dir=args.ssg_dir,
        )
        train_ids = set(train_ids_list)
        args.split_file.parent.mkdir(parents=True, exist_ok=True)
        args.split_file.write_text(
            json.dumps({
                "train": train_ids_list,
                "test": test_ids,
                "train_ratio": args.train_ratio,
                "seed": args.seed,
                "method": "repo_grouped",
            }, indent=2),
            encoding="utf-8",
        )
        logger.info(
            "Generated split: %d train + %d test (ratio=%.2f, from %d manifest tasks)",
            len(train_ids_list), len(test_ids), args.train_ratio, len(manifest),
        )

    train_tasks = [tg for tg in positive_tasks if tg.instance_id in train_ids]
    if not train_tasks:
        logger.error("No positive tasks in the training split.")
        sys.exit(1)

    best_ckpt = train_edge_predictor(train_tasks, cfg)
    logger.info("Training complete. Best checkpoint: %s", best_ckpt)


def _run_leave_one_out(
    positive_tasks: list, cfg: "TrainConfig", args: argparse.Namespace
) -> None:
    from token_sniper.edge_pred.train import TrainConfig, train_edge_predictor

    folds: list[dict] = []
    fold_dir = cfg.output_dir / "loo_folds"
    fold_dir.mkdir(parents=True, exist_ok=True)

    all_ids = [tg.instance_id for tg in positive_tasks]
    args.split_file.parent.mkdir(parents=True, exist_ok=True)
    args.split_file.write_text(
        json.dumps({"train": all_ids, "test": all_ids, "mode": "leave_one_out"}, indent=2),
        encoding="utf-8",
    )

    for i, held_out in enumerate(positive_tasks):
        train_fold = [tg for j, tg in enumerate(positive_tasks) if j != i]
        if not train_fold:
            logger.warning("Fold %d: no training data when holding out %s", i, held_out.instance_id)
            continue
        logger.info(
            "=== LOO fold %d/%d: hold out %s, train on %d tasks ===",
            i + 1, len(positive_tasks), held_out.instance_id, len(train_fold),
        )
        fold_cfg = TrainConfig(
            output_dir=fold_dir / held_out.instance_id.replace("/", "__"),
            epochs=cfg.epochs,
            lr=cfg.lr,
            hidden_dim=cfg.hidden_dim,
            out_dim=cfg.out_dim,
            negative_ratio=cfg.negative_ratio,
            patience=cfg.patience,
            seed=cfg.seed,
            device=cfg.device,
            use_focal_loss=cfg.use_focal_loss,
        )
        ckpt = train_edge_predictor(train_fold, fold_cfg)
        folds.append({"held_out": held_out.instance_id, "checkpoint": str(ckpt)})

    logger.info("=== Training full model on all %d tasks ===", len(positive_tasks))
    full_ckpt = train_edge_predictor(positive_tasks, cfg)
    folds_manifest = {
        "mode": "leave_one_out",
        "folds": folds,
        "full_checkpoint": str(full_ckpt),
    }
    (cfg.output_dir / "loo_manifest.json").write_text(
        json.dumps(folds_manifest, indent=2), encoding="utf-8"
    )
    logger.info("LOO training complete. Full model: %s", full_ckpt)


if __name__ == "__main__":
    main()

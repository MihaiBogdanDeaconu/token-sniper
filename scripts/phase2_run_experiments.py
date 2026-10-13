from __future__ import annotations

import argparse
import gc
import json
import logging
from concurrent.futures import BrokenExecutor, ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

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


def _build_one_task(
    task_dict: dict,
    ssg_path: str,
    trace_path: str,
    feature_cache: str,
    candidate_cache: str,
) -> "TaskGraph | None":  # noqa: F821
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


def _load_test_tasks(args: argparse.Namespace):
    from token_sniper.vgib_selector.features import ChromaSourceEncoder, SentenceTransformerSourceEncoder

    split = json.loads(args.split_file.read_text(encoding="utf-8"))
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    by_id = {t["instance_id"]: t for t in manifest}

    excluded = {r.strip() for r in getattr(args, "exclude_repos", "").split(",") if r.strip()}
    if excluded:
        logger.info("Excluding repos from evaluation: %s", sorted(excluded))

    chromadb_dir = getattr(args, "chromadb_dir", None)
    encoder_type = "chromadb" if chromadb_dir is not None else "sentence-transformer"
    chromadb_dir_str = str(chromadb_dir) if chromadb_dir is not None else None

    def _prepare(instance_id: str):
        t = by_id.get(instance_id)
        if t is None:
            return None
        if t["repo"] in excluded:
            return None
        repo_slug = t["repo"].replace("/", "__")
        commit_short = t["base_commit"][:12]
        ssg_path = args.ssg_dir / repo_slug / commit_short / "ssg.json"
        trace_path = args.traces_dir / f"{instance_id}.json"
        if not ssg_path.exists() or not trace_path.exists():
            missing = []
            if not ssg_path.exists():
                missing.append(f"ssg ({ssg_path})")
            if not trace_path.exists():
                missing.append(f"trace ({trace_path})")
            logger.warning("[%s] skipped: missing %s", instance_id, ", ".join(missing))
            return None
        try:
            td = json.loads(trace_path.read_text(encoding="utf-8"))
            if len(td.get("edges", [])) == 0:
                logger.warning("[%s] skipped: trace has 0 edges", instance_id)
                return None
        except Exception:
            logger.warning("[%s] skipped: unreadable trace file", instance_id)
            return None
        return t, ssg_path, trace_path

    workers = getattr(args, "loader_workers", 1) or 1
    feature_cache_str = str(args.feature_cache)
    candidate_cache_str = str(args.candidate_cache)

    submit_items: list[tuple[str, dict, str, str]] = []
    for iid in split["test"]:
        prepped = _prepare(iid)
        if prepped is not None:
            t, ssg_path, trace_path = prepped
            submit_items.append((iid, t, str(ssg_path), str(trace_path)))

    logger.info("Loading %d test tasks with %d worker processes...", len(submit_items), workers)
    tasks = []

    if workers <= 1:
        from token_sniper.edge_pred.dataset import load_task_graph

        enc = ChromaSourceEncoder(chromadb_dir=chromadb_dir) if chromadb_dir else SentenceTransformerSourceEncoder()
        for i, (iid, t, ssg_str, trace_str) in enumerate(submit_items):
            try:
                tg = load_task_graph(
                    instance_id=t["instance_id"], repo=t["repo"],
                    ssg_path=Path(ssg_str), trace_path=Path(trace_str),
                    encoder=enc,
                    feature_cache_dir=args.feature_cache,
                    candidate_cache_dir=args.candidate_cache,
                )
                if tg is not None:
                    tasks.append(tg)
            except Exception:
                logger.exception("[%s] failed to build TaskGraph", iid)
            if (i + 1) % 25 == 0 or (i + 1) == len(submit_items):
                gc.collect()
                logger.info("Loaded %d/%d test task graphs", i + 1, len(submit_items))
    else:
        batch_size = 50
        for batch_start in range(0, len(submit_items), batch_size):
            batch = submit_items[batch_start : batch_start + batch_size]
            try:
                with ProcessPoolExecutor(
                    max_workers=workers,
                    initializer=_init_worker,
                    initargs=(encoder_type, chromadb_dir_str),
                ) as pool:
                    futures = {}
                    for iid, t, ssg_str, trace_str in batch:
                        fut = pool.submit(
                            _build_one_task, t, ssg_str, trace_str,
                            feature_cache_str, candidate_cache_str,
                        )
                        futures[fut] = iid
                    for fut in as_completed(futures):
                        iid = futures[fut]
                        try:
                            tg = fut.result()
                        except Exception:
                            logger.exception("[%s] failed to build TaskGraph", iid)
                            continue
                        if tg is not None:
                            tasks.append(tg)
            except BrokenExecutor:
                logger.error("ProcessPool crashed on batch %d — continuing", batch_start)
            gc.collect()
            logger.info("Loaded %d test task graphs so far", len(tasks))

    if chromadb_dir is not None:
        encoder = ChromaSourceEncoder(chromadb_dir=chromadb_dir)
    else:
        encoder = SentenceTransformerSourceEncoder()
    return tasks, encoder


def run_2a(args: argparse.Namespace) -> dict[str, Any]:
    import torch
    from token_sniper.edge_pred.evaluate import aggregate_metrics, evaluate_edge_predictor, load_checkpoint

    tasks, _ = _load_test_tasks(args)
    from token_sniper.edge_pred.features import _pick_device
    device = torch.device(args.device or _pick_device())

    loo_manifest_path = args.checkpoint.parent / "loo_manifest.json"
    if loo_manifest_path.exists():
        loo = json.loads(loo_manifest_path.read_text(encoding="utf-8"))
        fold_map = {f["held_out"]: f["checkpoint"] for f in loo.get("folds", [])}
        logger.info("Using leave-one-out checkpoints for 2A (%d folds)", len(fold_map))
        all_metrics = []
        for tg in tasks:
            ckpt_path = fold_map.get(tg.instance_id)
            if ckpt_path is None:
                logger.info("[%s] no LOO fold checkpoint, using full model", tg.instance_id)
                ckpt_path = str(args.checkpoint)
            model = load_checkpoint(ckpt_path, device)
            fold_metrics = evaluate_edge_predictor(model, [tg], device=device)
            all_metrics.extend(fold_metrics)
        metrics = all_metrics
    else:
        model = load_checkpoint(args.checkpoint, device)
        metrics = evaluate_edge_predictor(model, tasks, device=device)

    agg = aggregate_metrics(metrics)

    out = args.output_dir / "2A_edge_quality.md"
    lines = ["# Phase 2 — Experiment 2A: Edge Prediction Quality", ""]
    lines.append(f"- **Tasks evaluated**: {int(agg.get('task_count', 0))}")
    lines.append(f"- **Mean AUC-ROC**: {agg.get('mean_auc_roc', 0):.4f}")
    if "mean_precision_at_50" in agg:
        lines.append("")
        lines.append("| k | Precision@k | Recall@k |")
        lines.append("|---|---|---|")
        for k in (50, 100, 200, 500):
            p = agg.get(f"mean_precision_at_{k}", 0.0)
            r = agg.get(f"mean_recall_at_{k}", 0.0)
            lines.append(f"| {k} | {p:.4f} | {r:.4f} |")
    lines.append("")
    killed = agg.get("mean_auc_roc", 0.0) < args.auc_threshold
    lines.append("## Kill criterion")
    lines.append(f"Threshold: AUC-ROC ≥ {args.auc_threshold}")
    lines.append(f"Result: **{'KILL' if killed else 'GO'}**")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines), encoding="utf-8")

    (args.output_dir / "2A_edge_quality.json").write_text(
        json.dumps({"aggregate": agg, "per_task": [m.__dict__ for m in metrics]}, indent=2, default=str),
        encoding="utf-8",
    )
    logger.info("2A complete. AUC=%.4f Kill=%s", agg.get("mean_auc_roc", 0), killed)
    return {"aggregate": agg, "killed": killed}


def run_2b(args: argparse.Namespace) -> dict[str, Any]:
    import torch

    from token_sniper.edge_pred.augment import augment_graph_with_top_m, predict_edge_scores
    from token_sniper.edge_pred.evaluate import load_checkpoint
    from token_sniper.evaluation.phase1 import compute_file_level_metrics
    from token_sniper.evaluation.swebench import load_swebench_task_metadata
    from token_sniper.retrieval.config import EmbeddingConfig
    from token_sniper.retrieval.embeddings import create_embedder
    from token_sniper.retrieval.graph_retrievers import GraphPPRRetriever
    from token_sniper.retrieval.vectorstore import EntityVectorStore

    tasks, _ = _load_test_tasks(args)
    from token_sniper.edge_pred.features import _pick_device
    device = torch.device(args.device or _pick_device())
    model = load_checkpoint(args.checkpoint, device)

    embedder = create_embedder(EmbeddingConfig(
        provider="sentence-transformers", name="jinaai/jina-code-embeddings-0.5b",
    ))
    vector_store = EntityVectorStore(persist_directory=str(args.persist_dir), collection_prefix="token_sniper")

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    by_id = {t["instance_id"]: t for t in manifest}
    k_values = [10, 20, 50, 100]
    m_values = args.augmentation_sizes

    rows: list[dict[str, Any]] = []
    for tg in tasks:
        t = by_id[tg.instance_id]
        repo_slug = t["repo"].replace("/", "__")
        commit_short = t["base_commit"][:12]
        collection_repo = f"{t['repo']}@{commit_short}"
        meta = load_swebench_task_metadata(tg.instance_id)
        gold_files = t.get("gold_patch_files", [])
        issue_text = meta.problem_statement

        from token_sniper.graph.io import read_ssg_json
        graph = read_ssg_json(args.ssg_dir / repo_slug / commit_short / "ssg.json")
        scores = predict_edge_scores(model, tg, device=device)

        for m in [0] + list(m_values):
            g = graph if m == 0 else augment_graph_with_top_m(graph, tg, scores, m=m)
            retr = GraphPPRRetriever(
                graph=g,
                entry_point_method="embedding",
                alpha=0.15,
                n_entry_points=3,
                embedder=embedder,
                vector_store=vector_store,
            )
            result, _ = retr.retrieve(issue_text=issue_text, repo=collection_repo, k=max(k_values))
            prec, rec = compute_file_level_metrics(
                retrieved_files=result.file_paths,
                gold_files=gold_files,
                k_values=k_values,
            )
            rows.append({
                "instance_id": tg.instance_id,
                "m": m,
                **{f"P@{k}": prec[k] for k in k_values},
                **{f"R@{k}": rec[k] for k in k_values},
            })

    out_json = args.output_dir / "2B_retrieval_impact.json"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(rows, indent=2), encoding="utf-8")

    agg: dict[int, dict[str, float]] = {}
    for m in [0] + list(m_values):
        subset = [r for r in rows if r["m"] == m]
        if not subset:
            continue
        agg[m] = {k: sum(r[k] for r in subset) / len(subset) for k in subset[0] if k.startswith(("P@", "R@"))}

    lines = ["# Phase 2 — Experiment 2B: Retrieval Impact", ""]
    lines.append("| m | " + " | ".join(f"P@{k}" for k in k_values) + " | " + " | ".join(f"R@{k}" for k in k_values) + " |")
    lines.append("|" + "---|" * (1 + 2 * len(k_values)))
    for m, metrics in sorted(agg.items()):
        row = [str(m)]
        row.extend(f"{metrics[f'P@{k}']:.4f}" for k in k_values)
        row.extend(f"{metrics[f'R@{k}']:.4f}" for k in k_values)
        lines.append("| " + " | ".join(row) + " |")

    (args.output_dir / "2B_retrieval_impact.md").write_text("\n".join(lines), encoding="utf-8")
    logger.info("2B complete. Aggregates: %s", agg)
    return {"aggregate": agg}


def run_2c(args: argparse.Namespace) -> dict[str, Any]:
    split = json.loads(args.split_file.read_text(encoding="utf-8"))
    manifest_path = args.output_dir / "2C_agent_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "conditions": ["ssg_only", "augmented"],
        "m_for_augmented": args.augmentation_sizes[0] if args.augmentation_sizes else 200,
        "test_tasks": split["test"],
        "instructions": (
            "Use scripts/run_agent_task.py for each (task, condition) pair, "
            "feeding the retrieved context from Phase 2B as the agent prompt. "
            "Then aggregate Pass@1 via token_sniper.evaluation.metrics.pass_at_k."
        ),
    }
    manifest_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    logger.info(
        "2C manifest written to %s. Run the agent harness next (see PHASE2_RUN_GUIDE.md).",
        manifest_path,
    )
    return {"manifest": str(manifest_path)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 2 experiments")
    parser.add_argument("--experiment", choices=["2A", "2B", "2C", "all"], nargs="+", default=["all"])
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=Path("configs/phases/phase0_tasks.json"))
    parser.add_argument("--split-file", type=Path, default=Path("data/edge_pred/splits.json"))
    parser.add_argument("--ssg-dir", type=Path, default=Path("data/ssg"))
    parser.add_argument("--traces-dir", type=Path, default=Path("data/traces"))
    parser.add_argument("--feature-cache", type=Path, default=Path("data/edge_pred/features"))
    parser.add_argument("--candidate-cache", type=Path, default=Path("data/edge_pred/candidates"))
    parser.add_argument("--output-dir", type=Path, default=Path("experiments/phase2"))
    parser.add_argument("--persist-dir", type=Path, default=Path("data/chromadb"))
    parser.add_argument("--chromadb-dir", type=Path, default=None,
                        help="ChromaDB dir with Jina embeddings (default: use SentenceTransformer)")
    parser.add_argument("--augmentation-sizes", type=int, nargs="+", default=[50, 100, 200, 500])
    parser.add_argument("--auc-threshold", type=float, default=0.65)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--loader-workers", type=int, default=1,
                        help="Worker processes for building test task graphs (0 or 1 = sequential)")
    parser.add_argument("--exclude-repos", type=str,
                        default="aws-cloudformation/cfn-lint,pylint-dev/pylint",
                        help="Comma-separated repos to skip during evaluation")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    results: dict[str, Any] = {}
    if "2A" in args.experiment or "all" in args.experiment:
        results["2A"] = run_2a(args)
    if "2B" in args.experiment or "all" in args.experiment:
        results["2B"] = run_2b(args)
    if "2C" in args.experiment or "all" in args.experiment:
        results["2C"] = run_2c(args)

    decision_lines = ["# Phase 2 — DECISION", ""]
    if "2A" in results:
        auc = results["2A"]["aggregate"].get("mean_auc_roc", 0.0)
        killed = results["2A"]["killed"]
        decision_lines.append(f"- Experiment 2A AUC-ROC: {auc:.4f} → {'KILL' if killed else 'GO'}")
    if "2B" in results:
        agg = results["2B"]["aggregate"]
        if 0 in agg:
            baseline_p10 = agg[0].get("P@10", 0.0)
            best_delta = max(
                (m_vals["P@10"] - baseline_p10)
                for m_key, m_vals in agg.items() if m_key != 0
            ) if len(agg) > 1 else 0.0
            decision_lines.append(f"- Experiment 2B best ΔP@10 over SSG-only: {best_delta:+.4f}")
    (args.output_dir / "DECISION.md").write_text("\n".join(decision_lines), encoding="utf-8")


if __name__ == "__main__":
    main()

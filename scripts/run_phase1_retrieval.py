from __future__ import annotations

import argparse
import json
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Phase 1 retrieval evaluation")
    parser.add_argument("--manifest", type=Path, default=Path("configs/phases/phase0_tasks.json"))
    parser.add_argument("--base-config", type=Path, default=Path("configs/base.yaml"))
    parser.add_argument("--phase-config", type=Path, default=Path("configs/phases/phase1_static_graph.yaml"))
    parser.add_argument("--ssg-dir", type=Path, default=Path("data/ssg"))
    parser.add_argument("--persist-dir", type=str, default="data/chromadb/")
    parser.add_argument("--output-dir", type=Path, default=Path("experiments/phase1"))
    parser.add_argument("--provider", choices=["voyageai", "nomic-local"], default="voyageai")
    parser.add_argument("--collection-prefix", default="token_sniper")
    parser.add_argument(
        "--conditions",
        nargs="+",
        choices=["graph_bfs", "graph_ppr", "embedding_knn"],
        default=["graph_bfs", "graph_ppr", "embedding_knn"],
    )
    parser.add_argument(
        "--entry-point-method",
        choices=["embedding", "llm_prompt"],
        default="embedding",
        help="Method for identifying graph entry points",
    )
    parser.add_argument("--k-values", nargs="+", type=int, default=[10, 20, 50, 100])
    parser.add_argument("--max-hops", type=int, default=2, help="BFS max hops")
    parser.add_argument("--ppr-alpha", type=float, default=0.15, help="PPR restart probability")
    parser.add_argument("--n-entry-points", type=int, default=3, help="Number of entry points")
    parser.add_argument("--max-k", type=int, default=100, help="Max k to retrieve per strategy")
    args = parser.parse_args()

    run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    logger.info("Phase 1 retrieval evaluation — run_id=%s", run_id)

    manifest_data = json.loads(args.manifest.read_text(encoding="utf-8"))
    logger.info("Loaded %d tasks from %s", len(manifest_data), args.manifest)

    _ = _load_yaml(args.phase_config)
    k_values = args.k_values

    from token_sniper.retrieval.config import EmbeddingConfig
    from token_sniper.retrieval.embeddings import create_embedder
    from token_sniper.retrieval.vectorstore import EntityVectorStore

    embedding_config = EmbeddingConfig(
        provider=args.provider,
        name="voyage-code-3" if args.provider == "voyageai" else "nomic-embed-code-v1",
    )
    embedder = create_embedder(embedding_config)
    vector_store = EntityVectorStore(
        persist_directory=args.persist_dir,
        collection_prefix=args.collection_prefix,
    )

    llm_client = None
    if args.entry_point_method == "llm_prompt":
        llm_client = _create_llm_client(args.base_config)

    from token_sniper.evaluation.phase1 import (
        ConditionSummary,
        Phase1Summary,
        TaskRetrievalMetrics,
        aggregate_condition_metrics,
        compute_file_level_metrics,
        evaluate_kill_criterion,
        write_phase1_decision,
        write_phase1_summary_json,
    )

    all_task_metrics: dict[str, list[TaskRetrievalMetrics]] = {
        cond: [] for cond in args.conditions
    }

    for task_idx, task in enumerate(manifest_data):
        instance_id = task["instance_id"]
        repo = task["repo"]
        commit = task["base_commit"]
        commit_short = commit[:12]
        difficulty = task.get("difficulty", "")
        gold_files = task.get("gold_patch_files", [])
        collection_repo = f"{repo}@{commit_short}"

        logger.info(
            "[%d/%d] Processing %s (%s @ %s, gold_files=%d)",
            task_idx + 1,
            len(manifest_data),
            instance_id,
            repo,
            commit_short,
            len(gold_files),
        )

        repo_slug = repo.replace("/", "__")
        ssg_path = args.ssg_dir / repo_slug / commit_short / "ssg.json"
        if not ssg_path.exists():
            logger.warning("SSG not found: %s — skipping task %s", ssg_path, instance_id)
            for cond in args.conditions:
                all_task_metrics[cond].append(
                    TaskRetrievalMetrics(
                        instance_id=instance_id,
                        condition=cond,
                        repo=repo,
                        difficulty=difficulty,
                        gold_files=gold_files,
                    )
                )
            continue

        from token_sniper.graph.io import read_ssg_json

        graph = read_ssg_json(ssg_path)

        issue_text = _load_issue_text(instance_id)

        pre_ep: Any | None = None
        graph_conditions = [c for c in args.conditions if c in ("graph_bfs", "graph_ppr")]
        if args.entry_point_method == "llm_prompt" and llm_client is not None and graph_conditions:
            from token_sniper.retrieval.entry_points import identify_entry_points
            pre_ep = identify_entry_points(
                method="llm_prompt",
                issue_text=issue_text,
                repo=collection_repo,
                graph=graph,
                n_candidates=args.n_entry_points,
                embedder=embedder,
                vector_store=vector_store,
                llm_client=llm_client,
            )
            logger.info(
                "  Pre-computed LLM entry points: matched=%d, fqns=%s",
                len(pre_ep.matched_node_ids),
                pre_ep.matched_node_ids,
            )

        for cond in args.conditions:
            try:
                result, ep_log = _run_condition(
                    condition=cond,
                    issue_text=issue_text,
                    repo=collection_repo,
                    graph=graph,
                    embedder=embedder,
                    vector_store=vector_store,
                    llm_client=llm_client,
                    entry_point_method=args.entry_point_method,
                    max_hops=args.max_hops,
                    ppr_alpha=args.ppr_alpha,
                    n_entry_points=args.n_entry_points,
                    max_k=args.max_k,
                    pre_computed_ep=pre_ep,
                )

                retrieved_files = result.file_paths
                prec, rec = compute_file_level_metrics(
                    retrieved_files=retrieved_files,
                    gold_files=gold_files,
                    k_values=k_values,
                )

                task_metric = TaskRetrievalMetrics(
                    instance_id=instance_id,
                    condition=cond,
                    repo=repo,
                    difficulty=difficulty,
                    gold_files=gold_files,
                    retrieved_files=retrieved_files[:max(k_values)],
                    entry_point_fqns=ep_log.matched_node_ids if ep_log else [],
                    entry_point_method=ep_log.method if ep_log else "",
                    latency_ms=result.latency_ms,
                    precision_at_k=prec,
                    recall_at_k=rec,
                )
                all_task_metrics[cond].append(task_metric)

                logger.info(
                    "  %s: P@10=%.4f, R@10=%.4f, files=%d, latency=%.0fms",
                    cond,
                    prec.get(10, 0.0),
                    rec.get(10, 0.0),
                    len(retrieved_files),
                    result.latency_ms,
                )

            except Exception as exc:
                logger.error("  %s: FAILED — %s", cond, exc)
                all_task_metrics[cond].append(
                    TaskRetrievalMetrics(
                        instance_id=instance_id,
                        condition=cond,
                        repo=repo,
                        difficulty=difficulty,
                        gold_files=gold_files,
                    )
                )

    logger.info("Aggregating results...")
    condition_summaries: dict[str, ConditionSummary] = {}
    all_results: list[TaskRetrievalMetrics] = []

    for cond in args.conditions:
        condition_summaries[cond] = aggregate_condition_metrics(
            all_task_metrics[cond],
            condition=cond,
            k_values=k_values,
        )
        all_results.extend(all_task_metrics[cond])
        s = condition_summaries[cond]
        logger.info(
            "  %s: mean P@10=%.4f, mean R@10=%.4f, tasks=%d",
            cond,
            s.mean_precision_at_k.get(10, 0.0),
            s.mean_recall_at_k.get(10, 0.0),
            s.task_count,
        )

    killed, detail = evaluate_kill_criterion(condition_summaries, k_values)
    logger.info("Kill criterion: %s", "KILL" if killed else "GO")
    logger.info("%s", detail)

    summary = Phase1Summary(
        run_id=run_id,
        task_count=len(manifest_data),
        k_values=k_values,
        conditions=condition_summaries,
        task_results=all_results,
        kill_criterion_met=killed,
        kill_criterion_detail=detail,
    )

    output_dir = args.output_dir / run_id
    write_phase1_summary_json(output_dir=output_dir, summary=summary)
    write_phase1_decision(output_dir=output_dir, summary=summary)

    canonical_dir = args.output_dir
    write_phase1_decision(output_dir=canonical_dir, summary=summary)
    write_phase1_summary_json(output_dir=canonical_dir, summary=summary)

    logger.info("Phase 1 evaluation complete. Results in %s", output_dir)
    logger.info("DECISION: %s", "KILL" if killed else "GO → proceed to Phase 2")


def _run_condition(
    *,
    condition: str,
    issue_text: str,
    repo: str,
    graph: Any,
    embedder: Any,
    vector_store: Any,
    llm_client: Any,
    entry_point_method: str,
    max_hops: int,
    ppr_alpha: float,
    n_entry_points: int,
    max_k: int,
    pre_computed_ep: Any | None = None,
) -> tuple[Any, Any]:
    from token_sniper.retrieval.entry_points import EntryPointResult
    from token_sniper.retrieval.graph_retrievers import GraphBFSRetriever, GraphPPRRetriever
    from token_sniper.retrieval.retriever import DenseRetriever

    if condition == "embedding_knn":
        retriever = DenseRetriever(embedder=embedder, vector_store=vector_store)
        result = retriever.retrieve(issue_text=issue_text, repo=repo, k=max_k)
        ep_log = EntryPointResult(method="none", issue_text_preview=issue_text[:200])
        return result, ep_log

    if condition == "graph_bfs":
        retriever = GraphBFSRetriever(
            graph=graph,
            entry_point_method=entry_point_method,
            max_hops=max_hops,
            n_entry_points=n_entry_points,
            embedder=embedder,
            vector_store=vector_store,
            llm_client=llm_client,
        )
        return retriever.retrieve(
            issue_text=issue_text, repo=repo, k=max_k, pre_computed_entry_points=pre_computed_ep
        )

    if condition == "graph_ppr":
        retriever = GraphPPRRetriever(
            graph=graph,
            entry_point_method=entry_point_method,
            alpha=ppr_alpha,
            n_entry_points=n_entry_points,
            embedder=embedder,
            vector_store=vector_store,
            llm_client=llm_client,
        )
        return retriever.retrieve(
            issue_text=issue_text, repo=repo, k=max_k, pre_computed_entry_points=pre_computed_ep
        )

    raise ValueError(f"Unknown condition: {condition!r}")


def _load_issue_text(instance_id: str) -> str:
    try:
        from token_sniper.evaluation.swebench import load_swebench_task_metadata

        metadata = load_swebench_task_metadata(instance_id)
        return metadata.problem_statement
    except Exception as exc:
        logger.warning("Could not load issue text for %s: %s", instance_id, exc)
        return f"Issue: {instance_id}"


def _load_yaml(path: Path) -> dict[str, Any]:
    import yaml

    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def _create_llm_client(base_config_path: Path) -> Any:
    from token_sniper.agent.config import load_base_config
    from token_sniper.agent.providers import OpenAIChatModelClient

    config = load_base_config(base_config_path)
    return OpenAIChatModelClient(config.models.agent)


if __name__ == "__main__":
    main()

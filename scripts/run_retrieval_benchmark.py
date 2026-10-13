from __future__ import annotations

import argparse
import json
import logging
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import UTC, datetime
from functools import partial
from pathlib import Path
from typing import Any, Literal

from token_sniper.agent.config import AgentModelConfig, load_base_config, load_yaml_config
from token_sniper.agent.providers import OpenAIChatModelClient
from token_sniper.agent.runner import AgentRunResult, AgentRunSettings, run_agent_session
from token_sniper.evaluation.config import HarnessConfig
from token_sniper.evaluation.metrics import token_usage_from_session_log
from token_sniper.evaluation.models import (
    ConditionRunSummary,
    SWEbenchEvaluationAttempt,
    TaskEvaluationResult,
)
from token_sniper.evaluation.phase0 import (
    Phase0Task,
    ensure_repo_cached,
    load_phase0_manifest,
    materialize_repo_snapshot,
)
from token_sniper.evaluation.swebench import (
    HarnessPatchSubmission,
    SWEbenchHarnessRunner,
    load_swebench_task_metadata,
)
from token_sniper.evaluation.tracking import ExperimentTracker

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

VALID_CONDITIONS = ("ssg_ppr", "augmented_ppr")


@dataclass(frozen=True)
class TaskGenerationOutput:
    task_result: TaskEvaluationResult
    harness_submission: HarnessPatchSubmission | None = None


@dataclass(frozen=True)
class HarnessBatchOutcome:
    batch_index: int
    submissions: list[HarnessPatchSubmission]
    attempts_by_instance: dict[str, SWEbenchEvaluationAttempt] | None = None
    error_message: str | None = None


_MAX_CONTEXT_CHARS = 100_000
_MAX_FILE_CHARS = 20_000


def _build_retrieval_context(
    *,
    repo_root: Path,
    retrieved_files: list[str],
    strategy: str,
) -> str:
    sections = [
        f"Retrieval mode: {strategy}. "
        f"The following file(s) were identified as relevant "
        f"by Graph-PPR retrieval and are preloaded into context.",
        "If more surrounding context is needed, inspect the repository with shell tools.",
    ]
    total_chars = sum(len(s) for s in sections)
    missing: list[str] = []
    included_count = 0

    for rel_path in retrieved_files:
        abs_path = (repo_root / rel_path).resolve()
        if not abs_path.is_file():
            missing.append(rel_path)
            continue
        try:
            content = abs_path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            missing.append(rel_path)
            continue

        if len(content) > _MAX_FILE_CHARS:
            content = content[:_MAX_FILE_CHARS] + f"\n... (truncated, {len(content)} chars total)"

        section = f"Path: {rel_path}\n```\n{content}\n```"
        if total_chars + len(section) > _MAX_CONTEXT_CHARS:
            remaining = len(retrieved_files) - included_count - len(missing)
            if remaining > 0:
                sections.append(
                    f"{remaining} more retrieved file(s) omitted due to context budget. "
                    "Use shell tools to inspect them:\n"
                    + "\n".join(f"- {f}" for f in retrieved_files[included_count + len(missing):])
                )
            break
        sections.append(section)
        total_chars += len(section)
        included_count += 1

    if missing:
        sections.append("These retrieved files do not exist at the base commit:\n"
                         + "\n".join(f"- {p}" for p in missing))
    return "\n\n".join(sections)


def _retrieve_files_ppr(
    *,
    task: Phase0Task,
    ssg_dir: Path,
    issue_text: str,
    embedder: Any,
    vector_store: Any,
    k: int = 20,
    alpha: float = 0.15,
    n_entry_points: int = 3,
    augmented_graph: Any | None = None,
) -> list[str]:
    from token_sniper.graph.io import read_ssg_json
    from token_sniper.retrieval.graph_retrievers import GraphPPRRetriever

    repo_slug = task.repo.replace("/", "__")
    commit_short = task.base_commit[:12]
    ssg_path = ssg_dir / repo_slug / commit_short / "ssg.json"
    if not ssg_path.exists():
        logger.warning("SSG not found at %s, skipping retrieval for %s", ssg_path, task.instance_id)
        return []

    graph = augmented_graph if augmented_graph is not None else read_ssg_json(ssg_path)
    collection_repo = f"{task.repo}@{commit_short}"

    retriever = GraphPPRRetriever(
        graph=graph,
        entry_point_method="embedding",
        alpha=alpha,
        n_entry_points=n_entry_points,
        embedder=embedder,
        vector_store=vector_store,
    )
    result, ep_result = retriever.retrieve(
        issue_text=issue_text, repo=collection_repo, k=k,
    )
    logger.info(
        "[%s] PPR retrieved %d files, %d entities, %d entry points",
        task.instance_id, len(result.file_paths), len(result.entities),
        len(ep_result.matched_node_ids),
    )
    return result.file_paths


def _generate_task_output(
    task: Phase0Task,
    *,
    condition: str,
    dataset_name: str,
    split: str,
    base_config: dict[str, Any],
    run_dir: Path,
    swebench_cache: Path,
    ssg_dir: Path,
    agent_model_config: AgentModelConfig,
    embedder: Any,
    vector_store: Any,
    checkpoint_path: Path | None = None,
    retrieval_k: int = 20,
) -> TaskGenerationOutput:
    logger.info("[%s] Processing task %s", condition, task.instance_id)
    metadata = load_swebench_task_metadata(task.instance_id, dataset_name=dataset_name)

    work_dir = run_dir / "work" / task.instance_id
    agent_output_dir = run_dir / "agent_outputs" / task.instance_id
    repo_work_dir = materialize_repo_snapshot(
        repo=task.repo,
        base_commit=task.base_commit,
        cache_dir=swebench_cache / "repos",
        work_dir=work_dir,
    )

    augmented_graph = None
    if condition == "augmented_ppr" and checkpoint_path is not None:
        augmented_graph = _build_augmented_graph(
            task=task, ssg_dir=ssg_dir, checkpoint_path=checkpoint_path,
        )

    retrieved_files = _retrieve_files_ppr(
        task=task,
        ssg_dir=ssg_dir,
        issue_text=metadata.problem_statement,
        embedder=embedder,
        vector_store=vector_store,
        k=retrieval_k,
        augmented_graph=augmented_graph,
    )

    if retrieved_files:
        oracle_context = _build_retrieval_context(
            repo_root=repo_work_dir,
            retrieved_files=retrieved_files,
            strategy=condition,
        )
        mode: Literal["blind", "oracle"] = "oracle"
    else:
        oracle_context = None
        mode = "blind"
        logger.warning("[%s] No files retrieved, falling back to blind mode", task.instance_id)

    settings = AgentRunSettings(
        repo_root=str(repo_work_dir),
        issue_description=metadata.problem_statement,
        output_dir=str(agent_output_dir),
        mode=mode,
        oracle_files=retrieved_files if mode == "oracle" else [],
        oracle_context=oracle_context,
        max_turns=base_config.get("models", {}).get("agent", {}).get("max_turns", 12),
        context_window=base_config.get("models", {}).get("agent", {}).get("context_window", 131072),
        max_output_tokens=base_config.get("models", {}).get("agent", {}).get("max_tokens", 16384),
    )

    try:
        agent_result = run_agent_session(settings, OpenAIChatModelClient(agent_model_config))
    except Exception as exc:
        logger.warning("[%s] Agent run failed for %s: %s", condition, task.instance_id, exc)
        agent_result = _load_agent_result(agent_output_dir=agent_output_dir, settings=settings)

    patch_summary = _summarize_patch_validation(agent_result)
    task_result = TaskEvaluationResult(
        instance_id=task.instance_id,
        phase="retrieval_benchmark",
        condition=condition,
        dataset_name=dataset_name,
        split=split,
        resolved=False,
        repo=task.repo,
        base_commit=task.base_commit,
        difficulty=task.difficulty,
        gold_patch_files=task.gold_patch_files,
        attempts=[],
        metadata={
            "agent_status": agent_result.status,
            "agent_error": agent_result.error_message,
            "evaluation_error": None,
            "session_log_path": agent_result.session_log_path,
            "patch_path": agent_result.patch_path,
            "retrieved_files": retrieved_files,
            "retrieved_file_count": len(retrieved_files),
            "retrieval_hit_count": len(set(retrieved_files) & set(task.gold_patch_files)),
            **patch_summary,
        },
    )
    if not agent_result.patch_path:
        logger.warning("[%s] No patch produced for %s", condition, task.instance_id)
        return TaskGenerationOutput(task_result=task_result)

    return TaskGenerationOutput(
        task_result=task_result,
        harness_submission=HarnessPatchSubmission(
            instance_id=task.instance_id,
            patch_file=Path(agent_result.patch_path),
            model_name=base_config.get("models", {}).get("agent", {}).get("name", "agent"),
            session_log_path=Path(agent_result.session_log_path),
            token_usage=token_usage_from_session_log(agent_result.session_log_path),
        ),
    )


def _build_augmented_graph(
    *,
    task: Phase0Task,
    ssg_dir: Path,
    checkpoint_path: Path,
    m: int = 50,
) -> Any:
    import torch

    from token_sniper.edge_pred.augment import augment_graph_with_top_m, predict_edge_scores
    from token_sniper.edge_pred.dataset import load_task_graph
    from token_sniper.edge_pred.evaluate import load_checkpoint
    from token_sniper.edge_pred.features import _pick_device
    from token_sniper.graph.io import read_ssg_json

    repo_slug = task.repo.replace("/", "__")
    commit_short = task.base_commit[:12]
    ssg_path = ssg_dir / repo_slug / commit_short / "ssg.json"
    if not ssg_path.exists():
        logger.warning("SSG not found for augmentation: %s", ssg_path)
        return None

    device = torch.device(_pick_device())
    model = load_checkpoint(checkpoint_path, device)
    graph = read_ssg_json(ssg_path)

    try:
        tg = load_task_graph(task.instance_id, ssg_dir=ssg_dir, traces_dir=ssg_dir.parent / "traces")
    except Exception as exc:
        logger.warning("Could not load task graph for %s: %s", task.instance_id, exc)
        return graph

    scores = predict_edge_scores(model, tg, device=device)
    return augment_graph_with_top_m(graph, tg, scores, m=m)


def run_condition(
    *,
    condition: str,
    manifest_tasks: list[Phase0Task],
    base_config: dict[str, Any],
    tracker: ExperimentTracker,
    swebench_cache: Path,
    ssg_dir: Path,
    agent_model_config: AgentModelConfig,
    agent_workers: int,
    eval_batch_size: int,
    run_id: str,
    embedder: Any,
    vector_store: Any,
    checkpoint_path: Path | None = None,
    retrieval_k: int = 20,
) -> dict[str, Any]:
    dataset_name = "SWE-bench-Live/SWE-bench-Live"
    logger.info("Starting %s run %s", condition, run_id)

    with tracker.start_run(
        phase="retrieval_benchmark",
        condition=condition,
        dataset_name=dataset_name,
        split="test",
        config_snapshot={"base": base_config, "condition": condition},
        run_name=run_id,
    ) as run:
        harness_config = HarnessConfig(**base_config.get("evaluation", {}).get("harness", {}))
        harness_runner = SWEbenchHarnessRunner(harness_config)
        task_results: dict[str, TaskEvaluationResult] = {}
        harness_submissions: list[HarnessPatchSubmission] = []
        harness_run_id = f"{condition}_{run_id}"

        _prewarm_repos(manifest_tasks, dataset_name, swebench_cache / "repos")

        worker = partial(
            _generate_task_output,
            condition=condition,
            dataset_name=dataset_name,
            split="test",
            base_config=base_config,
            run_dir=run.run_dir,
            swebench_cache=swebench_cache,
            ssg_dir=ssg_dir,
            agent_model_config=agent_model_config,
            embedder=embedder,
            vector_store=vector_store,
            checkpoint_path=checkpoint_path,
            retrieval_k=retrieval_k,
        )

        pending_submissions: list[HarnessPatchSubmission] = []
        pending_batches: list[Future[HarnessBatchOutcome]] = []
        batch_index = 0

        with ThreadPoolExecutor(max_workers=1) as eval_executor:
            for output in _iter_outputs(manifest_tasks, worker, agent_workers):
                task_result = output.task_result
                run.log_task_result(task_result)
                task_results[task_result.instance_id] = task_result
                if output.harness_submission is not None:
                    harness_submissions.append(output.harness_submission)
                    pending_submissions.append(output.harness_submission)
                    if len(pending_submissions) >= eval_batch_size:
                        pending_batches.append(
                            eval_executor.submit(
                                _evaluate_batch, harness_runner, run.run_dir,
                                harness_run_id, list(pending_submissions), batch_index,
                            )
                        )
                        batch_index += 1
                        pending_submissions = []
                pending_batches = _drain_done(pending_batches, task_results, run)

            if pending_submissions:
                pending_batches.append(
                    eval_executor.submit(
                        _evaluate_batch, harness_runner, run.run_dir,
                        harness_run_id, list(pending_submissions), batch_index,
                    )
                )
            for future in pending_batches:
                _apply_outcome(future.result(), task_results, run)

        summary = run.finalize()
        return _build_summary(condition, manifest_tasks, task_results, summary)


def _iter_outputs(tasks: list[Phase0Task], worker: Any, workers: int) -> Any:
    if len(tasks) <= 1 or workers == 1:
        for task in tasks:
            yield worker(task)
        return
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(worker, task): task for task in tasks}
        for future in as_completed(futures):
            yield future.result()


def _prewarm_repos(tasks: list[Phase0Task], dataset_name: str, repo_cache: Path) -> None:
    if not tasks:
        return
    load_swebench_task_metadata(tasks[0].instance_id, dataset_name=dataset_name)
    seen: set[tuple[str, str]] = set()
    for task in tasks:
        key = (task.repo, task.base_commit)
        if key in seen:
            continue
        ensure_repo_cached(repo=task.repo, base_commit=task.base_commit, cache_dir=repo_cache)
        seen.add(key)


def _evaluate_batch(
    harness_runner: SWEbenchHarnessRunner,
    run_dir: Path,
    run_id: str,
    submissions: list[HarnessPatchSubmission],
    batch_index: int,
) -> HarnessBatchOutcome:
    try:
        attempts = harness_runner.evaluate_patches(
            work_dir=run_dir / "harness" / f"batch_{batch_index:03d}",
            submissions=submissions,
            run_id=f"{run_id}_batch_{batch_index:03d}",
        )
    except Exception as exc:
        return HarnessBatchOutcome(
            batch_index=batch_index, submissions=submissions,
            error_message=f"{type(exc).__name__}: {exc}",
        )
    return HarnessBatchOutcome(
        batch_index=batch_index, submissions=submissions, attempts_by_instance=attempts,
    )


def _drain_done(
    pending: list[Future[HarnessBatchOutcome]],
    results: dict[str, TaskEvaluationResult],
    run: Any,
) -> list[Future[HarnessBatchOutcome]]:
    remaining: list[Future[HarnessBatchOutcome]] = []
    for f in pending:
        if f.done():
            _apply_outcome(f.result(), results, run)
        else:
            remaining.append(f)
    return remaining


def _apply_outcome(
    outcome: HarnessBatchOutcome,
    results: dict[str, TaskEvaluationResult],
    run: Any,
) -> None:
    if outcome.error_message:
        logger.error("Harness batch %s failed: %s", outcome.batch_index, outcome.error_message)
        for sub in outcome.submissions:
            results[sub.instance_id] = results[sub.instance_id].model_copy(
                update={"resolved": False, "metadata": {
                    **dict(results[sub.instance_id].metadata),
                    "evaluation_error": outcome.error_message,
                }},
            )
            run.log_task_result(results[sub.instance_id])
        return

    for sub in outcome.submissions:
        attempt = (outcome.attempts_by_instance or {}).get(sub.instance_id)
        if attempt is None:
            results[sub.instance_id] = results[sub.instance_id].model_copy(
                update={"resolved": False, "metadata": {
                    **dict(results[sub.instance_id].metadata),
                    "evaluation_error": "No attempt returned by harness.",
                }},
            )
        else:
            results[sub.instance_id] = results[sub.instance_id].model_copy(
                update={"resolved": attempt.resolved, "attempts": [attempt]},
            )
        run.log_task_result(results[sub.instance_id])


def _build_summary(
    condition: str,
    tasks: list[Phase0Task],
    results: dict[str, TaskEvaluationResult],
    summary: ConditionRunSummary,
) -> dict[str, Any]:
    resolved = [r for r in results.values() if r.resolved]
    attempted = [r for r in results.values() if r.attempts]
    total_tokens = sum(
        (r.metadata.get("total_prompt_tokens", 0) or 0) + (r.metadata.get("total_completion_tokens", 0) or 0)
        for r in results.values()
    )
    return {
        "condition": condition,
        "run_id": summary.run_id,
        "run_dir": summary.run_dir,
        "task_count": len(tasks),
        "attempted_count": len(attempted),
        "resolved_count": len(resolved),
        "pass_at_1": len(resolved) / max(len(tasks), 1),
        "total_tokens": total_tokens,
        "avg_tokens_per_task": total_tokens / max(len(tasks), 1),
        "resolved_ids": sorted(r.instance_id for r in resolved),
        "metrics": summary.metrics,
        "retrieval_stats": _retrieval_stats(results),
    }


def _retrieval_stats(results: dict[str, TaskEvaluationResult]) -> dict[str, Any]:
    hits = sum(r.metadata.get("retrieval_hit_count", 0) for r in results.values())
    total_gold = sum(len(r.gold_patch_files) for r in results.values())
    total_retrieved = sum(r.metadata.get("retrieved_file_count", 0) for r in results.values())
    return {
        "total_gold_files": total_gold,
        "total_retrieved_files": total_retrieved,
        "total_retrieval_hits": hits,
        "retrieval_recall": hits / max(total_gold, 1),
    }


def _load_agent_result(*, agent_output_dir: Path, settings: AgentRunSettings) -> AgentRunResult:
    session_path = agent_output_dir / "session.json"
    if session_path.is_file():
        return AgentRunResult.model_validate_json(session_path.read_text(encoding="utf-8"))
    return AgentRunResult(
        mode=settings.mode,
        repo_root=settings.repo_root,
        output_dir=str(agent_output_dir),
        session_log_path=str(session_path),
        total_prompt_tokens=0,
        total_completion_tokens=0,
        total_wall_time_seconds=0.0,
        turns=[],
        status="failed",
        error_message="Agent failed before writing a session log.",
    )


def _summarize_patch_validation(result: AgentRunResult) -> dict[str, Any]:
    validations = [t.patch_validation for t in result.turns if t.patch_validation is not None]
    valid = sum(1 for v in validations if v.validation_status == "accepted")
    return {
        "patch_attempt_count": len(validations),
        "valid_patch_attempt_count": valid,
        "invalid_patch_attempt_count": len(validations) - valid,
        "total_prompt_tokens": result.total_prompt_tokens,
        "total_completion_tokens": result.total_completion_tokens,
        "total_wall_time_seconds": result.total_wall_time_seconds,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Retrieval-augmented agent benchmark")
    parser.add_argument("--base-config", type=Path, default=Path("configs/base.yaml"))
    parser.add_argument("--manifest", type=Path, default=Path("configs/phases/phase0_tasks.json"))
    parser.add_argument("--ssg-dir", type=Path, required=True, help="SSG cache directory")
    parser.add_argument("--chromadb-dir", type=Path, required=True, help="ChromaDB persist directory")
    parser.add_argument("--checkpoint", type=Path, default=None, help="Edge predictor checkpoint (for augmented_ppr)")
    parser.add_argument("--conditions", nargs="+", choices=VALID_CONDITIONS, default=["ssg_ppr"])
    parser.add_argument("--retrieval-k", type=int, default=20, help="Number of files to retrieve via PPR")
    parser.add_argument("--agent-workers", type=int, default=4)
    parser.add_argument("--eval-batch-size", type=int, default=8)
    parser.add_argument("--max-workers", type=int, default=4, help="SWE-bench harness max_workers")
    args = parser.parse_args()

    base_config = load_yaml_config(args.base_config)
    agent_config = load_base_config(args.base_config)

    if args.max_workers:
        base_config.setdefault("evaluation", {}).setdefault("harness", {})["max_workers"] = args.max_workers

    manifest = load_phase0_manifest(args.manifest)
    experiments_dir = Path(base_config.get("paths", {}).get("experiments_dir", "experiments")).resolve()
    swebench_cache = Path(base_config.get("paths", {}).get("swebench_cache", "data/swebench")).resolve()
    tracking_uri = base_config.get("mlflow", {}).get("tracking_uri", "experiments/mlflow")
    experiment_prefix = base_config.get("mlflow", {}).get("experiment_prefix", "token-sniper")

    from token_sniper.retrieval.config import EmbeddingConfig
    from token_sniper.retrieval.embeddings import create_embedder
    from token_sniper.retrieval.vectorstore import EntityVectorStore

    embedder = create_embedder(EmbeddingConfig(provider="voyageai", name="voyage-code-3"))
    vector_store = EntityVectorStore(
        persist_directory=str(args.chromadb_dir), collection_prefix="token_sniper",
    )

    tracker = ExperimentTracker(
        experiments_dir=experiments_dir,
        tracking_uri=tracking_uri,
        experiment_prefix=experiment_prefix,
    )
    run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")

    all_summaries: dict[str, dict[str, Any]] = {}
    for condition in args.conditions:
        if condition == "augmented_ppr" and args.checkpoint is None:
            logger.warning("Skipping augmented_ppr: no --checkpoint provided")
            continue
        summary = run_condition(
            condition=condition,
            manifest_tasks=manifest.tasks,
            base_config=base_config,
            tracker=tracker,
            swebench_cache=swebench_cache,
            ssg_dir=args.ssg_dir,
            agent_model_config=agent_config.models.agent,
            agent_workers=args.agent_workers,
            eval_batch_size=args.eval_batch_size,
            run_id=run_id,
            embedder=embedder,
            vector_store=vector_store,
            checkpoint_path=args.checkpoint,
            retrieval_k=args.retrieval_k,
        )
        all_summaries[condition] = summary

    output_dir = experiments_dir / "retrieval_benchmark"
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps({
        "run_id": run_id,
        "task_count": len(manifest.tasks),
        "conditions": all_summaries,
    }, indent=2), encoding="utf-8")
    logger.info("Benchmark complete. Results at %s", summary_path)


if __name__ == "__main__":
    main()

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
from token_sniper.evaluation.metrics import compare_task_resolution, token_usage_from_session_log
from token_sniper.evaluation.models import (
    ConditionRunSummary,
    SWEbenchEvaluationAttempt,
    TaskEvaluationResult,
)
from token_sniper.evaluation.phase0 import (
    build_oracle_collapsed_context,
    ensure_repo_cached,
    load_phase0_manifest,
    materialize_repo_snapshot,
    write_decision_document,
)
from token_sniper.evaluation.swebench import (
    HarnessPatchSubmission,
    SWEbenchHarnessRunner,
    load_swebench_task_metadata,
)
from token_sniper.evaluation.tracking import ExperimentTracker

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ConditionExecution:
    summary: ConditionRunSummary
    task_results: list[TaskEvaluationResult]
    difficulty_breakdown: dict[str, dict[str, float | int]]


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


def run_condition(
    *,
    condition: str,
    manifest_tasks: list[Any],
    base_config: dict[str, Any],
    phase_config: dict[str, Any],
    tracker: ExperimentTracker,
    swebench_cache: Path,
    agent_model_config: AgentModelConfig,
    agent_workers: int,
    eval_batch_size: int,
    run_id: str,
) -> ConditionExecution:
    dataset_name = phase_config.get("dataset", {}).get("benchmark", "swe-bench-verified")
    if dataset_name == "swe-bench-verified":
        dataset_name = "princeton-nlp/SWE-bench_Verified"

    logger.info("Starting %s run %s", condition, run_id)

    with tracker.start_run(
        phase="phase0",
        condition=condition,
        dataset_name=dataset_name,
        split="test",
        config_snapshot={"base": base_config, "phase": phase_config},
        run_name=run_id,
    ) as run:
        harness_config = HarnessConfig(**base_config.get("evaluation", {}).get("harness", {}))
        harness_runner = SWEbenchHarnessRunner(harness_config)
        task_results: dict[str, TaskEvaluationResult] = {}
        harness_submissions: list[HarnessPatchSubmission] = []
        harness_run_id = f"{condition}_{run_id}"
        _prewarm_condition_inputs(
            manifest_tasks=manifest_tasks,
            dataset_name=dataset_name,
            repo_cache_dir=swebench_cache / "repos",
        )

        generation_worker = partial(
            _generate_task_output,
            condition=condition,
            dataset_name=dataset_name,
            split="test",
            base_config=base_config,
            run_dir=run.run_dir,
            swebench_cache=swebench_cache,
            agent_model_config=agent_model_config,
        )

        pending_submissions: list[HarnessPatchSubmission] = []
        pending_batches: list[Future[HarnessBatchOutcome]] = []
        batch_index = 0
        with ThreadPoolExecutor(max_workers=1) as evaluation_executor:
            for output in _iter_condition_outputs(
                manifest_tasks=manifest_tasks,
                worker_fn=generation_worker,
                agent_workers=agent_workers,
            ):
                task_result = output.task_result
                run.log_task_result(task_result)
                task_results[task_result.instance_id] = task_result
                if output.harness_submission is not None:
                    harness_submissions.append(output.harness_submission)
                    pending_submissions.append(output.harness_submission)
                    if len(pending_submissions) >= eval_batch_size:
                        pending_batches.append(
                            _submit_harness_batch(
                                evaluation_executor=evaluation_executor,
                                condition=condition,
                                harness_runner=harness_runner,
                                run_dir=run.run_dir,
                                run_id=harness_run_id,
                                submissions=pending_submissions,
                                batch_index=batch_index,
                            )
                        )
                        batch_index += 1
                        pending_submissions = []
                pending_batches = _drain_completed_harness_batches(
                    pending_batches=pending_batches,
                    task_results=task_results,
                    run=run,
                )

            if pending_submissions:
                pending_batches.append(
                    _submit_harness_batch(
                        evaluation_executor=evaluation_executor,
                        condition=condition,
                        harness_runner=harness_runner,
                        run_dir=run.run_dir,
                        run_id=harness_run_id,
                        submissions=pending_submissions,
                        batch_index=batch_index,
                    )
                )

            for future in pending_batches:
                _apply_harness_batch_outcome(
                    outcome=future.result(),
                    task_results=task_results,
                    run=run,
                )

        summary = run.finalize()
        return ConditionExecution(
            summary=summary,
            task_results=list(task_results.values()),
            difficulty_breakdown=_load_difficulty_breakdown(summary),
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-config", type=Path, default=Path("configs/base.yaml"))
    parser.add_argument(
        "--phase-config",
        type=Path,
        default=Path("configs/phases/phase0_oracle.yaml"),
    )
    parser.add_argument("--manifest", type=Path, default=Path("configs/phases/phase0_tasks.json"))
    parser.add_argument(
        "--max-workers",
        type=int,
        default=None,
        help="Override SWE-bench harness max_workers for this run.",
    )
    parser.add_argument(
        "--agent-workers",
        type=int,
        default=None,
        help="Override concurrent agent sessions per condition for this run.",
    )
    parser.add_argument(
        "--condition-workers",
        type=int,
        default=None,
        help="Override concurrent condition runs for this invocation.",
    )
    parser.add_argument(
        "--eval-batch-size",
        type=int,
        default=None,
        help="Override the number of successful patches submitted per harness batch.",
    )
    parser.add_argument(
        "--conditions",
        nargs="+",
        choices=["blind_agent", "oracle_retrieval"],
        default=["blind_agent", "oracle_retrieval"],
        help="Subset of Phase 0 conditions to execute.",
    )
    args = parser.parse_args()
    if args.max_workers is not None and args.max_workers <= 0:
        parser.error("--max-workers must be positive.")
    if args.agent_workers is not None and args.agent_workers <= 0:
        parser.error("--agent-workers must be positive.")
    if args.condition_workers is not None and args.condition_workers <= 0:
        parser.error("--condition-workers must be positive.")
    if args.eval_batch_size is not None and args.eval_batch_size <= 0:
        parser.error("--eval-batch-size must be positive.")

    base_config_raw = load_yaml_config(args.base_config)
    phase_config_raw = load_yaml_config(args.phase_config)
    if args.max_workers is not None:
        base_config_raw.setdefault("evaluation", {}).setdefault("harness", {})["max_workers"] = (
            args.max_workers
        )
    if args.agent_workers is not None:
        base_config_raw.setdefault("evaluation", {})["agent_workers"] = args.agent_workers
    if args.condition_workers is not None:
        base_config_raw.setdefault("evaluation", {})["condition_workers"] = args.condition_workers
    if args.eval_batch_size is not None:
        base_config_raw.setdefault("evaluation", {}).setdefault("harness", {})[
            "submission_batch_size"
        ] = args.eval_batch_size

    agent_config = load_base_config(args.base_config)
    manifest = load_phase0_manifest(args.manifest)
    selected_conditions = _normalize_conditions(args.conditions)
    agent_workers = _coerce_positive_int(
        base_config_raw.get("evaluation", {}).get("agent_workers", 1),
        default=1,
        field_name="agent_workers",
    )
    condition_workers = _coerce_positive_int(
        base_config_raw.get("evaluation", {}).get("condition_workers", 1),
        default=1,
        field_name="condition_workers",
    )
    eval_batch_size = _coerce_positive_int(
        base_config_raw.get("evaluation", {})
        .get("harness", {})
        .get(
            "submission_batch_size",
            base_config_raw.get("evaluation", {}).get("harness", {}).get("max_workers", 1),
        ),
        default=1,
        field_name="eval_batch_size",
    )

    experiments_dir = Path(
        base_config_raw.get("paths", {}).get("experiments_dir", "experiments")
    ).resolve()
    swebench_cache = Path(
        base_config_raw.get("paths", {}).get("swebench_cache", "data/swebench")
    ).resolve()
    tracking_uri = base_config_raw.get("mlflow", {}).get("tracking_uri", "experiments/mlflow")
    experiment_prefix = base_config_raw.get("mlflow", {}).get("experiment_prefix", "token-sniper")

    tracker = ExperimentTracker(
        experiments_dir=experiments_dir,
        tracking_uri=tracking_uri,
        experiment_prefix=experiment_prefix,
    )

    run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    executions = _execute_conditions(
        conditions=selected_conditions,
        manifest_tasks=manifest.tasks,
        base_config=base_config_raw,
        phase_config=phase_config_raw,
        tracker=tracker,
        swebench_cache=swebench_cache,
        agent_model_config=agent_config.models.agent,
        agent_workers=agent_workers,
        condition_workers=condition_workers,
        eval_batch_size=eval_batch_size,
        run_id=run_id,
    )

    combined_metrics: dict[str, float | int] = {}
    for condition, execution in executions.items():
        for metric_name, metric_value in execution.summary.metrics.items():
            combined_metrics[f"{condition}.{metric_name}"] = metric_value

    paired_comparison: dict[str, Any] | None = None
    if {"blind_agent", "oracle_retrieval"}.issubset(executions):
        blind_pass = float(executions["blind_agent"].summary.metrics.get("pass_at_1", 0.0))
        oracle_pass = float(executions["oracle_retrieval"].summary.metrics.get("pass_at_1", 0.0))
        combined_metrics["retrieval_ceiling_gap"] = oracle_pass - blind_pass
        paired_comparison = compare_task_resolution(
            "blind_agent",
            "oracle_retrieval",
            executions["blind_agent"].task_results,
            executions["oracle_retrieval"].task_results,
        ).model_dump(mode="json")

    phase0_dir = experiments_dir / "phase0"
    phase0_dir.mkdir(parents=True, exist_ok=True)
    final_summary_path = phase0_dir / "summary.json"
    summary_data = {
        "phase": "phase0",
        "run_id": run_id,
        "requested_task_count": len(manifest.tasks),
        "requested_task_ids": [task.instance_id for task in manifest.tasks],
        "metrics": combined_metrics,
        "conditions": {
            condition: _build_condition_summary(
                manifest_tasks=manifest.tasks,
                execution=execution,
            )
            for condition, execution in executions.items()
        },
        "paired_comparison": paired_comparison,
    }
    final_summary_path.write_text(json.dumps(summary_data, indent=2), encoding="utf-8")

    if paired_comparison is not None:
        decision_path = phase0_dir / "DECISION.md"
        write_decision_document(summary_path=final_summary_path, decision_path=decision_path)
        logger.info("Phase 0 complete. See %s for results.", decision_path)
    else:
        logger.info("Phase 0 partial run complete. See %s for results.", final_summary_path)


def _execute_conditions(
    *,
    conditions: list[str],
    manifest_tasks: list[Any],
    base_config: dict[str, Any],
    phase_config: dict[str, Any],
    tracker: ExperimentTracker,
    swebench_cache: Path,
    agent_model_config: AgentModelConfig,
    agent_workers: int,
    condition_workers: int,
    eval_batch_size: int,
    run_id: str,
) -> dict[str, ConditionExecution]:
    if len(conditions) <= 1 or condition_workers == 1:
        return {
            condition: run_condition(
                condition=condition,
                manifest_tasks=manifest_tasks,
                base_config=base_config,
                phase_config=phase_config,
                tracker=tracker,
                swebench_cache=swebench_cache,
                agent_model_config=agent_model_config,
                agent_workers=agent_workers,
                eval_batch_size=eval_batch_size,
                run_id=run_id,
            )
            for condition in conditions
        }

    logger.info(
        "Running %s condition(s) with %s concurrent condition worker(s)",
        len(conditions),
        min(len(conditions), condition_workers),
    )
    executions: dict[str, ConditionExecution] = {}
    with ThreadPoolExecutor(max_workers=min(len(conditions), condition_workers)) as executor:
        future_to_condition = {
            executor.submit(
                run_condition,
                condition=condition,
                manifest_tasks=manifest_tasks,
                base_config=base_config,
                phase_config=phase_config,
                tracker=tracker,
                swebench_cache=swebench_cache,
                agent_model_config=agent_model_config,
                agent_workers=agent_workers,
                eval_batch_size=eval_batch_size,
                run_id=run_id,
            ): condition
            for condition in conditions
        }
        for future in as_completed(future_to_condition):
            executions[future_to_condition[future]] = future.result()
    return executions


def _iter_condition_outputs(
    *,
    manifest_tasks: list[Any],
    worker_fn: Any,
    agent_workers: int,
) -> Any:
    if len(manifest_tasks) <= 1 or agent_workers == 1:
        for task in manifest_tasks:
            yield worker_fn(task)
        return

    logger.info(
        "Running %s task(s) with %s concurrent agent worker(s)",
        len(manifest_tasks),
        agent_workers,
    )
    with ThreadPoolExecutor(max_workers=agent_workers) as executor:
        future_to_task = {executor.submit(worker_fn, task): task for task in manifest_tasks}
        for future in as_completed(future_to_task):
            yield future.result()


def _generate_task_output(
    task: Any,
    *,
    condition: str,
    dataset_name: str,
    split: str,
    base_config: dict[str, Any],
    run_dir: Path,
    swebench_cache: Path,
    agent_model_config: AgentModelConfig,
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

    mode: Literal["blind", "oracle"] = "oracle" if condition == "oracle_retrieval" else "blind"
    oracle_files = task.gold_patch_files if mode == "oracle" else []
    oracle_context = (
        build_oracle_collapsed_context(
            repo_root=repo_work_dir,
            gold_patch=metadata.gold_patch,
            oracle_files=oracle_files,
        )
        if mode == "oracle"
        else None
    )
    settings = AgentRunSettings(
        repo_root=str(repo_work_dir),
        issue_description=metadata.problem_statement,
        output_dir=str(agent_output_dir),
        mode=mode,
        oracle_files=oracle_files,
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

    patch_validation_summary = _summarize_patch_validation(agent_result)
    task_result = TaskEvaluationResult(
        instance_id=task.instance_id,
        phase="phase0",
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
            **patch_validation_summary,
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


def _prewarm_condition_inputs(
    *,
    manifest_tasks: list[Any],
    dataset_name: str,
    repo_cache_dir: Path,
) -> None:
    if not manifest_tasks:
        return
    load_swebench_task_metadata(manifest_tasks[0].instance_id, dataset_name=dataset_name)
    seen_repo_commits: set[tuple[str, str]] = set()
    for task in manifest_tasks:
        repo_commit = (str(task.repo), str(task.base_commit))
        if repo_commit in seen_repo_commits:
            continue
        ensure_repo_cached(
            repo=repo_commit[0], base_commit=repo_commit[1], cache_dir=repo_cache_dir
        )
        seen_repo_commits.add(repo_commit)


def _submit_harness_batch(
    *,
    evaluation_executor: ThreadPoolExecutor,
    condition: str,
    harness_runner: SWEbenchHarnessRunner,
    run_dir: Path,
    run_id: str,
    submissions: list[HarnessPatchSubmission],
    batch_index: int,
) -> Future[HarnessBatchOutcome]:
    logger.info(
        "[%s] Evaluating harness batch %s with %s patch(es) and max_workers=%s",
        condition,
        batch_index,
        len(submissions),
        harness_runner._config.max_workers,
    )
    return evaluation_executor.submit(
        _evaluate_harness_batch,
        harness_runner=harness_runner,
        run_dir=run_dir,
        run_id=run_id,
        submissions=list(submissions),
        batch_index=batch_index,
    )


def _evaluate_harness_batch(
    *,
    harness_runner: SWEbenchHarnessRunner,
    run_dir: Path,
    run_id: str,
    submissions: list[HarnessPatchSubmission],
    batch_index: int,
) -> HarnessBatchOutcome:
    try:
        attempts_by_instance = harness_runner.evaluate_patches(
            work_dir=run_dir / "harness" / f"batch_{batch_index:03d}",
            submissions=submissions,
            run_id=f"{run_id}_batch_{batch_index:03d}",
        )
    except Exception as exc:
        return HarnessBatchOutcome(
            batch_index=batch_index,
            submissions=submissions,
            error_message=f"{type(exc).__name__}: {exc}",
        )

    return HarnessBatchOutcome(
        batch_index=batch_index,
        submissions=submissions,
        attempts_by_instance=attempts_by_instance,
    )


def _drain_completed_harness_batches(
    *,
    pending_batches: list[Future[HarnessBatchOutcome]],
    task_results: dict[str, TaskEvaluationResult],
    run: Any,
) -> list[Future[HarnessBatchOutcome]]:
    remaining_batches: list[Future[HarnessBatchOutcome]] = []
    for future in pending_batches:
        if future.done():
            _apply_harness_batch_outcome(
                outcome=future.result(),
                task_results=task_results,
                run=run,
            )
            continue
        remaining_batches.append(future)
    return remaining_batches


def _apply_harness_batch_outcome(
    *,
    outcome: HarnessBatchOutcome,
    task_results: dict[str, TaskEvaluationResult],
    run: Any,
) -> None:
    if outcome.error_message is not None:
        logger.error("Harness batch %s failed: %s", outcome.batch_index, outcome.error_message)
        for submission in outcome.submissions:
            task_results[submission.instance_id] = _with_evaluation_error(
                task_results[submission.instance_id],
                outcome.error_message,
            )
            run.log_task_result(task_results[submission.instance_id])
        return

    attempts_by_instance = outcome.attempts_by_instance or {}
    for submission in outcome.submissions:
        attempt = attempts_by_instance.get(submission.instance_id)
        if attempt is None:
            task_results[submission.instance_id] = _with_evaluation_error(
                task_results[submission.instance_id],
                "Harness batch completed without returning an attempt for the instance.",
            )
        else:
            task_results[submission.instance_id] = _with_attempt(
                task_results[submission.instance_id],
                attempt,
            )
        run.log_task_result(task_results[submission.instance_id])


def _coerce_positive_int(value: Any, *, default: int, field_name: str) -> int:
    if value is None:
        return default
    try:
        resolved = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be an integer, got {value!r}.") from exc
    if resolved <= 0:
        raise ValueError(f"{field_name} must be positive, got {resolved}.")
    return resolved


def _normalize_conditions(conditions: list[str]) -> list[str]:
    normalized: list[str] = []
    for condition in conditions:
        if condition not in normalized:
            normalized.append(condition)
    if not normalized:
        raise ValueError("At least one condition must be provided.")
    return normalized


def _with_attempt(
    task_result: TaskEvaluationResult,
    attempt: SWEbenchEvaluationAttempt,
) -> TaskEvaluationResult:
    metadata = dict(task_result.metadata)
    evaluation_error = attempt.metadata.get("evaluation_error")
    metadata["evaluation_error"] = evaluation_error or None
    return task_result.model_copy(
        update={
            "resolved": attempt.resolved,
            "attempts": [attempt],
            "metadata": metadata,
        }
    )


def _with_evaluation_error(
    task_result: TaskEvaluationResult,
    error_message: str,
) -> TaskEvaluationResult:
    metadata = dict(task_result.metadata)
    metadata["evaluation_error"] = error_message
    return task_result.model_copy(update={"resolved": False, "attempts": [], "metadata": metadata})


def _build_condition_summary(
    *,
    manifest_tasks: list[Any],
    execution: ConditionExecution,
) -> dict[str, Any]:
    requested_task_ids = [task.instance_id for task in manifest_tasks]
    logged_task_ids = {task_result.instance_id for task_result in execution.task_results}
    missing_logged_task_ids = sorted(set(requested_task_ids) - logged_task_ids)

    attempted_task_ids = sorted(
        task_result.instance_id for task_result in execution.task_results if task_result.attempts
    )
    resolved_task_ids = sorted(
        {task_result.instance_id for task_result in execution.task_results if task_result.resolved}
    )
    unresolved_evaluated_task_ids = sorted(
        {
            task_result.instance_id
            for task_result in execution.task_results
            if task_result.attempts and not task_result.resolved
        }
    )
    patch_attempted_task_ids = sorted(
        task_result.instance_id
        for task_result in execution.task_results
        if int(task_result.metadata.get("patch_attempt_count", 0)) > 0
    )
    valid_patch_task_ids = sorted(
        task_result.instance_id
        for task_result in execution.task_results
        if int(task_result.metadata.get("valid_patch_attempt_count", 0)) > 0
    )
    invalid_patch_task_ids = sorted(
        task_result.instance_id
        for task_result in execution.task_results
        if not task_result.attempts
        and task_result.metadata.get("agent_status") == "failed"
        and int(task_result.metadata.get("invalid_patch_attempt_count", 0)) > 0
        and int(task_result.metadata.get("valid_patch_attempt_count", 0)) == 0
    )
    no_patch_task_ids = sorted(
        task_result.instance_id
        for task_result in execution.task_results
        if not task_result.attempts
        and task_result.metadata.get("agent_status") == "failed"
        and int(task_result.metadata.get("patch_attempt_count", 0)) == 0
    )
    agent_failed_task_ids = sorted(
        {
            task_result.instance_id
            for task_result in execution.task_results
            if not task_result.attempts and task_result.metadata.get("agent_status") == "failed"
        }
    )
    evaluation_error_task_ids = sorted(
        {
            task_result.instance_id
            for task_result in execution.task_results
            if task_result.metadata.get("evaluation_error")
        }
    )
    unattempted_task_ids = sorted(
        {
            task_result.instance_id
            for task_result in execution.task_results
            if not task_result.attempts
        }
        | set(missing_logged_task_ids)
    )
    not_resolved_task_ids = sorted(
        (set(requested_task_ids) - set(resolved_task_ids)) | set(missing_logged_task_ids)
    )

    return {
        "run_id": execution.summary.run_id,
        "run_dir": execution.summary.run_dir,
        "summary_path": str(Path(execution.summary.run_dir) / "summary.json"),
        "metrics": execution.summary.metrics,
        "requested_task_count": len(requested_task_ids),
        "logged_task_count": len(execution.task_results),
        "attempted_task_count": len(attempted_task_ids),
        "resolved_task_count": len(resolved_task_ids),
        "unresolved_evaluated_task_count": len(unresolved_evaluated_task_ids),
        "patch_attempted_task_count": len(patch_attempted_task_ids),
        "valid_patch_task_count": len(valid_patch_task_ids),
        "invalid_patch_task_count": len(invalid_patch_task_ids),
        "no_patch_task_count": len(no_patch_task_ids),
        "agent_failed_task_count": len(agent_failed_task_ids),
        "evaluation_error_task_count": len(evaluation_error_task_ids),
        "unattempted_task_count": len(unattempted_task_ids),
        "missing_task_count": len(missing_logged_task_ids),
        "not_resolved_task_count": len(not_resolved_task_ids),
        "attempted_task_ids": attempted_task_ids,
        "resolved_task_ids": resolved_task_ids,
        "unresolved_evaluated_task_ids": unresolved_evaluated_task_ids,
        "patch_attempted_task_ids": patch_attempted_task_ids,
        "valid_patch_task_ids": valid_patch_task_ids,
        "invalid_patch_task_ids": invalid_patch_task_ids,
        "no_patch_task_ids": no_patch_task_ids,
        "agent_failed_task_ids": agent_failed_task_ids,
        "evaluation_error_task_ids": evaluation_error_task_ids,
        "unattempted_task_ids": unattempted_task_ids,
        "missing_task_ids": missing_logged_task_ids,
        "not_resolved_task_ids": not_resolved_task_ids,
        "difficulty_breakdown": execution.difficulty_breakdown,
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
        error_message="RuntimeError: Agent failed before writing a session log.",
    )


def _summarize_patch_validation(agent_result: AgentRunResult) -> dict[str, Any]:
    validation_logs = [
        turn.patch_validation for turn in agent_result.turns if turn.patch_validation is not None
    ]
    patch_attempt_count = len(validation_logs)
    valid_patch_attempt_count = sum(
        1 for log in validation_logs if log.validation_status == "accepted"
    )
    invalid_patch_attempt_count = patch_attempt_count - valid_patch_attempt_count
    invalid_logs = [log for log in validation_logs if log.validation_status == "rejected"]
    last_invalid_log = invalid_logs[-1] if invalid_logs else None
    repair_turn_count = sum(1 for log in validation_logs if log.repair_prompt_sent)
    return {
        "patch_attempt_count": patch_attempt_count,
        "valid_patch_attempt_count": valid_patch_attempt_count,
        "invalid_patch_attempt_count": invalid_patch_attempt_count,
        "repair_turn_count": repair_turn_count,
        "patch_validation_failure_stage": (
            last_invalid_log.validation_stage if last_invalid_log is not None else None
        ),
        "patch_validation_failure_message": (
            last_invalid_log.validation_message if last_invalid_log is not None else None
        ),
    }


def _load_difficulty_breakdown(
    summary: ConditionRunSummary,
) -> dict[str, dict[str, float | int]]:
    path = Path(summary.run_dir) / "artifacts" / "difficulty_breakdown.json"
    if not path.is_file():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        return {}
    return {
        str(difficulty): {
            str(metric_name): metric_value
            for metric_name, metric_value in metrics.items()
            if isinstance(metric_value, int | float)
        }
        for difficulty, metrics in data.items()
        if isinstance(metrics, dict)
    }


if __name__ == "__main__":
    main()

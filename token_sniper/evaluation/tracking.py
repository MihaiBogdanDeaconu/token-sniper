from __future__ import annotations

import json
import shutil
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import mlflow
import yaml

from token_sniper.evaluation.metrics import summarize_condition_metrics, to_jsonl
from token_sniper.evaluation.models import ConditionRunSummary, TaskEvaluationResult

if TYPE_CHECKING:
    from collections.abc import Mapping


class ExperimentTracker:
    def __init__(
        self,
        *,
        experiments_dir: Path | str,
        tracking_uri: str,
        experiment_prefix: str,
    ) -> None:
        self._experiments_dir = Path(experiments_dir).resolve()
        self._tracking_uri = _resolve_tracking_uri(tracking_uri)
        self._experiment_prefix = experiment_prefix

    def start_run(
        self,
        *,
        phase: str,
        condition: str,
        dataset_name: str,
        split: str,
        config_snapshot: Mapping[str, Any],
        run_name: str | None = None,
        tags: Mapping[str, str] | None = None,
        retrieval_k_values: list[int] | None = None,
    ) -> TrackedEvaluationRun:
        timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        run_id = run_name or timestamp
        run_dir = self._experiments_dir / phase / condition / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        return TrackedEvaluationRun(
            phase=phase,
            condition=condition,
            dataset_name=dataset_name,
            split=split,
            run_id=run_id,
            run_dir=run_dir,
            tracking_uri=self._tracking_uri,
            experiment_name=f"{self._experiment_prefix}-{phase}",
            config_snapshot=dict(config_snapshot),
            tags=dict(tags or {}),
            retrieval_k_values=list(retrieval_k_values or []),
        )


class TrackedEvaluationRun:
    def __init__(
        self,
        *,
        phase: str,
        condition: str,
        dataset_name: str,
        split: str,
        run_id: str,
        run_dir: Path,
        tracking_uri: str,
        experiment_name: str,
        config_snapshot: Mapping[str, Any],
        tags: Mapping[str, str],
        retrieval_k_values: list[int],
    ) -> None:
        self.phase = phase
        self.condition = condition
        self.dataset_name = dataset_name
        self.split = split
        self.run_id = run_id
        self.run_dir = run_dir
        self.tasks_dir = run_dir / "tasks"
        self.artifacts_dir = run_dir / "artifacts"
        self.plots_dir = run_dir / "plots"
        self.logs_dir = run_dir / "logs"
        self.tasks_dir.mkdir(parents=True, exist_ok=True)
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)
        self.plots_dir.mkdir(parents=True, exist_ok=True)
        self.logs_dir.mkdir(parents=True, exist_ok=True)

        self._retrieval_k_values = retrieval_k_values
        self._task_results: dict[str, TaskEvaluationResult] = {}
        self._closed = False

        self.config_path = self.run_dir / "config.yaml"
        self.config_path.write_text(
            yaml.safe_dump(dict(config_snapshot), sort_keys=False),
            encoding="utf-8",
        )

        self.metadata_path = self.run_dir / "metadata.json"
        metadata = {
            "phase": phase,
            "condition": condition,
            "dataset_name": dataset_name,
            "split": split,
            "run_id": run_id,
            "created_at": datetime.now(UTC).isoformat(),
            "git_branch": _safe_git_output(["git", "branch", "--show-current"]),
            "git_commit": _safe_git_output(["git", "rev-parse", "HEAD"]),
            "tags": dict(tags),
        }
        self.metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment_name)
        self._mlflow_run = mlflow.start_run(run_name=f"{condition}-{run_id}")
        mlflow.set_tags(
            {
                "phase": phase,
                "condition": condition,
                "dataset_name": dataset_name,
                "split": split,
                **dict(tags),
            }
        )
        mlflow.log_artifact(str(self.config_path), artifact_path="config")
        mlflow.log_artifact(str(self.metadata_path), artifact_path="metadata")

    def __enter__(self) -> TrackedEvaluationRun:
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        del exc_type, exc, tb
        self.close()

    def log_task_result(self, task_result: TaskEvaluationResult) -> Path:
        self._task_results[task_result.instance_id] = task_result
        task_path = self.tasks_dir / f"{task_result.instance_id}.json"
        task_path.write_text(task_result.model_dump_json(indent=2), encoding="utf-8")
        return task_path

    def log_artifact(self, source_path: Path | str, *, artifact_subdir: str = "artifacts") -> Path:
        source = Path(source_path).resolve()
        target_dir = self.run_dir / artifact_subdir
        target_dir.mkdir(parents=True, exist_ok=True)
        target = target_dir / source.name
        if source != target:
            shutil.copy2(source, target)
        mlflow.log_artifact(str(target), artifact_path=artifact_subdir)
        return target

    def log_plot(self, source_path: Path | str) -> Path:
        return self.log_artifact(source_path, artifact_subdir="plots")

    def finalize(self) -> ConditionRunSummary:
        task_results = list(self._task_results.values())
        metrics = summarize_condition_metrics(
            task_results,
            retrieval_k_values=self._retrieval_k_values,
        )
        metrics_path = self.run_dir / "metrics.json"
        metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

        task_results_path = self.run_dir / "task_results.jsonl"
        task_results_path.write_text(
            to_jsonl(task.model_dump(mode="json") for task in task_results),
            encoding="utf-8",
        )

        repo_breakdown = _group_breakdown(
            task_results,
            key="repo",
            retrieval_k_values=self._retrieval_k_values,
        )
        difficulty_breakdown = _group_breakdown(
            task_results,
            key="difficulty",
            retrieval_k_values=self._retrieval_k_values,
        )
        repo_breakdown_path = self.artifacts_dir / "repo_breakdown.json"
        difficulty_breakdown_path = self.artifacts_dir / "difficulty_breakdown.json"
        repo_breakdown_path.write_text(json.dumps(repo_breakdown, indent=2), encoding="utf-8")
        difficulty_breakdown_path.write_text(
            json.dumps(difficulty_breakdown, indent=2),
            encoding="utf-8",
        )

        mlflow.log_metrics({k: float(v) for k, v in metrics.items() if isinstance(v, (int, float))})
        mlflow.log_artifact(str(metrics_path), artifact_path="metrics")
        mlflow.log_artifact(str(task_results_path), artifact_path="metrics")
        mlflow.log_artifact(str(repo_breakdown_path), artifact_path="artifacts")
        mlflow.log_artifact(str(difficulty_breakdown_path), artifact_path="artifacts")

        summary = ConditionRunSummary(
            phase=self.phase,
            condition=self.condition,
            run_id=self.run_id,
            dataset_name=self.dataset_name,
            split=self.split,
            run_dir=str(self.run_dir),
            config_path=str(self.config_path),
            metrics_path=str(metrics_path),
            task_results_path=str(task_results_path),
            total_tasks=len(task_results),
            metrics=metrics,
            tags={
                key: str(value)
                for key, value in mlflow.get_run(self._mlflow_run.info.run_id).data.tags.items()
            },
        )
        summary_path = self.run_dir / "summary.json"
        summary_path.write_text(summary.model_dump_json(indent=2), encoding="utf-8")
        mlflow.log_artifact(str(summary_path), artifact_path="metrics")
        self.close()
        return summary

    def close(self) -> None:
        if self._closed:
            return
        mlflow.end_run()
        self._closed = True


def _group_breakdown(
    task_results: list[TaskEvaluationResult],
    *,
    key: str,
    retrieval_k_values: list[int],
) -> dict[str, dict[str, float | int]]:
    groups: dict[str, list[TaskEvaluationResult]] = {}
    for task in task_results:
        raw_value = getattr(task, key)
        group_name = str(raw_value or "unknown")
        groups.setdefault(group_name, []).append(task)
    return {
        group: summarize_condition_metrics(results, retrieval_k_values=retrieval_k_values)
        for group, results in sorted(groups.items())
    }


def _resolve_tracking_uri(tracking_uri: str) -> str:
    if "://" in tracking_uri:
        return tracking_uri
    return Path(tracking_uri).resolve().as_uri()


def _safe_git_output(command: list[str]) -> str:
    try:
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return completed.stdout.strip() or "unknown"

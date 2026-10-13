from __future__ import annotations

import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from token_sniper.evaluation.metrics import token_usage_from_session_log
from token_sniper.evaluation.models import TaskEvaluationResult
from token_sniper.evaluation.phase0 import Phase0Task
from token_sniper.evaluation.swebench import HarnessPatchSubmission

logger = logging.getLogger(__name__)


def patch_path(run_dir: Path, instance_id: str) -> Path:
    return run_dir / "agent_outputs" / instance_id / "patch.diff"


def has_patch_on_disk(run_dir: Path, instance_id: str) -> bool:
    patch_file = patch_path(run_dir, instance_id)
    if not patch_file.is_file():
        return False
    return bool(patch_file.read_text(encoding="utf-8").strip())


def load_task_result_if_exists(run_dir: Path, instance_id: str) -> TaskEvaluationResult | None:
    task_path = run_dir / "tasks" / f"{instance_id}.json"
    if not task_path.is_file():
        return None
    try:
        return TaskEvaluationResult.model_validate_json(
            task_path.read_text(encoding="utf-8"),
        )
    except Exception as exc:
        logger.warning("Failed to load task result %s: %s", task_path, exc)
        return None


def is_task_complete(run_dir: Path, instance_id: str) -> bool:
    if has_patch_on_disk(run_dir, instance_id):
        return True
    existing = load_task_result_if_exists(run_dir, instance_id)
    return existing is not None and bool(existing.attempts)


def find_latest_run_id(output_dir: Path) -> str | None:
    candidates = [
        child for child in output_dir.iterdir()
        if child.is_dir() and (
            (child / "agent_outputs").is_dir() or (child / "tasks").is_dir()
        )
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda path: path.stat().st_mtime).name


def write_task_result(run_dir: Path, task_result: TaskEvaluationResult) -> Path:
    tasks_dir = run_dir / "tasks"
    tasks_dir.mkdir(parents=True, exist_ok=True)
    task_path = tasks_dir / f"{task_result.instance_id}.json"
    task_path.write_text(task_result.model_dump_json(indent=2), encoding="utf-8")
    return task_path


def resolve_run_id(
    output_dir: Path,
    *,
    resume: bool,
    run_id: str | None,
) -> str:
    if run_id is not None:
        run_dir = output_dir / run_id
        if resume and not run_dir.is_dir():
            raise SystemExit(f"--run-id {run_id} not found under {output_dir}")
        return run_id
    if resume:
        latest = find_latest_run_id(output_dir)
        if latest is None:
            raise SystemExit(
                f"--resume requested but no prior run found under {output_dir}",
            )
        return latest
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")


def load_existing_task_results(run_dir: Path) -> dict[str, TaskEvaluationResult]:
    results: dict[str, TaskEvaluationResult] = {}
    tasks_dir = run_dir / "tasks"
    if not tasks_dir.is_dir():
        return results
    for task_path in tasks_dir.glob("*.json"):
        try:
            task_result = TaskEvaluationResult.model_validate_json(
                task_path.read_text(encoding="utf-8"),
            )
        except Exception as exc:
            logger.warning("Skipping unreadable task result %s: %s", task_path, exc)
            continue
        results[task_result.instance_id] = task_result
    return results


def build_harness_submissions(
    tasks: list[Phase0Task],
    *,
    run_dir: Path,
    task_results: dict[str, TaskEvaluationResult],
    base_config: dict[str, Any],
) -> list[HarnessPatchSubmission]:
    model_name = base_config.get("models", {}).get("agent", {}).get("name", "agent")
    submissions: list[HarnessPatchSubmission] = []
    for task in tasks:
        existing = task_results.get(task.instance_id)
        if existing is not None and existing.attempts:
            continue
        patch_file = patch_path(run_dir, task.instance_id)
        if not patch_file.is_file() or not patch_file.read_text(encoding="utf-8").strip():
            continue
        session_log = run_dir / "agent_outputs" / task.instance_id / "session.json"
        token_usage = None
        if session_log.is_file():
            try:
                token_usage = token_usage_from_session_log(str(session_log))
            except Exception:
                pass
        submissions.append(
            HarnessPatchSubmission(
                instance_id=task.instance_id,
                patch_file=patch_file,
                model_name=model_name,
                session_log_path=session_log if session_log.is_file() else None,
                token_usage=token_usage,
            )
        )
    return submissions


def pending_tasks(tasks: list[Phase0Task], run_dir: Path) -> list[Phase0Task]:
    return [task for task in tasks if not is_task_complete(run_dir, task.instance_id)]

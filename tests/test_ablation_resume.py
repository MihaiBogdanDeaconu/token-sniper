from __future__ import annotations

import os
from pathlib import Path

from token_sniper.evaluation.ablation_resume import (
    find_latest_run_id,
    is_task_complete,
    patch_path,
    pending_tasks,
)
from token_sniper.evaluation.models import SWEbenchEvaluationAttempt, TaskEvaluationResult
from token_sniper.evaluation.phase0 import Phase0Task


def _sample_task(instance_id: str = "org__repo-1") -> Phase0Task:
    return Phase0Task(
        instance_id=instance_id,
        repo="org/repo",
        base_commit="abc123",
        difficulty="medium",
        gold_patch_files=["src/foo.py"],
        gold_patch_file_count=1,
        bucket="single_file",
    )


def test_is_task_complete_with_patch(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    instance_id = "org__repo-1"
    patch = patch_path(run_dir, instance_id)
    patch.parent.mkdir(parents=True)
    patch.write_text("diff --git a/foo b/foo\n", encoding="utf-8")

    assert is_task_complete(run_dir, instance_id) is True


def test_is_task_complete_with_harness_attempt(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    instance_id = "org__repo-2"
    tasks_dir = run_dir / "tasks"
    tasks_dir.mkdir(parents=True)
    result = TaskEvaluationResult(
        instance_id=instance_id,
        phase="ablation",
        condition="dense_knn",
        dataset_name="SWE-bench-Live/SWE-bench-Live",
        split="verified",
        resolved=True,
        attempts=[
            SWEbenchEvaluationAttempt(
                attempt_index=1,
                model_name="agent",
                patch_path="patch.diff",
                patch_sha256="abc",
                run_id="run",
                resolved=True,
                patch_exists=True,
                patch_successfully_applied=True,
                summary_report_path="summary.json",
                instance_report_path="report.json",
                harness_stdout_path="stdout.log",
                harness_stderr_path="stderr.log",
            )
        ],
    )
    (tasks_dir / f"{instance_id}.json").write_text(
        result.model_dump_json(indent=2),
        encoding="utf-8",
    )

    assert is_task_complete(run_dir, instance_id) is True


def test_is_task_complete_failed_agent_is_pending(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    instance_id = "org__repo-3"
    tasks_dir = run_dir / "tasks"
    tasks_dir.mkdir(parents=True)
    result = TaskEvaluationResult(
        instance_id=instance_id,
        phase="ablation",
        condition="dense_knn",
        dataset_name="SWE-bench-Live/SWE-bench-Live",
        split="verified",
        resolved=False,
        metadata={"agent_status": "failed", "agent_error": "connection error"},
    )
    (tasks_dir / f"{instance_id}.json").write_text(
        result.model_dump_json(indent=2),
        encoding="utf-8",
    )

    assert is_task_complete(run_dir, instance_id) is False


def test_pending_tasks_filters_complete(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    done = _sample_task("done__task-1")
    pending = _sample_task("pending__task-2")
    patch = patch_path(run_dir, done.instance_id)
    patch.parent.mkdir(parents=True)
    patch.write_text("diff\n", encoding="utf-8")

    result = pending_tasks([done, pending], run_dir)
    assert [task.instance_id for task in result] == [pending.instance_id]


def test_find_latest_run_id_from_agent_outputs(tmp_path: Path) -> None:
    run_dir = tmp_path / "20260102T000000Z"
    (run_dir / "agent_outputs" / "org__repo-1").mkdir(parents=True)
    os.utime(run_dir, (2_000, 2_000))

    assert find_latest_run_id(tmp_path) == "20260102T000000Z"


def test_find_latest_run_id(tmp_path: Path) -> None:
    older = tmp_path / "20260101T000000Z"
    newer = tmp_path / "20260102T000000Z"
    (older / "tasks").mkdir(parents=True)
    (newer / "tasks").mkdir(parents=True)
    os.utime(older, (1_000, 1_000))
    os.utime(newer, (2_000, 2_000))

    assert find_latest_run_id(tmp_path) == "20260102T000000Z"

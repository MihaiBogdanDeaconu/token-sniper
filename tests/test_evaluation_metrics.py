from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from token_sniper.agent.runner import AgentRunResult, TurnLog, TurnToolLog
from token_sniper.evaluation.metrics import (
    average_precision,
    compare_task_resolution,
    compute_retrieval_metrics,
    compute_token_efficiency,
    extract_modified_files_from_patch,
    mcnemar_test,
    pass_at_k,
    token_usage_from_session_log,
)
from token_sniper.evaluation.models import SWEbenchEvaluationAttempt, TaskEvaluationResult

if TYPE_CHECKING:
    from pathlib import Path


def test_extract_modified_files_from_patch_handles_diff_headers() -> None:
    patch = (
        "diff --git a/src/app.py b/src/app.py\n"
        "--- a/src/app.py\n"
        "+++ b/src/app.py\n"
        "@@ -1 +1 @@\n"
        "-print('old')\n"
        "+print('new')\n"
        "diff --git a/tests/test_app.py b/tests/test_app.py\n"
        "--- a/tests/test_app.py\n"
        "+++ b/tests/test_app.py\n"
        "@@ -1 +1 @@\n"
        "-assert True\n"
        "+assert False\n"
    )
    assert extract_modified_files_from_patch(patch) == ["src/app.py", "tests/test_app.py"]


def test_pass_at_k_and_retrieval_metrics() -> None:
    assert pass_at_k([[False, True], [False, False], [True]], k=1) == 1 / 3
    assert pass_at_k([[False, True], [False, False], [True]], k=3) == 2 / 3

    metrics = compute_retrieval_metrics(
        ["a.py", "b.py", "c.py"],
        {"b.py", "c.py"},
        k_values=[1, 2, 3],
    )
    assert metrics["retrieval_precision_at_1"] == 0.0
    assert metrics["retrieval_precision_at_2"] == 0.5
    assert metrics["retrieval_recall_at_3"] == 1.0
    assert average_precision(["a.py", "b.py", "c.py"], {"b.py", "c.py"}) == (1 / 2 + 2 / 3) / 2


def test_token_usage_from_session_log_and_efficiency(tmp_path: Path) -> None:
    session = AgentRunResult(
        mode="blind",
        repo_root=str(tmp_path),
        output_dir=str(tmp_path / "out"),
        patch_path=str(tmp_path / "out" / "patch.diff"),
        session_log_path=str(tmp_path / "out" / "session.json"),
        total_prompt_tokens=52,
        total_completion_tokens=26,
        total_wall_time_seconds=1.0,
        turns=[
            TurnLog(
                turn_index=1,
                prompt_tokens=10,
                completion_tokens=4,
                wall_time_seconds=0.1,
                tool_calls=[],
                assistant_text="Reasoning only.",
            ),
            TurnLog(
                turn_index=2,
                prompt_tokens=30,
                completion_tokens=16,
                wall_time_seconds=0.2,
                tool_calls=[
                    TurnToolLog(
                        id="tool-1",
                        name="cat",
                        arguments={"path": "app.py"},
                        result={"path": "app.py"},
                    )
                ],
                assistant_text="Need to inspect the file first.",
            ),
            TurnLog(
                turn_index=3,
                prompt_tokens=12,
                completion_tokens=6,
                wall_time_seconds=0.3,
                tool_calls=[],
                assistant_text="Patch is ready.",
            ),
        ],
    )
    session_path = tmp_path / "session.json"
    session_path.write_text(session.model_dump_json(indent=2), encoding="utf-8")

    breakdown = token_usage_from_session_log(session_path)
    assert breakdown.reasoning_tokens == 50
    assert breakdown.navigation_tokens == 28

    task_results = [
        TaskEvaluationResult(
            instance_id="repo__task-1",
            phase="phase0",
            condition="blind_agent",
            dataset_name="princeton-nlp/SWE-bench_Verified",
            split="test",
            resolved=True,
            attempts=[
                SWEbenchEvaluationAttempt(
                    attempt_index=1,
                    model_name="token-sniper",
                    patch_path="patch.diff",
                    patch_sha256="abc",
                    run_id="run-1",
                    resolved=True,
                    patch_exists=True,
                    patch_successfully_applied=True,
                    summary_report_path="summary.json",
                    instance_report_path="report.json",
                    harness_stdout_path="stdout.log",
                    harness_stderr_path="stderr.log",
                    token_usage=breakdown,
                )
            ],
        ),
        TaskEvaluationResult(
            instance_id="repo__task-2",
            phase="phase0",
            condition="blind_agent",
            dataset_name="princeton-nlp/SWE-bench_Verified",
            split="test",
            resolved=False,
            attempts=[],
            metadata={"session_log_path": str(session_path), "patch_path": None},
        ),
    ]

    metrics = compute_token_efficiency(task_results)
    assert metrics["resolved_tasks"] == 1
    assert metrics["total_tokens"] == 156
    assert metrics["token_efficiency_total"] == 1000 / 156


def test_mcnemar_and_paired_comparison() -> None:
    stats = mcnemar_test([True, True, False, False], [True, False, True, False])
    assert stats["baseline_only_correct"] == 1
    assert stats["candidate_only_correct"] == 1
    assert stats["discordant_pairs"] == 2

    baseline = [
        TaskEvaluationResult(
            instance_id="task-1",
            phase="phase5",
            condition="baseline",
            dataset_name="dataset",
            split="test",
            resolved=True,
        ),
        TaskEvaluationResult(
            instance_id="task-2",
            phase="phase5",
            condition="baseline",
            dataset_name="dataset",
            split="test",
            resolved=False,
        ),
    ]
    candidate = [
        TaskEvaluationResult(
            instance_id="task-1",
            phase="phase5",
            condition="candidate",
            dataset_name="dataset",
            split="test",
            resolved=False,
        ),
        TaskEvaluationResult(
            instance_id="task-2",
            phase="phase5",
            condition="candidate",
            dataset_name="dataset",
            split="test",
            resolved=True,
        ),
    ]

    comparison = compare_task_resolution("baseline", "candidate", baseline, candidate)
    assert comparison.paired_tasks == 2
    assert comparison.baseline_only_correct == 1
    assert comparison.candidate_only_correct == 1


def test_compare_task_resolution_requires_matching_tasks() -> None:
    baseline = [
        TaskEvaluationResult(
            instance_id="task-1",
            phase="phase5",
            condition="baseline",
            dataset_name="dataset",
            split="test",
            resolved=True,
        )
    ]
    candidate = [
        TaskEvaluationResult(
            instance_id="task-2",
            phase="phase5",
            condition="candidate",
            dataset_name="dataset",
            split="test",
            resolved=True,
        )
    ]

    with pytest.raises(ValueError, match="same task set"):
        compare_task_resolution("baseline", "candidate", baseline, candidate)

import json
from contextlib import contextmanager
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

from token_sniper.agent.runner import AgentRunResult
from token_sniper.evaluation.models import (
    ConditionRunSummary,
    SWEbenchEvaluationAttempt,
    TaskEvaluationResult,
    TokenUsageBreakdown,
)
from token_sniper.evaluation.phase0 import build_oracle_collapsed_context, write_decision_document
from token_sniper.evaluation.swebench import HarnessPatchSubmission


def test_write_decision_document(tmp_path: Path) -> None:
    summary_path = tmp_path / "summary.json"
    decision_path = tmp_path / "DECISION.md"

    summary_path.write_text(
        json.dumps({"metrics": {"blind_agent.pass_at_1": 0.1, "oracle_retrieval.pass_at_1": 0.2}})
    )
    write_decision_document(summary_path=summary_path, decision_path=decision_path)
    content = decision_path.read_text()
    assert "Retrieval Ceiling Gap:** 10.0%" in content
    assert "DECISION: GO" in content

    summary_path.write_text(
        json.dumps({"metrics": {"blind_agent.pass_at_1": 0.1, "oracle_retrieval.pass_at_1": 0.15}})
    )
    write_decision_document(summary_path=summary_path, decision_path=decision_path)
    content = decision_path.read_text()
    assert "Retrieval Ceiling Gap:** 5.0%" in content
    assert "DECISION: NO-GO / PIVOT" in content


def test_build_oracle_collapsed_context_limits_to_edit_windows(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    target_file = repo / "pkg" / "target.py"
    target_file.parent.mkdir(parents=True, exist_ok=True)
    target_file.write_text(
        "\n".join(f"line {line_number}" for line_number in range(1, 81)) + "\n",
        encoding="utf-8",
    )

    gold_patch = "\n".join(
        [
            "diff --git a/pkg/target.py b/pkg/target.py",
            "--- a/pkg/target.py",
            "+++ b/pkg/target.py",
            "@@ -30,2 +30,2 @@",
            "-line 30",
            "+updated line 30",
        ]
    )

    oracle_context = build_oracle_collapsed_context(
        repo_root=repo,
        gold_patch=gold_patch,
        oracle_files=["pkg/target.py", "pkg/created_later.py"],
        context_lines=2,
    )

    assert "Path: pkg/target.py" in oracle_context
    assert "Window 1 (lines 28-33)" in oracle_context
    assert "  28: line 28" in oracle_context
    assert "  33: line 33" in oracle_context
    assert "   1: line 1" not in oracle_context
    assert "pkg/created_later.py" in oracle_context
    assert "do not exist at the base commit" in oracle_context


@patch("scripts.run_phase0.load_phase0_manifest")
@patch("scripts.run_phase0._execute_conditions")
def test_main_writes_summary_and_passes_speedup_overrides(
    mock_execute_conditions: Any,
    mock_manifest: Any,
    tmp_path: Path,
) -> None:
    import scripts.run_phase0

    task_a = MagicMock(instance_id="task_a")
    task_b = MagicMock(instance_id="task_b")
    mock_manifest.return_value = MagicMock(tasks=[task_a, task_b])

    condition_run_dir = tmp_path / "run_dir"
    condition_run_dir.mkdir(parents=True, exist_ok=True)
    execution = scripts.run_phase0.ConditionExecution(
        summary=ConditionRunSummary(
            phase="phase0",
            condition="blind_agent",
            run_id="blind-run",
            dataset_name="princeton-nlp/SWE-bench_Verified",
            split="test",
            run_dir=str(condition_run_dir),
            config_path=str(condition_run_dir / "config.yaml"),
            metrics_path=str(condition_run_dir / "metrics.json"),
            task_results_path=str(condition_run_dir / "task_results.jsonl"),
            total_tasks=2,
            metrics={"pass_at_1": 0.5, "resolved_tasks": 1},
            tags={},
        ),
        task_results=[
            _task_result("task_a", "blind_agent", resolved=True, evaluation_error=None),
            _task_result("task_b", "blind_agent", resolved=False, evaluation_error="Harness error"),
        ],
        difficulty_breakdown={"easy": {"pass_at_1": 0.5, "num_tasks": 2}},
    )
    oracle_execution = scripts.run_phase0.ConditionExecution(
        summary=execution.summary.model_copy(
            update={
                "condition": "oracle_retrieval",
                "run_id": "oracle-run",
                "metrics": {"pass_at_1": 0.75, "resolved_tasks": 2},
            }
        ),
        task_results=[
            _task_result("task_a", "oracle_retrieval", resolved=True, evaluation_error=None),
            _task_result("task_b", "oracle_retrieval", resolved=True, evaluation_error=None),
        ],
        difficulty_breakdown={"easy": {"pass_at_1": 1.0, "num_tasks": 2}},
    )
    mock_execute_conditions.return_value = {
        "blind_agent": execution,
        "oracle_retrieval": oracle_execution,
    }

    experiments_dir = tmp_path / "exp"
    base_config = tmp_path / "base.yaml"
    base_config.write_text(
        "\n".join(
            [
                "models:",
                "  agent:",
                "    name: test",
                "    provider: local",
                "    base_url: http://localhost",
                "    api_key_env: KEY",
                "paths:",
                f"  experiments_dir: {experiments_dir.as_posix()}",
                "evaluation:",
                "  agent_workers: 2",
                "  condition_workers: 1",
                "  harness:",
                "    max_workers: 4",
                "    submission_batch_size: 8",
            ]
        ),
        encoding="utf-8",
    )
    phase_config = tmp_path / "phase.yaml"
    phase_config.write_text("dataset:\n  benchmark: swe-bench-verified\n", encoding="utf-8")
    manifest_file = tmp_path / "manifest.json"
    manifest_file.write_text("[]", encoding="utf-8")

    with patch(
        "sys.argv",
        [
            "run_phase0.py",
            "--base-config",
            str(base_config),
            "--phase-config",
            str(phase_config),
            "--manifest",
            str(manifest_file),
            "--conditions",
            "blind_agent",
            "oracle_retrieval",
            "--agent-workers",
            "6",
            "--condition-workers",
            "2",
            "--eval-batch-size",
            "3",
            "--max-workers",
            "10",
        ],
    ):
        scripts.run_phase0.main()

    execute_kwargs = mock_execute_conditions.call_args.kwargs
    assert execute_kwargs["conditions"] == ["blind_agent", "oracle_retrieval"]
    assert execute_kwargs["agent_workers"] == 6
    assert execute_kwargs["condition_workers"] == 2
    assert execute_kwargs["eval_batch_size"] == 3

    summary = json.loads((experiments_dir / "phase0" / "summary.json").read_text(encoding="utf-8"))
    assert summary["requested_task_count"] == 2
    assert summary["conditions"]["blind_agent"]["attempted_task_ids"] == ["task_a"]
    assert summary["conditions"]["blind_agent"]["unattempted_task_ids"] == ["task_b"]
    assert summary["conditions"]["blind_agent"]["evaluation_error_task_ids"] == ["task_b"]
    assert summary["conditions"]["oracle_retrieval"]["attempted_task_count"] == 2
    assert summary["metrics"]["retrieval_ceiling_gap"] == 0.25
    assert summary["paired_comparison"]["baseline_condition"] == "blind_agent"
    assert (experiments_dir / "phase0" / "DECISION.md").is_file()


def test_build_condition_summary_uses_mutually_exclusive_status_buckets() -> None:
    import scripts.run_phase0

    manifest_tasks = [
        MagicMock(instance_id="resolved"),
        MagicMock(instance_id="evaluated_unresolved"),
        MagicMock(instance_id="agent_failed"),
        MagicMock(instance_id="missing"),
    ]
    execution = scripts.run_phase0.ConditionExecution(
        summary=ConditionRunSummary(
            phase="phase0",
            condition="blind_agent",
            run_id="run-1",
            dataset_name="princeton-nlp/SWE-bench_Verified",
            split="test",
            run_dir="run-dir",
            config_path="config.yaml",
            metrics_path="metrics.json",
            task_results_path="task_results.jsonl",
            total_tasks=3,
            metrics={"resolved_tasks": 1},
            tags={},
        ),
        task_results=[
            _task_result("resolved", "blind_agent", resolved=True, evaluation_error=None),
            _task_result(
                "evaluated_unresolved",
                "blind_agent",
                resolved=False,
                evaluation_error=None,
            ),
            _task_result(
                "agent_failed", "blind_agent", resolved=False, evaluation_error=None
            ).model_copy(
                update={
                    "attempts": [],
                    "metadata": {
                        "agent_status": "failed",
                        "agent_error": "RuntimeError: max turns",
                        "evaluation_error": None,
                        "session_log_path": "session.json",
                        "patch_path": None,
                    },
                }
            ),
        ],
        difficulty_breakdown={},
    )

    summary = scripts.run_phase0._build_condition_summary(
        manifest_tasks=manifest_tasks,
        execution=execution,
    )

    assert summary["resolved_task_ids"] == ["resolved"]
    assert summary["unresolved_evaluated_task_ids"] == ["evaluated_unresolved"]
    assert summary["agent_failed_task_ids"] == ["agent_failed"]
    assert summary["missing_task_ids"] == ["missing"]
    assert summary["not_resolved_task_ids"] == [
        "agent_failed",
        "evaluated_unresolved",
        "missing",
    ]


def test_build_condition_summary_separates_patch_failures() -> None:
    import scripts.run_phase0

    manifest_tasks = [
        MagicMock(instance_id="resolved"),
        MagicMock(instance_id="evaluated_unresolved"),
        MagicMock(instance_id="invalid_patch"),
        MagicMock(instance_id="no_patch"),
    ]
    execution = scripts.run_phase0.ConditionExecution(
        summary=ConditionRunSummary(
            phase="phase0",
            condition="blind_agent",
            run_id="run-2",
            dataset_name="princeton-nlp/SWE-bench_Verified",
            split="test",
            run_dir="run-dir",
            config_path="config.yaml",
            metrics_path="metrics.json",
            task_results_path="task_results.jsonl",
            total_tasks=4,
            metrics={"resolved_tasks": 2},
            tags={},
        ),
        task_results=[
            _task_result(
                "resolved", "blind_agent", resolved=True, evaluation_error=None
            ).model_copy(
                update={
                    "metadata": {
                        "agent_status": "success",
                        "agent_error": None,
                        "evaluation_error": None,
                        "session_log_path": "session.json",
                        "patch_path": "patch.diff",
                        "patch_attempt_count": 1,
                        "valid_patch_attempt_count": 1,
                        "invalid_patch_attempt_count": 0,
                        "repair_turn_count": 0,
                    }
                }
            ),
            _task_result(
                "evaluated_unresolved",
                "blind_agent",
                resolved=False,
                evaluation_error=None,
            ).model_copy(
                update={
                    "metadata": {
                        "agent_status": "success",
                        "agent_error": None,
                        "evaluation_error": None,
                        "session_log_path": "session.json",
                        "patch_path": "patch.diff",
                        "patch_attempt_count": 1,
                        "valid_patch_attempt_count": 1,
                        "invalid_patch_attempt_count": 0,
                        "repair_turn_count": 0,
                    }
                }
            ),
            _task_result(
                "invalid_patch", "blind_agent", resolved=False, evaluation_error="Harness error"
            ).model_copy(
                update={
                    "attempts": [],
                    "metadata": {
                        "agent_status": "failed",
                        "agent_error": "RuntimeError: patch rejected",
                        "evaluation_error": None,
                        "session_log_path": "session.json",
                        "patch_path": None,
                        "patch_attempt_count": 1,
                        "valid_patch_attempt_count": 0,
                        "invalid_patch_attempt_count": 1,
                        "repair_turn_count": 1,
                    },
                }
            ),
            _task_result(
                "no_patch", "blind_agent", resolved=False, evaluation_error="Harness error"
            ).model_copy(
                update={
                    "attempts": [],
                    "metadata": {
                        "agent_status": "failed",
                        "agent_error": "RuntimeError: max turns",
                        "evaluation_error": None,
                        "session_log_path": "session.json",
                        "patch_path": None,
                        "patch_attempt_count": 0,
                        "valid_patch_attempt_count": 0,
                        "invalid_patch_attempt_count": 0,
                        "repair_turn_count": 0,
                    },
                }
            ),
        ],
        difficulty_breakdown={},
    )

    summary = scripts.run_phase0._build_condition_summary(
        manifest_tasks=manifest_tasks,
        execution=execution,
    )

    assert summary["patch_attempted_task_ids"] == [
        "evaluated_unresolved",
        "invalid_patch",
        "resolved",
    ]
    assert summary["valid_patch_task_ids"] == ["evaluated_unresolved", "resolved"]
    assert summary["invalid_patch_task_ids"] == ["invalid_patch"]
    assert summary["no_patch_task_ids"] == ["no_patch"]


@patch("scripts.run_phase0._prewarm_condition_inputs")
@patch("scripts.run_phase0._iter_condition_outputs")
@patch("scripts.run_phase0.SWEbenchHarnessRunner")
def test_run_condition_pipelines_harness_batches(
    mock_harness_cls: Any,
    mock_iter_outputs: Any,
    mock_prewarm: Any,
    tmp_path: Path,
) -> None:
    import scripts.run_phase0

    del mock_prewarm
    run_dir = tmp_path / "run"
    run_dir.mkdir(parents=True, exist_ok=True)
    artifacts_dir = run_dir / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    (artifacts_dir / "difficulty_breakdown.json").write_text(
        json.dumps({"easy": {"pass_at_1": 1.0, "num_tasks": 3}}),
        encoding="utf-8",
    )

    logged_results: list[TaskEvaluationResult] = []
    run = MagicMock()
    run.run_dir = run_dir
    run.log_task_result.side_effect = logged_results.append
    run.finalize.return_value = ConditionRunSummary(
        phase="phase0",
        condition="blind_agent",
        run_id="run-1",
        dataset_name="princeton-nlp/SWE-bench_Verified",
        split="test",
        run_dir=str(run_dir),
        config_path=str(run_dir / "config.yaml"),
        metrics_path=str(run_dir / "metrics.json"),
        task_results_path=str(run_dir / "task_results.jsonl"),
        total_tasks=3,
        metrics={"pass_at_1": 1.0, "resolved_tasks": 3},
        tags={},
    )
    tracker = MagicMock()
    tracker.start_run.return_value = _context_manager(run)

    task_ids = ["task_a", "task_b", "task_c"]
    mock_iter_outputs.return_value = [
        _generation_output(tmp_path, "task_a"),
        _generation_output(tmp_path, "task_b"),
        _generation_output(tmp_path, "task_c"),
    ]

    harness_instance = MagicMock()
    batch_sizes: list[int] = []

    def evaluate_patches(
        *, work_dir: Path, submissions: list[HarnessPatchSubmission], run_id: str
    ) -> dict[str, SWEbenchEvaluationAttempt]:
        batch_sizes.append(len(submissions))
        return {
            submission.instance_id: SWEbenchEvaluationAttempt(
                attempt_index=1,
                model_name="test-model",
                patch_path=str(submission.patch_file),
                patch_sha256=submission.instance_id,
                run_id=run_id,
                resolved=True,
                patch_exists=True,
                patch_successfully_applied=True,
                summary_report_path=str(work_dir / "summary.json"),
                instance_report_path=str(work_dir / submission.instance_id / "report.json"),
                harness_stdout_path=str(work_dir / "stdout.log"),
                harness_stderr_path=str(work_dir / "stderr.log"),
                token_usage=TokenUsageBreakdown(),
                metadata={"evaluation_error": None, "harness_status": "reported"},
            )
            for submission in submissions
        }

    harness_instance.evaluate_patches.side_effect = evaluate_patches
    harness_instance._config.max_workers = 12
    mock_harness_cls.return_value = harness_instance

    execution = scripts.run_phase0.run_condition(
        condition="blind_agent",
        manifest_tasks=[MagicMock(instance_id=task_id) for task_id in task_ids],
        base_config={"evaluation": {"harness": {"max_workers": 12}}},
        phase_config={"dataset": {"benchmark": "swe-bench-verified"}},
        tracker=tracker,
        swebench_cache=tmp_path / "cache",
        agent_model_config=MagicMock(),
        agent_workers=8,
        eval_batch_size=2,
        run_id="phase0-run",
    )

    assert batch_sizes == [2, 1]
    assert execution.summary.metrics["pass_at_1"] == 1.0
    assert {result.instance_id for result in execution.task_results} == set(task_ids)
    assert all(result.attempts for result in execution.task_results)


@patch("scripts.run_phase0.OpenAIChatModelClient")
@patch("scripts.run_phase0.run_agent_session")
@patch("scripts.run_phase0.materialize_repo_snapshot")
@patch("scripts.run_phase0.load_swebench_task_metadata")
def test_generate_task_output_uses_collapsed_oracle_context(
    mock_load_metadata: Any,
    mock_materialize_repo_snapshot: Any,
    mock_run_agent_session: Any,
    mock_client_cls: Any,
    tmp_path: Path,
) -> None:
    import scripts.run_phase0

    repo = tmp_path / "repo"
    target_file = repo / "pkg" / "target.py"
    target_file.parent.mkdir(parents=True, exist_ok=True)
    target_file.write_text(
        "\n".join(f"line {line_number}" for line_number in range(1, 41)) + "\n",
        encoding="utf-8",
    )
    mock_materialize_repo_snapshot.return_value = repo
    mock_load_metadata.return_value = MagicMock(
        problem_statement="Fix the target function.",
        gold_patch="\n".join(
            [
                "diff --git a/pkg/target.py b/pkg/target.py",
                "--- a/pkg/target.py",
                "+++ b/pkg/target.py",
                "@@ -10,1 +10,1 @@",
                "-line 10",
                "+updated line 10",
            ]
        ),
    )
    mock_client_cls.return_value = MagicMock()
    mock_run_agent_session.return_value = AgentRunResult(
        mode="oracle",
        repo_root=str(repo),
        output_dir=str(tmp_path / "artifacts"),
        patch_path=None,
        session_log_path=str(tmp_path / "artifacts" / "session.json"),
        total_prompt_tokens=0,
        total_completion_tokens=0,
        total_wall_time_seconds=0.1,
        turns=[],
        status="failed",
        error_message="No patch",
    )

    task = MagicMock(
        instance_id="task_a",
        repo="test/repo",
        base_commit="abc123",
        difficulty="easy",
        gold_patch_files=["pkg/target.py"],
    )

    scripts.run_phase0._generate_task_output(
        task,
        condition="oracle_retrieval",
        dataset_name="princeton-nlp/SWE-bench_Verified",
        split="test",
        base_config={
            "models": {"agent": {"max_turns": 12, "context_window": 65536, "max_tokens": 8192}}
        },
        run_dir=tmp_path / "run",
        swebench_cache=tmp_path / "cache",
        agent_model_config=MagicMock(),
    )

    run_settings = mock_run_agent_session.call_args.args[0]
    assert run_settings.mode == "oracle"
    assert run_settings.oracle_context is not None
    assert "Path: pkg/target.py" in run_settings.oracle_context
    assert "Window 1" in run_settings.oracle_context


def _task_result(
    instance_id: str,
    condition: str,
    *,
    resolved: bool,
    evaluation_error: str | None,
) -> TaskEvaluationResult:
    attempts = []
    if resolved or evaluation_error is None:
        attempts = [
            SWEbenchEvaluationAttempt(
                attempt_index=1,
                model_name="test-model",
                patch_path="patch.diff",
                patch_sha256=instance_id,
                run_id="run-1",
                resolved=resolved,
                patch_exists=True,
                patch_successfully_applied=resolved,
                summary_report_path="summary.json",
                instance_report_path="instance.json",
                harness_stdout_path="stdout.log",
                harness_stderr_path="stderr.log",
                token_usage=TokenUsageBreakdown(),
                metadata={"evaluation_error": evaluation_error, "harness_status": "reported"},
            )
        ]
    return TaskEvaluationResult(
        instance_id=instance_id,
        phase="phase0",
        condition=condition,
        dataset_name="princeton-nlp/SWE-bench_Verified",
        split="test",
        resolved=resolved,
        repo="test/repo",
        base_commit="abc123",
        difficulty="easy",
        gold_patch_files=[],
        attempts=attempts,
        metadata={
            "agent_status": "success",
            "agent_error": None,
            "evaluation_error": evaluation_error,
            "session_log_path": "session.json",
            "patch_path": "patch.diff",
        },
    )


def _generation_output(tmp_path: Path, instance_id: str) -> Any:
    import scripts.run_phase0

    patch_path = tmp_path / f"{instance_id}.diff"
    patch_path.write_text(f"diff --git a/{instance_id}.py b/{instance_id}.py\n", encoding="utf-8")
    return scripts.run_phase0.TaskGenerationOutput(
        task_result=_task_result(
            instance_id,
            "blind_agent",
            resolved=False,
            evaluation_error=None,
        ).model_copy(update={"attempts": []}),
        harness_submission=HarnessPatchSubmission(
            instance_id=instance_id,
            patch_file=patch_path,
            model_name="test-model",
            token_usage=TokenUsageBreakdown(),
        ),
    )


@contextmanager
def _context_manager(value: Any) -> Any:
    yield value

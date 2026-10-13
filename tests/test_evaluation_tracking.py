from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any, cast

import mlflow

from token_sniper.evaluation.models import (
    SWEbenchEvaluationAttempt,
    TaskEvaluationResult,
    TokenUsageBreakdown,
)
from token_sniper.evaluation.tracking import ExperimentTracker

if TYPE_CHECKING:
    from pathlib import Path


def test_experiment_tracker_writes_local_artifacts_and_mlflow_metrics(tmp_path: Path) -> None:
    tracker = ExperimentTracker(
        experiments_dir=tmp_path / "experiments",
        tracking_uri=str(tmp_path / "mlruns"),
        experiment_prefix="token-sniper",
    )

    with tracker.start_run(
        phase="phase0",
        condition="blind_agent",
        dataset_name="princeton-nlp/SWE-bench_Verified",
        split="test",
        config_snapshot={"seed": 42, "runtime": {"condition": "blind_agent"}},
        run_name="unit-test-run",
        tags={"seed": "42"},
        retrieval_k_values=[1, 3],
    ) as run:
        task_result = TaskEvaluationResult(
            instance_id="sympy__sympy-20590",
            phase="phase0",
            condition="blind_agent",
            dataset_name="princeton-nlp/SWE-bench_Verified",
            split="test",
            resolved=True,
            repo="sympy/sympy",
            difficulty="<15 min fix",
            retrieval_metrics={
                "retrieval_precision_at_1": 1.0,
                "retrieval_precision_at_3": 2 / 3,
                "retrieval_recall_at_1": 0.5,
                "retrieval_recall_at_3": 1.0,
                "retrieval_map": 0.75,
            },
            attempts=[
                SWEbenchEvaluationAttempt(
                    attempt_index=1,
                    model_name="token-sniper",
                    patch_path="patch.diff",
                    patch_sha256="abc",
                    run_id="unit-test-run",
                    resolved=True,
                    patch_exists=True,
                    patch_successfully_applied=True,
                    summary_report_path="summary.json",
                    instance_report_path="report.json",
                    harness_stdout_path="stdout.log",
                    harness_stderr_path="stderr.log",
                    fail_to_pass_success=["tests/test_issue.py::test_fix"],
                    pass_to_pass_success=["tests/test_safe.py::test_safe"],
                    token_usage=TokenUsageBreakdown(
                        navigation_prompt_tokens=10,
                        navigation_completion_tokens=4,
                        reasoning_prompt_tokens=6,
                        reasoning_completion_tokens=2,
                    ),
                )
            ],
        )
        run.log_task_result(task_result)
        plot_path = tmp_path / "plot.txt"
        plot_path.write_text("plot", encoding="utf-8")
        run.log_plot(plot_path)
        summary = run.finalize()

    run_dir = tmp_path / "experiments" / "phase0" / "blind_agent" / "unit-test-run"
    assert (run_dir / "config.yaml").is_file()
    assert (run_dir / "metrics.json").is_file()
    assert (run_dir / "task_results.jsonl").is_file()
    assert (run_dir / "tasks" / "sympy__sympy-20590.json").is_file()
    assert (run_dir / "plots" / "plot.txt").is_file()
    assert summary.metrics["pass_at_1"] == 1.0

    mlflow.set_tracking_uri((tmp_path / "mlruns").resolve().as_uri())
    experiment = mlflow.get_experiment_by_name("token-sniper-phase0")
    assert experiment is not None
    runs = cast("Any", mlflow.search_runs(experiment_ids=[experiment.experiment_id]))
    assert len(runs) == 1
    first_run = runs.iloc[0]
    assert float(first_run["metrics.pass_at_1"]) == 1.0
    assert first_run["tags.condition"] == "blind_agent"

    repo_breakdown = json.loads((run_dir / "artifacts" / "repo_breakdown.json").read_text())
    assert repo_breakdown["sympy/sympy"]["pass_at_1"] == 1.0

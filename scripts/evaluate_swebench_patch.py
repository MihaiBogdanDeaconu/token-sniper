from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import yaml

from token_sniper.evaluation.config import load_evaluation_base_config
from token_sniper.evaluation.metrics import (
    compute_retrieval_metrics,
    extract_modified_files_from_patch,
    token_usage_from_session_log,
)
from token_sniper.evaluation.models import TaskEvaluationResult
from token_sniper.evaluation.swebench import SWEbenchHarnessRunner, load_swebench_task_metadata
from token_sniper.evaluation.tracking import ExperimentTracker


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate a patch against one SWE-bench task and track the run.",
    )
    parser.add_argument("--task-id", required=True, help="SWE-bench instance id to evaluate.")
    parser.add_argument(
        "--patch-file", type=Path, required=True, help="Unified diff patch to submit."
    )
    parser.add_argument("--phase", required=True, help="Phase label, e.g. phase0 or phase5.")
    parser.add_argument("--condition", required=True, help="Condition label, e.g. shell_only.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/base.yaml"),
        help="Shared project config with MLflow and harness defaults.",
    )
    parser.add_argument(
        "--phase-config",
        type=Path,
        help="Optional phase config to freeze alongside the base config.",
    )
    parser.add_argument("--run-name", help="Optional stable run id under the phase/condition tree.")
    parser.add_argument(
        "--model-name",
        default="token-sniper",
        help="Label written into the SWE-bench predictions file.",
    )
    parser.add_argument(
        "--session-log",
        type=Path,
        help="Optional agent session log to derive navigation vs reasoning token usage.",
    )
    parser.add_argument(
        "--retrieved-file",
        action="append",
        default=[],
        help="Repo-relative file retrieved for this task. Can be repeated.",
    )
    parser.add_argument(
        "--tag",
        action="append",
        default=[],
        help="Extra MLflow tag in key=value form. Can be repeated.",
    )
    args = parser.parse_args()

    base_config = load_evaluation_base_config(args.config)
    phase_config = _load_optional_yaml(args.phase_config)
    config_snapshot: dict[str, Any] = {
        "base_config_path": str(args.config.resolve()),
        "base_config": yaml.safe_load(args.config.read_text(encoding="utf-8")) or {},
        "phase_config_path": str(args.phase_config.resolve()) if args.phase_config else None,
        "phase_config": phase_config,
        "runtime": {
            "task_id": args.task_id,
            "phase": args.phase,
            "condition": args.condition,
            "model_name": args.model_name,
            "patch_file": str(args.patch_file.resolve()),
            "session_log": str(args.session_log.resolve()) if args.session_log else None,
            "retrieved_files": list(args.retrieved_file),
        },
    }

    task = load_swebench_task_metadata(
        args.task_id,
        dataset_name=base_config.evaluation.harness.dataset_name,
        split=base_config.evaluation.harness.split,
    )

    tracker = ExperimentTracker(
        experiments_dir=base_config.paths.experiments_dir,
        tracking_uri=base_config.mlflow.tracking_uri,
        experiment_prefix=base_config.mlflow.experiment_prefix,
    )
    with tracker.start_run(
        phase=args.phase,
        condition=args.condition,
        dataset_name=base_config.evaluation.harness.dataset_name,
        split=base_config.evaluation.harness.split,
        config_snapshot=config_snapshot,
        run_name=args.run_name,
        tags=_parse_tags(args.tag),
        retrieval_k_values=base_config.retrieval.top_k_values,
    ) as run:
        token_usage = (
            token_usage_from_session_log(args.session_log) if args.session_log is not None else None
        )
        harness_runner = SWEbenchHarnessRunner(base_config.evaluation.harness)
        attempt = harness_runner.evaluate_patch(
            work_dir=run.run_dir / "harness",
            instance_id=args.task_id,
            patch_file=args.patch_file,
            model_name=args.model_name,
            run_id=run.run_id,
            attempt_index=1,
            session_log_path=args.session_log,
            token_usage=token_usage,
        )

        gold_patch_files = extract_modified_files_from_patch(task.gold_patch)
        retrieved_files = list(dict.fromkeys(args.retrieved_file))
        retrieval_metrics = compute_retrieval_metrics(
            retrieved_files,
            gold_patch_files,
            k_values=base_config.retrieval.top_k_values,
        )
        task_result = TaskEvaluationResult(
            instance_id=task.instance_id,
            phase=args.phase,
            condition=args.condition,
            dataset_name=base_config.evaluation.harness.dataset_name,
            split=base_config.evaluation.harness.split,
            resolved=attempt.resolved,
            repo=task.repo,
            base_commit=task.base_commit,
            difficulty=task.difficulty,
            fail_to_pass_tests=task.fail_to_pass_tests,
            pass_to_pass_tests=task.pass_to_pass_tests,
            gold_patch_files=gold_patch_files,
            retrieved_files=retrieved_files,
            retrieval_metrics=retrieval_metrics,
            attempts=[attempt],
            metadata={
                "problem_statement_chars": len(task.problem_statement),
                "phase_config_path": str(args.phase_config.resolve())
                if args.phase_config
                else None,
            },
        )
        run.log_task_result(task_result)
        summary = run.finalize()
        print(json.dumps(summary.model_dump(mode="json"), indent=2))


def _load_optional_yaml(path: Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def _parse_tags(tag_args: list[str]) -> dict[str, str]:
    tags: dict[str, str] = {}
    for raw in tag_args:
        if "=" not in raw:
            raise ValueError(f"Invalid tag {raw!r}. Expected key=value.")
        key, value = raw.split("=", 1)
        tags[key] = value
    return tags


if __name__ == "__main__":
    main()

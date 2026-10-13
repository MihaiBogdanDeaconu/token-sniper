from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from dataclasses import dataclass
from functools import lru_cache
from typing import TYPE_CHECKING, Any

from token_sniper.evaluation.metrics import compute_patch_sha256
from token_sniper.evaluation.models import (
    SWEbenchEvaluationAttempt,
    SWEbenchTaskMetadata,
    TokenUsageBreakdown,
)

if TYPE_CHECKING:
    from pathlib import Path

    from token_sniper.evaluation.config import HarnessConfig


@dataclass(frozen=True)
class HarnessPatchSubmission:
    instance_id: str
    patch_file: Path
    model_name: str
    attempt_index: int = 1
    session_log_path: Path | None = None
    token_usage: TokenUsageBreakdown | None = None


class SWEbenchHarnessRunner:
    def __init__(self, config: HarnessConfig, *, python_executable: str | None = None) -> None:
        self._config = config
        self._python_executable = python_executable or sys.executable

    def build_command(
        self,
        *,
        predictions_path: Path,
        instance_ids: list[str],
        run_id: str,
    ) -> list[str]:
        cmd = [
            self._python_executable,
            "-m",
            "swebench.harness.run_evaluation",
            "--dataset_name",
            self._config.dataset_name,
            "--split",
            self._config.split,
            "--predictions_path",
            str(predictions_path),
            "--max_workers",
            str(self._config.max_workers),
            "--force_rebuild",
            str(self._config.force_rebuild).lower(),
            "--cache_level",
            self._config.cache_level,
            "--clean",
            str(self._config.clean).lower(),
            "--open_file_limit",
            str(self._config.open_file_limit),
            "--run_id",
            run_id,
            "--timeout",
            str(self._config.timeout_seconds),
            "--rewrite_reports",
            str(self._config.rewrite_reports).lower(),
            "--modal",
            str(self._config.modal).lower(),
            "--instance_image_tag",
            self._config.instance_image_tag,
        ]
        if not self._config.dataset_name.startswith("SWE-bench-Live"):
            cmd.extend(["--env_image_tag", self._config.env_image_tag])
        if self._config.namespace is not None:
            cmd.extend(["--namespace", self._config.namespace])
        if instance_ids:
            cmd.append("--instance_ids")
            cmd.extend(instance_ids)
        return cmd

    def evaluate_patch(
        self,
        *,
        work_dir: Path,
        instance_id: str,
        patch_file: Path,
        model_name: str,
        run_id: str,
        attempt_index: int = 1,
        session_log_path: Path | None = None,
        token_usage: TokenUsageBreakdown | None = None,
    ) -> SWEbenchEvaluationAttempt:
        return self.evaluate_patches(
            work_dir=work_dir,
            submissions=[
                HarnessPatchSubmission(
                    instance_id=instance_id,
                    patch_file=patch_file,
                    model_name=model_name,
                    attempt_index=attempt_index,
                    session_log_path=session_log_path,
                    token_usage=token_usage,
                )
            ],
            run_id=run_id,
        )[instance_id]

    def evaluate_patches(
        self,
        *,
        work_dir: Path,
        submissions: list[HarnessPatchSubmission],
        run_id: str,
    ) -> dict[str, SWEbenchEvaluationAttempt]:
        self._ensure_harness_available()
        if not submissions:
            return {}

        work_dir = work_dir.resolve()
        work_dir.mkdir(parents=True, exist_ok=True)

        model_names = {submission.model_name for submission in submissions}
        if len(model_names) != 1:
            raise ValueError("Batched harness evaluation expects a single model name.")
        model_name = submissions[0].model_name
        model_key = _sanitize_model_name(model_name)

        patch_texts: dict[str, str] = {}
        prediction_lines: list[str] = []
        instance_ids: list[str] = []
        for submission in submissions:
            patch_path = submission.patch_file.resolve()
            patch_text = patch_path.read_text(encoding="utf-8")
            patch_texts[submission.instance_id] = patch_text
            prediction_lines.append(
                json.dumps(
                    {
                        "instance_id": submission.instance_id,
                        "model_name_or_path": submission.model_name,
                        "model_patch": patch_text,
                    }
                )
            )
            instance_ids.append(submission.instance_id)

        predictions_path = work_dir / "predictions.jsonl"
        predictions_path.write_text("\n".join(prediction_lines) + "\n", encoding="utf-8")

        command = self.build_command(
            predictions_path=predictions_path,
            instance_ids=instance_ids,
            run_id=run_id,
        )
        process = subprocess.run(
            command,
            cwd=work_dir,
            capture_output=True,
            text=True,
            check=False,
        )

        stdout_path = work_dir / "harness.stdout.log"
        stderr_path = work_dir / "harness.stderr.log"
        stdout_path.write_text(process.stdout, encoding="utf-8")
        stderr_path.write_text(process.stderr, encoding="utf-8")
        if process.returncode != 0:
            raise RuntimeError(
                "SWE-bench harness failed. "
                f"See {stderr_path} for stderr and {stdout_path} for stdout."
            )

        summary_report_path = work_dir / f"{model_key}.{run_id}.json"
        if not summary_report_path.exists():
            raise FileNotFoundError(f"Expected harness summary report at {summary_report_path}")
        summary_report = json.loads(summary_report_path.read_text(encoding="utf-8"))

        attempts: dict[str, SWEbenchEvaluationAttempt] = {}
        for submission in submissions:
            instance_report_path = (
                work_dir
                / "logs"
                / "run_evaluation"
                / run_id
                / model_key
                / submission.instance_id
                / "report.json"
            )
            if instance_report_path.exists():
                attempts[submission.instance_id] = _attempt_from_instance_report(
                    submission=submission,
                    patch_text=patch_texts[submission.instance_id],
                    run_id=run_id,
                    summary_report_path=summary_report_path,
                    instance_report_path=instance_report_path,
                    stdout_path=stdout_path,
                    stderr_path=stderr_path,
                    command=command,
                )
                continue

            attempts[submission.instance_id] = _attempt_without_instance_report(
                submission=submission,
                patch_text=patch_texts[submission.instance_id],
                run_id=run_id,
                summary_report_path=summary_report_path,
                instance_report_path=instance_report_path,
                stdout_path=stdout_path,
                stderr_path=stderr_path,
                command=command,
                summary_report=summary_report,
            )
        return attempts

    @staticmethod
    def _ensure_harness_available() -> None:
        if importlib.util.find_spec("swebench") is None:
            raise RuntimeError(
                "The 'swebench' package is required for issue #17 evaluation runs. "
                "Install the project with the 'experiments' or 'research' extra."
            )


DEFAULT_DATASET_NAME = "SWE-bench-Live/SWE-bench-Live"
DEFAULT_SPLIT = "verified"


def load_swebench_task_metadata(
    instance_id: str,
    *,
    dataset_name: str = DEFAULT_DATASET_NAME,
    split: str = DEFAULT_SPLIT,
) -> SWEbenchTaskMetadata:
    dataset = _load_dataset_rows(dataset_name, split)
    try:
        row = dataset[instance_id]
    except KeyError as exc:
        raise KeyError(f"Task {instance_id!r} was not found in {dataset_name}:{split}") from exc

    return SWEbenchTaskMetadata(
        instance_id=str(row["instance_id"]),
        repo=str(row["repo"]),
        base_commit=str(row["base_commit"]),
        problem_statement=str(row.get("problem_statement") or ""),
        gold_patch=str(row.get("patch") or ""),
        test_patch=_optional_text(row.get("test_patch")),
        fail_to_pass_tests=_parse_test_list(row.get("FAIL_TO_PASS")),
        pass_to_pass_tests=_parse_test_list(row.get("PASS_TO_PASS")),
        difficulty=_optional_text(row.get("difficulty")),
        hints_text=_optional_text(row.get("hints_text")),
        version=_optional_text(row.get("version")),
        environment_setup_commit=_optional_text(row.get("environment_setup_commit")),
    )


@lru_cache(maxsize=4)
def _load_dataset_rows(dataset_name: str, split: str) -> dict[str, dict[str, Any]]:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError(
            "The 'datasets' package is required to load SWE-bench task metadata. "
            "Install the project with the 'data', 'experiments', or 'research' extra."
        ) from exc

    dataset = load_dataset(dataset_name, split=split)
    return {str(row["instance_id"]): dict(row) for row in dataset}


def _parse_test_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        try:
            loaded = json.loads(value)
        except json.JSONDecodeError:
            return [value]
        if isinstance(loaded, list):
            return [str(item) for item in loaded]
        return [str(loaded)]
    if isinstance(value, list):
        return [str(item) for item in value]
    return [str(value)]


def _optional_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value)
    return text if text else None


def _sanitize_model_name(model_name: str) -> str:
    return model_name.replace("/", "__")


def _attempt_from_instance_report(
    *,
    submission: HarnessPatchSubmission,
    patch_text: str,
    run_id: str,
    summary_report_path: Path,
    instance_report_path: Path,
    stdout_path: Path,
    stderr_path: Path,
    command: list[str],
) -> SWEbenchEvaluationAttempt:
    instance_report = json.loads(instance_report_path.read_text(encoding="utf-8"))
    report_entry = instance_report.get(submission.instance_id, {})
    tests_status = report_entry.get("tests_status", {})
    fail_to_pass = tests_status.get("FAIL_TO_PASS", {})
    pass_to_pass = tests_status.get("PASS_TO_PASS", {})

    return SWEbenchEvaluationAttempt(
        attempt_index=submission.attempt_index,
        model_name=submission.model_name,
        patch_path=str(submission.patch_file.resolve()),
        patch_sha256=compute_patch_sha256(patch_text),
        run_id=run_id,
        resolved=bool(report_entry.get("resolved", False)),
        patch_exists=bool(report_entry.get("patch_exists", False)),
        patch_successfully_applied=bool(report_entry.get("patch_successfully_applied", False)),
        summary_report_path=str(summary_report_path),
        instance_report_path=str(instance_report_path),
        harness_stdout_path=str(stdout_path),
        harness_stderr_path=str(stderr_path),
        fail_to_pass_success=list(fail_to_pass.get("success", [])),
        fail_to_pass_failure=list(fail_to_pass.get("failure", [])),
        pass_to_pass_success=list(pass_to_pass.get("success", [])),
        pass_to_pass_failure=list(pass_to_pass.get("failure", [])),
        token_usage=submission.token_usage or TokenUsageBreakdown(),
        session_log_path=str(submission.session_log_path.resolve())
        if submission.session_log_path
        else None,
        metadata={
            "submitted_instance_ids": [submission.instance_id],
            "command": command,
            "harness_status": "reported",
            "evaluation_error": None,
        },
    )


def _attempt_without_instance_report(
    *,
    submission: HarnessPatchSubmission,
    patch_text: str,
    run_id: str,
    summary_report_path: Path,
    instance_report_path: Path,
    stdout_path: Path,
    stderr_path: Path,
    command: list[str],
    summary_report: dict[str, Any],
) -> SWEbenchEvaluationAttempt:
    status = _summary_status_for_instance(summary_report, submission.instance_id)
    patch_exists = bool(patch_text.strip()) and status != "empty_patch"
    error_message = _summary_error_message(status)

    return SWEbenchEvaluationAttempt(
        attempt_index=submission.attempt_index,
        model_name=submission.model_name,
        patch_path=str(submission.patch_file.resolve()),
        patch_sha256=compute_patch_sha256(patch_text),
        run_id=run_id,
        resolved=False,
        patch_exists=patch_exists,
        patch_successfully_applied=False,
        summary_report_path=str(summary_report_path),
        instance_report_path=str(instance_report_path),
        harness_stdout_path=str(stdout_path),
        harness_stderr_path=str(stderr_path),
        token_usage=submission.token_usage or TokenUsageBreakdown(),
        session_log_path=str(submission.session_log_path.resolve())
        if submission.session_log_path
        else None,
        metadata={
            "submitted_instance_ids": [submission.instance_id],
            "command": command,
            "harness_status": status,
            "evaluation_error": error_message,
        },
    )


def _summary_status_for_instance(summary_report: dict[str, Any], instance_id: str) -> str:
    for status, key in (
        ("resolved", "resolved_ids"),
        ("unresolved", "unresolved_ids"),
        ("error", "error_ids"),
        ("incomplete", "incomplete_ids"),
        ("empty_patch", "empty_patch_ids"),
        ("completed", "completed_ids"),
    ):
        raw_ids = summary_report.get(key, [])
        if isinstance(raw_ids, list) and instance_id in raw_ids:
            return status
    return "missing_report"


def _summary_error_message(status: str) -> str:
    messages = {
        "error": "Harness reported the instance as an evaluation error.",
        "incomplete": "Harness marked the instance as incomplete.",
        "empty_patch": "Harness reported an empty patch submission.",
        "missing_report": "Harness finished without producing an instance report.",
    }
    return messages.get(status, "Harness summary did not include an instance report.")

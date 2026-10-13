from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

from token_sniper.evaluation.config import HarnessConfig
from token_sniper.evaluation.models import TokenUsageBreakdown
from token_sniper.evaluation.swebench import HarnessPatchSubmission, SWEbenchHarnessRunner

if TYPE_CHECKING:
    import pytest


def test_build_command_includes_instance_ids() -> None:
    runner = SWEbenchHarnessRunner(
        HarnessConfig(dataset_name="dataset", split="test", max_workers=2)
    )
    command = runner.build_command(
        predictions_path=Path("predictions.jsonl"),
        instance_ids=["sympy__sympy-20590"],
        run_id="run-123",
    )
    assert command[1:3] == ["-m", "swebench.harness.run_evaluation"]
    assert "--instance_ids" in command
    assert "sympy__sympy-20590" in command
    assert "--predictions_path" in command


def test_build_command_omits_env_image_tag_for_swe_bench_live() -> None:
    runner = SWEbenchHarnessRunner(
        HarnessConfig(dataset_name="SWE-bench-Live/SWE-bench-Live", split="verified")
    )
    command = runner.build_command(
        predictions_path=Path("predictions.jsonl"),
        instance_ids=["conan-io__conan-17863"],
        run_id="run-live",
    )
    assert "--env_image_tag" not in command
    assert "--split" in command
    assert "verified" in command


def test_evaluate_patch_parses_summary_and_instance_reports(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    patch_path = tmp_path / "patch.diff"
    patch_path.write_text(
        "--- a/app.py\n+++ b/app.py\n@@ -1 +1 @@\n-print('old')\n+print('new')\n",
        encoding="utf-8",
    )
    work_dir = tmp_path / "harness"
    runner = SWEbenchHarnessRunner(HarnessConfig())

    monkeypatch.setattr(
        "token_sniper.evaluation.swebench.importlib.util.find_spec",
        lambda name: object() if name == "swebench" else None,
    )

    def fake_run(
        command: list[str], cwd: Path, capture_output: bool, text: bool, check: bool
    ) -> object:
        del command, capture_output, text, check
        cwd.mkdir(parents=True, exist_ok=True)
        summary_path = cwd / "token-sniper.run-123.json"
        summary_path.write_text(
            json.dumps(
                {
                    "resolved_instances": 1,
                    "resolved_ids": ["sympy__sympy-20590"],
                    "schema_version": 2,
                }
            ),
            encoding="utf-8",
        )
        report_dir = (
            cwd / "logs" / "run_evaluation" / "run-123" / "token-sniper" / "sympy__sympy-20590"
        )
        report_dir.mkdir(parents=True, exist_ok=True)
        (report_dir / "report.json").write_text(
            json.dumps(
                {
                    "sympy__sympy-20590": {
                        "patch_exists": True,
                        "patch_successfully_applied": True,
                        "resolved": True,
                        "tests_status": {
                            "FAIL_TO_PASS": {"success": ["a"], "failure": []},
                            "PASS_TO_PASS": {"success": ["b"], "failure": []},
                        },
                    }
                }
            ),
            encoding="utf-8",
        )

        class Result:
            returncode = 0
            stdout = "ok"
            stderr = ""

        return Result()

    monkeypatch.setattr("token_sniper.evaluation.swebench.subprocess.run", fake_run)

    attempt = runner.evaluate_patch(
        work_dir=work_dir,
        instance_id="sympy__sympy-20590",
        patch_file=patch_path,
        model_name="token-sniper",
        run_id="run-123",
        token_usage=TokenUsageBreakdown(reasoning_prompt_tokens=10),
    )

    assert attempt.resolved is True
    assert attempt.patch_successfully_applied is True
    assert attempt.fail_to_pass_success == ["a"]
    assert Path(attempt.summary_report_path).is_file()
    assert Path(attempt.instance_report_path).is_file()


def test_evaluate_patches_keeps_error_instances_without_reports(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    patch_a = tmp_path / "a.diff"
    patch_b = tmp_path / "b.diff"
    patch_a.write_text(
        "--- a/app.py\n+++ b/app.py\n@@ -1 +1 @@\n-print('old')\n+print('new')\n",
        encoding="utf-8",
    )
    patch_b.write_text(
        "--- a/core.py\n+++ b/core.py\n@@ -1 +1 @@\n-print('bad')\n+print('worse')\n",
        encoding="utf-8",
    )
    work_dir = tmp_path / "harness"
    runner = SWEbenchHarnessRunner(HarnessConfig(max_workers=4))

    monkeypatch.setattr(
        "token_sniper.evaluation.swebench.importlib.util.find_spec",
        lambda name: object() if name == "swebench" else None,
    )

    def fake_run(
        command: list[str], cwd: Path, capture_output: bool, text: bool, check: bool
    ) -> object:
        del command, capture_output, text, check
        cwd.mkdir(parents=True, exist_ok=True)
        summary_path = cwd / "token-sniper.run-456.json"
        summary_path.write_text(
            json.dumps(
                {
                    "resolved_ids": ["sympy__sympy-20590"],
                    "error_ids": ["sympy__sympy-99999"],
                    "completed_ids": ["sympy__sympy-20590"],
                    "schema_version": 2,
                }
            ),
            encoding="utf-8",
        )
        report_dir = (
            cwd / "logs" / "run_evaluation" / "run-456" / "token-sniper" / "sympy__sympy-20590"
        )
        report_dir.mkdir(parents=True, exist_ok=True)
        (report_dir / "report.json").write_text(
            json.dumps(
                {
                    "sympy__sympy-20590": {
                        "patch_exists": True,
                        "patch_successfully_applied": True,
                        "resolved": True,
                        "tests_status": {
                            "FAIL_TO_PASS": {"success": ["a"], "failure": []},
                            "PASS_TO_PASS": {"success": ["b"], "failure": []},
                        },
                    }
                }
            ),
            encoding="utf-8",
        )

        class Result:
            returncode = 0
            stdout = "ok"
            stderr = ""

        return Result()

    monkeypatch.setattr("token_sniper.evaluation.swebench.subprocess.run", fake_run)

    attempts = runner.evaluate_patches(
        work_dir=work_dir,
        submissions=[
            HarnessPatchSubmission(
                instance_id="sympy__sympy-20590",
                patch_file=patch_a,
                model_name="token-sniper",
                token_usage=TokenUsageBreakdown(reasoning_prompt_tokens=10),
            ),
            HarnessPatchSubmission(
                instance_id="sympy__sympy-99999",
                patch_file=patch_b,
                model_name="token-sniper",
                token_usage=TokenUsageBreakdown(reasoning_prompt_tokens=3),
            ),
        ],
        run_id="run-456",
    )

    assert set(attempts) == {"sympy__sympy-20590", "sympy__sympy-99999"}
    assert attempts["sympy__sympy-20590"].resolved is True
    assert attempts["sympy__sympy-99999"].resolved is False
    assert attempts["sympy__sympy-99999"].patch_successfully_applied is False
    assert attempts["sympy__sympy-99999"].metadata["harness_status"] == "error"
    assert attempts["sympy__sympy-99999"].metadata["evaluation_error"] is not None

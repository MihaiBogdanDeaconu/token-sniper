from __future__ import annotations

from pathlib import Path


def test_ci_and_task_runner_files_exist() -> None:
    assert Path("Makefile").is_file()
    assert Path(".pre-commit-config.yaml").is_file()
    assert Path(".github/workflows/ci.yml").is_file()


def test_readme_mentions_ci_badge() -> None:
    readme = Path("README.md").read_text(encoding="utf-8")
    assert "actions/workflows/ci.yml" in readme

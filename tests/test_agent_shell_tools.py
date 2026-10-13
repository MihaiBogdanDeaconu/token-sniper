from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from token_sniper.agent.shell_tools import ShellToolExecutor

if TYPE_CHECKING:
    from pathlib import Path


def test_grep_and_find_return_repo_relative_paths(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    pkg = repo / "pkg"
    pkg.mkdir(parents=True)
    (pkg / "mod.py").write_text("def helper():\n    return 'token'\n", encoding="utf-8")
    (repo / "README.md").write_text("token sniper\n", encoding="utf-8")

    executor = ShellToolExecutor(repo)
    grep_result = executor.execute("grep", {"pattern": "token", "path": ".", "glob": "*.py"})
    find_result = executor.execute("find", {"path": ".", "pattern": "*.py", "kind": "files"})

    assert grep_result["matches"][0]["path"] == "pkg/mod.py"
    assert find_result["matches"] == ["pkg/mod.py"]


def test_cat_head_tail_and_sed_slice_lines(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    file_path = repo / "sample.txt"
    file_path.write_text("a\nb\nc\nd\n", encoding="utf-8")

    executor = ShellToolExecutor(repo)

    assert executor.execute("cat", {"path": "sample.txt"})["content"] == "a\nb\nc\nd"
    assert executor.execute("head", {"path": "sample.txt", "lines": 2})["content"] == "a\nb"
    assert executor.execute("tail", {"path": "sample.txt", "lines": 2})["content"] == "c\nd"
    assert (
        executor.execute("sed", {"path": "sample.txt", "start_line": 2, "end_line": 3})["content"]
        == "b\nc"
    )


def test_path_escape_is_rejected(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "inside.txt").write_text("ok\n", encoding="utf-8")
    outside = tmp_path / "outside.txt"
    outside.write_text("nope\n", encoding="utf-8")

    executor = ShellToolExecutor(repo)

    with pytest.raises(ValueError):
        executor.execute("cat", {"path": "../outside.txt"})

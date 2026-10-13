from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from token_sniper.graph.io import write_ssg_json
from token_sniper.graph.ssg import build_ssg
from token_sniper.vgib_selector.token_counts import (
    count_ssg_tokens,
    line_owners,
    read_blobs,
    read_token_sidecar,
    sidecar_path,
    write_token_sidecar,
)


class CharTokenizer:
    def __call__(self, texts: list[str], add_special_tokens: bool = False) -> dict[str, list[list[str]]]:
        return {"input_ids": [list(text) for text in texts]}


SOURCE = (
    b'"""Module docstring."""\n'
    b"import os\n"
    b"\n"
    b"@staticmethod\n"
    b"def helper(x):\n"
    b"    return x\n"
    b"\n"
    b"class Box:\n"
    b"    size = 1\n"
    b"\n"
    b"    def grow(self):\n"
    b"        return self.size + 1\n"
    b"\n"
    b"SEP = '\r'\n"
    b"MAIN = 1\n"
)


def _commit(repo: Path, files: dict[str, bytes]) -> str:
    def git(*args: str) -> str:
        return subprocess.run(["git", *args], cwd=repo, check=True, capture_output=True, text=True).stdout

    for rel, content in files.items():
        path = repo / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
    git("init", "-q")
    git("add", "-A")
    git("-c", "user.name=t", "-c", "user.email=t@t", "commit", "-q", "-m", "init")
    return git("rev-parse", "HEAD").strip()


@pytest.fixture
def repo(tmp_path: Path) -> tuple[Path, str]:
    root = tmp_path / "repo"
    root.mkdir()
    commit = _commit(root, {"pkg/box.py": SOURCE, "data/config.json": b'{"a": 1}\n'})
    return root, commit


def test_counts_partition_every_file_exactly(repo: tuple[Path, str]) -> None:
    root, commit = repo
    graph = build_ssg(root)
    counts = count_ssg_tokens(graph, root / ".git", commit, CharTokenizer())

    assert counts.keys() == set(graph.nodes)
    by_file: dict[str, int] = {}
    for node_id, n in counts.items():
        path = graph.nodes[node_id]["file_path"]
        by_file[path] = by_file.get(path, 0) + n
    assert by_file == {
        "pkg/box.py": len(SOURCE.decode()),
        "data/config.json": len('{"a": 1}\n'),
    }


def test_class_excludes_its_methods(repo: tuple[Path, str]) -> None:
    root, commit = repo
    graph = build_ssg(root)
    counts = count_ssg_tokens(graph, root / ".git", commit, CharTokenizer())

    method = next(n for n in graph.nodes if n.endswith("Box.grow"))
    klass = next(n for n in graph.nodes if n.endswith(".Box"))
    rows = SOURCE.decode().split("\n")
    m = graph.nodes[method]
    k = graph.nodes[klass]
    method_text = "\n".join(rows[m["start_line"] - 1 : m["end_line"]]) + "\n"
    class_text = "\n".join(rows[k["start_line"] - 1 : k["end_line"]]) + "\n"
    assert counts[method] == len(method_text)
    assert counts[klass] == len(class_text) - len(method_text)


def test_lone_carriage_return_does_not_start_a_row() -> None:
    owners = line_owners([("A", 2, 2)], 3, "file:x.py")
    assert owners == ["file:x.py", "A", "file:x.py"]
    rows = b"a = '\r'\nclass A: pass\n".split(b"\n")
    assert len(rows) == 3


@pytest.mark.parametrize(
    "spans",
    [
        [("a", 1, 3), ("b", 2, 5)],
        [("a", 2, 4), ("b", 2, 4)],
        [("a", 1, 6)],
    ],
)
def test_inconsistent_spans_raise(spans: list[tuple[str, int, int]]) -> None:
    with pytest.raises(ValueError):
        line_owners(spans, 5, "file:x.py")


def test_symlinked_file_counts_its_target(tmp_path: Path) -> None:
    root = tmp_path / "linked"
    root.mkdir()
    (root / "backend").mkdir()
    (root / "backend" / "notes.md").write_bytes(b"# Notes\n\nline two\n")
    (root / "notes.md").symlink_to("backend/notes.md")
    commit = _commit(root, {})

    blobs = read_blobs(root / ".git", commit, ["notes.md", "backend/notes.md"])
    assert blobs["notes.md"] == blobs["backend/notes.md"] == b"# Notes\n\nline two\n"


def test_file_missing_at_commit_raises(repo: tuple[Path, str]) -> None:
    root, commit = repo
    graph = build_ssg(root)
    graph.add_node("file:gone.py", node_type="file", file_path="gone.py", start_line=1, end_line=1)
    with pytest.raises(RuntimeError, match="gone.py"):
        count_ssg_tokens(graph, root / ".git", commit, CharTokenizer())


def test_sidecar_is_bound_to_its_ssg_and_tokenizer(repo: tuple[Path, str], tmp_path: Path) -> None:
    root, commit = repo
    graph = build_ssg(root)
    ssg_path = tmp_path / "ssg" / "ssg.json"
    ssg_path.parent.mkdir()
    write_ssg_json(graph, ssg_path)

    with pytest.raises(FileNotFoundError):
        read_token_sidecar(ssg_path)

    counts = count_ssg_tokens(graph, root / ".git", commit, CharTokenizer())
    write_token_sidecar(ssg_path, counts)
    assert read_token_sidecar(ssg_path) == counts

    sidecar = sidecar_path(ssg_path)
    sidecar.write_text(sidecar.read_text().replace('"revision": "', '"revision": "x'))
    with pytest.raises(ValueError, match="built for"):
        read_token_sidecar(ssg_path)

    write_token_sidecar(ssg_path, counts)
    ssg_path.write_text(ssg_path.read_text() + " ")
    with pytest.raises(ValueError, match="built for"):
        read_token_sidecar(ssg_path)

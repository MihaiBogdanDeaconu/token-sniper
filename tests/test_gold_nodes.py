from __future__ import annotations

import subprocess
from typing import TYPE_CHECKING

import pytest

from token_sniper.graph.ssg import build_ssg
from token_sniper.vgib_selector.gold_nodes import gold_nodes, parse_patch

if TYPE_CHECKING:
    from pathlib import Path

SOURCE = (
    "import os\n"
    "\n"
    "def top(x):\n"
    "    return x\n"
    "\n"
    "class Box:\n"
    "    size = 1\n"
    "\n"
    "    def grow(self):\n"
    "        return self.size + 1\n"
    "\n"
    "    def shrink(self):\n"
    "        return self.size - 1\n"
    "\n"
    "MAIN = 1\n"
)
HEADER = "diff --git a/pkg/box.py b/pkg/box.py\n--- a/pkg/box.py\n+++ b/pkg/box.py\n"


@pytest.fixture
def graph(tmp_path: Path):
    root = tmp_path / "repo"
    (root / "pkg").mkdir(parents=True)
    (root / "pkg" / "box.py").write_text(SOURCE)
    for args in (["init", "-q"], ["add", "-A"], ["-c", "user.name=t", "-c", "user.email=t@t", "commit", "-q", "-m", "i"]):
        subprocess.run(["git", *args], cwd=root, check=True, capture_output=True)
    return build_ssg(root)


def node(graph, suffix: str) -> str:
    return next(n for n in graph.nodes if n.endswith(suffix))


def labels(graph, patch: str) -> set[str]:
    result = gold_nodes(graph, parse_patch(patch))
    assert result.missing_files == ()
    return set(result.nodes)


def test_rewritten_line_labels_its_function(graph):
    patch = HEADER + "@@ -9,2 +9,2 @@\n     def grow(self):\n-        return self.size + 1\n+        return self.size + 2\n"
    assert labels(graph, patch) == {node(graph, "Box.grow")}


def test_rewriting_the_last_line_does_not_escalate_to_the_class(graph):
    patch = HEADER + ("@@ -12,2 +12,3 @@\n     def shrink(self):\n-        return self.size - 1\n"
                      "+        self.size -= 1\n+        return self.size\n")
    assert labels(graph, patch) == {node(graph, "Box.shrink")}


def test_method_inserted_between_methods_labels_the_class(graph):
    patch = HEADER + ("@@ -10,2 +10,5 @@\n         return self.size + 1\n+\n+    def reset(self):\n"
                      "+        self.size = 0\n \n")
    assert labels(graph, patch) == {node(graph, ".Box")}


def test_class_attribute_edit_labels_the_class(graph):
    patch = HEADER + "@@ -7,1 +7,1 @@\n-    size = 1\n+    size = 2\n"
    assert labels(graph, patch) == {node(graph, ".Box")}


def test_append_at_end_of_file_labels_the_file(graph):
    patch = HEADER + "@@ -15,0 +16,2 @@\n+def extra():\n+    pass\n"
    assert labels(graph, patch) == {"file:pkg/box.py"}


def test_new_files_add_nothing_and_unknown_files_are_reported(graph):
    patch = (
        "diff --git a/pkg/new.py b/pkg/new.py\nnew file mode 100644\n--- /dev/null\n+++ b/pkg/new.py\n"
        "@@ -0,0 +1,2 @@\n+def f():\n+    pass\n"
        + HEADER + "@@ -4,1 +4,1 @@\n-    return x\n+    return -x\n"
        + "diff --git a/docs/x.py b/docs/x.py\n--- a/docs/x.py\n+++ b/docs/x.py\n@@ -1,1 +1,1 @@\n-a\n+b\n"
    )
    result = gold_nodes(graph, parse_patch(patch))
    assert set(result.nodes) == {node(graph, ".top")}
    assert result.missing_files == ("docs/x.py",)


def test_hunk_counts_that_do_not_match_raise():
    with pytest.raises(ValueError, match="ends inside a hunk"):
        parse_patch(HEADER + "@@ -1,3 +1,3 @@\n-a\n+b\n")

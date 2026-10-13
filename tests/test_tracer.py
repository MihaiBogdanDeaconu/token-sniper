from __future__ import annotations

import functools
from pathlib import Path
from typing import Any

import networkx as nx
import pytest

from token_sniper.traces.tracer import CallEdgeTracer, TraceResult


@pytest.fixture
def project_root(tmp_path: Path) -> Path:
    pkg = tmp_path / "mypkg"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("")
    (pkg / "core.py").write_text(
        "def helper():\n    return 42\n\n"
        "def caller():\n    return helper()\n"
    )
    (pkg / "decorated.py").write_text(
        "import functools\n\n"
        "def my_decorator(fn):\n"
        "    @functools.wraps(fn)\n"
        "    def wrapper(*a, **kw):\n"
        "        return fn(*a, **kw)\n"
        "    return wrapper\n\n"
        "@my_decorator\n"
        "def decorated_func():\n"
        "    return 99\n"
    )
    return tmp_path


class TestCallEdgeTracer:
    def test_traces_direct_call(self, project_root: Path) -> None:
        import importlib
        import sys

        sys.path.insert(0, str(project_root))
        try:
            mod = importlib.import_module("mypkg.core")
            tracer = CallEdgeTracer(project_root=project_root)
            with tracer.trace():
                mod.caller()
            result = tracer.result
            assert result.call_events > 0
            assert len(result.edges) > 0
            callee_fqns = {e[1] for e in result.edges}
            assert "mypkg.core.helper" in callee_fqns
        finally:
            sys.path.remove(str(project_root))
            sys.modules.pop("mypkg.core", None)
            sys.modules.pop("mypkg", None)

    def test_drops_out_of_repo(self, project_root: Path) -> None:
        tracer = CallEdgeTracer(project_root=project_root)
        with tracer.trace():
            sorted([3, 1, 2])
        assert tracer.result.dropped_out_of_repo > 0

    def test_drops_unmappable_lambda(self, project_root: Path) -> None:
        import importlib
        import sys

        pkg = project_root / "mypkg"
        (pkg / "lam.py").write_text("fn = lambda x: x + 1\n")
        sys.path.insert(0, str(project_root))
        try:
            mod = importlib.import_module("mypkg.lam")
            tracer = CallEdgeTracer(project_root=project_root)
            with tracer.trace():
                mod.fn(5)
            assert tracer.result.dropped_unmappable > 0
        finally:
            sys.path.remove(str(project_root))
            sys.modules.pop("mypkg.lam", None)

    def test_module_qualname_outside_root(self) -> None:
        tracer = CallEdgeTracer(project_root=Path("/nonexistent/project"))
        result = tracer._module_qualname("/usr/lib/python3.12/os.py")
        assert result is None

    def test_strip_locals_wrapper_no_locals(self) -> None:
        assert CallEdgeTracer._strip_locals_wrapper("mod.Cls.method") == "mod.Cls.method"

    def test_strip_locals_wrapper_strips(self) -> None:
        result = CallEdgeTracer._strip_locals_wrapper("mod.decorator.<locals>.wrapper")
        assert result == "mod.decorator"

    def test_strip_locals_wrapper_nested(self) -> None:
        result = CallEdgeTracer._strip_locals_wrapper(
            "mod.outer.<locals>.inner.<locals>.wrapper"
        )
        assert result == "mod.outer"

    def test_strip_locals_wrapper_preserves_real_locals(self) -> None:
        result = CallEdgeTracer._strip_locals_wrapper(
            "mod.factory.<locals>.RealClass.method"
        )
        assert "RealClass" in result
        assert "method" in result


class TestTraceResult:
    def test_defaults(self) -> None:
        r = TraceResult()
        assert r.call_events == 0
        assert r.dropped_out_of_repo == 0
        assert r.dropped_unmappable == 0
        assert len(r.edges) == 0
        assert len(r.executed_nodes) == 0


class TestSSGNodeIndex:
    @pytest.fixture
    def graph(self) -> nx.MultiDiGraph[Any]:
        g: nx.MultiDiGraph[Any] = nx.MultiDiGraph()
        g.add_node("file:pkg/mod.py", node_type="file")
        g.add_node("pkg.mod.ClassA", node_type="class", file_path="pkg/mod.py")
        g.add_node("pkg.mod.ClassA.method", node_type="method", file_path="pkg/mod.py")
        g.add_node("pkg.mod.helper", node_type="function", file_path="pkg/mod.py")
        g.add_node("pkg.utils.util_func", node_type="function", file_path="pkg/utils.py")
        g.add_node("pkg.__module_preamble__", node_type="module_preamble", file_path="pkg/__init__.py")
        return g

    def test_exact_match(self, graph: nx.MultiDiGraph[Any]) -> None:
        from token_sniper.traces.ssg_mapping import build_ssg_node_index

        idx = build_ssg_node_index(graph)
        assert idx.resolve("pkg.mod.ClassA") == "pkg.mod.ClassA"

    def test_suffix_match(self, graph: nx.MultiDiGraph[Any]) -> None:
        from token_sniper.traces.ssg_mapping import build_ssg_node_index

        idx = build_ssg_node_index(graph)
        assert idx.resolve("mod.helper") == "pkg.mod.helper"

    def test_last_segment_unique(self, graph: nx.MultiDiGraph[Any]) -> None:
        from token_sniper.traces.ssg_mapping import build_ssg_node_index

        idx = build_ssg_node_index(graph)
        assert idx.resolve("util_func") == "pkg.utils.util_func"

    def test_ambiguous_returns_none(self, graph: nx.MultiDiGraph[Any]) -> None:
        from token_sniper.traces.ssg_mapping import build_ssg_node_index

        g = graph.copy()
        g.add_node("other.mod.helper", node_type="function", file_path="other/mod.py")
        idx = build_ssg_node_index(g)
        assert idx.resolve("helper") is None

    def test_excludes_file_nodes(self, graph: nx.MultiDiGraph[Any]) -> None:
        from token_sniper.traces.ssg_mapping import build_ssg_node_index

        idx = build_ssg_node_index(graph)
        assert "file:pkg/mod.py" not in idx.by_fqn

    def test_excludes_module_preamble(self, graph: nx.MultiDiGraph[Any]) -> None:
        from token_sniper.traces.ssg_mapping import build_ssg_node_index

        idx = build_ssg_node_index(graph)
        assert "pkg.__module_preamble__" not in idx.by_fqn

    def test_is_static_edge(self, graph: nx.MultiDiGraph[Any]) -> None:
        from token_sniper.traces.ssg_mapping import build_ssg_node_index

        graph.add_edge("pkg.mod.ClassA.method", "pkg.mod.helper", edge_type="calls")
        idx = build_ssg_node_index(graph)
        assert idx.is_static_edge("pkg.mod.ClassA.method", "pkg.mod.helper") is True
        assert idx.is_static_edge("pkg.mod.helper", "pkg.mod.ClassA.method") is False

    def test_locals_stripping(self, graph: nx.MultiDiGraph[Any]) -> None:
        from token_sniper.traces.ssg_mapping import build_ssg_node_index

        idx = build_ssg_node_index(graph)
        result = idx.resolve("pkg.mod.ClassA.<locals>.wrapper.method")
        assert result == "pkg.mod.ClassA.method"

    def test_file_scoped_disambiguation(self, graph: nx.MultiDiGraph[Any]) -> None:
        from token_sniper.traces.ssg_mapping import build_ssg_node_index

        g = graph.copy()
        g.add_node("other.mod.helper", node_type="function", file_path="other/mod.py")
        idx = build_ssg_node_index(g)
        assert idx.resolve("helper") is None
        assert idx.resolve("helper", hint_file="pkg/mod.py") == "pkg.mod.helper"
        assert idx.resolve("helper", hint_file="other/mod.py") == "other.mod.helper"

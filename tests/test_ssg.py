from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from token_sniper.graph import build_ssg, read_ssg_json, ssg_quality_snapshot, write_ssg_json
from token_sniper.graph.python_static_edges import extract_call_targets
from token_sniper.graph.schema import SSG_META_UNRESOLVED_KEY

FIXTURE_ROOT = Path(__file__).resolve().parent / "fixtures" / "minimal_pkg"


def _edges(g: Any, edge_type: str) -> list[tuple[str, str, dict[str, Any]]]:
    out: list[tuple[str, str, dict]] = []
    for u, v, _k, data in g.edges(keys=True, data=True):
        if data.get("edge_type") == edge_type:
            out.append((u, v, dict(data)))
    return out


def test_build_ssg_contains_and_file_nodes() -> None:
    g = build_ssg(FIXTURE_ROOT)
    assert g.has_node("file:pack/base.py")
    assert g.has_node("pack.base.Base")
    assert ("file:pack/base.py", "pack.base.Base") in {(u, v) for u, v, _ in _edges(g, "contains")}


def test_contains_nested_parent_is_longest_prefix() -> None:
    g = build_ssg(FIXTURE_ROOT)
    method = "pack.base.Base.save"
    assert g.has_node(method)
    assert ("pack.base.Base", method) in {(u, v) for u, v, _ in _edges(g, "contains")}


def test_imports_resolved_to_entity_and_file() -> None:
    g = build_ssg(FIXTURE_ROOT)
    imp = _edges(g, "imports")
    mains = [e for e in imp if e[0] == "file:pack/main.py"]
    targets = {e[1] for e in mains}
    assert "pack.base.Base" in targets
    assert "pack.utils.helper" in targets
    assert "file:pack/dup_a.py" in targets


def test_inherits_resolved_and_external_unresolved() -> None:
    g = build_ssg(FIXTURE_ROOT)
    inh = _edges(g, "inherits")
    assert ("pack.main.Child", "pack.base.Base") in {(u, v) for u, v, _ in inh}
    unresolved = g.graph.get(SSG_META_UNRESOLVED_KEY, [])
    assert any(
        x.get("edge_type") == "inherits" and x.get("target_name") == "ABC" for x in unresolved
    )


def test_calls_high_confidence_unique_helper() -> None:
    g = build_ssg(FIXTURE_ROOT)
    calls = _edges(g, "calls")
    run_method = "pack.main.Child.run"
    hel = [e for e in calls if e[0] == run_method and e[1] == "pack.utils.helper"]
    assert len(hel) == 1
    assert hel[0][2].get("confidence") == "high"


def test_calls_no_edge_from_variable_reference() -> None:
    g = build_ssg(FIXTURE_ROOT)
    calls = _edges(g, "calls")
    assert not any(e[1].endswith(".x") or e[1] == "x" for e in calls)


def test_extract_call_targets_includes_qualified_hint() -> None:
    src = b"def f():\n    dup_a.dup()\n"
    targets = extract_call_targets(src)
    assert any(t.leaf == "dup" and t.qualified_hint == "dup_a.dup" for t in targets)


def test_qualified_attribute_call_matches_suffix_not_bare_leaf() -> None:
    g = build_ssg(FIXTURE_ROOT)
    calls = _edges(g, "calls")
    run = "pack.main.Child.run"
    assert any(e[0] == run and e[1] == "pack.dup_a.dup" for e in calls)
    assert not any(e[0] == run and e[1] == "pack.dup_b.dup" for e in calls)


def test_calls_low_confidence_when_multiple_targets() -> None:
    g = build_ssg(FIXTURE_ROOT)
    calls = _edges(g, "calls")
    amb = "pack.ambiguous.ambiguous_caller"
    dups = [e for e in calls if e[0] == amb and e[2].get("confidence") == "low"]
    targets = {e[1] for e in dups}
    assert "pack.dup_a.dup" in targets
    assert "pack.dup_b.dup" in targets


def test_json_round_trip_preserves_counts() -> None:
    g = build_ssg(FIXTURE_ROOT)
    path = Path(__file__).resolve().parent / "_ssg_roundtrip_test.json"
    try:
        write_ssg_json(g, path)
        g2 = read_ssg_json(path)
        assert g2.number_of_nodes() == g.number_of_nodes()
        assert g2.number_of_edges() == g.number_of_edges()
        assert len(g2.graph.get(SSG_META_UNRESOLVED_KEY, [])) == len(
            g.graph.get(SSG_META_UNRESOLVED_KEY, [])
        )
    finally:
        if path.exists():
            path.unlink()


def test_ssg_quality_snapshot_keys() -> None:
    g = build_ssg(FIXTURE_ROOT)
    snap = ssg_quality_snapshot(g)
    for key in (
        "num_nodes",
        "num_edges",
        "node_types",
        "edge_types",
        "unresolved_edges",
        "degree",
        "out_degree",
        "in_degree",
        "num_weakly_connected_components",
    ):
        assert key in snap
    assert snap["num_nodes"] == g.number_of_nodes()


@pytest.mark.parametrize("expensive", [False, True])
def test_ssg_quality_snapshot_expensive_flag(expensive: bool) -> None:
    g = build_ssg(FIXTURE_ROOT)
    snap = ssg_quality_snapshot(g, expensive=expensive)
    if expensive:
        assert "largest_wcc_nodes" in snap
        assert "largest_wcc_diameter" in snap

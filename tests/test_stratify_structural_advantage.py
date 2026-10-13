from __future__ import annotations

import json
from typing import TYPE_CHECKING

import networkx as nx
import numpy as np
import pytest

from scripts.stratify_structural_advantage import (
    assign_strata,
    cached_instance_scores,
    entity_similarities,
    gold_node_ids,
    hop_distances,
    label_instance,
    localization_by_stratum,
    resolve_by_stratum,
    similarity_rank_fraction,
    strata_ids,
)
from token_sniper.evaluation.telemetry import AgentRunRecord

if TYPE_CHECKING:
    from pathlib import Path


def graph() -> nx.MultiDiGraph:
    g = nx.MultiDiGraph()
    g.add_node("file:pkg/mod.py", file_path="pkg/mod.py", fully_qualified_name="", module_qualname="pkg.mod")
    for name in ["seed", "a", "A", "A.__init__", "B", "far", "island"]:
        g.add_node(f"pkg.mod.{name}", file_path="pkg/mod.py", fully_qualified_name=f"pkg.mod.{name}",
                   module_qualname="pkg.mod")
    g.add_edge("pkg.mod.seed", "pkg.mod.a", edge_type="calls")
    g.add_edge("pkg.mod.a", "pkg.mod.A.__init__", edge_type="calls")
    g.add_edge("pkg.mod.B", "pkg.mod.A", edge_type="inherits")
    g.add_edge("pkg.mod.far", "pkg.mod.seed", edge_type="contains")
    g.add_edge("pkg.mod.far", "pkg.mod.a", edge_type="imports")
    return g


def test_gold_node_ids_maps_a_class_key_to_the_class_and_its_init():
    nodes = gold_node_ids(graph(), ["pkg/mod.py:A", "pkg/mod.py:a", "pkg/mod.py:gone"])
    assert sorted(nodes["pkg/mod.py:A"]) == ["pkg.mod.A", "pkg.mod.A.__init__"]
    assert nodes["pkg/mod.py:a"] == ["pkg.mod.a"]
    assert nodes["pkg/mod.py:gone"] == []


def test_hop_distances_use_only_calls_and_inherits_undirected():
    distance = hop_distances(graph(), ["pkg.mod.A.__init__"])
    assert distance["pkg.mod.a"] == 1
    assert distance["pkg.mod.seed"] == 2
    assert "pkg.mod.far" not in distance
    assert "pkg.mod.A" not in distance
    assert hop_distances(graph(), ["pkg.mod.A"]) == {"pkg.mod.A": 0, "pkg.mod.B": 1}


def test_entity_similarities_take_the_best_chunk_cosine():
    sims = entity_similarities(np.array([1.0, 0.0]), {
        "x": np.array([[2.0, 0.0]]),
        "y": np.array([[0.0, 1.0], [1.0, 1.0]]),
    })
    assert sims["x"] == pytest.approx(1.0)
    assert sims["y"] == pytest.approx(1 / np.sqrt(2))


def test_similarity_rank_fraction_counts_strictly_more_similar_entities():
    sims = {"a": 0.9, "b": 0.5, "c": 0.5, "d": 0.1}
    assert similarity_rank_fraction(sims, 0.9) == 0.0
    assert similarity_rank_fraction(sims, 0.5) == 0.25
    assert similarity_rank_fraction(sims, 0.1) == 0.75


def sims_for(g: nx.MultiDiGraph, **overrides: float) -> dict[str, float]:
    sims = {nid: 0.5 for nid in g.nodes if not nid.startswith("file:")}
    sims.update({f"pkg.mod.{k}": v for k, v in overrides.items()})
    return sims


def test_label_instance_takes_the_nearest_and_most_similar_gold_node():
    g = graph()
    label = label_instance(graph=g, gold_entities=["pkg/mod.py:A"], seeds=["pkg.mod.seed"],
                           sims=sims_for(g, A=0.1, seed=0.9, a=0.8, **{"A.__init__": 0.3}))
    assert label["gold_nodes"] == ["pkg.mod.A", "pkg.mod.A.__init__"]
    assert label["min_hops"] == 2
    assert label["far"] is True
    assert label["gold_similarity"] == 0.3
    assert label["similarity_rank_fraction"] == pytest.approx(5 / 7)


def test_label_instance_near_and_unreachable_gold():
    g = graph()
    near = label_instance(graph=g, gold_entities=["pkg/mod.py:a"], seeds=["pkg.mod.seed"], sims=sims_for(g))
    assert (near["min_hops"], near["far"]) == (1, False)
    island = label_instance(graph=g, gold_entities=["pkg/mod.py:island"], seeds=["pkg.mod.seed"], sims=sims_for(g))
    assert (island["min_hops"], island["far"]) == (None, True)
    seed = label_instance(graph=g, gold_entities=["pkg/mod.py:seed"], seeds=["pkg.mod.seed"], sims=sims_for(g))
    assert (seed["min_hops"], seed["far"]) == (0, False)


def test_label_instance_reports_why_it_cannot_label():
    g = graph()
    sims = sims_for(g)
    assert "no gold" in label_instance(graph=g, gold_entities=[], seeds=["pkg.mod.seed"], sims=sims)["unresolved"]
    assert "no seed" in label_instance(graph=g, gold_entities=["pkg/mod.py:a"], seeds=[], sims=sims)["unresolved"]
    missing = label_instance(graph=g, gold_entities=["pkg/mod.py:a", "pkg/mod.py:gone"], seeds=["pkg.mod.seed"],
                             sims=sims)
    assert missing == {"unresolved": "gold entities without an SSG node: ['pkg/mod.py:gone']"}
    del sims["pkg.mod.a"]
    unembedded = label_instance(graph=g, gold_entities=["pkg/mod.py:a"], seeds=["pkg.mod.seed"], sims=sims)
    assert unembedded == {"unresolved": "gold entities without a ChromaDB embedding: ['pkg/mod.py:a']"}


def test_label_instance_needs_one_embedded_node_per_gold_key():
    g = graph()
    sims = sims_for(g, **{"A.__init__": 0.2})
    del sims["pkg.mod.A"]
    label = label_instance(graph=g, gold_entities=["pkg/mod.py:A"], seeds=["pkg.mod.seed"], sims=sims)
    assert label["gold_similarity"] == 0.2
    assert label["min_hops"] == 2


def resolved_label(iid: str, far: bool, rank: float) -> dict:
    return {"instance_id": iid, "far": far, "similarity_rank_fraction": rank}


def test_assign_strata_uses_the_bottom_similarity_tercile_of_labeled_instances():
    labels = [
        resolved_label("i0", True, 0.0),
        resolved_label("i1", True, 0.1),
        resolved_label("i2", False, 0.2),
        resolved_label("i3", True, 0.3),
        resolved_label("i4", False, 0.4),
        resolved_label("i5", True, 0.5),
        {"instance_id": "i6", "unresolved": "no SSG"},
    ]
    threshold = assign_strata(labels)
    assert threshold == pytest.approx(1 / 3)
    assert [label["low_similarity"] for label in labels[:6]] == [False, False, False, False, True, True]
    assert strata_ids(labels) == {"structural": ["i5"], "rest": ["i0", "i1", "i2", "i3", "i4"]}
    assert labels[6]["stratum"] == "unresolved"


def test_assign_strata_raises_when_nothing_is_labeled():
    with pytest.raises(ValueError, match="no instance"):
        assign_strata([{"instance_id": "i0", "unresolved": "no SSG"}])


def write_predictions(path: Path, ids: list[str]) -> Path:
    path.write_text("\n".join(json.dumps({"instance_id": i, "found_files": []}) for i in ids) + "\n")
    return path


def test_localization_by_stratum_scores_each_stratum_with_acc_cis(tmp_path: Path):
    preds = write_predictions(tmp_path / "p.jsonl", ["i0", "i1", "i2"])
    hit = {"i0": 1.0, "i1": 0.0, "i2": 1.0, "i9": 0.0}
    calls = []

    def score(path: Path, members: list[str]) -> dict:
        calls.append((path, members))
        rate = round(sum(hit[i] for i in members) / len(members), 4)
        return {"function": {"Acc@5": rate, "NDCG@5": 0.3}}

    def instance_scores(path: Path, members: list[str]) -> dict:
        return {i: {"function": {"Acc@5": hit[i], "NDCG@5": 0.3}} for i in members}

    out = localization_by_stratum({"structural": ["i0", "i1", "i9"], "rest": ["i2"]}, {"ts": preds},
                                  score, instance_scores)
    structural = out["ts"]["structural"]
    assert structural["instances"] == 3
    assert structural["without_prediction"] == 1
    assert structural["scores"] == {"function": {"Acc@5": 0.3333, "NDCG@5": 0.3}}
    acc = structural["acc"]["function"]
    assert list(structural["acc"]["function"]) == ["Acc@5"]
    assert acc["Acc@5"]["rate"] == pytest.approx(1 / 3)
    assert acc["Acc@5"]["hits"] == 1
    assert 0.0 <= acc["Acc@5"]["bootstrap_ci_95"][0] <= 1 / 3 <= acc["Acc@5"]["bootstrap_ci_95"][1] <= 1.0
    assert out["ts"]["rest"]["acc"]["function"]["Acc@5"] == {"rate": 1.0, "hits": 1, "bootstrap_ci_95": (1.0, 1.0)}
    assert calls == [(preds, ["i0", "i1", "i9"]), (preds, ["i2"])]

    with pytest.raises(ValueError, match="empty"):
        localization_by_stratum({"structural": [], "rest": ["i1"]}, {"ts": preds}, score, instance_scores)


def test_localization_by_stratum_checks_hits_against_the_aggregate(tmp_path: Path):
    preds = write_predictions(tmp_path / "p.jsonl", ["i0", "i1"])
    ids = {"structural": ["i0"], "rest": ["i1"]}

    def instance_scores(path: Path, members: list[str]) -> dict:
        return {i: {"file": {"Acc@1": 1.0}} for i in members}

    with pytest.raises(ValueError, match="!= LocAgent aggregate"):
        localization_by_stratum(ids, {"ts": preds}, lambda p, m: {"file": {"Acc@1": 0.0}}, instance_scores)
    with pytest.raises(ValueError, match="not 0/1"):
        localization_by_stratum(ids, {"ts": preds}, lambda p, m: {"file": {"Acc@1": 0.5}},
                                lambda p, m: {i: {"file": {"Acc@1": 0.5}} for i in m})


def test_cached_instance_scores_scores_each_instance_once_per_file_content(tmp_path: Path):
    preds = write_predictions(tmp_path / "p.jsonl", ["i0", "i1"])
    cache = tmp_path / "cache"
    scored = []

    def score_one(path: Path, iid: str) -> dict:
        scored.append(iid)
        return {"file": {"Acc@1": 1.0 if iid == "i0" else 0.0}}

    kwargs = {"cache_dir": cache, "header": {"dataset_key": "lite_274"}, "score_one": score_one, "workers": 2}
    first = cached_instance_scores(preds, ["i0", "i1"], **kwargs)
    assert first == {"i0": {"file": {"Acc@1": 1.0}}, "i1": {"file": {"Acc@1": 0.0}}}
    assert sorted(scored) == ["i0", "i1"]

    assert cached_instance_scores(preds, ["i1", "i0"], **kwargs) == first
    cached_instance_scores(preds, ["i0", "i2"], **kwargs)
    assert sorted(scored) == ["i0", "i1", "i2"]

    write_predictions(preds, ["i0"])
    cached_instance_scores(preds, ["i0"], **kwargs)
    assert sorted(scored) == ["i0", "i0", "i1", "i2"]
    assert len(list(cache.iterdir())) == 2

    with pytest.raises(ValueError, match="was built from"):
        cached_instance_scores(preds, ["i0"], **{**kwargs, "header": {"dataset_key": "locbench_560"}})


def run(iid: str, arm: str, seed: int, resolved: bool) -> AgentRunRecord:
    return AgentRunRecord(
        instance_id=iid, dataset="swebench_verified", arm=arm, seed=seed, model="m", git_ref="g",
        selector_checkpoint_sha256=None, beta=None, retrieved_files=[], retrieved_spans=[], retained_span_count=0,
        fallback_triggered=False, gold_files=[], file_recall=0.0, function_recall=0.0, selector_latency_ms=None,
        first_turn_context_tokens=0, prompt_tokens=0, completion_tokens=0, cached_tokens=0, turns=0, tool_calls=0,
        wall_seconds=0.0, usd_cost=None, applied=True, resolved=resolved, error_class=None,
    )


def test_resolve_by_stratum_per_seed_mean_std_and_ci():
    ids = {"structural": ["s0", "s1"], "rest": ["r0"]}
    records = [
        run("s0", "ts", 0, True), run("s1", "ts", 0, False),
        run("s0", "ts", 1, True), run("s1", "ts", 1, True),
        run("r0", "ts", 0, True),
        run("s0", "shell", 0, False),
        run("other", "ts", 0, True),
    ]
    out = resolve_by_stratum(ids, records)
    assert out["records_outside_labeled_instances"] == 1

    ts = out["arms"]["ts"]["structural"]
    assert ts["instances"] == 2
    assert ts["per_seed"] == {0: {"n": 2, "resolved": 1, "rate": 0.5}, 1: {"n": 2, "resolved": 2, "rate": 1.0}}
    assert ts["mean_over_seeds"] == 0.75
    assert ts["std_over_seeds"] == pytest.approx(np.std([0.5, 1.0], ddof=1))
    assert 0.5 <= ts["bootstrap_ci_95"][0] <= ts["bootstrap_ci_95"][1] <= 1.0

    assert out["arms"]["ts"]["rest"]["std_over_seeds"] is None
    assert out["arms"]["shell"]["rest"] == {"instances": 0}
    assert out["arms"]["shell"]["structural"]["mean_over_seeds"] == 0.0


def test_resolve_by_stratum_rejects_duplicate_runs():
    with pytest.raises(ValueError, match="duplicate"):
        resolve_by_stratum({"structural": ["s0"], "rest": []}, [run("s0", "ts", 0, True), run("s0", "ts", 0, False)])

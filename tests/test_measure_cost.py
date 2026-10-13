from __future__ import annotations

import json
import sys
from dataclasses import asdict
from typing import TYPE_CHECKING

import numpy as np
import pytest
import torch
import xgboost as xgb

from scripts import eval_candidates, measure_cost, setup_phase1
from scripts.eval_retention import retention_header
from token_sniper.evaluation.eval_stages import (
    candidates_path,
    retention_path,
    sha256_file,
    write_record,
)
from token_sniper.vgib_selector.backbone import GraphSAGEConfig
from token_sniper.vgib_selector.selector import VGIBSelector

if TYPE_CHECKING:
    from pathlib import Path

COMMIT_A = "a" * 40
COMMIT_B = "b" * 40
TASKS = [
    ("o__r-1", "o/r", COMMIT_A),
    ("o__r-2", "o/r", COMMIT_A),
    ("o__r-3", "o/r", COMMIT_B),
    ("o__r-4", "o/r", COMMIT_B),
]
KEYS = {"selector_params", "reranker_trees", "latency_ms_per_instance", "amortized_build_seconds_per_instance",
        "gpu_hour_usd", "cpu_hour_usd", "usd_per_instance", "instances", "source"}


@pytest.fixture
def env(tmp_path: Path) -> dict[str, Path]:
    manifest = tmp_path / "tasks.json"
    manifest.write_text(json.dumps([
        {"instance_id": iid, "repo": repo, "base_commit": commit, "gold_patch_file_count": 1, "bucket": "single_file"}
        for iid, repo, commit in TASKS
    ]))

    cfg = GraphSAGEConfig(num_files=4, source_emb_dim=8, file_emb_dim=4, hidden_dim=8, out_dim=8, task_emb_dim=8)
    checkpoint = tmp_path / "vgib.pt"
    torch.save({"model_cfg": asdict(cfg), "model_state": VGIBSelector(cfg).state_dict()}, checkpoint)

    reranker = tmp_path / "xgb.json"
    rng = np.random.default_rng(0)
    xgb.train({"objective": "binary:logistic"}, xgb.DMatrix(rng.random((20, 3)), label=rng.integers(0, 2, 20)),
              num_boost_round=5).save_model(str(reranker))

    cache = tmp_path / "cache"
    for (iid, _, _), latency in zip(TASKS, [10.0, 20.0, 30.0, None], strict=True):
        cpath = candidates_path(cache, iid)
        if latency is None:
            write_record(cpath, {}, {"excluded": "no SSG"})
            continue
        write_record(cpath, {}, {
            "candidate_files": ["a.py"], "stage_a_latency_ms": 10 * latency, "stage_a_device": "cuda",
            "stage_a_components_ms": {"dense_retrieval": 5 * latency, "bm25_retrieval": 3 * latency,
                                      "ssg_load_entry_points_features": 2 * latency},
        })
        header = retention_header(sha256_file(checkpoint), sha256_file(reranker), sha256_file(cpath), 20)
        write_record(retention_path(cache, sha256_file(checkpoint), iid), header,
                     {"ranked_files": [], "pool_tokens": 0, "nodes": [], "selector_latency_ms": latency,
                      "vgib_input_ms": 5.0, "selector_device": "cuda"})

    ssg_dir = tmp_path / "ssg"
    for commit, ssg_s, index_s in [(COMMIT_A, 100.0, 50.0), (COMMIT_B, 200.0, 250.0)]:
        commit_dir = ssg_dir / "o__r" / commit[:12]
        setup_phase1._write_timing(commit_dir / setup_phase1.SSG_TIMING,
                                   {"repo": "o/r", "commit": commit, "seconds": ssg_s})
        setup_phase1._write_timing(commit_dir / setup_phase1.INDEX_TIMING,
                                   {"repo": "o/r", "commit": commit, "seconds": index_s,
                                    "provider": "sentence-transformers"})
    return {"manifest": manifest, "checkpoint": checkpoint, "reranker": reranker, "cache": cache, "ssg": ssg_dir,
            "output": tmp_path / "out" / "cost.json"}


def _run(monkeypatch, env: dict[str, Path], *extra: str) -> dict:
    monkeypatch.setattr(sys, "argv", [
        "measure_cost.py", "--manifest", str(env["manifest"]), "--cache", str(env["cache"]),
        "--reranker-model", str(env["reranker"]), "--checkpoint", str(env["checkpoint"]),
        "--ssg-dir", str(env["ssg"]), "--output", str(env["output"]), *extra,
    ])
    measure_cost.main()
    return json.loads(env["output"].read_text())


def test_cost_json(monkeypatch, env):
    cost = _run(monkeypatch, env, "--gpu-hour-usd", "3.6", "--cpu-hour-usd", "0.36")
    assert set(cost) == KEYS
    cfg = GraphSAGEConfig(num_files=4, source_emb_dim=8, file_emb_dim=4, hidden_dim=8, out_dim=8, task_emb_dim=8)
    assert cost["selector_params"] == sum(p.numel() for p in VGIBSelector(cfg).parameters())
    assert cost["reranker_trees"] == 5
    assert cost["instances"] == 3
    assert cost["latency_ms_per_instance"] == {"mean": 225.0, "p50": 225.0, "p95": pytest.approx(324.0)}
    assert cost["amortized_build_seconds_per_instance"] == pytest.approx(600.0 / 3)
    assert (cost["gpu_hour_usd"], cost["cpu_hour_usd"]) == (3.6, 0.36)
    assert cost["usd_per_instance"] == pytest.approx(225.0 / 3.6e6 * 3.6 + 600 * 3.6 / 3600 / 3)
    for part in ("dense_retrieval 100.0", "bm25_retrieval 60.0", "ssg_load_entry_points_features 40.0",
                 "VGIB input", "all at the GPU rate", "bare repo clones", "token-count sidecars",
                 "VGIB feature caches", "failed builds"):
        assert part in cost["source"]


def test_prices_are_required(monkeypatch, env):
    with pytest.raises(SystemExit):
        _run(monkeypatch, env, "--gpu-hour-usd", "3.6")


@pytest.mark.parametrize("field", ["selector_latency_ms", "vgib_input_ms"])
def test_stage_b_record_without_timing_raises(monkeypatch, env, field):
    ckpt_sha = sha256_file(env["checkpoint"])
    path = retention_path(env["cache"], ckpt_sha, "o__r-2")
    record = json.loads(path.read_text())
    del record[field]
    path.write_text(json.dumps(record))
    with pytest.raises(ValueError, match="1 instances .* without timings"):
        _run(monkeypatch, env, "--gpu-hour-usd", "1", "--cpu-hour-usd", "1")


def test_stage_a_record_without_timing_raises(env):
    tasks = measure_cost.load_phase0_manifest(env["manifest"]).tasks
    reranker_sha, ckpt_sha = sha256_file(env["reranker"]), sha256_file(env["checkpoint"])
    cpath = candidates_path(env["cache"], "o__r-1")
    write_record(cpath, {}, {"candidate_files": ["a.py"]})
    header = retention_header(ckpt_sha, reranker_sha, sha256_file(cpath), 20)
    write_record(retention_path(env["cache"], ckpt_sha, "o__r-1"), header,
                 {"selector_latency_ms": 1.0, "vgib_input_ms": 1.0, "selector_device": "cuda"})
    with pytest.raises(ValueError, match="1 instances .* without timings"):
        measure_cost.query_latencies(env["cache"], tasks, ckpt_sha, reranker_sha, 20)


def test_missing_build_timing_raises(monkeypatch, env):
    (env["ssg"] / "o__r" / COMMIT_B[:12] / setup_phase1.INDEX_TIMING).unlink()
    with pytest.raises(FileNotFoundError, match="1 build timings missing"):
        _run(monkeypatch, env, "--gpu-hour-usd", "1", "--cpu-hour-usd", "1")


def test_api_embedding_provider_raises(env):
    path = env["ssg"] / "o__r" / COMMIT_A[:12] / setup_phase1.INDEX_TIMING
    path.write_text(json.dumps({"repo": "o/r", "commit": COMMIT_A, "seconds": 1.0, "provider": "voyageai"}))
    tasks = measure_cost.load_phase0_manifest(env["manifest"]).tasks
    with pytest.raises(ValueError, match="voyageai is an API"):
        measure_cost.build_seconds(env["ssg"], tasks)


def test_mixed_devices_raise():
    with pytest.raises(ValueError, match="one device per measurement"):
        measure_cost.cost_summary(
            selector_params=1, reranker_trees=1, latency_rows=[_row(1.0)], devices={"cpu", "cuda"},
            ssg_seconds=0.0, index_seconds=0.0, commits=1, gpu_hour_usd=1.0, cpu_hour_usd=1.0, source="x",
        )


def test_cpu_selector_uses_cpu_rate():
    cost = measure_cost.cost_summary(
        selector_params=1, reranker_trees=1, latency_rows=[_row(3600.0)], devices={"cpu"},
        ssg_seconds=1800.0, index_seconds=1800.0, commits=1, gpu_hour_usd=100.0, cpu_hour_usd=1.0, source="x",
    )
    assert cost["usd_per_instance"] == pytest.approx(1.0 / 1000 + 100.0)


def _row(selector_ms: float) -> dict[str, float]:
    return {"dense_retrieval": 0.0, "stage_a": 0.0, "vgib_input": 0.0, "selector": selector_ms}


def test_timed_retriever_accumulates_and_reraises():
    class Retriever:
        def retrieve(self, **kwargs):
            if kwargs["fail"]:
                raise RuntimeError("bm25 down")
            return "ok"

    timed = eval_candidates._TimedRetriever(Retriever())
    assert timed.retrieve(fail=False) == "ok"
    with pytest.raises(RuntimeError, match="bm25 down"):
        timed.retrieve(fail=True)
    assert timed.ms > 0


def test_ssg_build_writes_timing(tmp_path, monkeypatch):
    repo_root = tmp_path / "snapshot"
    repo_root.mkdir()
    (repo_root / "m.py").write_text("def f():\n    return 1\n")
    monkeypatch.setattr("token_sniper.evaluation.phase0.materialize_repo_snapshot", lambda **_: repo_root)
    ssg_dir = tmp_path / "ssg"
    _, _, ok, msg = setup_phase1._build_one_ssg(("o/r", COMMIT_A, tmp_path / "repos", ssg_dir))
    assert ok, msg
    timing = json.loads((ssg_dir / "o__r" / COMMIT_A[:12] / setup_phase1.SSG_TIMING).read_text())
    assert (timing["repo"], timing["commit"]) == ("o/r", COMMIT_A)
    assert timing["seconds"] >= timing["build_ssg_seconds"] > 0
    assert timing["nodes"] > 0

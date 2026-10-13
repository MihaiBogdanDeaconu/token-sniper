from __future__ import annotations

import json
import os
import shlex
import subprocess
import sys
from dataclasses import replace
from typing import TYPE_CHECKING

import pytest
import torch
import yaml

from scripts.collect_beta_sweep import MISSING, collect, main, resolve_rate
from scripts.submit_beta_sweep import REPO_ROOT, beta_tag, check_submit_args, job_name, submit_sweep
from token_sniper.evaluation.eval_stages import sha256_file
from token_sniper.evaluation.telemetry import AgentRunRecord, append_record

if TYPE_CHECKING:
    from pathlib import Path

SUBMIT_ARGS = ["--kubeconfig", "kube.yaml", "--workspace-pvc", "ws-pvc", "--epochs", "50", "--new-model-flag", "x"]


def test_beta_tag_and_job_names():
    assert [beta_tag(b) for b in (0.001, 0.03, 0.3, 1.0, 0.0)] == ["0p001", "0p03", "0p3", "1", "0"]
    assert job_name("vgib-beta", "20260918120000", 0.001) == "vgib-beta-20260918120000-b0p001"
    with pytest.raises(ValueError, match="63"):
        job_name("x" * 50, "20260918120000", 0.1)


@pytest.mark.parametrize("flag", ["--beta", "--job-name", "--wait", "--beta=0.1"])
def test_owned_flags_rejected(flag):
    with pytest.raises(ValueError, match=flag.split("=")[0]):
        check_submit_args([*SUBMIT_ARGS, flag])


def test_workspace_pvc_required():
    with pytest.raises(SystemExit):
        check_submit_args(["--kubeconfig", "kube.yaml"])


def test_submit_sweep_passes_flags_through_and_writes_manifest(tmp_path: Path):
    calls = []
    out = tmp_path / "sweep.json"
    manifest = submit_sweep(
        [0.01, 0.1], SUBMIT_ARGS, name_prefix="vgib-beta", stamp="20260918120000", dataset="lite_274",
        manifest_out=out, dry_run=False, git_ref="abc123", run=lambda cmd, **kw: calls.append(cmd),
    )
    assert len(calls) == 2
    first = calls[0]
    assert first[2:2 + len(SUBMIT_ARGS)] == SUBMIT_ARGS
    assert first[-6:] == ["--git-ref", "abc123", "--job-name", "vgib-beta-20260918120000-b0p01", "--beta", "0.01"]
    assert "--dry-run" not in first
    assert json.loads(out.read_text()) == manifest
    run = manifest["runs"][1]
    assert run == {
        "beta": 0.1,
        "job_name": "vgib-beta-20260918120000-b0p1",
        "checkpoint_dir": "/workspace/runs/vgib-beta-20260918120000-b0p1/repo/experiments/vgib",
        "models_dir": "/models/runs/vgib-beta-20260918120000-b0p1/repo/experiments",
        "eval_job_name": "eval-vgib-beta-20260918120000-b0p1",
        "eval_command": "./venv/bin/python scripts/submit_eval_retrieval_k8s.py --kubeconfig kube.yaml "
                        "--models-pvc ws-pvc --job-name eval-vgib-beta-20260918120000-b0p1 --datasets lite_274 "
                        "--models-dir /models/runs/vgib-beta-20260918120000-b0p1/repo/experiments "
                        "--checkpoints vgib",
    }
    assert manifest["git_ref"] == "abc123"
    assert manifest["workspace_pvc"] == "ws-pvc"


def test_submit_sweep_keeps_explicit_git_ref_and_records_partial_sweep(tmp_path: Path):
    out = tmp_path / "sweep.json"

    def run(cmd, **_kw):
        if cmd[-1] == "0.1":
            raise RuntimeError("kubectl failed")

    with pytest.raises(RuntimeError):
        submit_sweep([0.01, 0.1], [*SUBMIT_ARGS, "--git-ref", "pinned"], name_prefix="p", stamp="1",
                     dataset="lite_274", manifest_out=out, dry_run=False, git_ref="head", run=run)
    written = json.loads(out.read_text())
    assert written["git_ref"] == "pinned"
    assert [r["beta"] for r in written["runs"]] == [0.01]


def test_submit_sweep_rejects_duplicate_betas():
    with pytest.raises(ValueError, match="duplicate"):
        submit_sweep([0.1, 0.1], SUBMIT_ARGS, name_prefix="p", stamp="1", dataset="lite_274",
                     manifest_out=None, dry_run=True, git_ref="h", run=lambda *a, **k: None)


def test_dry_run_renders_real_training_jobs(tmp_path: Path, capfd):
    args = ["--kubeconfig", str(_kubeconfig(tmp_path)), "--workspace-pvc", "ws-pvc", "--repo-url", "https://example/r.git"]
    submit_sweep([0.003], args, name_prefix="vgib-beta", stamp="20260918120000", dataset="lite_274",
                 manifest_out=None, dry_run=True, git_ref="abc123")
    rendered = capfd.readouterr().out
    job = yaml.safe_load(rendered[rendered.index("apiVersion"):])
    assert job["metadata"]["name"] == "vgib-beta-20260918120000-b0p003"
    script = job["spec"]["template"]["spec"]["containers"][0]["command"][-1]
    assert "--beta 0.003" in script
    assert "REPO=/workspace/runs/vgib-beta-20260918120000-b0p003/repo" in script


def _kubeconfig(tmp_path: Path) -> Path:
    path = tmp_path / "kube.yaml"
    path.write_text(yaml.safe_dump({"current-context": "c", "contexts": [
        {"name": "c", "context": {"namespace": "ns"}}]}))
    return path


@pytest.mark.parametrize(("extra", "expected"), [([], ["vgib"]), (["--run-rl"], ["vgib", "grpo"])])
def test_eval_command_scores_grpo_only_with_run_rl(tmp_path: Path, extra, expected):
    args = ["--kubeconfig", str(_kubeconfig(tmp_path)), "--workspace-pvc", "ws-pvc", *extra]
    manifest = submit_sweep([0.1], args, name_prefix="vgib-beta", stamp="20260918120000", dataset="lite_274",
                            manifest_out=None, dry_run=False, git_ref="abc123", run=lambda *a, **k: None)
    command = shlex.split(manifest["runs"][0]["eval_command"])
    assert command[command.index("--checkpoints") + 1:] == expected

    rendered = subprocess.run(
        [sys.executable, *command[1:], "--dry-run", "--repo-url", "https://example/r.git", "--git-ref", "abc123"],
        cwd=REPO_ROOT, env={**os.environ, "PYTHONPATH": str(REPO_ROOT)},
        check=True, capture_output=True, text=True,
    ).stdout
    job = yaml.safe_load(rendered)
    assert job["metadata"]["name"] == "eval-vgib-beta-20260918120000-b0p1"
    script = job["spec"]["template"]["spec"]["containers"][0]["command"][-1]
    assert "cp /models/runs/vgib-beta-20260918120000-b0p1/repo/experiments/vgib/best.pt" in script
    assert ("rl_grpo/best_rl.pt" in script) == ("grpo" in expected)
    assert ("--checkpoint grpo=" in script) == ("grpo" in expected)


def _stage_c(root: Path, run: dict, beta_in_ckpt: float, *, reranker: str = "r") -> str:
    eval_dir = root / run["eval_job_name"]
    (eval_dir / "models").mkdir(parents=True)
    ckpt = eval_dir / "models" / "vgib_best.pt"
    torch.save({"beta": beta_in_ckpt, "epoch": 7, "val_token_ratio": 0.2, "best_val_func_at_10": 0.6,
                "best_val_gold_percentile": 0.9, "best_val_loss": -0.5}, ckpt)
    sha = sha256_file(ckpt)
    stage_c = eval_dir / "lite_274"
    stage_c.mkdir()
    (stage_c / "inputs.json").write_text(json.dumps({
        "manifest_instances": 274, "top_k": 20, "reranker_sha256": reranker, "cache": "/eval/cache/lite_274",
        "checkpoints_sha256": {"vgib": sha, "grpo": "g"},
    }))
    (stage_c / "selection.json").write_text(json.dumps({"keep_threshold": 0.5, "vgib": {
        "instances": 270, "gold_files_kept_rate": 0.8, "gold_functions_in_topk_rate": 0.7,
        "gold_functions_kept_rate": 0.5, "mean_function_recall": 0.6, "mean_kept_ratio_of_topk": 0.3,
        "mean_kept_ratio_of_pool": 0.1,
    }}))
    (stage_c / "scores.json").write_text(json.dumps({"vgib": {"all_instances": {
        "file": {"Acc@1": 0.5}, "function": {"Acc@5": 0.3, "Acc@10": 0.4}}}}))
    return sha


def _manifest(*betas: float) -> dict:
    runs = [{"beta": b, "job_name": f"j{b}", "eval_job_name": f"eval-j{b}"} for b in betas]
    return {"sweep_id": "s", "eval_dataset": "lite_274", "runs": runs}


def _record(sha: str, **overrides) -> AgentRunRecord:
    base = AgentRunRecord(
        instance_id="t1", dataset="swebench_verified", arm="dense_reranker_vgib", seed=0, model="m",
        git_ref="g", selector_checkpoint_sha256=sha, beta=0.1, retrieved_files=[], retrieved_spans=[],
        retained_span_count=0, fallback_triggered=False, gold_files=[], file_recall=1.0, function_recall=1.0,
        selector_latency_ms=None, first_turn_context_tokens=0, prompt_tokens=0, completion_tokens=0,
        cached_tokens=0, turns=1, tool_calls=0, wall_seconds=1.0, usd_cost=None, applied=True, resolved=True,
        error_class=None,
    )
    return replace(base, **overrides)


def test_collect_rows_sorted_with_dash_without_telemetry(tmp_path: Path):
    manifest = _manifest(0.1, 0.01)
    for run in manifest["runs"]:
        _stage_c(tmp_path, run, run["beta"])
    rows = collect(manifest, tmp_path, "lite_274", [])
    assert [r["beta"] for r in rows] == [0.01, 0.1]
    assert rows[0]["val_token_ratio"] == 0.2
    assert rows[0]["best_epoch"] == 7
    assert rows[0]["token_retention_topk"] == 0.3
    assert rows[0]["gold_functions_kept_rate"] == 0.5
    assert rows[0]["func_acc_at_10"] == 0.4
    assert rows[0]["resolve_rate"] == MISSING
    assert rows[0]["resolve_n"] == MISSING


def test_collect_fails_on_missing_run(tmp_path: Path):
    manifest = _manifest(0.01, 0.1)
    _stage_c(tmp_path, manifest["runs"][0], 0.01)
    with pytest.raises(FileNotFoundError, match="beta=0.1"):
        collect(manifest, tmp_path, "lite_274", [])


def test_collect_fails_on_beta_mismatch(tmp_path: Path):
    manifest = _manifest(0.01)
    _stage_c(tmp_path, manifest["runs"][0], 0.3)
    with pytest.raises(ValueError, match="trained with beta=0.3"):
        collect(manifest, tmp_path, "lite_274", [])


def test_collect_fails_when_runs_scored_differently(tmp_path: Path):
    manifest = _manifest(0.01, 0.1)
    _stage_c(tmp_path, manifest["runs"][0], 0.01)
    _stage_c(tmp_path, manifest["runs"][1], 0.1, reranker="other")
    with pytest.raises(ValueError, match="must share"):
        collect(manifest, tmp_path, "lite_274", [])


def test_resolve_rate_from_matching_records_only():
    records = [_record("a", instance_id="t1"), _record("a", instance_id="t2", resolved=False),
               _record("b", instance_id="t1", resolved=False)]
    assert resolve_rate(records, "a", 0.1) == (0.5, 2)
    assert resolve_rate(records, "zzz", 0.1) == (MISSING, MISSING)


@pytest.mark.parametrize(("second", "match"), [
    ({"arm": "dense_reranker_vgib_rl"}, "several"),
    ({"beta": 0.3}, "beta"),
    ({}, "duplicate"),
])
def test_resolve_rate_rejects_mixed_records(second, match):
    with pytest.raises(ValueError, match=match):
        resolve_rate([_record("a"), _record("a", **second)], "a", 0.1)


def test_main_writes_csv_with_resolve_rate(tmp_path: Path, capsys):
    manifest = _manifest(0.1)
    sha = _stage_c(tmp_path / "results", manifest["runs"][0], 0.1)
    manifest_path = tmp_path / "sweep.json"
    manifest_path.write_text(json.dumps(manifest))
    telemetry = tmp_path / "telemetry.jsonl"
    append_record(telemetry, _record(sha))
    out = tmp_path / "f3.csv"
    main(["--manifest", str(manifest_path), "--results-root", str(tmp_path / "results"),
          "--telemetry", str(telemetry), "--output", str(out)])
    lines = out.read_text().splitlines()
    assert lines[0].startswith("beta,job_name,best_epoch,val_token_ratio")
    assert lines[1].endswith(",1.0,1")
    assert capsys.readouterr().out == out.read_text()

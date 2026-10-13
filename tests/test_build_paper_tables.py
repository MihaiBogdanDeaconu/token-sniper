from __future__ import annotations

import csv
import json
import shutil
import subprocess
from dataclasses import replace
from typing import TYPE_CHECKING

import pytest

from scripts import build_paper_tables as bpt
from token_sniper.evaluation.telemetry import AgentRunRecord, append_record

if TYPE_CHECKING:
    from pathlib import Path

LITE_ACC = {"file": {"Acc@1": 0.9, "Acc@3": 0.5, "Acc@5": 0.6},
            "module": {"Acc@5": 0.5, "Acc@10": 0.6}, "function": {"Acc@5": 0.4, "Acc@10": 0.5}}
LOCBENCH_ACC = {"file": {"Acc@5": 0.7, "Acc@10": 0.8}, "module": {"Acc@10": 0.6, "Acc@15": 0.65},
                "function": {"Acc@10": 0.55, "Acc@15": 0.6}}
SELECTION = {"instances": 3, "gold_files_kept_rate": 0.8, "gold_functions_in_topk_rate": 0.9,
             "gold_functions_kept_rate": 0.7, "mean_function_recall": 0.75, "mean_kept_ratio_of_topk": 0.3,
             "mean_kept_ratio_of_pool": 0.12, "median_kept_tokens": 1, "median_topk_tokens": 2,
             "median_pool_tokens": 3}
COST = {"selector_params": 2_100_000, "reranker_trees": 200,
        "latency_ms_per_instance": {"mean": 40.0, "p50": 35.0, "p95": 90.0},
        "amortized_build_seconds_per_instance": 12.5, "gpu_hour_usd": 2.0, "cpu_hour_usd": 0.05,
        "usd_per_instance": 0.0004, "instances": 274, "source": "cluster run"}


def fake_ci(values):
    return (min(values), max(values))


def fake_mcnemar(comparisons):
    out = {}
    for name, (main, arm) in comparisons.items():
        assert len(main) == len(arm)
        out[name] = {"baseline_only_correct": sum(m and not a for m, a in zip(main, arm, strict=True)),
                     "candidate_only_correct": sum(a and not m for m, a in zip(main, arm, strict=True)),
                     "p_value": 0.5, "p_holm": 0.0004, "significant": True}
    return out


def results_dir(root: Path, name: str, acc: dict, instances: int, checkpoints=("vgib",)) -> Path:
    out = root / name
    out.mkdir(parents=True)
    (out / "inputs.json").write_text(json.dumps({"manifest_instances": instances}))
    (out / "scores.json").write_text(json.dumps(
        {c: {"all_instances": acc, "indexed_only": acc} for c in ("reranker", *checkpoints)}))
    (out / "selection.json").write_text(json.dumps({"keep_threshold": 0.5, **{c: SELECTION for c in checkpoints}}))
    return out


def record(instance_id: str, arm: str, seed: int, **overrides) -> AgentRunRecord:
    base = AgentRunRecord(
        instance_id=instance_id, dataset="swebench_verified", arm=arm, seed=seed, model="m", git_ref="abc",
        selector_checkpoint_sha256=None, beta=None, retrieved_files=[], retrieved_spans=[],
        retained_span_count=0, fallback_triggered=False, gold_files=[], file_recall=0.0, function_recall=0.0,
        selector_latency_ms=None, first_turn_context_tokens=1000, prompt_tokens=5000, completion_tokens=500,
        cached_tokens=0, turns=3, tool_calls=2, wall_seconds=60.0, usd_cost=None, applied=True, resolved=False,
        error_class=None,
    )
    return replace(base, **overrides)


def write_runs(path: Path, records) -> Path:
    for r in records:
        append_record(path, r)
    return path


def t4_records() -> list[AgentRunRecord]:
    resolved_main = {"t1": True, "t2": True, "t3": True, "t4": True}
    resolved_shell = {"t1": True, "t2": False, "t3": False, "t4": False}
    records = []
    for task in ("t1", "t2", "t3", "t4"):
        for seed in (0, 1):
            if not (task == "t4" and seed == 0):
                records.append(record(task, "main", seed, resolved=resolved_main[task],
                                      first_turn_context_tokens=800))
            records.append(record(task, "shell", seed, resolved=resolved_shell[task],
                                  error_class="timeout" if (task, seed) == ("t3", 1) else None,
                                  first_turn_context_tokens=4000))
    return records


def test_t1_has_every_published_row_and_bolds_our_best(tmp_path: Path):
    lite = results_dir(tmp_path, "lite", LITE_ACC, 274)
    latex = bpt.build_t1([("Token Sniper", bpt.LocRef(lite, "vgib"))])
    for _, method, _, _ in bpt.T1_PUBLISHED:
        assert bpt.tex(method) in latex
    ours = next(line for line in latex.splitlines() if line.startswith("Graph"))
    assert ours == (r"Graph & Token Sniper & --- & \textbf{90.00} & 50.00 & 60.00 & 50.00 & 60.00 & 40.00 "
                    r"& 50.00 \\")
    assert "83.21 & 94.89" in latex
    assert r"\textbf{97.08}" in latex


def test_t1_rejects_wrong_manifest_size(tmp_path: Path):
    lite = results_dir(tmp_path, "lite", LITE_ACC, 300)
    with pytest.raises(ValueError, match="274"):
        bpt.build_t1([("Token Sniper", bpt.LocRef(lite, "vgib"))])


def test_t1_unknown_checkpoint_raises(tmp_path: Path):
    lite = results_dir(tmp_path, "lite", LITE_ACC, 274)
    with pytest.raises(KeyError, match="grpo"):
        bpt.build_t1([("Token Sniper", bpt.LocRef(lite, "grpo"))])


def test_t2_uses_locbench_columns_and_footnotes_coderankembed(tmp_path: Path):
    locbench = results_dir(tmp_path, "locbench", LOCBENCH_ACC, 560)
    latex = bpt.build_t2([("Token Sniper", bpt.LocRef(locbench, "vgib"))])
    assert "File@5 & File@10 & Mod@10 & Mod@15 & Func@10 & Func@15" in latex
    assert r"CodeRankEmbed (137M)$^\dagger$" in latex
    assert "43.39" in latex
    assert r"Token Sniper & --- & 70.00 & 80.00 & 60.00 & 65.00 & 55.00 & 60.00 \\" in latex


def test_t3_joins_cost_with_lite_func_at_10(tmp_path: Path):
    lite = results_dir(tmp_path, "lite", LITE_ACC, 274)
    cost = tmp_path / "cost.json"
    cost.write_text(json.dumps(COST))
    latex = bpt.build_t3([("Token Sniper", cost)], [("Token Sniper", bpt.LocRef(lite, "vgib"))])
    assert r"Token Sniper & 2.1M + 200-tree XGB & \$0.0004 & 1 & 50.00 & 1250.0 & 35 \\" in latex
    assert "latency p95 90" in latex and "jina-code-embeddings-0.5b" in latex
    assert r"LocAgent (Claude-3.5) & agentic LLM & \$0.66 & 7 & 77.37 & 1.2 & --- \\" in latex
    assert r"\textasciitilde{}\$0 (precomputable)" in latex
    assert "12.5" in latex and "274 instances" in latex


def test_t3_rejects_bad_cost_or_missing_join(tmp_path: Path):
    lite = results_dir(tmp_path, "lite", LITE_ACC, 274)
    cost = tmp_path / "cost.json"
    cost.write_text(json.dumps({k: v for k, v in COST.items() if k != "gpu_hour_usd"}))
    with pytest.raises(ValueError, match="gpu_hour_usd"):
        bpt.build_t3([("Token Sniper", cost)], [("Token Sniper", bpt.LocRef(lite, "vgib"))])
    cost.write_text(json.dumps(COST))
    with pytest.raises(ValueError, match="no --lite row"):
        bpt.build_t3([("Other", cost)], [("Token Sniper", bpt.LocRef(lite, "vgib"))])


ARMS = [("shell", "Shell-only"), ("main", "Token Sniper")]


def downstream(records=(), arms=ARMS, main="main", mcnemar=fake_mcnemar, **manifests):
    return bpt.build_downstream(records=records, arms=arms, main_arm=main, ci_fn=fake_ci,
                                mcnemar_fn=mcnemar, **manifests)


def test_t4_uniform_denominators_and_majority_pairs():
    calls = {}

    def mcnemar(comparisons):
        calls.update(comparisons)
        return fake_mcnemar(comparisons)

    latex, audit, f2 = downstream(t4_records(), mcnemar=mcnemar).t4["swebench_verified"]
    assert audit["kept_tasks"] == ["t1", "t2"]
    assert audit["dropped_tasks"] == ["t3", "t4"]
    assert "2 of 4 tasks are dropped" in latex
    assert calls == {"shell": ([True, True], [True, False])}
    shell = next(line for line in latex.splitlines() if line.startswith("Shell-only"))
    assert shell == (r"Shell-only & 50.0 $\pm$ 0.0 & [0.0, 100.0] & 4{,}000 & 5{,}500 & --- & --- & 60 "
                     r"& 0 / 1 & $<$0.001$^*$ \\")
    assert r"Token Sniper & 100.0 $\pm$ 0.0 & [100.0, 100.0] & 800" in latex
    assert "(main) & ---" in latex
    assert [(r["label"], r["first_turn_context_tokens"], r["resolve_mean"]) for r in f2] == [
        ("Shell-only", 4000, 0.5), ("Token Sniper", 800, 1.0)]


def test_t4_majority_tie_raises():
    records = [replace(r, resolved=r.seed == 0) if (r.instance_id, r.arm) == ("t1", "shell") else r
               for r in t4_records()]
    with pytest.raises(ValueError, match="exactly half"):
        downstream(records)


def test_t4_prints_cost_only_when_every_run_is_priced():
    records = [replace(r, usd_cost=0.10) for r in t4_records()]
    latex, audit, _ = downstream(records).t4["swebench_verified"]
    assert r"Shell-only & 50.0 $\pm$ 0.0 & [0.0, 100.0] & 4{,}000 & 5{,}500 & \$0.10 & \$0.20" in latex
    assert audit["arms"]["main"]["usd_per_resolved"] == pytest.approx(0.10)
    records[0] = replace(records[0], usd_cost=None)
    with pytest.raises(ValueError, match="usd_cost"):
        downstream(records)


def test_t4_rejects_inconsistent_inputs():
    with pytest.raises(ValueError, match="without an --arm label"):
        downstream(t4_records(), arms=[("main", "Main")])
    with pytest.raises(ValueError, match="different seeds"):
        downstream([r for r in t4_records() if not (r.arm == "shell" and r.seed == 1)])
    with pytest.raises(ValueError, match="duplicate"):
        downstream([*t4_records(), record("t1", "main", 0)])
    with pytest.raises(ValueError, match="no task succeeded"):
        downstream([replace(r, error_class="boom") for r in t4_records()])
    with pytest.raises(KeyError, match="p_holm"):
        downstream(t4_records(), mcnemar=lambda c: {k: {"p_value": 0.1} for k in c})


def write_manifests(tmp_path: Path, sweep_error: tuple[str, float, int] | None = None) -> tuple[Path, Path, Path]:
    results_dir(tmp_path, "lite", LITE_ACC, 274, checkpoints=("vgib", "no_edges", "b0.01", "b0.1"))
    runs = write_runs(tmp_path / "runs" / "telemetry.jsonl", t4_records())
    write_runs(tmp_path / "sweep" / "telemetry.jsonl", [
        record(t, "vgib", s, beta=b, resolved=(t == "t1" or b == 0.01), first_turn_context_tokens=int(1e5 * b),
               error_class="oom" if (t, b, s) == sweep_error else None)
        for b in (0.01, 0.1) for t in ("t1", "t2") for s in (0, 1)])
    t5 = tmp_path / "t5.json"
    t5.write_text(json.dumps({"rows": [
        {"label": "Token Sniper (full)", "localization": {"results": "lite", "checkpoint": "vgib"},
         "telemetry": {"path": "runs/telemetry.jsonl", "arm": "main"}},
        {"label": "- graph edges", "localization": {"results": "lite", "checkpoint": "no_edges"},
         "telemetry": {"path": "runs/telemetry.jsonl", "arm": "shell"}},
        {"label": "- GRPO", "localization": None, "telemetry": None},
    ]}))
    f3 = tmp_path / "f3.json"
    f3.write_text(json.dumps({"points": [
        {"beta": 0.01, "localization": {"results": "lite", "checkpoint": "b0.01"},
         "telemetry": {"path": "sweep/telemetry.jsonl", "arm": "vgib", "beta": 0.01}},
        {"beta": 0.1, "localization": {"results": "lite", "checkpoint": "b0.1"},
         "telemetry": {"path": "sweep/telemetry.jsonl", "arm": "vgib", "beta": 0.1}},
        {"beta": 1.0, "localization": None, "telemetry": None},
    ]}))
    return runs, t5, f3


def test_t5_from_manifest(tmp_path: Path):
    _, t5, _ = write_manifests(tmp_path)
    latex = downstream(t5_manifest=t5).t5
    assert r"Token Sniper (full) & 50.00 & 12.0 & 100.0 $\pm$ 0.0 \\" in latex
    assert r"- graph edges & 50.00 & 12.0 & 50.0 $\pm$ 0.0 \\" in latex
    assert r"- GRPO & --- & --- & --- \\" in latex
    assert "2 of 4 tasks are dropped" in latex


def test_one_drop_set_shared_by_t4_t5_f2_f3(tmp_path: Path):
    runs, t5, f3 = write_manifests(tmp_path, sweep_error=("t2", 0.1, 0))
    down = downstream(bpt.read_records(runs), t5_manifest=t5, f3_manifest=f3)
    assert down.uniforms["swebench_verified"].tasks == ["t1"]
    latex, audit, _ = down.t4["swebench_verified"]
    assert audit["dropped_tasks"] == ["t2", "t3", "t4"]
    assert "3 of 4 tasks are dropped" in latex and "3 of 4 tasks are dropped" in down.t5
    assert r"Shell-only & 100.0 $\pm$ 0.0" in latex
    assert r"- graph edges & 50.00 & 12.0 & 100.0 $\pm$ 0.0 \\" in down.t5
    assert {row["n_tasks"] for row in down.f2} == {1}
    assert [row["resolve_mean"] for row in down.f3[:2]] == [1.0, 1.0]


def test_t5_manifest_must_be_explicit(tmp_path: Path):
    manifest = tmp_path / "t5.json"
    manifest.write_text(json.dumps({"rows": [{"label": "x", "localization": None}]}))
    with pytest.raises(ValueError, match="telemetry"):
        bpt.load_t5(manifest)


def test_f3_rejects_beta_mismatch(tmp_path: Path):
    write_runs(tmp_path / "sweep.jsonl", [record("t1", "vgib", 0, beta=0.3)])
    manifest = tmp_path / "f3.json"
    manifest.write_text(json.dumps({"points": [
        {"beta": 0.1, "localization": None, "telemetry": {"path": "sweep.jsonl", "arm": "vgib"}}]}))
    with pytest.raises(ValueError, match="beta 0.3"):
        bpt.load_f3(manifest)


def test_end_to_end_with_real_statistics(tmp_path: Path):
    runs, t5, f3 = write_manifests(tmp_path)
    out = tmp_path / "out"
    args = bpt.parse_args(["--output-dir", str(out), "--telemetry", str(runs), "--arm", "shell=Shell-only",
                           "--arm", "main=Token Sniper", "--main-arm", "main", "--t5-manifest", str(t5),
                           "--f3-manifest", str(f3)])
    bpt.build(args)
    audit = json.loads((out / "t4_swebench_verified.json").read_text())
    test = audit["mcnemar_holm"]["shell"]
    assert (test["baseline_only_correct"], test["candidate_only_correct"]) == (1, 0)
    assert test["p_holm"] == pytest.approx(1.0) and test["significant"] is False
    low, high = audit["arms"]["shell"]["ci"]
    assert 0.0 <= low <= 0.5 <= high <= 1.0
    assert "& 0 / 1 & 1.000 \\\\" in (out / "t4_swebench_verified.tex").read_text()


def test_cli_end_to_end(tmp_path: Path):
    runs, t5, f3 = write_manifests(tmp_path)
    results_dir(tmp_path, "locbench", LOCBENCH_ACC, 560)
    (tmp_path / "cost.json").write_text(json.dumps(COST))
    out = tmp_path / "out"
    args = bpt.parse_args([
        "--output-dir", str(out),
        "--lite", f"Token Sniper={tmp_path / 'lite'}:vgib", "--lite", f"Token Sniper + GRPO={tmp_path / 'lite'}:b0.1",
        "--locbench", f"Token Sniper={tmp_path / 'locbench'}:vgib",
        "--cost", f"Token Sniper={tmp_path / 'cost.json'}",
        "--telemetry", str(runs), "--arm", "shell=Shell-only", "--arm", "main=+ VGIB selector ($\\beta^*$)",
        "--main-arm", "main", "--t5-manifest", str(t5), "--f3-manifest", str(f3),
    ])
    written = bpt.build(args, ci_fn=fake_ci, mcnemar_fn=fake_mcnemar)
    assert sorted(p.name for p in written) == sorted([
        "t1_lite_274.tex", "t2_locbench_560.tex", "t3_cost.tex", "t4_swebench_verified.tex",
        "t4_swebench_verified.json", "t5_ablation.tex", "f3_beta.csv", "f2_pareto.csv"])
    assert r"+ VGIB selector ($\beta^*$) & 100.0" in (out / "t4_swebench_verified.tex").read_text()

    f3_rows = list(csv.DictReader((out / "f3_beta.csv").open()))
    assert [r["beta"] for r in f3_rows] == ["0.01", "0.1", "1.0"]
    assert f3_rows[0]["token_retention_pool"] == "0.12" and f3_rows[0]["resolve_mean"] == "1.0"
    assert f3_rows[1]["resolve_mean"] == "0.5" and f3_rows[1]["func_acc_at_10"] == "0.5"
    assert f3_rows[2]["token_retention_pool"] == "" and f3_rows[2]["resolve_mean"] == ""

    f2_rows = list(csv.DictReader((out / "f2_pareto.csv").open()))
    assert [(r["series"], r["label"], r["first_turn_context_tokens"]) for r in f2_rows] == [
        ("arm", "Shell-only", "4000.0"), ("arm", "+ VGIB selector ($\\beta^*$)", "800.0"),
        ("beta", "beta=0.01", "1000.0"), ("beta", "beta=0.1", "10000.0")]

    if shutil.which("pdflatex") is None:
        pytest.skip("pdflatex not installed")
    tables = "\n".join(p.read_text() for p in written if p.suffix == ".tex")
    doc = out / "doc.tex"
    doc.write_text("\\documentclass{article}\\usepackage{booktabs}\\begin{document}\n" + tables
                   + "\\end{document}\n")
    proc = subprocess.run(["pdflatex", "-interaction=nonstopmode", "-halt-on-error", doc.name],
                          cwd=out, capture_output=True, text=True, check=False)
    assert proc.returncode == 0, proc.stdout[-2000:]


def test_cli_requires_inputs(tmp_path: Path):
    with pytest.raises(ValueError, match="no inputs"):
        bpt.build(bpt.parse_args(["--output-dir", str(tmp_path)]), ci_fn=fake_ci, mcnemar_fn=fake_mcnemar)
    with pytest.raises(ValueError, match="--main-arm"):
        bpt.build(bpt.parse_args(["--output-dir", str(tmp_path), "--telemetry", "x.jsonl"]),
                  ci_fn=fake_ci, mcnemar_fn=fake_mcnemar)

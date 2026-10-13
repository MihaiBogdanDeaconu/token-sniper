from __future__ import annotations

import json
import subprocess
from types import SimpleNamespace
from typing import TYPE_CHECKING

import pytest

from scripts.eval_ablation_swebench import (
    Pricing,
    build_record,
    check_context_sent_whole,
    grade,
    parse_args,
)
from token_sniper.evaluation.arms.client import _SeededOpenAI, read_usage
from token_sniper.evaluation.arms.context import (
    CONTEXT_HEADER,
    Span,
    gold_units,
    line_prefix,
    max_under_budget,
    normalize,
    pack_to_budget,
    patch_files,
    recall,
    render,
    truncate_to_budget,
)
from token_sniper.evaluation.arms.selection import (
    Selection,
    SelectionConfig,
    Selector,
    checkpoint_identity,
    task_input,
)
from token_sniper.evaluation.eval_stages import (
    candidates_path,
    retention_path,
    sha256_file,
    write_record,
)
from token_sniper.evaluation.phase0 import Phase0Task
from token_sniper.graph.ssg import build_ssg

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
OTHER = "".join(f"x{i} = {i}\n" for i in range(1, 31))
HEADER = "diff --git a/pkg/box.py b/pkg/box.py\n--- a/pkg/box.py\n+++ b/pkg/box.py\n"
GROW_PATCH = HEADER + "@@ -9,2 +9,2 @@\n     def grow(self):\n-        return self.size + 1\n+        return self.size + 2\n"
MAIN_PATCH = HEADER + "@@ -15 +15 @@\n-MAIN = 1\n+MAIN = 2\n"
NEW_FILE_PATCH = "diff --git a/pkg/new.py b/pkg/new.py\n--- /dev/null\n+++ b/pkg/new.py\n@@ -0,0 +1 @@\n+A = 1\n"


class WordTokenizer:
    def __call__(self, text: str, add_special_tokens: bool = False) -> dict:
        return {"input_ids": text.split()}


def words(text: str) -> int:
    return len(text.split())


@pytest.fixture
def repo(tmp_path: Path) -> SimpleNamespace:
    root = tmp_path / "work"
    (root / "pkg").mkdir(parents=True)
    (root / "pkg" / "box.py").write_text(SOURCE)
    (root / "pkg" / "other.py").write_text(OTHER)
    for args in (["init", "-q"], ["add", "-A"], ["-c", "user.name=t", "-c", "user.email=t@t", "commit", "-q", "-m", "i"]):
        subprocess.run(["git", *args], cwd=root, check=True, capture_output=True)
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, check=True, capture_output=True,
                            text=True).stdout.strip()
    repos = tmp_path / "repos"
    subprocess.run(["git", "clone", "-q", "--bare", str(root), str(repos / "o__r")], check=True, capture_output=True)
    task = Phase0Task(instance_id="o__r-1", repo="o/r", base_commit=commit, gold_patch_files=["pkg/box.py"],
                      gold_patch_file_count=1, bucket="single_file")
    return SimpleNamespace(graph=build_ssg(root), task=task, repos=repos, tmp=tmp_path,
                           sources={"pkg/box.py": SOURCE.split("\n"), "pkg/other.py": OTHER.split("\n")})


def node(graph, suffix: str) -> str:
    return next(n for n in graph.nodes if n.endswith(suffix))


def test_normalize_merges_near_spans_and_keeps_file_order(repo):
    spans = [Span("pkg/other.py", 1, 2, "lines"), Span("pkg/box.py", 9, 10, "method"),
             Span("pkg/box.py", 3, 4, "function"), Span("pkg/box.py", 6, 7, "class")]
    groups = normalize(spans, repo.sources)
    assert [(g.file_path, g.start_line, g.end_line) for g in groups] == [
        ("pkg/other.py", 1, 2), ("pkg/box.py", 3, 10)]
    assert groups[1].node_type == "class,function,method"
    text = render(groups, repo.sources)
    assert text.startswith(CONTEXT_HEADER)
    assert "--- pkg/box.py (lines 3-10) ---\ndef top(x):" in text


def test_whole_file_renders_verbatim(repo):
    groups = normalize([Span("pkg/box.py", 1, 16, "file")], repo.sources)
    assert render(groups, repo.sources) == f"{CONTEXT_HEADER}\n\n--- pkg/box.py ---\n{SOURCE}"


def test_span_outside_file_raises(repo):
    with pytest.raises(ValueError, match="outside"):
        normalize([Span("pkg/box.py", 10, 99, "function")], repo.sources)


def test_truncate_to_budget_takes_the_longest_row_prefix(repo):
    groups = normalize([Span("pkg/other.py", 1, 31, "file"), Span("pkg/box.py", 1, 16, "file")], repo.sources)
    budget = words(render(line_prefix(groups, 12), repo.sources))
    kept = truncate_to_budget(groups, repo.sources, WordTokenizer(), budget)
    assert kept == line_prefix(groups, 12)
    assert kept[-1].node_type == "lines"
    assert truncate_to_budget(groups, repo.sources, WordTokenizer(), 10**6) == groups
    with pytest.raises(ValueError, match="header"):
        truncate_to_budget(groups, repo.sources, WordTokenizer(), 3)


def test_pack_to_budget_skips_spans_that_do_not_fit(repo):
    big, small = Span("pkg/other.py", 1, 30, "lines"), Span("pkg/box.py", 3, 4, "function")
    budget = words(render(normalize([small], repo.sources), repo.sources))
    kept, groups = pack_to_budget([big, small], repo.sources, WordTokenizer(), budget)
    assert kept == [small]
    assert groups == normalize([small], repo.sources)


def test_max_under_budget_bisects_and_raises_when_nothing_fits():
    x, text = max_under_budget(lambda v: "w " * int(v * 100), words, 37, lo=0.0, hi=1.0, rounds=12)
    assert words(text) <= 37 and x > 0.36
    with pytest.raises(ValueError, match="budget"):
        max_under_budget(lambda v: "w " * (10 + int(v * 100)), words, 5, lo=0.0, hi=1.0)


def test_patch_files_lists_old_and_created_paths():
    assert patch_files(GROW_PATCH + NEW_FILE_PATCH) == ["pkg/box.py", "pkg/new.py"]


def test_entity_counts_only_when_shown_whole(repo):
    units = gold_units(repo.graph, GROW_PATCH)
    assert units == {node(repo.graph, "Box.grow"): frozenset({9, 10})}
    gold = patch_files(GROW_PATCH)
    assert recall(normalize([Span("pkg/box.py", 9, 10, "method")], repo.sources), gold, units,
                  repo.graph) == (1.0, 1.0)
    assert recall(normalize([Span("pkg/box.py", 10, 12, "lines")], repo.sources), gold, units,
                  repo.graph) == (1.0, 0.0)
    assert recall([], gold, units, repo.graph) == (0.0, 0.0)


def test_module_level_edit_counts_through_its_rows(repo):
    units = gold_units(repo.graph, MAIN_PATCH)
    assert units == {node(repo.graph, "__module_preamble__"): frozenset({15})}
    shown = normalize([Span("pkg/box.py", 15, 15, "lines")], repo.sources)
    assert recall(shown, ["pkg/box.py"], units, repo.graph) == (1.0, 1.0)


def test_patch_that_only_creates_files_raises(repo):
    with pytest.raises(ValueError, match="only creates files"):
        recall([], ["pkg/new.py"], gold_units(repo.graph, NEW_FILE_PATCH), repo.graph)


def _selector(repo, **cfg) -> Selector:
    config = SelectionConfig(ssg_dir=repo.tmp, repos_dir=repo.repos, cache_root=repo.tmp / "cache",
                             cache_name="ds", fallback_top_k=1, **cfg)
    return Selector(config, cache_dir=repo.tmp / "selections", tokenizer=WordTokenizer())


def _stage_records(repo, p_grow: float, *, latency: bool = True):
    from scripts.eval_retention import retention_header

    cache = repo.tmp / "cache" / "ds"
    cpath = candidates_path(cache, repo.task.instance_id)
    write_record(cpath, {"ssg_sha256": "s"}, {
        "candidate_files": ["pkg/box.py", "pkg/other.py"], "feature_matrix": [[0.0] * 18] * 2,
        "entities": [{"file_path": "pkg/other.py", "rank": 1}, {"file_path": "pkg/box.py", "rank": 2}],
        "expand_depth": 0,
    })
    grow, box = node(repo.graph, "Box.grow"), node(repo.graph, ".Box")
    body = {"ranked_files": [["pkg/other.py", 2.0], ["pkg/box.py", 1.0]], "pool_tokens": 100,
            "nodes": [{"id": grow, "file": "pkg/box.py", "p_v": p_grow, "tokens": 5},
                      {"id": box, "file": "pkg/box.py", "p_v": 0.1, "tokens": 9},
                      {"id": "file:pkg/box.py", "file": "pkg/box.py", "p_v": 0.9, "tokens": 3}]}
    if latency:
        body["selector_latency_ms"] = 4.5
    write_record(retention_path(cache, "c" * 64, repo.task.instance_id),
                 retention_header("c" * 64, "r" * 64, sha256_file(cpath), 20), body)


def _ti(repo, patch: str = GROW_PATCH):
    return task_input(repo.task, issue="grow is wrong", gold_patch=patch, graph=repo.graph)


def _with_vgib(selector: Selector) -> Selector:
    selector._resources["booster"] = (None, "r" * 64)
    selector._resources["vgib_checkpoint"] = ("c" * 64, 0.01)
    return selector


def test_vgib_arm_shows_kept_entity_spans(repo):
    _stage_records(repo, p_grow=0.8)
    sel = _with_vgib(_selector(repo, vgib_checkpoint=repo.tmp / "unused.pt")).get("dense_reranker_vgib", _ti(repo))
    assert not sel.fallback_triggered
    assert sel.retained_span_count == 1
    assert sel.retrieved_spans == [{"file_path": "pkg/box.py", "start_line": 9, "end_line": 10,
                                    "node_type": "method"}]
    assert (sel.file_recall, sel.function_recall) == (1.0, 1.0)
    assert (sel.selector_latency_ms, sel.selector_checkpoint_sha256, sel.beta) == (4.5, "c" * 64, 0.01)
    assert sel.context_tokens == words(sel.context)


def test_vgib_fallback_is_recorded(repo):
    _stage_records(repo, p_grow=0.2)
    sel = _with_vgib(_selector(repo, vgib_checkpoint=repo.tmp / "unused.pt")).get("dense_reranker_vgib", _ti(repo))
    assert sel.fallback_triggered
    assert sel.retained_span_count == 0
    assert sel.retrieved_files == ["pkg/other.py"]
    assert sel.retrieved_spans == []
    assert (sel.file_recall, sel.function_recall) == (0.0, 0.0)


def test_vgib_record_without_latency_raises(repo):
    _stage_records(repo, p_grow=0.8, latency=False)
    selector = _with_vgib(_selector(repo, vgib_checkpoint=repo.tmp / "unused.pt"))
    with pytest.raises(ValueError, match="selector_latency_ms"):
        selector.get("dense_reranker_vgib", _ti(repo))


def test_cached_selection_with_other_inputs_raises(repo):
    _stage_records(repo, p_grow=0.8)
    selector = _with_vgib(_selector(repo, vgib_checkpoint=repo.tmp / "unused.pt"))
    selector.get("dense_reranker_vgib", _ti(repo))
    other = _with_vgib(_selector(repo, vgib_checkpoint=repo.tmp / "unused.pt", max_context_tokens=123))
    with pytest.raises(ValueError, match="was built from"):
        other.get("dense_reranker_vgib", _ti(repo))


def test_hybrid_and_oracle_arms(repo):
    _stage_records(repo, p_grow=0.8)
    selector = _selector(repo, top_k_files=1)
    assert selector.get("hybrid", _ti(repo)).retrieved_files == ["pkg/other.py"]
    oracle = selector.get("oracle_gold_files", _ti(repo, GROW_PATCH + NEW_FILE_PATCH))
    assert oracle.retrieved_files == ["pkg/box.py"]
    assert oracle.gold_files == ["pkg/box.py", "pkg/new.py"]
    assert (oracle.file_recall, oracle.function_recall, oracle.selector_latency_ms) == (0.5, 1.0, None)


def test_shell_only_has_no_context(repo):
    sel = _selector(repo).get("shell_only", _ti(repo))
    assert (sel.mode, sel.context, sel.context_tokens, sel.function_recall) == ("blind", None, 0, 0.0)


def test_budget_arms_fit_the_budget(repo):
    _stage_records(repo, p_grow=0.8)
    selector = _selector(repo, budget_tokens=40, swe_pruner_model=repo.tmp)
    selector._resources["booster"] = (None, "r" * 64)
    selector._reranked = lambda ti: (["pkg/box.py", "pkg/other.py"], 1.0)

    class Pruner:
        def kept_rows(self, query, rows, threshold):
            return [i for i in range(1, len(rows) + 1) if i % 10 >= threshold * 10]

    class Lingua:
        def compress(self, text, rate):
            return " ".join(text.split()[: int(len(text.split()) * rate)])

    selector._resources["swe_pruner"] = Pruner()
    selector._resources["llmlingua2"] = Lingua()
    for arm in ("truncate_to_budget", "swe_pruner", "llmlingua2"):
        sel = selector.get(arm, _ti(repo))
        assert 0 < sel.context_tokens <= 40, arm
        assert sel.context_tokens == words(sel.context)
    trunc = selector.get("truncate_to_budget", _ti(repo))
    assert trunc.retrieved_files == ["pkg/box.py"] and trunc.retrieved_spans[0]["start_line"] == 1
    assert selector.get("llmlingua2", _ti(repo)).recall_on == "compressor_input"


def test_budget_matches_another_arm(repo):
    _stage_records(repo, p_grow=0.8)
    selector = _with_vgib(_selector(repo, vgib_checkpoint=repo.tmp / "unused.pt"))
    selector._reranked = lambda ti: (["pkg/box.py", "pkg/other.py"], 1.0)
    vgib = selector.get("dense_reranker_vgib", _ti(repo))
    trunc = selector.get("truncate_to_budget", _ti(repo))
    assert trunc.context_tokens <= vgib.context_tokens
    assert (trunc.selector_checkpoint_sha256, trunc.beta) == (None, None)


def test_checkpoint_identity_reads_the_stored_beta(tmp_path):
    torch = pytest.importorskip("torch")
    vgib, nested, grpo = tmp_path / "v.pt", tmp_path / "n.pt", tmp_path / "g.pt"
    torch.save({"beta": 0.03, "model_state": {}}, vgib)
    torch.save({"train_cfg": {"beta": 0.1}}, nested)
    torch.save({"mode": "grpo", "grpo_cfg": {"reward_beta": 0.5}}, grpo)
    assert checkpoint_identity(vgib, require_beta=True) == (sha256_file(vgib), 0.03)
    assert checkpoint_identity(nested, require_beta=True)[1] == 0.1
    assert checkpoint_identity(grpo, require_beta=False) == (sha256_file(grpo), None)
    with pytest.raises(ValueError, match="stores no beta"):
        checkpoint_identity(grpo, require_beta=True)


class FakeCompletions:
    def __init__(self, cached):
        self.kwargs = None
        self.cached = cached

    def create(self, **kwargs):
        self.kwargs = kwargs
        details = None if self.cached is None else SimpleNamespace(cached_tokens=self.cached)
        return SimpleNamespace(usage=SimpleNamespace(prompt_tokens=100, completion_tokens=7,
                                                     prompt_tokens_details=details))


def test_client_sends_the_seed_and_logs_usage(tmp_path):
    inner = FakeCompletions(cached=60)
    client = _SeededOpenAI(SimpleNamespace(chat=SimpleNamespace(completions=inner)), 3, tmp_path / "u.jsonl")
    client.chat.completions.create(model="m", messages=[])
    client.chat.completions.create(model="m", messages=[])
    assert inner.kwargs["seed"] == 3
    assert read_usage(tmp_path / "u.jsonl") == {"prompt_tokens": 200, "completion_tokens": 14,
                                                "cached_tokens": 120, "calls": 2}


def test_client_without_cached_tokens_raises(tmp_path):
    client = _SeededOpenAI(SimpleNamespace(chat=SimpleNamespace(completions=FakeCompletions(cached=None))), 0,
                           tmp_path / "u.jsonl")
    with pytest.raises(RuntimeError, match="enable-prompt-tokens-details"):
        client.chat.completions.create(model="m", messages=[])


def test_runner_truncation_is_refused():
    from token_sniper.agent.runner import AgentRunSettings

    def settings(context):
        return AgentRunSettings(repo_root="/testbed", issue_description="bug", output_dir="/tmp/x", mode="oracle",
                                oracle_context=context, max_output_tokens=16_384)

    def selection(context):
        return SimpleNamespace(mode="oracle", context=context, context_tokens=0, arm="a")

    check_context_sent_whole(settings("small"), selection("small"))
    huge = "x" * 1_000_000
    with pytest.raises(RuntimeError, match="max-context-tokens"):
        check_context_sent_whole(settings(huge), selection(huge))


def _attempt(tmp_path, status, log=""):
    report = tmp_path / "logs" / "report.json"
    report.parent.mkdir(parents=True, exist_ok=True)
    (report.parent / "run_instance.log").write_text(log)
    return SimpleNamespace(metadata={"harness_status": status}, resolved=status == "reported",
                           patch_successfully_applied=status == "reported", instance_report_path=str(report))


def test_grade_classifies_harness_outcomes(tmp_path):
    assert grade(_attempt(tmp_path, "reported")) == (
        {"resolved": True, "applied": True, "harness_status": "reported"}, None)
    assert grade(_attempt(tmp_path, "error", ">>>>> Patch Apply Failed:\n..."))[1] is None
    assert grade(_attempt(tmp_path, "error", "docker died"))[1] == "harness_error"


STATE = {"error_class": None, "error": None, "patch": True, "turns": 5, "tool_calls": 9, "wall_seconds": 12.5,
         "prompt_tokens": 1_000_000, "completion_tokens": 2_000, "cached_tokens": 600_000,
         "grade": {"resolved": True, "applied": True, "harness_status": "reported"}}


def _selection(**overrides) -> Selection:
    base = dict(arm="dense_reranker_vgib", mode="oracle", context="ctx", context_tokens=321,
                retrieved_files=["a.py"], retrieved_spans=[], retained_span_count=2, fallback_triggered=True,
                selector_latency_ms=4.0, selector_checkpoint_sha256="c", beta=0.01, gold_files=["a.py"],
                file_recall=1.0, function_recall=0.5, recall_on="shown")
    return Selection(**{**base, **overrides})


def test_build_record_carries_every_measured_field():
    record = build_record(STATE, _selection(), instance_id="i", gold_files=["a.py"], dataset="swebench_verified",
                          arm="dense_reranker_vgib", seed=2, model="q", git_ref="g", arm_checkpoint=("c", 0.01),
                          pricing=Pricing(prompt=1.0, cached=0.1, completion=2.0))
    assert (record.first_turn_context_tokens, record.cached_tokens, record.fallback_triggered) == (321, 600_000, True)
    assert (record.resolved, record.applied, record.error_class, record.seed) == (True, True, None, 2)
    assert record.usd_cost == pytest.approx((400_000 * 1.0 + 600_000 * 0.1 + 2_000 * 2.0) / 1e6)
    unpriced = build_record(STATE, _selection(), instance_id="i", gold_files=["a.py"], dataset="d",
                            arm="dense_reranker_vgib", seed=2, model="q", git_ref="g",
                            arm_checkpoint=("c", 0.01), pricing=None)
    assert unpriced.usd_cost is None
    with pytest.raises(ValueError, match="checkpoint"):
        build_record(STATE, _selection(), instance_id="i", gold_files=[], dataset="d", arm="a", seed=0,
                     model="q", git_ref="g", arm_checkpoint=("other", 0.01), pricing=None)


def test_build_record_for_a_selection_error():
    state = {**STATE, "error_class": "selection_error", "patch": False, "grade": None, "prompt_tokens": 0,
             "completion_tokens": 0, "cached_tokens": 0, "turns": 0, "tool_calls": 0, "wall_seconds": 0.0}
    record = build_record(state, "Traceback ...", instance_id="i", gold_files=["a.py"], dataset="d",
                          arm="dense_reranker_vgib", seed=0, model="q", git_ref="g", arm_checkpoint=("c", 0.01),
                          pricing=None)
    assert (record.error_class, record.resolved, record.gold_files, record.beta) == (
        "selection_error", False, ["a.py"], 0.01)
    assert json.loads(json.dumps(record.__dict__))["retrieved_files"] == []


def test_cli_rules():
    base = ["--dataset", "swebench_verified", "--manifest", "m.json", "--output-root", "o",
            "--ssg-dir", "s", "--repos-dir", "r"]
    assert parse_args([*base, "--arm", "bm25", "--seed", "1"]).seed == 1
    assert parse_args([*base, "--select-only", "--arm", "bm25", "hybrid"]).arm == ["bm25", "hybrid"]
    for bad in (["--arm", "bm25"], ["--arm", "bm25", "hybrid", "--seed", "0"],
                ["--arm", "bm25", "--seed", "0", "--usd-per-mtok-prompt", "1"],
                ["--arm", "ppr_topk", "--seed", "0", "--budget-arm", "llmlingua2"]):
        with pytest.raises(SystemExit):
            parse_args([*base, *bad])

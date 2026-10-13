from __future__ import annotations

import threading
from pathlib import Path

import networkx as nx
import pytest
import torch

from tests.rl.test_rewards import _make_task
from token_sniper.evaluation.models import SWEbenchEvaluationAttempt, SWEbenchTaskMetadata
from token_sniper.graph.io import write_ssg_json
from token_sniper.rl import verified
from token_sniper.rl.rewards import LLMJudgeReward
from token_sniper.rl.verified import (
    AgentRollout,
    NodeSpan,
    VerifiedRolloutError,
    build_span_table,
    fail_to_pass_verdict,
    render_span_context,
    spans_for_mask,
)

COMMIT = "abcdef1234567890"
F2P = ["tests/test_x.py::test_a", "tests/test_x.py::test_b"]


def _graph(num_nodes: int = 10, num_files: int = 3) -> nx.MultiDiGraph:
    g = nx.MultiDiGraph()
    for f in range(num_files):
        g.add_node(f"file:file_{f}.py", node_type="file", file_path=f"file_{f}.py",
                   start_line=1, end_line=100)
    for i in range(num_nodes):
        g.add_node(f"module.func_{i}", node_type="function", file_path=f"file_{i % num_files}.py",
                   start_line=10 * i + 1, end_line=10 * i + 5)
    return g


def _metadata(instance_id: str = "test-reward") -> SWEbenchTaskMetadata:
    return SWEbenchTaskMetadata(
        instance_id=instance_id, repo="test/repo", base_commit=COMMIT,
        problem_statement="it is broken", gold_patch="", fail_to_pass_tests=F2P,
    )


def _attempt(tmp: Path, *, status="reported", applied=True, success=(), failure=(), resolved=False):
    return SWEbenchEvaluationAttempt(
        attempt_index=1, model_name="m", patch_path=str(tmp / "p.diff"), patch_sha256="x",
        run_id="r", resolved=resolved, patch_exists=True, patch_successfully_applied=applied,
        summary_report_path="s", instance_report_path="i", harness_stdout_path="o",
        harness_stderr_path="e", fail_to_pass_success=list(success),
        fail_to_pass_failure=list(failure), metadata={"harness_status": status, "evaluation_error": None},
    )


class FakeBackend:
    def __init__(self, patch: str | None = "diff --git a/x b/x\n", error: Exception | None = None):
        self.patch = patch
        self.error = error
        self.calls: list[list[NodeSpan]] = []
        self._lock = threading.Lock()

    def run(self, *, metadata, spans, rollout_id, output_dir):
        with self._lock:
            self.calls.append(spans)
        if self.error is not None:
            raise self.error
        return AgentRollout(
            patch_text=self.patch, context_chars=123, prompt_tokens=10,
            completion_tokens=5, wall_time_s=1.0, session_log_path=None,
        )


class FakeHarness:
    def __init__(self, tmp: Path, **attempt_kwargs):
        self.tmp = tmp
        self.kwargs = attempt_kwargs
        self.calls = 0

    def evaluate_patch(self, *, work_dir, instance_id, patch_file, model_name, run_id):
        self.calls += 1
        assert patch_file.read_text(encoding="utf-8")
        return _attempt(self.tmp, **self.kwargs)


@pytest.fixture
def judge_env(tmp_path, monkeypatch):
    write_ssg_json(_graph(), tmp_path / "ssg" / "test__repo" / COMMIT[:12] / "ssg.json")
    monkeypatch.setattr(
        "token_sniper.evaluation.swebench.load_swebench_task_metadata",
        lambda instance_id, **_: _metadata(instance_id),
    )
    return tmp_path


def _judge(tmp: Path, backend, harness, **kwargs) -> LLMJudgeReward:
    return LLMJudgeReward(
        agent_backend=backend, harness=harness, model_name="agent/model",
        ssg_dir=tmp / "ssg", base_commits={"test-reward": COMMIT}, work_dir=tmp / "work",
        dataset_name="d", split="s", **kwargs,
    )


class TestSpans:
    def test_span_table_skips_file_nodes_and_non_candidates(self):
        g = _graph()
        ids = ["file:file_0.py", "module.func_0", "module.func_1"]
        table = build_span_table(g, ids, candidate_files=["file_0.py"])
        assert table[0] is None
        assert table[1] == NodeSpan("file_0.py", 1, 5, "function", "module.func_0")
        assert table[2] is None

    def test_span_table_raises_on_node_missing_from_ssg(self):
        with pytest.raises(KeyError, match="module.gone"):
            build_span_table(_graph(), ["module.gone"], candidate_files=["file_0.py"])

    def test_spans_for_mask_keeps_retained_in_node_order(self):
        g = _graph()
        ids = [f"module.func_{i}" for i in range(6)]
        table = build_span_table(g, ids, candidate_files=["file_0.py", "file_1.py", "file_2.py"])
        spans = spans_for_mask(table, torch.tensor([0.0, 1, 0, 1, 1, 0]))
        assert [s.node_id for s in spans] == ["module.func_1", "module.func_3", "module.func_4"]

    def test_spans_for_mask_length_mismatch(self):
        with pytest.raises(ValueError, match="span table"):
            spans_for_mask([None, None], torch.ones(3))

    def test_render_matches_eval_condition(self):
        from token_sniper.evaluation.arms.context import Span, normalize, render, source_rows

        sources = {
            "pkg/b.py": source_rows(("\n".join(f"b line {i}" for i in range(1, 21)) + "\n").encode()),
            "a.py": source_rows(("\n".join(f"a line {i}" for i in range(1, 41)) + "\n").encode()),
        }
        spans = [
            NodeSpan("pkg/b.py", 2, 3, "method", "b.m"),
            NodeSpan("a.py", 7, 10, "class", "a.C"),
            NodeSpan("a.py", 1, 5, "function", "a.f"),
            NodeSpan("a.py", 30, 40, "function", "a.g"),
        ]
        eval_spans = [Span(s.file_path, s.start_line, s.end_line, s.node_type) for s in spans]
        rendered = render_span_context(spans, sources)
        assert rendered == render(normalize(eval_spans, sources), sources)
        assert rendered.index("pkg/b.py") < rendered.index("a.py")

    def test_span_outside_its_file_raises(self):
        from token_sniper.evaluation.arms.context import source_rows

        sources = {"a.py": source_rows(b"x = 1\n")}
        with pytest.raises(ValueError, match="outside"):
            render_span_context([NodeSpan("a.py", 1, 9, "function", "a.f")], sources)


class TestVerdict:
    def test_all_fail_to_pass_pass(self, tmp_path):
        assert fail_to_pass_verdict(_attempt(tmp_path, success=F2P), F2P) is True

    def test_one_failure_is_zero(self, tmp_path):
        assert fail_to_pass_verdict(_attempt(tmp_path, success=F2P[:1], failure=F2P[1:]), F2P) is False

    def test_patch_not_applied_is_zero(self, tmp_path):
        assert fail_to_pass_verdict(_attempt(tmp_path, applied=False), F2P) is False

    @pytest.mark.parametrize("status", ["error", "incomplete", "missing_report", "empty_patch"])
    def test_no_report_raises(self, tmp_path, status):
        with pytest.raises(VerifiedRolloutError, match=status):
            fail_to_pass_verdict(_attempt(tmp_path, status=status), F2P)

    def test_report_missing_a_test_raises(self, tmp_path):
        with pytest.raises(VerifiedRolloutError, match="does not account"):
            fail_to_pass_verdict(_attempt(tmp_path, success=F2P[:1]), F2P)

    def test_no_expected_tests_raises(self, tmp_path):
        with pytest.raises(ValueError, match="FAIL_TO_PASS"):
            fail_to_pass_verdict(_attempt(tmp_path), [])


class TestLLMJudgeReward:
    def test_pass_gives_one(self, judge_env):
        harness = FakeHarness(judge_env, success=F2P, resolved=True)
        judge = _judge(judge_env, FakeBackend(), harness)
        task = _make_task()
        signal = judge.compute(task, torch.ones(10))
        assert signal.task_reward == 1.0
        assert signal.source == "verified"
        assert signal.recall == 1.0 and signal.token_ratio == 1.0
        assert harness.calls == 1
        outcomes = list((judge_env / "work" / "test-reward").glob("*/outcome.json"))
        assert len(outcomes) == 1

    def test_fail_gives_zero(self, judge_env):
        judge = _judge(judge_env, FakeBackend(), FakeHarness(judge_env, failure=F2P))
        assert judge.compute(_make_task(), torch.ones(10)).task_reward == 0.0

    def test_no_patch_is_zero_without_harness(self, judge_env):
        harness = FakeHarness(judge_env, success=F2P)
        judge = _judge(judge_env, FakeBackend(patch=None), harness)
        outcome = judge.run_many([(_make_task(), torch.ones(10))])[0]
        assert outcome.passed is False and outcome.patch_produced is False
        assert harness.calls == 0

    def test_infra_failure_raises_not_zero(self, judge_env):
        judge = _judge(judge_env, FakeBackend(error=VerifiedRolloutError("docker died")),
                       FakeHarness(judge_env, success=F2P))
        with pytest.raises(VerifiedRolloutError, match="docker died"):
            judge.compute(_make_task(), torch.ones(10))

    def test_harness_error_raises(self, judge_env):
        judge = _judge(judge_env, FakeBackend(), FakeHarness(judge_env, status="error"))
        with pytest.raises(VerifiedRolloutError):
            judge.compute(_make_task(), torch.ones(10))

    def test_token_beta(self, judge_env):
        judge = _judge(judge_env, FakeBackend(), FakeHarness(judge_env, success=F2P), token_beta=0.5)
        mask = torch.zeros(10)
        mask[:5] = 1.0
        assert judge.compute(_make_task(), mask).task_reward == pytest.approx(1.0 - 0.5 * 0.5)

    def test_compute_many_parallel_keeps_order_and_spans(self, judge_env):
        backend = FakeBackend()
        judge = _judge(judge_env, backend, FakeHarness(judge_env, success=F2P), max_workers=4)
        task = _make_task()
        masks = [torch.zeros(10) for _ in range(4)]
        for i, m in enumerate(masks):
            m[: i + 1] = 1.0
        signals = judge.compute_many([(task, m) for m in masks])
        assert [s.token_ratio for s in signals] == pytest.approx([0.1, 0.2, 0.3, 0.4])
        assert sorted(len(c) for c in backend.calls) == [1, 2, 3, 4]


class TestDockerAgentBackend:
    @pytest.fixture
    def patched(self, monkeypatch, tmp_path):
        state = {"stopped": [], "running": True, "session": None}

        def fake_start(name, *, image_tag):
            state["image"] = image_tag
            return f"swebench-{name}"

        monkeypatch.setattr("token_sniper.agent.docker_tools.start_task_container", fake_start)
        monkeypatch.setattr("token_sniper.agent.docker_tools.stop_task_container",
                            lambda c: state["stopped"].append(c))
        monkeypatch.setattr("token_sniper.evaluation.swebench._load_dataset_rows",
                            lambda *_: {"test-reward": {"instance_id": "test-reward"}})
        monkeypatch.setattr(verified, "extract_test_cmd", lambda row, ns: "pytest -x")
        monkeypatch.setattr(verified, "swebench_live_image", lambda row, ns: "img:latest")
        monkeypatch.setattr(verified, "_read_container_file", lambda c, fp: ["x = 1"])

        def fake_require(container):
            if not state["running"]:
                raise VerifiedRolloutError(f"container {container} is not running")

        monkeypatch.setattr(verified, "_require_running", fake_require)

        def fake_session(settings, client, *, tool_executor):
            out = Path(settings.output_dir)
            out.mkdir(parents=True, exist_ok=True)
            (out / "session.json").write_text(
                '{"total_prompt_tokens": 7, "total_completion_tokens": 3}', encoding="utf-8")
            state["settings"] = settings
            if isinstance(state["session"], BaseException):
                raise state["session"]
            (out / "patch.diff").write_text("diff\n", encoding="utf-8")

            class R:
                patch_path = str(out / "patch.diff")
                total_prompt_tokens = 7
                total_completion_tokens = 3
            return R()

        monkeypatch.setattr("token_sniper.agent.runner.run_agent_session", fake_session)
        state["tmp"] = tmp_path
        return state

    def _run(self, state):
        from token_sniper.agent.config import AgentModelConfig

        backend = verified.DockerAgentBackend(
            client=object(), model_config=AgentModelConfig(
                name="m", provider="local", base_url="http://x/v1", api_key_env="K", max_tokens=1024,
            ),
            max_turns=5, dataset_name="d", split="s",
        )
        return backend.run(
            metadata=_metadata(), spans=[NodeSpan("a.py", 1, 1, "function", "a.f")],
            rollout_id="r1", output_dir=state["tmp"] / "agent",
        )

    def test_patch(self, patched):
        out = self._run(patched)
        assert out.patch_text == "diff\n"
        assert patched["stopped"] == ["swebench-test-reward.r1"]
        assert patched["settings"].test_cmd == "pytest -x"
        assert patched["settings"].fail_to_pass_tests == F2P
        assert "a.py" in patched["settings"].oracle_context

    @pytest.mark.parametrize("msg", sorted(verified._NO_PATCH_ERRORS))
    def test_no_patch_is_not_an_error(self, patched, msg):
        patched["session"] = RuntimeError(msg)
        out = self._run(patched)
        assert out.patch_text is None
        assert (out.prompt_tokens, out.completion_tokens) == (7, 3)

    def test_model_server_error_raises(self, patched):
        patched["session"] = ConnectionError("vllm unreachable")
        with pytest.raises(VerifiedRolloutError, match="vllm unreachable"):
            self._run(patched)
        assert patched["stopped"], "container must be stopped on failure"

    def test_other_runtime_error_raises(self, patched):
        patched["session"] = RuntimeError("something else")
        with pytest.raises(VerifiedRolloutError, match="something else"):
            self._run(patched)

    def test_dead_container_raises(self, patched):
        patched["session"] = RuntimeError(sorted(verified._NO_PATCH_ERRORS)[0])
        patched["running"] = False
        with pytest.raises(VerifiedRolloutError, match="not running"):
            self._run(patched)


def test_extract_test_cmd_raises_without_marker(monkeypatch):
    class Spec:
        eval_script = "set -e\necho hi\n"

    monkeypatch.setattr("swebench.harness.test_spec.test_spec.make_test_spec", lambda row, namespace: Spec())
    with pytest.raises(ValueError, match="no test command"):
        verified.extract_test_cmd({"instance_id": "x"})


def test_extract_test_cmd_reads_line_after_marker(monkeypatch):
    class Spec:
        eval_script = "a\n: '>>>>> Start Test Output'\npytest -rA tests/x.py\n: end\n"

    monkeypatch.setattr("swebench.harness.test_spec.test_spec.make_test_spec", lambda row, namespace: Spec())
    assert verified.extract_test_cmd({"instance_id": "x"}) == "pytest -rA tests/x.py"


def test_image_name_comes_from_the_harness(monkeypatch):
    seen = {}

    class Spec:
        instance_image_key = "starryzhang/sweb.eval.x86_64.textualize_1776_textual-5743:latest"

    def fake_spec(row, namespace):
        seen["args"] = (row["instance_id"], namespace)
        return Spec()

    monkeypatch.setattr("swebench.harness.test_spec.test_spec.make_test_spec", fake_spec)
    row = {"instance_id": "Textualize__textual-5743"}
    assert verified.swebench_live_image(row) == Spec.instance_image_key
    assert seen["args"] == ("Textualize__textual-5743", verified.SWEBENCH_LIVE_NAMESPACE)

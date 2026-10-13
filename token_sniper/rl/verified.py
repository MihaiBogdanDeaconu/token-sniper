from __future__ import annotations

import json
import logging
import subprocess
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:
    from token_sniper.agent.config import AgentModelConfig
    from token_sniper.agent.providers import ChatModelClient
    from token_sniper.evaluation.models import SWEbenchEvaluationAttempt, SWEbenchTaskMetadata

logger = logging.getLogger(__name__)

TESTBED = "/testbed"
SWEBENCH_LIVE_NAMESPACE = "starryzhang"

_NO_PATCH_ERRORS = frozenset({
    "Agent reached the maximum number of turns without completing.",
    "Agent completed but produced no changes (empty git diff).",
})


class VerifiedRolloutError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class NodeSpan:
    file_path: str
    start_line: int
    end_line: int
    node_type: str
    node_id: str


@dataclass(frozen=True)
class AgentRollout:
    patch_text: str | None
    """The agent's diff, or ``None`` when it produced no patch."""

    context_chars: int
    prompt_tokens: int
    completion_tokens: int
    wall_time_s: float
    session_log_path: str | None


@dataclass(frozen=True)
class VerifiedOutcome:
    instance_id: str
    rollout_id: str
    passed: bool
    """Every FAIL_TO_PASS test passed with the agent's patch."""

    resolved: bool
    """Harness verdict (FAIL_TO_PASS and PASS_TO_PASS); diagnostics only."""

    patch_produced: bool
    patch_applied: bool
    n_spans: int
    n_files: int
    context_chars: int
    prompt_tokens: int
    completion_tokens: int
    agent_wall_s: float
    harness_wall_s: float
    rollout_dir: str


class AgentBackend(Protocol):
    def run(
        self,
        *,
        metadata: SWEbenchTaskMetadata,
        spans: list[NodeSpan],
        rollout_id: str,
        output_dir: Path,
    ) -> AgentRollout: ...


class PatchEvaluator(Protocol):
    def evaluate_patch(
        self,
        *,
        work_dir: Path,
        instance_id: str,
        patch_file: Path,
        model_name: str,
        run_id: str,
    ) -> SWEbenchEvaluationAttempt: ...


def build_span_table(
    graph: Any,
    node_ids: list[str],
    candidate_files: list[str],
) -> list[NodeSpan | None]:
    from token_sniper.graph.schema import is_file_node_id

    candidate_set = set(candidate_files)
    table: list[NodeSpan | None] = []
    for nid in node_ids:
        if is_file_node_id(nid):
            table.append(None)
            continue
        if nid not in graph.nodes:
            raise KeyError(f"node {nid!r} of the task subgraph is not in the SSG")
        data = graph.nodes[nid]
        fp = data["file_path"]
        if fp not in candidate_set:
            table.append(None)
            continue
        table.append(NodeSpan(
            file_path=fp,
            start_line=int(data["start_line"]),
            end_line=int(data["end_line"]),
            node_type=str(data["node_type"]),
            node_id=nid,
        ))
    return table


def spans_for_mask(table: list[NodeSpan | None], actions: Any) -> list[NodeSpan]:
    if len(table) != len(actions):
        raise ValueError(f"span table has {len(table)} nodes, mask has {len(actions)}")
    return [
        span for span, keep in zip(table, actions.tolist(), strict=True)
        if keep > 0.5 and span is not None
    ]


def render_span_context(spans: list[NodeSpan], sources: dict[str, list[str]]) -> str:
    from token_sniper.evaluation.arms.context import Span, normalize, render

    shown = [Span(s.file_path, s.start_line, s.end_line, s.node_type) for s in spans]
    return render(normalize(shown, sources), sources)


def fail_to_pass_verdict(
    attempt: SWEbenchEvaluationAttempt,
    expected_fail_to_pass: list[str],
) -> bool:
    if not expected_fail_to_pass:
        raise ValueError("task has no FAIL_TO_PASS tests; nothing to verify")
    status = attempt.metadata.get("harness_status")
    if status != "reported":
        raise VerifiedRolloutError(
            f"harness gave no verdict (status={status!r}: "
            f"{attempt.metadata.get('evaluation_error')}); see {attempt.harness_stderr_path}"
        )
    if not attempt.patch_successfully_applied:
        return False
    expected = set(expected_fail_to_pass)
    reported = set(attempt.fail_to_pass_success) | set(attempt.fail_to_pass_failure)
    missing = expected - reported
    if missing:
        raise VerifiedRolloutError(
            f"harness report {attempt.instance_report_path} does not account for "
            f"{len(missing)} FAIL_TO_PASS test(s), e.g. {sorted(missing)[0]!r}"
        )
    return expected <= set(attempt.fail_to_pass_success)


def new_rollout_id() -> str:
    return uuid.uuid4().hex[:12]


def swebench_live_image(row: dict[str, Any], namespace: str = SWEBENCH_LIVE_NAMESPACE) -> str:
    from swebench.harness.test_spec.test_spec import make_test_spec

    return make_test_spec(row, namespace=namespace).instance_image_key


def extract_test_cmd(row: dict[str, Any], namespace: str = SWEBENCH_LIVE_NAMESPACE) -> str:
    from swebench.harness.test_spec.test_spec import make_test_spec

    lines = (make_test_spec(row, namespace=namespace).eval_script or "").splitlines()
    for i, line in enumerate(lines[:-1]):
        if "Start Test Output" in line:
            cmd = lines[i + 1].strip()
            if cmd and not cmd.startswith((":", "#")):
                return cmd
    raise ValueError(f"{row['instance_id']}: no test command after 'Start Test Output'")


class DockerAgentBackend:
    def __init__(
        self,
        *,
        client: ChatModelClient,
        model_config: AgentModelConfig,
        max_turns: int,
        dataset_name: str,
        split: str,
        namespace: str = SWEBENCH_LIVE_NAMESPACE,
    ) -> None:
        self._client = client
        self._model_config = model_config
        self._max_turns = max_turns
        self._dataset_name = dataset_name
        self._split = split
        self._namespace = namespace

    def run(
        self,
        *,
        metadata: SWEbenchTaskMetadata,
        spans: list[NodeSpan],
        rollout_id: str,
        output_dir: Path,
    ) -> AgentRollout:
        from token_sniper.agent.docker_tools import (
            DockerStartError,
            DockerToolExecutor,
            start_task_container,
            stop_task_container,
        )
        from token_sniper.agent.runner import AgentRunSettings, run_agent_session
        from token_sniper.evaluation.swebench import _load_dataset_rows

        instance_id = metadata.instance_id
        row = _load_dataset_rows(self._dataset_name, self._split)[instance_id]
        test_cmd = extract_test_cmd(row, self._namespace)
        try:
            container = start_task_container(
                f"{instance_id}.{rollout_id}", image_tag=swebench_live_image(row, self._namespace),
            )
        except DockerStartError as exc:
            raise VerifiedRolloutError(f"{instance_id}: {exc}: {exc.stderr.strip()}") from exc

        t0 = time.perf_counter()
        try:
            sources = {
                fp: _read_container_file(container, fp)
                for fp in sorted({s.file_path for s in spans})
            }
            context = render_span_context(spans, sources)
            settings = AgentRunSettings(
                repo_root=TESTBED,
                issue_description=metadata.problem_statement,
                output_dir=str(output_dir),
                mode="oracle",
                oracle_context=context,
                max_turns=self._max_turns,
                context_window=self._model_config.context_window,
                max_output_tokens=self._model_config.max_tokens,
                fail_to_pass_tests=metadata.fail_to_pass_tests,
                test_cmd=test_cmd,
            )
            executor = DockerToolExecutor(container, testbed=TESTBED)
            try:
                result = run_agent_session(settings, self._client, tool_executor=executor)
                patch_text: str | None = Path(result.patch_path).read_text(encoding="utf-8")
                prompt_tokens = result.total_prompt_tokens
                completion_tokens = result.total_completion_tokens
            except RuntimeError as exc:
                if str(exc) not in _NO_PATCH_ERRORS:
                    raise VerifiedRolloutError(f"{instance_id}: agent session failed: {exc!r}") from exc
                patch_text = None
                prompt_tokens, completion_tokens = _session_tokens(output_dir / "session.json")
            except Exception as exc:
                raise VerifiedRolloutError(f"{instance_id}: agent session failed: {exc!r}") from exc
            _require_running(container)
        finally:
            stop_task_container(container)

        return AgentRollout(
            patch_text=patch_text,
            context_chars=len(context),
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            wall_time_s=time.perf_counter() - t0,
            session_log_path=str(output_dir / "session.json"),
        )


def _read_container_file(container: str, file_path: str) -> list[str]:
    proc = subprocess.run(
        ["docker", "exec", container, "cat", f"{TESTBED}/{file_path}"],
        capture_output=True, check=False, timeout=60,
    )
    if proc.returncode != 0:
        raise VerifiedRolloutError(
            f"cannot read {file_path} in {container}: {proc.stderr.decode(errors='replace').strip()}"
        )
    from token_sniper.evaluation.arms.context import source_rows

    return source_rows(proc.stdout)


def _require_running(container: str) -> None:
    proc = subprocess.run(
        ["docker", "inspect", "-f", "{{.State.Running}}", container],
        capture_output=True, text=True, check=False, timeout=30,
    )
    if proc.returncode != 0 or proc.stdout.strip() != "true":
        raise VerifiedRolloutError(
            f"container {container} is not running after the agent session: "
            f"{(proc.stderr or proc.stdout).strip()}"
        )


def _session_tokens(session_log: Path) -> tuple[int, int]:
    data = json.loads(session_log.read_text(encoding="utf-8"))
    return int(data["total_prompt_tokens"]), int(data["total_completion_tokens"])

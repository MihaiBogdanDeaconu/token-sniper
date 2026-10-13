from __future__ import annotations

import json
import logging
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

from token_sniper.rl.verified import (
    NodeSpan,
    VerifiedOutcome,
    build_span_table,
    fail_to_pass_verdict,
    new_rollout_id,
    spans_for_mask,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    import torch

    from token_sniper.rl.verified import AgentBackend, PatchEvaluator
    from token_sniper.vgib_selector.dataset import RerankedVGIBTask

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RewardSignal:
    task_reward: float
    """Primary scalar reward (higher is better context selection)."""

    recall: float
    """Fraction of gold nodes retained by the action mask."""

    token_ratio: float
    """Fraction of total tokens retained (0 = total compression)."""

    trace_connectivity: float
    """Fraction of trace edges with both endpoints retained."""

    source: str
    """Origin of this reward: 'proxy' or 'verified'."""


class RewardFunction(Protocol):
    def compute(
        self,
        task: RerankedVGIBTask,
        actions: torch.Tensor,
    ) -> RewardSignal: ...

    def compute_many(
        self,
        items: Sequence[tuple[RerankedVGIBTask, torch.Tensor]],
    ) -> list[RewardSignal]:
        ...


def recall_and_token_ratio(task: RerankedVGIBTask, actions: torch.Tensor) -> tuple[float, float]:
    if task.gold_node_mask is None:
        raise ValueError(f"{task.instance_id}: no gold_node_mask; RL needs gold node labels")
    gold_mask = task.gold_node_mask.to(actions.device) > 0.5
    if not gold_mask.any():
        raise ValueError(f"{task.instance_id}: no gold node in this subgraph")
    if task.token_counts is None:
        raise ValueError(f"{task.instance_id}: no token_counts; RL needs the token sidecar")
    recall = (actions > 0.5)[gold_mask].float().mean().item()
    token_counts = task.token_counts.to(actions.device)
    retained = (actions.float() * token_counts).sum()
    token_ratio = (retained / token_counts.sum().clamp(min=1.0)).item()
    return recall, token_ratio


class ProxyReward:
    def __init__(
        self,
        beta: float = 0.5,
        trace_weight: float = 0.1,
        recall_bonus: float = 0.5,
    ) -> None:
        self.beta = beta
        self.trace_weight = trace_weight
        self.recall_bonus = recall_bonus

    def compute(
        self,
        task: RerankedVGIBTask,
        actions: torch.Tensor,
    ) -> RewardSignal:
        actions_bool = actions > 0.5

        recall, token_ratio = recall_and_token_ratio(task, actions)

        trace_connectivity = 0.0
        if (
            hasattr(task, "trace_edge_mask")
            and task.trace_edge_mask is not None
            and task.trace_edge_mask.numel() > 0
        ):
            trace_edges = task.trace_edge_mask.to(actions.device)
            src_retained = actions_bool[trace_edges[:, 0]]
            dst_retained = actions_bool[trace_edges[:, 1]]
            both_retained = (src_retained & dst_retained).float()
            trace_connectivity = both_retained.mean().item()

        reward = recall - self.beta * token_ratio

        reward += self.trace_weight * trace_connectivity

        reward += self.recall_bonus * recall * (1.0 - token_ratio)

        return RewardSignal(
            task_reward=reward,
            recall=recall,
            token_ratio=token_ratio,
            trace_connectivity=trace_connectivity,
            source="proxy",
        )

    def compute_many(
        self,
        items: Sequence[tuple[RerankedVGIBTask, torch.Tensor]],
    ) -> list[RewardSignal]:
        return [self.compute(task, actions) for task, actions in items]


class LLMJudgeReward:
    def __init__(
        self,
        *,
        agent_backend: AgentBackend,
        harness: PatchEvaluator,
        model_name: str,
        ssg_dir: Path,
        base_commits: dict[str, str],
        work_dir: Path,
        dataset_name: str,
        split: str,
        max_workers: int = 8,
        token_beta: float = 0.0,
    ) -> None:
        if max_workers < 1:
            raise ValueError("max_workers must be >= 1")
        self._agent = agent_backend
        self._harness = harness
        self._model_name = model_name
        self._ssg_dir = Path(ssg_dir)
        self._base_commits = base_commits
        self._work_dir = Path(work_dir)
        self._dataset_name = dataset_name
        self._split = split
        self._max_workers = max_workers
        self.token_beta = token_beta
        self._span_tables: dict[str, list[NodeSpan | None]] = {}

    def compute(
        self,
        task: RerankedVGIBTask,
        actions: torch.Tensor,
    ) -> RewardSignal:
        return self.compute_many([(task, actions)])[0]

    def compute_many(
        self,
        items: Sequence[tuple[RerankedVGIBTask, torch.Tensor]],
    ) -> list[RewardSignal]:
        outcomes = self.run_many(items)
        signals = []
        for (task, actions), outcome in zip(items, outcomes, strict=True):
            recall, token_ratio = recall_and_token_ratio(task, actions)
            signals.append(RewardSignal(
                task_reward=float(outcome.passed) - self.token_beta * token_ratio,
                recall=recall,
                token_ratio=token_ratio,
                trace_connectivity=0.0,
                source="verified",
            ))
        return signals

    def run_many(
        self,
        items: Sequence[tuple[RerankedVGIBTask, torch.Tensor]],
    ) -> list[VerifiedOutcome]:
        jobs = [(task, spans_for_mask(self._span_table(task), actions.detach().cpu()))
                for task, actions in items]
        if self._max_workers == 1 or len(jobs) == 1:
            return [self._run_one(task, spans) for task, spans in jobs]
        with ThreadPoolExecutor(max_workers=min(self._max_workers, len(jobs))) as pool:
            futures = [pool.submit(self._run_one, task, spans) for task, spans in jobs]
            try:
                return [f.result() for f in futures]
            except BaseException:
                for f in futures:
                    f.cancel()
                raise

    def _span_table(self, task: RerankedVGIBTask) -> list[NodeSpan | None]:
        table = self._span_tables.get(task.instance_id)
        if table is None:
            from token_sniper.graph.io import read_ssg_json

            commit = self._base_commits[task.instance_id]
            ssg = self._ssg_dir / task.repo.replace("/", "__") / commit[:12] / "ssg.json"
            table = build_span_table(read_ssg_json(ssg), task.features.node_ids, task.candidate_files)
            self._span_tables[task.instance_id] = table
        return table

    def _run_one(self, task: RerankedVGIBTask, spans: list[NodeSpan]) -> VerifiedOutcome:
        import time

        from token_sniper.evaluation.swebench import load_swebench_task_metadata

        metadata = load_swebench_task_metadata(
            task.instance_id, dataset_name=self._dataset_name, split=self._split,
        )
        if not metadata.fail_to_pass_tests:
            raise ValueError(f"{task.instance_id}: no FAIL_TO_PASS tests; cannot verify")
        rollout_id = new_rollout_id()
        rollout_dir = self._work_dir / task.instance_id / rollout_id
        rollout_dir.mkdir(parents=True, exist_ok=False)

        agent = self._agent.run(
            metadata=metadata, spans=spans, rollout_id=rollout_id, output_dir=rollout_dir / "agent",
        )

        passed = resolved = applied = False
        harness_wall = 0.0
        if agent.patch_text is not None:
            patch_file = rollout_dir / "patch.diff"
            patch_file.write_text(agent.patch_text, encoding="utf-8")
            t0 = time.perf_counter()
            attempt = self._harness.evaluate_patch(
                work_dir=rollout_dir / "harness",
                instance_id=task.instance_id,
                patch_file=patch_file,
                model_name=self._model_name,
                run_id=f"rl-{rollout_id}",
            )
            harness_wall = time.perf_counter() - t0
            passed = fail_to_pass_verdict(attempt, metadata.fail_to_pass_tests)
            resolved = attempt.resolved
            applied = attempt.patch_successfully_applied

        outcome = VerifiedOutcome(
            instance_id=task.instance_id,
            rollout_id=rollout_id,
            passed=passed,
            resolved=resolved,
            patch_produced=agent.patch_text is not None,
            patch_applied=applied,
            n_spans=len(spans),
            n_files=len({s.file_path for s in spans}),
            context_chars=agent.context_chars,
            prompt_tokens=agent.prompt_tokens,
            completion_tokens=agent.completion_tokens,
            agent_wall_s=agent.wall_time_s,
            harness_wall_s=harness_wall,
            rollout_dir=str(rollout_dir),
        )
        (rollout_dir / "outcome.json").write_text(
            json.dumps(asdict(outcome), indent=2), encoding="utf-8",
        )
        logger.info(
            "verified %s/%s: passed=%s patch=%s spans=%d files=%d agent=%.0fs harness=%.0fs",
            task.instance_id, rollout_id, passed, outcome.patch_produced,
            outcome.n_spans, outcome.n_files, agent.wall_time_s, harness_wall,
        )
        return outcome

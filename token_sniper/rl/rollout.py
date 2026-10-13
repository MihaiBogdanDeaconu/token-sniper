from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from collections.abc import Sequence

    from token_sniper.rl.policy import VGIBPolicy
    from token_sniper.rl.rewards import RewardFunction, RewardSignal
    from token_sniper.vgib_selector.dataset import RerankedVGIBTask

logger = logging.getLogger(__name__)


@dataclass
class Trajectory:
    instance_id: str
    """Task identifier for grouping."""

    actions: torch.Tensor
    """[N] binary retention mask (detached)."""

    log_prob: float
    """log π(a|s) under the *behavior* policy at collection time."""

    reward: float
    """Scalar reward r(s, a) from the reward function."""

    reward_signal: RewardSignal | None = None
    """Full reward breakdown for diagnostics."""

    retention_probs: torch.Tensor | None = None
    """[N] p_v at collection time (for KL computation, detached)."""

    advantage: float = 0.0
    """GRPO-normalised advantage (set post-hoc)."""


@dataclass
class RolloutBuffer:
    group_size: int = 8

    _groups: dict[str, list[Trajectory]] = field(
        default_factory=lambda: defaultdict(list),
        repr=False,
    )

    def clear(self) -> None:
        self._groups.clear()

    @property
    def all_trajectories(self) -> list[Trajectory]:
        out: list[Trajectory] = []
        for group in self._groups.values():
            out.extend(group)
        return out

    @property
    def num_groups(self) -> int:
        return len(self._groups)

    @property
    def num_trajectories(self) -> int:
        return sum(len(g) for g in self._groups.values())

    def collect_group(
        self,
        policy: VGIBPolicy,
        task: RerankedVGIBTask,
        reward_fn: RewardFunction,
        device: torch.device,
        rollout_temperature: float = 2.0,
    ) -> list[Trajectory]:
        return self.collect_groups(
            policy, [task], reward_fn, device, rollout_temperature=rollout_temperature,
        )[0]

    def collect_groups(
        self,
        policy: VGIBPolicy,
        tasks: Sequence[RerankedVGIBTask],
        reward_fn: RewardFunction,
        device: torch.device,
        rollout_temperature: float = 2.0,
    ) -> list[list[Trajectory]]:
        saved_sampling_temperature = policy.sampling_temperature
        policy.set_sampling_temperature(rollout_temperature)
        policy.train()

        samples: list[tuple[RerankedVGIBTask, torch.Tensor, float, torch.Tensor]] = []
        try:
            with torch.no_grad():
                for task in tasks:
                    for _ in range(self.group_size):
                        output = policy(task, device)
                        samples.append((
                            task,
                            output.actions.detach().cpu(),
                            output.log_prob.detach().item(),
                            output.retention_probs.detach().cpu(),
                        ))
        finally:
            policy.set_sampling_temperature(saved_sampling_temperature)

        signals = reward_fn.compute_many([(task, actions) for task, actions, _, _ in samples])

        groups: list[list[Trajectory]] = []
        for i, task in enumerate(tasks):
            lo = i * self.group_size
            trajectories = [
                Trajectory(
                    instance_id=task.instance_id,
                    actions=actions,
                    log_prob=log_prob,
                    reward=signal.task_reward,
                    reward_signal=signal,
                    retention_probs=probs,
                )
                for (_, actions, log_prob, probs), signal in zip(
                    samples[lo:lo + self.group_size], signals[lo:lo + self.group_size], strict=True,
                )
            ]
            _compute_group_advantages(trajectories)
            self._groups[task.instance_id] = trajectories
            groups.append(trajectories)
        return groups

    def get_group(self, instance_id: str) -> list[Trajectory]:
        return self._groups.get(instance_id, [])

    def mean_reward(self) -> float:
        all_rewards = [t.reward for t in self.all_trajectories]
        if not all_rewards:
            return 0.0
        return sum(all_rewards) / len(all_rewards)

    def mean_recall(self) -> float:
        recalls = [
            t.reward_signal.recall
            for t in self.all_trajectories
            if t.reward_signal is not None
        ]
        if not recalls:
            return 0.0
        return sum(recalls) / len(recalls)

    def mean_token_ratio(self) -> float:
        ratios = [
            t.reward_signal.token_ratio
            for t in self.all_trajectories
            if t.reward_signal is not None
        ]
        if not ratios:
            return 0.0
        return sum(ratios) / len(ratios)

    def mean_group_std(self) -> float:
        stds = [group_reward_std(g) for g in self._groups.values() if len(g) > 1]
        if not stds:
            return 0.0
        return sum(stds) / len(stds)

    def stats(self) -> dict[str, float]:
        all_t = self.all_trajectories
        if not all_t:
            return {
                "mean_reward": 0.0,
                "std_reward": 0.0,
                "mean_recall": 0.0,
                "mean_token_ratio": 0.0,
                "mean_group_std": 0.0,
                "num_trajectories": 0,
                "num_groups": 0,
            }
        rewards = [t.reward for t in all_t]
        mean_r = sum(rewards) / len(rewards)
        var_r = sum((r - mean_r) ** 2 for r in rewards) / max(len(rewards) - 1, 1)
        return {
            "mean_reward": mean_r,
            "std_reward": var_r ** 0.5,
            "mean_recall": self.mean_recall(),
            "mean_token_ratio": self.mean_token_ratio(),
            "mean_group_std": self.mean_group_std(),
            "num_trajectories": len(all_t),
            "num_groups": self.num_groups,
        }


def group_reward_std(group: list[Trajectory]) -> float:
    return torch.tensor([t.reward for t in group], dtype=torch.float64).std().item()


def _compute_group_advantages(group: list[Trajectory]) -> None:
    if len(group) <= 1:
        for t in group:
            t.advantage = 0.0
        return

    rewards = torch.tensor([t.reward for t in group], dtype=torch.float64)
    mean_r = rewards.mean()
    std_r = rewards.std().clamp(min=1e-6)

    for i, t in enumerate(group):
        t.advantage = ((rewards[i] - mean_r) / std_r).item()

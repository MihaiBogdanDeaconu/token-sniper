from __future__ import annotations

import pytest
import torch

from token_sniper.rl.rollout import RolloutBuffer, Trajectory, _compute_group_advantages


class TestGRPOAdvantages:
    def test_advantages_sum_to_zero(self):
        trajectories = [
            Trajectory(instance_id="t1", actions=torch.zeros(5),
                       log_prob=-1.0, reward=r)
            for r in [0.5, 0.7, 0.3, 0.9, 0.1]
        ]

        _compute_group_advantages(trajectories)

        advantages = [t.advantage for t in trajectories]
        assert abs(sum(advantages)) < 1e-6, f"Sum of advantages: {sum(advantages)}"

    def test_highest_reward_gets_positive_advantage(self):
        trajectories = [
            Trajectory(instance_id="t1", actions=torch.zeros(5),
                       log_prob=-1.0, reward=r)
            for r in [0.1, 0.2, 0.9, 0.3]
        ]

        _compute_group_advantages(trajectories)

        assert trajectories[2].advantage > 0
        assert trajectories[0].advantage < 0

    def test_single_trajectory_gets_zero_advantage(self):
        trajectories = [
            Trajectory(instance_id="t1", actions=torch.zeros(5),
                       log_prob=-1.0, reward=0.5)
        ]

        _compute_group_advantages(trajectories)

        assert trajectories[0].advantage == 0.0

    def test_equal_rewards_get_zero_advantages(self):
        trajectories = [
            Trajectory(instance_id="t1", actions=torch.zeros(5),
                       log_prob=-1.0, reward=0.5)
            for _ in range(4)
        ]

        _compute_group_advantages(trajectories)

        for t in trajectories:
            assert abs(t.advantage) < 1e-6

    def test_advantages_are_normalised(self):
        import math

        trajectories = [
            Trajectory(instance_id="t1", actions=torch.zeros(5),
                       log_prob=-1.0, reward=r)
            for r in [0.0, 0.25, 0.5, 0.75, 1.0]
        ]

        _compute_group_advantages(trajectories)

        advantages = [t.advantage for t in trajectories]
        mean_a = sum(advantages) / len(advantages)
        var_a = sum((a - mean_a) ** 2 for a in advantages) / (len(advantages) - 1)
        std_a = math.sqrt(var_a)

        assert abs(std_a - 1.0) < 0.1, f"Std of advantages: {std_a}"


class TestRolloutBuffer:
    def test_empty_buffer_stats(self):
        buffer = RolloutBuffer(group_size=4)
        stats = buffer.stats()

        assert stats["num_trajectories"] == 0
        assert stats["num_groups"] == 0
        assert stats["mean_reward"] == 0.0

    def test_clear_empties_buffer(self):
        buffer = RolloutBuffer(group_size=4)
        buffer._groups["t1"] = [
            Trajectory(instance_id="t1", actions=torch.zeros(5),
                       log_prob=-1.0, reward=0.5)
        ]

        assert buffer.num_trajectories == 1
        buffer.clear()
        assert buffer.num_trajectories == 0

    def test_collect_group_produces_correct_count(self):
        from token_sniper.rl.policy import VGIBPolicy
        from token_sniper.rl.rewards import ProxyReward
        from token_sniper.vgib_selector.backbone import GraphSAGEConfig
        from token_sniper.vgib_selector.selector import VGIBSelector

        cfg = GraphSAGEConfig(
            num_files=3, source_emb_dim=32, file_emb_dim=8,
            hidden_dim=64, out_dim=32, num_sage_layers=2,
            dropout=0.0, task_emb_dim=32,
        )
        selector = VGIBSelector(cfg)
        policy = VGIBPolicy(selector, temperature=1.0)

        from tests.rl.test_rewards import _make_task
        task = _make_task(num_nodes=10, num_files=3, num_gold=1)

        reward_fn = ProxyReward(beta=0.3, trace_weight=0.0)
        buffer = RolloutBuffer(group_size=4)

        group = buffer.collect_group(policy, task, reward_fn, torch.device("cpu"))

        assert len(group) == 4
        assert buffer.num_trajectories == 4
        assert buffer.num_groups == 1

        for t in group:
            assert t.reward_signal is not None
            assert isinstance(t.advantage, float)

    def test_mean_reward(self):
        buffer = RolloutBuffer(group_size=2)
        buffer._groups["t1"] = [
            Trajectory(instance_id="t1", actions=torch.zeros(5),
                       log_prob=-1.0, reward=0.4),
            Trajectory(instance_id="t1", actions=torch.zeros(5),
                       log_prob=-1.0, reward=0.6),
        ]

        assert buffer.mean_reward() == pytest.approx(0.5, abs=1e-6)


class _CountingReward:
    def __init__(self):
        from token_sniper.rl.rewards import ProxyReward

        self.proxy = ProxyReward()
        self.batches: list[int] = []

    def compute(self, task, actions):
        raise AssertionError("collect_group must score the group with compute_many")

    def compute_many(self, items):
        self.batches.append(len(items))
        return self.proxy.compute_many(items)


def test_collect_group_scores_whole_group_in_one_call():
    from tests.rl.test_rewards import _make_task
    from tests.rl.test_trainer import _make_selector
    from token_sniper.rl.policy import VGIBPolicy

    reward_fn = _CountingReward()
    buffer = RolloutBuffer(group_size=5)
    buffer.collect_group(VGIBPolicy(_make_selector()), _make_task(num_nodes=8),
                         reward_fn, torch.device("cpu"))
    assert reward_fn.batches == [5]


def test_mean_group_std_ignores_between_task_spread():
    buffer = RolloutBuffer(group_size=2)
    buffer._groups["a"] = [Trajectory("a", torch.zeros(1), 0.0, 0.1),
                           Trajectory("a", torch.zeros(1), 0.0, 0.1)]
    buffer._groups["b"] = [Trajectory("b", torch.zeros(1), 0.0, 0.9),
                           Trajectory("b", torch.zeros(1), 0.0, 0.9)]
    assert buffer.mean_group_std() == 0.0
    assert buffer.stats()["std_reward"] > 0.4
    buffer._groups["b"][1].reward = 0.7
    assert buffer.mean_group_std() == pytest.approx(((0.2 ** 2 / 2) ** 0.5) / 2)


def test_collect_groups_one_call_across_tasks():
    from tests.rl.test_rewards import _make_task
    from tests.rl.test_trainer import _make_selector
    from token_sniper.rl.policy import VGIBPolicy

    tasks = [_make_task(num_nodes=8) for _ in range(3)]
    for i, t in enumerate(tasks):
        t.instance_id = f"t{i}"
    reward_fn = _CountingReward()
    buffer = RolloutBuffer(group_size=4)
    groups = buffer.collect_groups(VGIBPolicy(_make_selector()), tasks, reward_fn, torch.device("cpu"))
    assert reward_fn.batches == [12]
    assert [g[0].instance_id for g in groups] == ["t0", "t1", "t2"]
    assert all(len(g) == 4 for g in groups) and buffer.num_groups == 3
    for g in groups:
        assert sum(t.advantage for t in g) == pytest.approx(0.0, abs=1e-6)

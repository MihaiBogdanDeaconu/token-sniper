from __future__ import annotations

import pytest
import torch

from token_sniper.rl.rewards import ProxyReward, RewardSignal


def _make_task(
    num_nodes: int = 10,
    num_files: int = 3,
    num_gold: int = 1,
    with_trace: bool = False,
):
    from token_sniper.vgib_selector.dataset import RerankedVGIBTask
    from token_sniper.vgib_selector.features import VGIBFeatureBundle

    node_ids = [f"module.func_{i}" for i in range(num_nodes)]
    features = VGIBFeatureBundle(
        node_ids=node_ids,
        type_onehot=torch.zeros(num_nodes, 5),
        signature_emb=torch.randn(num_nodes, 32),
        degree_log=torch.rand(num_nodes),
        clustering=torch.rand(num_nodes),
        file_idx=torch.randint(0, num_files, (num_nodes,)),
        file_vocab={f"file_{i}.py": i for i in range(num_files)},
    )

    edge_index = torch.zeros(2, 0, dtype=torch.long)
    task_embedding = torch.randn(32)

    gold_file_mask = torch.zeros(num_files)
    for i in range(min(num_gold, num_files)):
        gold_file_mask[i] = 1.0

    node_to_file = torch.zeros(num_nodes, dtype=torch.long)
    for i in range(num_nodes):
        node_to_file[i] = i % num_files

    token_counts = torch.ones(num_nodes) * 100.0
    gold_node_mask = (node_to_file < num_gold).float()

    trace_edge_mask = None
    if with_trace and num_nodes >= 4:
        trace_edge_mask = torch.tensor([[0, 1], [2, 3]], dtype=torch.long)

    return RerankedVGIBTask(
        instance_id="test-reward",
        repo="test/repo",
        gold_files=[f"file_{i}.py" for i in range(num_gold)],
        candidate_files=[f"file_{i}.py" for i in range(num_files)],
        features=features,
        edge_index=edge_index,
        task_embedding=task_embedding,
        gold_file_mask=gold_file_mask,
        node_to_file=node_to_file,
        token_counts=token_counts,
        gold_node_mask=gold_node_mask,
        trace_edge_mask=trace_edge_mask,
    )


class TestProxyReward:
    def test_all_retain_gives_high_recall(self):
        reward_fn = ProxyReward(beta=0.3, trace_weight=0.0)
        task = _make_task(num_nodes=10, num_files=3, num_gold=1)

        actions = torch.ones(10)
        signal = reward_fn.compute(task, actions)

        assert isinstance(signal, RewardSignal)
        assert signal.recall == 1.0
        assert signal.token_ratio == pytest.approx(1.0, abs=1e-5)
        assert signal.source == "proxy"

    def test_all_drop_gives_zero_recall(self):
        reward_fn = ProxyReward(beta=0.3, trace_weight=0.0)
        task = _make_task(num_nodes=10, num_files=3, num_gold=1)

        actions = torch.zeros(10)
        signal = reward_fn.compute(task, actions)

        assert signal.recall == 0.0
        assert signal.token_ratio == 0.0

    def test_retain_all_reward_greater_than_drop_all(self):
        reward_fn = ProxyReward(beta=0.1, trace_weight=0.0)
        task = _make_task(num_nodes=10, num_files=3, num_gold=1)

        retain_all = reward_fn.compute(task, torch.ones(10))
        drop_all = reward_fn.compute(task, torch.zeros(10))

        assert retain_all.task_reward > drop_all.task_reward

    def test_compression_penalised(self):
        task = _make_task(num_nodes=10, num_files=3, num_gold=1)
        actions = torch.ones(10)

        low_beta = ProxyReward(beta=0.1, trace_weight=0.0)
        high_beta = ProxyReward(beta=1.0, trace_weight=0.0)

        r_low = low_beta.compute(task, actions)
        r_high = high_beta.compute(task, actions)

        assert r_low.task_reward > r_high.task_reward

    def test_perfect_recall_bonus(self):
        reward_fn = ProxyReward(beta=0.0, trace_weight=0.0, recall_bonus=0.5)
        task = _make_task(num_nodes=10, num_files=3, num_gold=1)

        actions = torch.ones(10)
        signal = reward_fn.compute(task, actions)

        assert signal.task_reward == pytest.approx(1.0, abs=0.01)

    def test_recall_bonus_increases_with_compression(self):
        reward_fn = ProxyReward(beta=0.0, trace_weight=0.0, recall_bonus=0.5)
        task = _make_task(num_nodes=10, num_files=3, num_gold=1)

        r_all = reward_fn.compute(task, torch.ones(10))

        partial = torch.zeros(10)
        partial[task.gold_node_mask > 0.5] = 1.0
        r_partial = reward_fn.compute(task, partial)

        assert r_partial.task_reward > r_all.task_reward

    def test_trace_connectivity_with_all_retained(self):
        reward_fn = ProxyReward(beta=0.0, trace_weight=0.1)
        task = _make_task(num_nodes=10, num_files=3, num_gold=1, with_trace=True)

        actions = torch.ones(10)
        signal = reward_fn.compute(task, actions)

        assert signal.trace_connectivity == 1.0

    def test_trace_connectivity_with_none_retained(self):
        reward_fn = ProxyReward(beta=0.0, trace_weight=0.1)
        task = _make_task(num_nodes=10, num_files=3, num_gold=1, with_trace=True)

        actions = torch.zeros(10)
        signal = reward_fn.compute(task, actions)

        assert signal.trace_connectivity == 0.0

    def test_no_trace_edges_gives_zero_connectivity(self):
        reward_fn = ProxyReward(beta=0.0, trace_weight=0.1)
        task = _make_task(num_nodes=10, num_files=3, num_gold=1, with_trace=False)

        actions = torch.ones(10)
        signal = reward_fn.compute(task, actions)

        assert signal.trace_connectivity == 0.0

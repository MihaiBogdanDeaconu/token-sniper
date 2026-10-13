from __future__ import annotations

import torch

from token_sniper.rl.policy import PolicyOutput, VGIBPolicy
from token_sniper.vgib_selector.backbone import GraphSAGEConfig
from token_sniper.vgib_selector.selector import VGIBSelector


def _make_selector(num_nodes: int = 10, num_files: int = 3) -> VGIBSelector:
    cfg = GraphSAGEConfig(
        num_files=num_files,
        source_emb_dim=32,
        file_emb_dim=8,
        hidden_dim=64,
        out_dim=32,
        num_sage_layers=2,
        dropout=0.0,
        task_emb_dim=32,
    )
    return VGIBSelector(cfg)


def _make_task(num_nodes: int = 10, num_files: int = 3, num_gold: int = 1):
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

    for i in range(num_nodes):
        features.type_onehot[i, i % 5] = 1.0

    src = list(range(num_nodes - 1))
    dst = list(range(1, num_nodes))
    edge_index = torch.tensor([src + dst, dst + src], dtype=torch.long)

    task_embedding = torch.randn(32)

    gold_file_mask = torch.zeros(num_files)
    for i in range(min(num_gold, num_files)):
        gold_file_mask[i] = 1.0

    node_to_file = torch.randint(0, num_files, (num_nodes,))
    token_counts = torch.ones(num_nodes) * 100.0
    gold_node_mask = (node_to_file < num_gold).float()

    from token_sniper.vgib_selector.dataset import RerankedVGIBTask

    return RerankedVGIBTask(
        instance_id="test-001",
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
    )


class TestVGIBPolicy:
    def test_act_returns_policy_output(self):
        selector = _make_selector()
        policy = VGIBPolicy(selector, temperature=1.0)
        task = _make_task()

        policy.train()
        output = policy(task, torch.device("cpu"))

        assert isinstance(output, PolicyOutput)
        assert output.actions.shape == (10,)
        assert output.log_prob.shape == ()
        assert output.retention_probs.shape == (10,)
        assert output.entropy.shape == ()
        assert output.node_embeddings.shape == (10, 32)

    def test_actions_are_binary_in_train_mode(self):
        selector = _make_selector()
        policy = VGIBPolicy(selector, temperature=1.0)
        task = _make_task()

        policy.train()
        output = policy(task, torch.device("cpu"))

        unique_vals = output.actions.unique()
        for v in unique_vals:
            assert v.item() in (0.0, 1.0), f"Got non-binary action: {v}"

    def test_actions_are_deterministic_in_eval_mode(self):
        selector = _make_selector()
        policy = VGIBPolicy(selector, temperature=1.0)
        task = _make_task()

        policy.eval()
        with torch.no_grad():
            output1 = policy(task, torch.device("cpu"))
            output2 = policy(task, torch.device("cpu"))

        torch.testing.assert_close(output1.actions, output2.actions)

    def test_log_prob_is_differentiable(self):
        selector = _make_selector()
        policy = VGIBPolicy(selector, temperature=1.0)
        task = _make_task()

        policy.train()
        output = policy(task, torch.device("cpu"))

        output.log_prob.backward()

        has_grads = any(
            p.grad is not None and p.grad.abs().sum() > 0
            for p in policy.parameters()
        )
        assert has_grads, "No gradients flowed through log_prob"

    def test_entropy_is_non_negative(self):
        selector = _make_selector()
        policy = VGIBPolicy(selector, temperature=1.0)
        task = _make_task()

        policy.train()
        output = policy(task, torch.device("cpu"))

        assert output.entropy.item() >= 0.0

    def test_retention_probs_are_in_01(self):
        selector = _make_selector()
        policy = VGIBPolicy(selector, temperature=1.0)
        task = _make_task()

        policy.train()
        output = policy(task, torch.device("cpu"))

        assert (output.retention_probs >= 0.0).all()
        assert (output.retention_probs <= 1.0).all()

    def test_temperature_setting(self):
        selector = _make_selector()
        policy = VGIBPolicy(selector, temperature=1.0)

        policy.set_temperature(0.5)
        assert policy.temperature == 0.5

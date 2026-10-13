from __future__ import annotations

import pytest
import torch

from scripts.rl_reward_correlation import _auc, summarize
from scripts.rl_tau_variance import tau_variance_rows
from tests.rl.test_rewards import _make_task
from tests.rl.test_trainer import _make_selector
from token_sniper.rl.policy import VGIBPolicy
from token_sniper.rl.rewards import ProxyReward


def _row(iid, idx, proxy, verified, recall=0.5):
    return {"instance_id": iid, "rollout_idx": str(idx), "proxy_reward": str(proxy),
            "recall": str(recall), "verified": str(verified), "patch_produced": "True"}


def test_auc():
    assert _auc([0.9, 0.8], [0.1, 0.2]) == 1.0
    assert _auc([0.5], [0.5]) == 0.5


def test_summarize_aligned_proxy():
    rows = [_row("a", 0, 0.9, 1), _row("a", 1, 0.2, 0), _row("b", 0, 0.8, 1),
            _row("b", 1, 0.7, 1), _row("c", 0, 0.1, 0), _row("c", 1, 0.3, 0)]
    s = summarize(rows)
    assert s["n_rollouts"] == 6 and s["n_tasks"] == 3
    assert s["pass_rate"] == pytest.approx(0.5)
    assert s["proxy_auc"] == 1.0
    assert s["proxy_pointbiserial_r"] > 0.8
    assert (s["groups_all_pass"], s["groups_all_fail"], s["groups_mixed"]) == (1, 1, 1)
    assert s["frac_groups_with_verified_signal"] == pytest.approx(1 / 3)
    assert s["within_group_proxy_auc_mean"] == 1.0
    assert "recall_auc" not in s and "recall_note" in s


def test_summarize_constant_outcome_has_no_correlation():
    s = summarize([_row("a", 0, 0.9, 0), _row("a", 1, 0.2, 0)])
    assert "proxy_auc" not in s and "note" in s
    assert s["within_group_proxy_auc_mean"] is None


def test_tau_variance_rows():
    policy = VGIBPolicy(_make_selector())
    tasks = [_make_task(num_nodes=8) for _ in range(3)]
    for i, t in enumerate(tasks):
        t.instance_id = f"t{i}"
    per_task, summary = tau_variance_rows(
        policy, tasks, taus=[1.0, 3.0], group_size=4, reward_fn=ProxyReward(),
        device=torch.device("cpu"), seed=0,
    )
    assert len(per_task) == 6 and [r["tau_s"] for r in summary] == [1.0, 3.0]
    for r in summary:
        assert r["n_tasks"] == 3 and 0.0 <= r["frac_zero_std_groups"] <= 1.0
        assert 1.0 <= r["mean_distinct_masks"] <= 4.0


def test_tau_variance_saturated_policy_needs_temperature():
    selector = _make_selector()
    policy = VGIBPolicy(selector)
    task = _make_task(num_nodes=8)

    def saturate(module, inputs, output):
        output["retention_probs"] = torch.full_like(output["retention_probs"], 0.9999)
        return output

    selector.register_forward_hook(saturate)
    _, summary = tau_variance_rows(
        policy, [task], taus=[1.0, 50.0], group_size=8, reward_fn=ProxyReward(),
        device=torch.device("cpu"), seed=0,
    )
    assert summary[0]["frac_zero_std_groups"] == 1.0
    assert summary[1]["mean_distinct_masks"] > 1

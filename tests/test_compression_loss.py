from __future__ import annotations

import pytest
import torch

from token_sniper.vgib_selector.loss import (
    RecallConstrainedTokenLoss,
    bernoulli_kl,
    structural_prior,
)
from token_sniper.vgib_selector.train import gold_rank_percentile, gold_within_file_topk


def _inputs(p_v: list[float], gold: list[float], tokens: list[float]):
    n = len(p_v)
    return {
        "retention_probs": torch.tensor(p_v),
        "gold_node_mask": torch.tensor(gold),
        "token_counts": torch.tensor(tokens),
        "lambda_soft": torch.tensor(p_v),
        "node_embeddings": torch.randn(n, 8),
    }


def test_recall_is_the_mean_over_gold_nodes_not_their_file():
    loss_fn = RecallConstrainedTokenLoss(beta=0.0, gamma=0.0)

    one_of_two = loss_fn(**_inputs(p_v=[1.0, 0.0, 0.0], gold=[1.0, 1.0, 0.0], tokens=[10, 10, 10]))
    both = loss_fn(**_inputs(p_v=[1.0, 1.0, 0.0], gold=[1.0, 1.0, 0.0], tokens=[10, 10, 10]))

    assert one_of_two["recall"].item() == pytest.approx(0.5)
    assert both["recall"].item() == pytest.approx(1.0)
    assert both["total"].item() < one_of_two["total"].item()


def test_dropping_a_non_gold_node_is_always_rewarded():
    loss_fn = RecallConstrainedTokenLoss(beta=10.0, gamma=0.0)
    kept = loss_fn(**_inputs(p_v=[1.0, 1.0], gold=[1.0, 0.0], tokens=[10, 90]))
    dropped = loss_fn(**_inputs(p_v=[1.0, 0.0], gold=[1.0, 0.0], tokens=[10, 90]))

    assert dropped["token_ratio"].item() == pytest.approx(0.1)
    assert dropped["total"].item() < kept["total"].item()


def test_a_subgraph_without_gold_nodes_raises():
    loss_fn = RecallConstrainedTokenLoss()
    with pytest.raises(ValueError, match="no gold node"):
        loss_fn(**_inputs(p_v=[0.5, 0.5], gold=[0.0, 0.0], tokens=[10, 10]))


def test_func_at_k_ranks_within_each_file():
    p_v = torch.tensor([0.9, 0.8, 0.1, 0.95])
    node_to_file = torch.tensor([0, 0, 0, 1])
    gold = torch.tensor([0.0, 0.0, 1.0, 0.0])

    assert gold_within_file_topk(p_v, node_to_file, gold, k=3) is True
    assert gold_within_file_topk(p_v, node_to_file, gold, k=2) is False

    every_gold_must_fit = torch.tensor([0.0, 1.0, 1.0, 0.0])
    assert gold_within_file_topk(p_v, node_to_file, every_gold_must_fit, k=3) is True
    assert gold_within_file_topk(p_v, node_to_file, every_gold_must_fit, k=1) is False


def test_gold_percentile_has_a_hard_null_at_one_half():
    one_file = torch.tensor([0, 0, 0, 0])
    gold = torch.tensor([0.0, 0.0, 0.0, 1.0])

    flat = torch.zeros(4)
    assert gold_rank_percentile(flat, one_file, gold) == pytest.approx(0.5)
    assert gold_rank_percentile(torch.ones(4), one_file, gold) == pytest.approx(0.5)

    gold_on_top = torch.tensor([0.1, 0.2, 0.3, 0.9])
    assert gold_rank_percentile(gold_on_top, one_file, gold) == pytest.approx(1.0)

    gold_at_bottom = torch.tensor([0.9, 0.8, 0.7, 0.1])
    assert gold_rank_percentile(gold_at_bottom, one_file, gold) == pytest.approx(0.0)

    middle = torch.tensor([0.9, 0.8, 0.1, 0.5])
    assert gold_rank_percentile(middle, one_file, gold) == pytest.approx(1 / 3)


def test_gold_percentile_ignores_single_node_files_and_averages_gold_nodes():
    node_to_file = torch.tensor([0, 1, 1, 1])
    p_v = torch.tensor([0.5, 0.9, 0.4, 0.1])

    alone = torch.tensor([1.0, 0.0, 0.0, 0.0])
    assert gold_rank_percentile(p_v, node_to_file, alone) == pytest.approx(0.5)

    best_and_worst = torch.tensor([0.0, 1.0, 0.0, 1.0])
    assert gold_rank_percentile(p_v, node_to_file, best_and_worst) == pytest.approx(0.5)


def test_a_collapsed_model_scores_zero_not_a_free_pass():
    flat = torch.zeros(4)
    node_to_file = torch.tensor([0, 0, 0, 0])
    gold = torch.tensor([0.0, 0.0, 0.0, 1.0])

    assert gold_within_file_topk(flat, node_to_file, gold, k=3) is False
    assert gold_within_file_topk(flat, node_to_file, gold, k=4) is True


def test_structural_prior_follows_ppr_and_seed_neighbourhood():
    ppr = torch.tensor([0.0, 2e-5, 4e-5, 0.0])
    seed = torch.tensor([0.0, 0.0, 1.0, 1.0])

    pi = structural_prior(ppr, seed, alpha=2.0, gamma_s=1.0)
    assert pi[0].item() == pytest.approx(0.5)
    assert pi[1].item() == pytest.approx(torch.sigmoid(torch.tensor(1.0)).item())
    assert pi[2].item() == pytest.approx(torch.sigmoid(torch.tensor(3.0)).item())
    assert pi[3].item() == pytest.approx(torch.sigmoid(torch.tensor(1.0)).item())

    uniform = structural_prior(ppr, seed, alpha=0.0, gamma_s=0.0, base_rate=0.1)
    assert torch.allclose(uniform, torch.full((4,), 0.1))


def test_prior_kl_is_zero_at_the_prior_and_off_by_default():
    p = torch.tensor([0.9, 0.2, 0.5])
    assert torch.allclose(bernoulli_kl(p, p), torch.zeros(3), atol=1e-6)
    assert (bernoulli_kl(p, torch.full((3,), 0.5)) > 0).sum() == 2

    inputs = _inputs(p_v=[0.9, 0.2], gold=[1.0, 0.0], tokens=[10, 10])
    off = RecallConstrainedTokenLoss(beta=0.0, gamma=0.0)(**inputs)
    assert off["prior_kl"].item() == 0.0

    on = RecallConstrainedTokenLoss(beta=0.0, gamma=0.0, prior_weight=2.0)
    far = on(**inputs, prior_probs=torch.tensor([0.1, 0.9]))
    near = on(**inputs, prior_probs=torch.tensor([0.9, 0.2]))
    assert near["prior_kl"].item() == pytest.approx(0.0, abs=1e-6)
    assert far["total"].item() == pytest.approx(off["total"].item() + 2.0 * far["prior_kl"].item())


def test_prior_weight_without_a_prior_raises():
    loss_fn = RecallConstrainedTokenLoss(prior_weight=1.0)
    with pytest.raises(ValueError, match="structural prior"):
        loss_fn(**_inputs(p_v=[0.5, 0.5], gold=[1.0, 0.0], tokens=[10, 10]))

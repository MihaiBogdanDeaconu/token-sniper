from __future__ import annotations

import pytest

from token_sniper.evaluation.statistics import bootstrap_ci, holm_adjust, mcnemar_holm


def test_bootstrap_ci_of_constant_values_is_the_constant():
    assert bootstrap_ci([0.4] * 7) == pytest.approx((0.4, 0.4))


def test_bootstrap_ci_of_two_points_matches_the_resampling_distribution():
    assert bootstrap_ci([0.0, 1.0]) == (0.0, 1.0)
    assert bootstrap_ci([0.0, 1.0], alpha=0.6) == (0.5, 0.5)


def test_bootstrap_ci_is_seeded_and_brackets_the_mean():
    values = [0.3, 1.7, 0.2, 2.9, 1.1, 0.8, 2.2, 0.5, 1.4, 0.9]
    low, high = bootstrap_ci(values, seed=3)
    assert bootstrap_ci(values, seed=3) == (low, high)
    assert low < 1.2 < high
    assert bootstrap_ci(values, seed=4) != (low, high)


@pytest.mark.parametrize("kwargs", [{"n_resamples": 0}, {"alpha": 0.0}, {"alpha": 1.0}])
def test_bootstrap_ci_rejects_bad_arguments(kwargs):
    with pytest.raises(ValueError):
        bootstrap_ci([1.0, 0.0], **kwargs)


def test_bootstrap_ci_rejects_empty_values():
    with pytest.raises(ValueError, match="at least one value"):
        bootstrap_ci([])


def test_holm_adjust_by_hand():
    assert holm_adjust([0.04, 0.01, 0.02]) == pytest.approx([0.04, 0.03, 0.04])


def test_holm_adjust_is_monotone_step_down():
    assert holm_adjust([0.011, 0.01]) == pytest.approx([0.02, 0.02])


def test_holm_adjust_ties_get_the_same_adjusted_p():
    assert holm_adjust([0.02, 0.02, 0.5]) == pytest.approx([0.06, 0.06, 0.5])


def test_holm_adjust_caps_at_one():
    assert holm_adjust([0.6, 0.9]) == pytest.approx([1.0, 1.0])


def test_mcnemar_holm_by_hand():
    comparisons = {
        "a": ([False] * 10, [True] * 10),
        "b": ([True] + [False] * 9 + [True, False], [False] + [True] * 9 + [True, False]),
        "c": ([True, False, True], [True, False, True]),
    }
    result = mcnemar_holm(comparisons)

    assert result["a"]["baseline_only_correct"] == 0
    assert result["a"]["candidate_only_correct"] == 10
    assert result["a"]["discordant_pairs"] == 10
    assert result["a"]["p_value"] == pytest.approx(2 / 1024)
    assert result["a"]["p_holm"] == pytest.approx(3 * 2 / 1024)
    assert result["a"]["significant"] is True

    assert result["b"]["baseline_only_correct"] == 1
    assert result["b"]["candidate_only_correct"] == 9
    assert result["b"]["p_value"] == pytest.approx(22 / 1024)
    assert result["b"]["p_holm"] == pytest.approx(44 / 1024)
    assert result["b"]["significant"] is True

    assert result["c"]["discordant_pairs"] == 0
    assert result["c"]["p_value"] == 1.0
    assert result["c"]["p_holm"] == 1.0
    assert result["c"]["significant"] is False


def test_mcnemar_holm_tied_comparisons_share_the_adjusted_p():
    flags = ([False] * 6, [True] * 6)
    result = mcnemar_holm({"x": flags, "y": flags}, alpha=0.05)
    assert result["x"]["p_holm"] == result["y"]["p_holm"] == pytest.approx(4 / 64)
    assert result["x"]["significant"] is False
    assert result["y"]["significant"] is False


def test_mcnemar_holm_alpha_decides_significance():
    flags = ([False] * 6, [True] * 6)
    assert mcnemar_holm({"x": flags}, alpha=0.05)["x"]["significant"] is True
    assert mcnemar_holm({"x": flags}, alpha=0.01)["x"]["significant"] is False


def test_mcnemar_holm_rejects_unpaired_and_empty_input():
    with pytest.raises(ValueError, match="paired"):
        mcnemar_holm({"x": ([True, False], [True])})
    with pytest.raises(ValueError, match="at least one comparison"):
        mcnemar_holm({})

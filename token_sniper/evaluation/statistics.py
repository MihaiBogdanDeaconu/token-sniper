from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from token_sniper.evaluation.metrics import mcnemar_test

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence


def bootstrap_ci(
    values: Sequence[float],
    *,
    n_resamples: int = 10_000,
    alpha: float = 0.05,
    seed: int = 0,
) -> tuple[float, float]:
    if len(values) == 0:
        raise ValueError("bootstrap_ci needs at least one value")
    if n_resamples < 1:
        raise ValueError(f"n_resamples must be >= 1, got {n_resamples}")
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")

    data = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    means = data[rng.integers(0, len(data), size=(n_resamples, len(data)))].mean(axis=1)
    low, high = np.quantile(means, [alpha / 2, 1 - alpha / 2])
    return float(low), float(high)


def holm_adjust(p_values: Sequence[float]) -> list[float]:
    m = len(p_values)
    adjusted = [0.0] * m
    running = 0.0
    for rank, index in enumerate(sorted(range(m), key=lambda i: p_values[i])):
        running = max(running, min(1.0, (m - rank) * p_values[index]))
        adjusted[index] = running
    return adjusted


def mcnemar_holm(
    comparisons: Mapping[str, tuple[Sequence[bool], Sequence[bool]]],
    *,
    alpha: float = 0.05,
) -> dict[str, dict[str, float | int | bool]]:
    if not comparisons:
        raise ValueError("mcnemar_holm needs at least one comparison")
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")

    names = list(comparisons)
    tests = {}
    for name in names:
        first, second = comparisons[name]
        if len(first) != len(second):
            raise ValueError(f"comparison {name!r}: {len(first)} vs {len(second)} tasks, expected paired flags")
        tests[name] = mcnemar_test(first, second)
    adjusted = holm_adjust([float(tests[name]["p_value"]) for name in names])
    return {
        name: {**tests[name], "p_holm": p_holm, "significant": p_holm <= alpha}
        for name, p_holm in zip(names, adjusted, strict=True)
    }

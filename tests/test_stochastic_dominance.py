"""Tests for second-order stochastic dominance constraints."""

from __future__ import annotations

import numpy as np
import pandas as pd

from portfolio_optimisation.optim import ssd_constrained_weights, ssd_dominates


def _returns_and_benchmark(seed: int = 19, t: int = 200, n: int = 4):
    rng = np.random.default_rng(seed)
    means = rng.uniform(0.0002, 0.0008, size=n)
    vols = rng.uniform(0.005, 0.015, size=n)
    r = rng.normal(loc=means, scale=vols, size=(t, n))
    returns = pd.DataFrame(r, columns=[f"A{i}" for i in range(n)])
    benchmark = returns.mean(axis=1)
    return returns, benchmark


def test_ssd_solution_satisfies_dominance() -> None:
    returns, benchmark = _returns_and_benchmark()
    weights = ssd_constrained_weights(returns, benchmark)
    portfolio = returns @ weights
    # The LP solution should empirically SSD-dominate (allow small slack).
    assert ssd_dominates(portfolio, benchmark)


def test_ssd_solution_is_long_only_simplex() -> None:
    returns, benchmark = _returns_and_benchmark()
    weights = ssd_constrained_weights(returns, benchmark)
    assert np.isclose(weights.sum(), 1.0, atol=1e-6)
    assert (weights >= -1e-9).all()


def test_ssd_dominates_self_is_true() -> None:
    _, benchmark = _returns_and_benchmark()
    assert ssd_dominates(benchmark, benchmark)


def test_ssd_dominates_inferior_is_false() -> None:
    rng = np.random.default_rng(0)
    benchmark = pd.Series(rng.normal(0.001, 0.01, size=200))
    inferior = pd.Series(rng.normal(-0.001, 0.02, size=200))
    assert not ssd_dominates(inferior, benchmark)


def test_ssd_rejects_length_mismatch() -> None:
    returns, _ = _returns_and_benchmark()
    short_bench = pd.Series(np.zeros(50))
    import pytest

    with pytest.raises(ValueError, match="same length"):
        ssd_constrained_weights(returns, short_bench)

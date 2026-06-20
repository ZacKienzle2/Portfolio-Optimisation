"""Tests for the Chekhlov-Uryasev minimum-CDaR LP."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from portfolio_optimisation.optim import cdar, min_cdar_weights


def _drifting_returns(seed: int = 17, t: int = 300, n: int = 5) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    drift = rng.uniform(0.0, 0.0005, size=n)
    vols = rng.uniform(0.005, 0.02, size=n)
    cor = np.full((n, n), 0.2)
    np.fill_diagonal(cor, 1.0)
    chol = np.linalg.cholesky(cor)
    z = rng.normal(size=(t, n))
    data = drift + (vols * (z @ chol.T))
    return pd.DataFrame(data, columns=[f"A{i}" for i in range(n)])


def test_cdar_metric_is_nonnegative_and_alpha_monotone() -> None:
    returns = _drifting_returns()
    portfolio = returns.mean(axis=1)
    cdar_5 = cdar(portfolio, alpha=0.05)
    cdar_25 = cdar(portfolio, alpha=0.25)
    assert cdar_5 >= 0
    assert cdar_25 >= 0
    # Tighter alpha (smaller tail) should give a larger or equal CDaR.
    assert cdar_5 >= cdar_25 - 1e-9


def test_min_cdar_long_only_simplex() -> None:
    returns = _drifting_returns()
    weights = min_cdar_weights(returns, alpha=0.1)
    assert np.isclose(weights.sum(), 1.0, atol=1e-6)
    assert (weights >= -1e-9).all()


def test_min_cdar_reduces_cdar_vs_equal_weight() -> None:
    returns = _drifting_returns()
    equal_w = pd.Series(1.0 / returns.shape[1], index=returns.columns)
    opt_w = min_cdar_weights(returns, alpha=0.1)
    cdar_eq = cdar(returns @ equal_w, alpha=0.1)
    cdar_opt = cdar(returns @ opt_w, alpha=0.1)
    assert cdar_opt <= cdar_eq + 1e-6


def test_min_cdar_rejects_invalid_alpha() -> None:
    returns = _drifting_returns()
    with pytest.raises(ValueError, match="alpha"):
        min_cdar_weights(returns, alpha=0.0)


def test_min_cdar_target_return_constraint() -> None:
    returns = _drifting_returns()
    mu = returns.mean().to_numpy()
    weights = min_cdar_weights(returns, alpha=0.1, target_return=mu.min())
    assert np.isclose(weights.sum(), 1.0, atol=1e-6)
    assert float(returns.mean().to_numpy() @ weights.to_numpy()) >= mu.min() - 1e-6


def test_min_cdar_minimises_worst_alpha_tail() -> None:
    rng = np.random.default_rng(5)
    t_steps = 500
    asset_a = rng.normal(0.0003, 0.010, t_steps)
    asset_b = rng.normal(0.0002, 0.020, t_steps)
    returns = pd.DataFrame({"A": asset_a, "B": asset_b})
    alpha = 0.05
    grid = np.linspace(0.0, 1.0, 101)
    grid_cdar = [cdar(pd.Series(wa * asset_a + (1.0 - wa) * asset_b), alpha=alpha) for wa in grid]
    weights = min_cdar_weights(returns, alpha=alpha)
    achieved = cdar(returns @ weights, alpha=alpha)
    assert achieved <= min(grid_cdar) + 1e-3

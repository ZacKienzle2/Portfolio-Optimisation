"""Tests for the Lai (1991) polynomial goal programming four-moment portfolio."""

from __future__ import annotations

import numpy as np
import pandas as pd

from portfolio_optimisation.optim import (
    cokurtosis_tensor,
    coskewness_tensor,
    pgp_higher_moment_weights,
)


def _skewed_returns(seed: int = 41, t: int = 800, n: int = 4) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    df = 5.0
    data = rng.standard_t(df=df, size=(t, n)) * 0.01
    drift = rng.uniform(0.0001, 0.0005, size=n)
    return pd.DataFrame(data + drift, columns=[f"A{i}" for i in range(n)])


def test_coskewness_tensor_shape() -> None:
    returns = _skewed_returns()
    n = returns.shape[1]
    m3 = coskewness_tensor(returns)
    assert m3.shape == (n, n * n)


def test_cokurtosis_tensor_shape() -> None:
    returns = _skewed_returns()
    n = returns.shape[1]
    m4 = cokurtosis_tensor(returns)
    assert m4.shape == (n, n * n * n)


def test_pgp_weights_long_only_simplex() -> None:
    returns = _skewed_returns()
    result = pgp_higher_moment_weights(returns)
    assert (result.weights >= -1e-9).all()
    assert np.isclose(result.weights.sum(), 1.0, atol=1e-6)


def test_pgp_individual_optima_bound_achieved_moments() -> None:
    returns = _skewed_returns()
    result = pgp_higher_moment_weights(returns)
    # The achieved moments should respect the single-objective bounds, modulo
    # small SLSQP slack from the trade-off.
    assert result.achieved_mean <= result.mean_star + 1e-9
    assert result.achieved_variance >= result.variance_star - 1e-9
    assert result.achieved_skewness <= result.skewness_star + 1e-6
    assert result.achieved_kurtosis >= result.kurtosis_star - 1e-6


def test_pgp_preference_exponents_shift_allocation() -> None:
    returns = _skewed_returns()
    default = pgp_higher_moment_weights(returns)
    aggressive = pgp_higher_moment_weights(returns, gamma=4.0, beta=0.25)
    # Higher gamma penalises skew shortfall more sharply -> allocations differ.
    assert not np.allclose(default.weights.to_numpy(), aggressive.weights.to_numpy())

"""Tests for the Black-Litterman implementation."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from portfolio_optimisation.optim import (
    HRPModel,
    black_litterman_weights,
    implied_equilibrium_returns,
)


def _returns_and_hrp(seed: int = 21, t: int = 600, n: int = 6):
    rng = np.random.default_rng(seed)
    market = rng.normal(0.0, 0.005, size=t)
    data = np.column_stack([0.4 * market + rng.normal(0.0, 0.01, size=t) for _ in range(n)])
    returns = pd.DataFrame(data, columns=[f"A{i}" for i in range(n)])
    hrp = HRPModel(returns)
    hrp.optimize(linkage_method="ward")
    return returns, hrp.clean_weights()


def test_implied_equilibrium_returns_has_expected_shape() -> None:
    returns, hrp_weights = _returns_and_hrp()
    cov = returns.cov()
    pi = implied_equilibrium_returns(cov, hrp_weights, risk_aversion=2.5)
    assert pi.shape == (returns.shape[1],)
    assert list(pi.index) == list(cov.columns)


def test_black_litterman_without_views_recovers_market_weights() -> None:
    returns, hrp_weights = _returns_and_hrp()
    cov = returns.cov()
    result = black_litterman_weights(cov, hrp_weights)
    np.testing.assert_allclose(
        result.weights.to_numpy(),
        hrp_weights.to_numpy(),
        atol=1e-9,
    )


def test_black_litterman_with_absolute_view_tilts_towards_view() -> None:
    returns, hrp_weights = _returns_and_hrp()
    cov = returns.cov()
    n = returns.shape[1]
    # Absolute view: asset 0 expected return is much higher than equilibrium.
    p = np.zeros((1, n))
    p[0, 0] = 1.0
    pi = implied_equilibrium_returns(cov, hrp_weights)
    q = np.array([pi.iloc[0] + 0.05])
    result = black_litterman_weights(
        cov,
        hrp_weights,
        views_matrix=p,
        views_returns=q,
    )
    assert result.weights.iloc[0] > hrp_weights.iloc[0]


def test_black_litterman_long_only_simplex() -> None:
    returns, hrp_weights = _returns_and_hrp()
    cov = returns.cov()
    n = returns.shape[1]
    p = np.eye(n)[:2]
    pi = implied_equilibrium_returns(cov, hrp_weights)
    q = pi.to_numpy()[:2] + 0.01
    result = black_litterman_weights(
        cov,
        hrp_weights,
        views_matrix=p,
        views_returns=q,
    )
    assert (result.weights >= -1e-9).all()
    assert np.isclose(result.weights.sum(), 1.0, atol=1e-9)


def test_black_litterman_rejects_dimension_mismatch() -> None:
    returns, hrp_weights = _returns_and_hrp()
    cov = returns.cov()
    n = returns.shape[1]
    bad_p = np.zeros((1, n + 1))
    with pytest.raises(ValueError, match="column count"):
        black_litterman_weights(
            cov,
            hrp_weights,
            views_matrix=bad_p,
            views_returns=np.array([0.0]),
        )

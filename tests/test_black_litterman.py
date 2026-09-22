"""Tests for the Black-Litterman implementation."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from hypothesis import given
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

from portfolio_optimisation import baselines
from portfolio_optimisation.optim import (
    HRPModel,
    black_litterman_weights,
    implied_equilibrium_returns,
)
from portfolio_optimisation.optim.black_litterman import black_litterman_posterior


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


@st.composite
def _black_litterman_inputs(draw: st.DrawFn) -> dict[str, np.ndarray | float]:
    n = draw(st.integers(1, 8))
    k = draw(st.integers(1, n))
    unit = st.floats(-1.0, 1.0)
    factor = draw(arrays(np.float64, (n, n), elements=unit))
    return {
        "sigma": 1e-4 * (factor @ factor.T + 0.1 * np.eye(n)),
        "pi": draw(arrays(np.float64, n, elements=st.floats(-0.01, 0.01))),
        "p": draw(arrays(np.float64, (k, n), elements=unit)),
        "q": draw(arrays(np.float64, k, elements=st.floats(-0.01, 0.01))),
        "omega": np.diag(draw(arrays(np.float64, k, elements=st.floats(1e-6, 1e-3)))),
        "tau": draw(st.floats(0.01, 0.2)),
    }


@given(inputs=_black_litterman_inputs())
def test_equivalent_black_litterman_posterior(inputs: dict[str, np.ndarray | float]) -> None:
    for old, new in zip(
        baselines.black_litterman_posterior(**inputs),
        black_litterman_posterior(**inputs),
        strict=True,
    ):
        np.testing.assert_allclose(old, new, rtol=1e-6, atol=1e-12)

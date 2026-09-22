"""Tests for the most-diversified portfolio."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

from portfolio_optimisation.optim import diversification_ratio, max_diversification_weights

_TOLERANCE = 1e-5


def _returns(seed: int, t: int = 250, n: int = 8) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    loadings = rng.normal(size=(n, 2))
    factors = rng.normal(scale=0.01, size=(t, 2))
    noise = rng.normal(scale=rng.uniform(0.005, 0.03, size=n), size=(t, n))
    return pd.DataFrame(factors @ loadings.T + noise, columns=[f"A{i}" for i in range(n)])


@pytest.mark.parametrize("seed", range(5))
def test_core_property_of_the_most_diversified_portfolio(seed: int) -> None:
    returns = _returns(seed)
    covariance = returns.cov().to_numpy()
    weights = max_diversification_weights(returns, cov_matrix=returns.cov()).to_numpy()
    correlation = (
        covariance @ weights / np.sqrt(np.diag(covariance) * (weights @ covariance @ weights))
    )
    held = weights > _TOLERANCE
    assert np.ptp(correlation[held]) < 1e-4
    assert np.all(correlation[~held] >= correlation[held].max() - 1e-4)


@settings(max_examples=50, deadline=None)
@given(
    seed=st.integers(0, 2**16),
    mix=arrays(np.float64, 8, elements=st.floats(0.01, 1.0)),
)
def test_no_long_only_portfolio_is_more_diversified(seed: int, mix: np.ndarray) -> None:
    returns = _returns(seed)
    covariance = returns.cov()
    weights = max_diversification_weights(returns, cov_matrix=covariance).to_numpy()
    matrix = covariance.to_numpy()
    assert (
        diversification_ratio(mix / mix.sum(), matrix)
        <= diversification_ratio(weights, matrix) + 1e-6
    )


def test_default_covariance_gives_simplex_weights() -> None:
    weights = max_diversification_weights(_returns(0))
    assert weights.sum() == pytest.approx(1.0)
    assert (weights >= 0.0).all()

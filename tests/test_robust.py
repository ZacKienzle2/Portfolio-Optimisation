"""Tests for robust and resampled optimisation."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

from portfolio_optimisation.optim import (
    resampled_weights,
    robust_mean_variance_weights,
)


def _returns(seed: int = 0, n: int = 400, k: int = 5) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    factor = rng.normal(0.0006, 0.01, size=(n, 1))
    loadings = rng.uniform(0.5, 1.5, size=(1, k))
    idiosyncratic = rng.normal(0.0, 0.006, size=(n, k))
    data = factor @ loadings + idiosyncratic + 0.0003
    return pd.DataFrame(data, columns=[f"A{i}" for i in range(k)])


def _assert_simplex(weights: pd.Series, columns: list[str]) -> None:
    assert abs(float(weights.sum()) - 1.0) < 1e-8
    assert bool((weights.to_numpy() >= -1e-12).all())
    assert set(weights.index) == set(columns)


def test_resampled_min_variance_is_simplex() -> None:
    returns = _returns()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        weights = resampled_weights(returns, objective="min_variance", n_resamples=100, seed=0)
    _assert_simplex(weights, list(returns.columns))


def test_resampled_max_sharpe_is_simplex() -> None:
    returns = _returns()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        weights = resampled_weights(returns, objective="max_sharpe", n_resamples=100, seed=0)
    _assert_simplex(weights, list(returns.columns))


def test_resampled_is_seed_reproducible() -> None:
    returns = _returns()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        first = resampled_weights(returns, n_resamples=80, seed=42)
        second = resampled_weights(returns, n_resamples=80, seed=42)
    pd.testing.assert_series_equal(first, second)


def test_robust_mean_variance_is_simplex() -> None:
    returns = _returns()
    _assert_simplex(robust_mean_variance_weights(returns), list(returns.columns))


def test_uncertainty_changes_allocation() -> None:
    returns = _returns()
    plug_in = robust_mean_variance_weights(returns, uncertainty=0.0)
    conservative = robust_mean_variance_weights(returns, uncertainty=5.0)
    assert not np.allclose(plug_in.to_numpy(), conservative.to_numpy())


def test_invalid_objective_raises() -> None:
    returns = _returns()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with pytest.raises(ValueError, match="objective"):
            resampled_weights(returns, objective="bad", n_resamples=5)

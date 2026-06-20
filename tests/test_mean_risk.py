"""Tests for mean-CVaR and mean-EVaR optimisation."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from portfolio_optimisation.optim import (
    MeanRiskModel,
    PortfolioConstraints,
    mean_risk_weights,
    min_cvar_weights,
    min_evar_weights,
)


def _drifting_returns(seed: int = 23, t: int = 400, n: int = 5) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    drift = rng.uniform(0.0, 0.0006, size=n)
    vols = rng.uniform(0.006, 0.02, size=n)
    cor = np.full((n, n), 0.25)
    np.fill_diagonal(cor, 1.0)
    chol = np.linalg.cholesky(cor)
    z = rng.normal(size=(t, n))
    data = drift + (vols * (z @ chol.T))
    return pd.DataFrame(data, columns=[f"A{i}" for i in range(n)])


def _empirical_cvar(returns: pd.Series, alpha: float) -> float:
    losses = -returns.to_numpy(dtype=np.float64)
    var = float(np.quantile(losses, 1.0 - alpha))
    tail = losses[losses >= var]
    return float(tail.mean()) if tail.size else var


def test_min_cvar_long_only_simplex() -> None:
    weights = min_cvar_weights(_drifting_returns(), alpha=0.05)
    assert np.isclose(weights.sum(), 1.0, atol=1e-6)
    assert (weights >= -1e-9).all()


def test_min_cvar_reduces_cvar_vs_equal_weight() -> None:
    returns = _drifting_returns()
    equal_w = pd.Series(1.0 / returns.shape[1], index=returns.columns)
    opt_w = min_cvar_weights(returns, alpha=0.05)
    cvar_eq = _empirical_cvar(returns @ equal_w, alpha=0.05)
    cvar_opt = _empirical_cvar(returns @ opt_w, alpha=0.05)
    assert cvar_opt <= cvar_eq + 1e-6


def test_min_cvar_rejects_invalid_alpha() -> None:
    with pytest.raises(ValueError, match="alpha"):
        min_cvar_weights(_drifting_returns(), alpha=1.0)


def test_min_cvar_target_return_constraint() -> None:
    returns = _drifting_returns()
    mu = returns.mean().to_numpy(dtype=np.float64)
    spec = PortfolioConstraints(target_return=float(np.median(mu)))
    weights = min_cvar_weights(returns, alpha=0.1, constraints=spec)
    realised = float(mu @ weights.to_numpy(dtype=np.float64))
    assert realised >= float(np.median(mu)) - 1e-6


def test_min_cvar_respects_box_bounds() -> None:
    returns = _drifting_returns()
    spec = PortfolioConstraints(max_weight=0.4)
    weights = min_cvar_weights(returns, alpha=0.05, constraints=spec)
    assert (weights <= 0.4 + 1e-6).all()


def test_min_cvar_turnover_budget_caps_trade() -> None:
    returns = _drifting_returns()
    n = returns.shape[1]
    previous = np.full(n, 1.0 / n)
    spec = PortfolioConstraints(previous_weights=previous, max_turnover=0.2)
    weights = min_cvar_weights(returns, alpha=0.05, constraints=spec)
    turnover = float(np.abs(weights.to_numpy(dtype=np.float64) - previous).sum())
    assert turnover <= 0.2 + 1e-6


def test_min_cvar_long_short_leverage_cap() -> None:
    returns = _drifting_returns()
    spec = PortfolioConstraints(long_only=False, max_leverage=1.6)
    weights = min_cvar_weights(returns, alpha=0.05, constraints=spec)
    assert np.isclose(weights.sum(), 1.0, atol=1e-6)
    assert float(np.abs(weights.to_numpy(dtype=np.float64)).sum()) <= 1.6 + 1e-4


def test_min_evar_long_only_simplex() -> None:
    weights = min_evar_weights(_drifting_returns(), alpha=0.05)
    assert np.isclose(weights.sum(), 1.0, atol=1e-5)
    assert (weights >= -1e-6).all()


def test_min_evar_rejects_invalid_alpha() -> None:
    with pytest.raises(ValueError, match="alpha"):
        min_evar_weights(_drifting_returns(), alpha=0.0)


def test_mean_risk_dispatch_matches_direct_calls() -> None:
    returns = _drifting_returns()
    cvar_direct = min_cvar_weights(returns, alpha=0.05)
    cvar_dispatch = mean_risk_weights(returns, measure="cvar", alpha=0.05)
    pd.testing.assert_series_equal(cvar_direct, cvar_dispatch, atol=1e-7)


def test_mean_risk_rejects_unknown_measure() -> None:
    with pytest.raises(ValueError, match="measure"):
        mean_risk_weights(_drifting_returns(), measure="cdar")  # type: ignore[arg-type]


def test_mean_risk_model_caches_weights() -> None:
    model = MeanRiskModel(_drifting_returns(), measure="evar", alpha=0.05)
    weights = model.optimise()
    pd.testing.assert_series_equal(weights, model.weights)

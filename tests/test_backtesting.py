"""Tests for the VaR and Expected Shortfall backtests."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.stats import norm

from portfolio_optimisation.risk import (
    acerbi_szekely_z2,
    christoffersen_conditional_coverage_test,
    christoffersen_independence_test,
    kupiec_pof_test,
)

ALPHA = 0.05


def _normal_returns(seed: int = 0, t: int = 4000, sigma: float = 0.01) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.normal(0.0, sigma, size=t)


def _var_forecast(sigma: float) -> float:
    return sigma * float(norm.ppf(ALPHA))


def _es_forecast(sigma: float) -> float:
    return -sigma * float(norm.pdf(norm.ppf(ALPHA))) / ALPHA


def test_kupiec_does_not_reject_calibrated_model() -> None:
    sigma = 0.01
    returns = _normal_returns(sigma=sigma)
    var = np.full_like(returns, _var_forecast(sigma))
    result = kupiec_pof_test(returns, var, alpha=ALPHA)
    assert not result.reject
    assert abs(result.violations / result.observations - ALPHA) < 0.02


def test_kupiec_rejects_excessive_violations() -> None:
    returns = _normal_returns(sigma=0.01)
    var = np.zeros_like(returns)
    assert kupiec_pof_test(returns, var, alpha=ALPHA).reject


def test_independence_detects_clustered_violations() -> None:
    returns = np.concatenate([np.full(50, -0.10), np.full(950, 0.01)])
    var = np.full_like(returns, _var_forecast(0.01))
    assert christoffersen_independence_test(returns, var).reject


def test_calibrated_conditional_coverage_rejection_rate_is_controlled() -> None:
    sigma = 0.01
    var_level = _var_forecast(sigma)
    rejections = 0
    trials = 30
    for seed in range(trials):
        returns = _normal_returns(seed=seed, sigma=sigma)
        var = np.full_like(returns, var_level)
        if christoffersen_conditional_coverage_test(returns, var, alpha=ALPHA).reject:
            rejections += 1
    assert rejections / trials <= 0.2


def test_acerbi_szekely_near_zero_for_calibrated_es() -> None:
    sigma = 0.01
    returns = _normal_returns(sigma=sigma, t=8000)
    var = np.full_like(returns, _var_forecast(sigma))
    es = np.full_like(returns, _es_forecast(sigma))
    assert abs(acerbi_szekely_z2(returns, var, es, alpha=ALPHA)) < 0.15


def test_acerbi_szekely_positive_when_es_underestimated() -> None:
    sigma = 0.01
    returns = _normal_returns(sigma=sigma, t=8000)
    var = np.full_like(returns, _var_forecast(sigma))
    es = np.full_like(returns, _es_forecast(sigma) * 0.5)
    assert acerbi_szekely_z2(returns, var, es, alpha=ALPHA) > 0.3


def test_mismatched_lengths_raise() -> None:
    returns = _normal_returns()
    with pytest.raises(ValueError, match="same length"):
        kupiec_pof_test(returns[:-1], np.full_like(returns, -0.02), alpha=ALPHA)


def test_zero_es_forecast_raises() -> None:
    returns = _normal_returns()
    with pytest.raises(ValueError, match="non-zero"):
        acerbi_szekely_z2(
            returns,
            np.full_like(returns, -0.02),
            np.zeros_like(returns),
            alpha=ALPHA,
        )

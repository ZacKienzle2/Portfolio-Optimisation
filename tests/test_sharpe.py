"""Tests for PSR, DSR and bootstrap Sharpe CIs."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from portfolio_optimisation.risk import (
    deflated_sharpe_ratio,
    probabilistic_sharpe_ratio,
    stationary_bootstrap_sharpe_ci,
)


def _normal_returns(seed: int = 31, t: int = 1000) -> pd.Series:
    rng = np.random.default_rng(seed)
    return pd.Series(rng.normal(0.0005, 0.01, size=t))


def test_psr_in_unit_interval() -> None:
    returns = _normal_returns()
    psr = probabilistic_sharpe_ratio(returns, benchmark_sharpe=0.0)
    assert 0.0 <= psr <= 1.0


def test_psr_against_zero_benchmark_high_for_positive_sample_sharpe() -> None:
    returns = _normal_returns()
    psr = probabilistic_sharpe_ratio(returns, benchmark_sharpe=0.0)
    assert psr > 0.9


def test_psr_against_high_benchmark_low() -> None:
    returns = _normal_returns()
    psr = probabilistic_sharpe_ratio(returns, benchmark_sharpe=1.0)
    assert psr < 0.1


def test_dsr_requires_inputs() -> None:
    returns = _normal_returns()
    with pytest.raises(ValueError, match="candidate_sharpes or n_trials"):
        deflated_sharpe_ratio(returns)


def test_dsr_with_many_trials_deflates_psr() -> None:
    returns = _normal_returns()
    psr_naive = probabilistic_sharpe_ratio(returns)
    dsr_100 = deflated_sharpe_ratio(returns, n_trials=100)
    assert dsr_100 < psr_naive


def test_dsr_with_candidate_set_returns_value_in_unit_interval() -> None:
    returns = _normal_returns()
    candidates = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5])
    dsr = deflated_sharpe_ratio(returns, candidate_sharpes=candidates)
    assert 0.0 <= dsr <= 1.0


def test_stationary_bootstrap_ci_contains_sample_sharpe() -> None:
    returns = _normal_returns()
    sample_sharpe = float(returns.mean() / returns.std(ddof=1))
    lower, upper, samples = stationary_bootstrap_sharpe_ci(
        returns, n_resamples=400, confidence=0.95, seed=7
    )
    assert lower <= sample_sharpe <= upper
    assert samples.size == 400
    assert math.isfinite(lower)
    assert math.isfinite(upper)


def test_stationary_bootstrap_rejects_short_series() -> None:
    with pytest.raises(ValueError, match="at least 4"):
        stationary_bootstrap_sharpe_ci(pd.Series([0.0, 1.0, 2.0]))

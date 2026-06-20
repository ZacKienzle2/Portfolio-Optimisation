"""Tests for GARCH conditional volatility and VaR/ES forecasts."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

from portfolio_optimisation.risk import (
    conditional_volatility,
    garch_var_es,
    kupiec_pof_test,
)


def _returns(seed: int = 0, t: int = 2500, sigma: float = 0.01) -> pd.Series:
    rng = np.random.default_rng(seed)
    return pd.Series(rng.normal(0.0, sigma, size=t))


def test_conditional_volatility_is_positive_and_aligned() -> None:
    returns = _returns()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sigma = conditional_volatility(returns)
    assert len(sigma) == len(returns)
    assert bool((sigma.to_numpy() > 0).all())
    assert bool(np.isfinite(sigma.to_numpy()).all())


def test_var_es_ordering_and_shapes() -> None:
    returns = _returns()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        var, es = garch_var_es(returns, alpha=0.05, dist="normal")
    assert var.shape == es.shape == returns.shape
    assert bool(np.isfinite(var).all())
    assert bool((es <= var).all())
    assert float(var.mean()) < 0.0


def test_student_t_path_runs() -> None:
    returns = _returns()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        var, es = garch_var_es(returns, alpha=0.05, dist="t")
    assert bool((es <= var).all())
    assert bool(np.isfinite(es).all())


def test_gjr_leverage_model_fits() -> None:
    returns = _returns()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sigma = conditional_volatility(returns, vol="GJR")
    assert bool((sigma.to_numpy() > 0).all())


def test_forecasts_have_near_nominal_coverage() -> None:
    returns = _returns(seed=3, t=3000)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        var, _ = garch_var_es(returns, alpha=0.05, dist="normal")
    result = kupiec_pof_test(returns.to_numpy(), var, alpha=0.05)
    rate = result.violations / result.observations
    assert 0.025 <= rate <= 0.085


def test_rejects_invalid_alpha() -> None:
    returns = _returns()
    with pytest.raises(ValueError, match="alpha"):
        garch_var_es(returns, alpha=1.5)

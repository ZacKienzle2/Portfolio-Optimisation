"""Tests for the Extreme Value Theory tail-risk estimators."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

from portfolio_optimisation.risk import (
    evt_expected_shortfall,
    evt_value_at_risk,
    fit_peaks_over_threshold,
    hill_estimator,
)


def _heavy_tailed_returns(seed: int = 0, t: int = 5000, dof: int = 3) -> pd.Series:
    rng = np.random.default_rng(seed)
    return pd.Series(rng.standard_t(dof, size=t) * 0.01)


def test_pot_fit_returns_finite_parameters() -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fit = fit_peaks_over_threshold(_heavy_tailed_returns(), threshold_quantile=0.95)
    assert np.isfinite(fit.shape)
    assert fit.scale > 0.0
    assert 0.0 < fit.exceedance_rate < 1.0
    assert fit.n_exceedances > 0


def test_evt_var_and_es_are_negative_and_ordered() -> None:
    returns = _heavy_tailed_returns()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        var = evt_value_at_risk(returns, alpha=0.01)
        es = evt_expected_shortfall(returns, alpha=0.01)
    assert var < 0.0
    assert es < 0.0
    assert es <= var


def test_evt_var_deepens_with_smaller_alpha() -> None:
    returns = _heavy_tailed_returns()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        deep = evt_value_at_risk(returns, alpha=0.005)
        shallow = evt_value_at_risk(returns, alpha=0.05)
    assert deep <= shallow


def test_hill_estimator_positive_for_heavy_tail() -> None:
    assert hill_estimator(_heavy_tailed_returns(), k=200) > 0.0


def test_rejects_invalid_alpha() -> None:
    with pytest.raises(ValueError, match="alpha"):
        evt_value_at_risk(_heavy_tailed_returns(), alpha=0.0)


def test_rejects_invalid_k() -> None:
    with pytest.raises(ValueError, match="k"):
        hill_estimator(_heavy_tailed_returns(), k=10**9)

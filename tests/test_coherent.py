"""Tests for coherent risk measures."""

from __future__ import annotations

import numpy as np
import pytest

from portfolio_optimisation.risk import (
    entropic_value_at_risk,
    exponential_spectrum,
    power_spectrum,
    spectral_risk_measure,
    wang_transform_risk,
)


def _normal_returns(seed: int = 23, n: int = 5000) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.normal(0.0, 0.01, size=n)


def test_evar_bounded_below_by_cvar() -> None:
    """EVaR is a coherent upper bound on CVaR. Verify on normal sample."""
    returns = _normal_returns()
    cvar = -float(returns[returns <= np.quantile(returns, 0.05)].mean())
    evar = entropic_value_at_risk(returns, alpha=0.05)
    assert evar >= cvar - 1e-6


def test_evar_rejects_invalid_alpha() -> None:
    with pytest.raises(ValueError, match="alpha"):
        entropic_value_at_risk(np.array([0.0]), alpha=1.5)


def test_spectral_risk_with_exponential_spectrum_positive() -> None:
    returns = _normal_returns()
    risk = spectral_risk_measure(returns, exponential_spectrum(0.1))
    assert risk > 0


def test_spectral_risk_with_power_spectrum_positive() -> None:
    returns = _normal_returns()
    risk = spectral_risk_measure(returns, power_spectrum(2.0))
    assert risk > 0


def test_wang_transform_zero_lambda_recovers_mean_loss() -> None:
    returns = _normal_returns()
    risk = wang_transform_risk(returns, lam=0.0)
    np.testing.assert_allclose(risk, -returns.mean(), atol=1e-3)


def test_wang_transform_positive_lambda_inflates_tail() -> None:
    returns = _normal_returns()
    base = wang_transform_risk(returns, lam=0.0)
    distorted = wang_transform_risk(returns, lam=0.5)
    assert distorted > base


def test_exponential_spectrum_rejects_non_positive() -> None:
    with pytest.raises(ValueError, match="absolute_risk_aversion"):
        exponential_spectrum(0.0)


def test_power_spectrum_rejects_gamma_le_one() -> None:
    with pytest.raises(ValueError, match="gamma"):
        power_spectrum(0.5)

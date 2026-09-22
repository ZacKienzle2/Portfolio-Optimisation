"""Tests for the closed-form maximum-likelihood SDE fitter."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from scipy.optimize import minimize
from scipy.stats import norm

from portfolio_optimisation.sde import SDEFitter, simulate_gbm, simulate_ornstein_uhlenbeck
from portfolio_optimisation.sde.fitter import ou_estimates

_DT = 1.0 / 252.0


def _ou_log_likelihood(levels: np.ndarray, kappa: float, alpha: float, sigma: float) -> float:
    decay = np.exp(-kappa * _DT)
    mean = alpha + (levels[:-1] - alpha) * decay
    scale = sigma * np.sqrt(-np.expm1(-2.0 * kappa * _DT) / (2.0 * kappa))
    return float(norm.logpdf(levels[1:], mean, scale).sum())


def test_gbm_fit_recovers_the_simulated_volatility() -> None:
    paths = simulate_gbm(s0=100.0, mu=0.08, sigma=0.25, t=8.0, n_steps=2016, n_paths=2, seed=4)
    prices = pd.DataFrame(paths.T, columns=["X", "Y"])
    estimates = SDEFitter(prices).fit_gbm()
    assert list(estimates.index) == ["X", "Y"]
    assert {"mu", "sigma", "Log-Likelihood", "AIC", "BIC"} <= set(estimates.columns)
    assert np.allclose(estimates["sigma"], 0.25, rtol=0.05)


def test_ou_fit_recovers_price_level_volatility_beyond_the_old_bound() -> None:
    path = simulate_ornstein_uhlenbeck(
        x0=100.0, kappa=3.0, theta=100.0, sigma=20.0, t=20.0, n_steps=5040, n_paths=1, seed=9
    )[0]
    estimates = SDEFitter(pd.DataFrame({"X": path})).fit_ou()
    assert estimates.loc["X", "sigma"] == pytest.approx(20.0, rel=0.03)
    assert estimates.loc["X", "mu"] == pytest.approx(100.0, abs=3.0)


@settings(deadline=None, max_examples=25)
@given(
    kappa=st.floats(0.5, 20.0),
    sigma=st.floats(0.05, 5.0),
    seed=st.integers(0, 2**32 - 1),
)
def test_closed_form_attains_the_numerical_likelihood_maximum(
    kappa: float, sigma: float, seed: int
) -> None:
    levels = simulate_ornstein_uhlenbeck(
        x0=1.0, kappa=kappa, theta=1.0, sigma=sigma, t=4.0, n_steps=1008, n_paths=1, seed=seed
    )[0]
    closed = ou_estimates(levels, _DT)
    search = minimize(
        lambda p: -_ou_log_likelihood(levels, np.exp(p[0]), p[1], np.exp(p[2])),
        x0=np.array([0.0, levels.mean(), np.log(levels.std() + 1e-3)]),
        method="Nelder-Mead",
        options={"xatol": 1e-10, "fatol": 1e-10, "maxiter": 20_000},
    )
    assert closed["Log-Likelihood"] >= -search.fun - 1e-6
    assert closed["Log-Likelihood"] == pytest.approx(
        _ou_log_likelihood(levels, closed["kappa"], closed["mu"], closed["sigma"]), abs=1e-6
    )

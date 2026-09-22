"""Tests for the maximum-likelihood SDE fitter."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from portfolio_optimisation.sde import SDEFitter, simulate_gbm


# pymle's trust-region optimiser warns when a step leaves the gradient unchanged,
# which is its own convergence behaviour rather than a defect here.
@pytest.mark.filterwarnings("ignore:delta_grad == 0.0:UserWarning")
def test_gbm_fit_recovers_the_simulated_volatility() -> None:
    paths = simulate_gbm(s0=100.0, mu=0.08, sigma=0.25, t=8.0, n_steps=2016, n_paths=2, seed=4)
    prices = pd.DataFrame(paths.T, columns=["X", "Y"])
    estimates = SDEFitter(prices).fit_gbm()
    assert list(estimates.index) == ["X", "Y"]
    assert {"mu", "sigma", "Log-Likelihood", "AIC", "BIC"} <= set(estimates.columns)
    assert np.allclose(estimates["sigma"], 0.25, rtol=0.05)

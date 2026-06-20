"""Tests for the risk-parity (equal-risk-contribution) allocator."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from portfolio_optimisation.optim import RiskParityModel, risk_parity_weights
from portfolio_optimisation.risk import percentage_risk_contributions


def _returns(seed: int = 0, n: int = 400, k: int = 5) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    factor = rng.normal(0.0, 0.010, size=(n, 1))
    idiosyncratic = rng.normal(0.0, 0.008, size=(n, k))
    data = factor + idiosyncratic
    return pd.DataFrame(data, columns=[f"A{i}" for i in range(k)])


def test_equalises_risk_contributions() -> None:
    returns = _returns()
    cov = returns.cov()
    weights = risk_parity_weights(returns, cov_matrix=cov)
    percentage = percentage_risk_contributions(weights, cov).to_numpy()
    assert abs(float(weights.sum()) - 1.0) < 1e-8
    assert bool((weights.to_numpy() >= -1e-12).all())
    assert np.allclose(percentage, 1.0 / len(weights), atol=1e-3)


def test_respects_custom_budgets() -> None:
    returns = _returns()
    cov = returns.cov()
    budgets = np.array([0.4, 0.2, 0.2, 0.1, 0.1])
    weights = risk_parity_weights(returns, cov_matrix=cov, risk_budgets=budgets)
    percentage = percentage_risk_contributions(weights, cov).to_numpy()
    assert np.allclose(percentage, budgets, atol=2e-3)


def test_model_wrapper_matches_function() -> None:
    returns = _returns()
    model = RiskParityModel(returns)
    pd.testing.assert_series_equal(model.optimise(), risk_parity_weights(returns))


def test_invalid_budgets_raise() -> None:
    returns = _returns()
    with pytest.raises(ValueError, match="risk_budgets"):
        risk_parity_weights(returns, risk_budgets=np.array([1.0, 2.0]))

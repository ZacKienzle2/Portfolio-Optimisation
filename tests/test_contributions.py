"""Tests for the risk-contribution decomposition."""

from __future__ import annotations

import numpy as np
import pandas as pd

from portfolio_optimisation.risk import (
    component_risk_contributions,
    percentage_risk_contributions,
    portfolio_volatility,
    risk_concentration,
)

_TICKERS = ["A", "B", "C"]
_COV = pd.DataFrame(
    np.array([[0.04, 0.01, 0.00], [0.01, 0.09, 0.02], [0.00, 0.02, 0.16]]),
    index=_TICKERS,
    columns=_TICKERS,
)


def test_component_contributions_sum_to_volatility() -> None:
    weights = pd.Series([0.5, 0.3, 0.2], index=_TICKERS)
    component = component_risk_contributions(weights, _COV)
    assert abs(float(component.sum()) - portfolio_volatility(weights, _COV)) < 1e-12


def test_percentage_contributions_sum_to_one() -> None:
    weights = pd.Series([0.5, 0.3, 0.2], index=_TICKERS)
    assert abs(float(percentage_risk_contributions(weights, _COV).sum()) - 1.0) < 1e-12


def test_symmetric_case_has_equal_risk_contributions() -> None:
    cov = pd.DataFrame(np.eye(3) * 0.04, index=_TICKERS, columns=_TICKERS)
    weights = pd.Series([1 / 3, 1 / 3, 1 / 3], index=_TICKERS)
    percentage = percentage_risk_contributions(weights, cov).to_numpy()
    assert np.allclose(percentage, 1 / 3, atol=1e-12)
    assert abs(risk_concentration(weights, cov) - 1 / 3) < 1e-12


def test_single_asset_has_maximal_concentration() -> None:
    cov = pd.DataFrame(np.eye(3) * 0.04, index=_TICKERS, columns=_TICKERS)
    weights = pd.Series([1.0, 0.0, 0.0], index=_TICKERS)
    assert abs(risk_concentration(weights, cov) - 1.0) < 1e-12


def test_accepts_numpy_inputs() -> None:
    cov = np.eye(2) * 0.04
    weights = np.array([0.5, 0.5])
    component = component_risk_contributions(weights, cov)
    assert abs(float(component.sum()) - portfolio_volatility(weights, cov)) < 1e-12

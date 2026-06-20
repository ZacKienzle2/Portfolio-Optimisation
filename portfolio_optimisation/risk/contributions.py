"""Risk decomposition: marginal, component and percentage risk contributions.

For a long-short portfolio with weights ``w`` and covariance ``Sigma`` the
portfolio volatility is ``sigma_p = sqrt(w' Sigma w)``. Euler's theorem for the
(homogeneous degree-one) volatility splits it additively across assets:

    marginal_i   = d sigma_p / d w_i = (Sigma w)_i / sigma_p,
    component_i  = w_i * marginal_i,    sum_i component_i = sigma_p,
    percentage_i = component_i / sigma_p, sum_i percentage_i = 1.

These diagnostics show where portfolio risk actually concentrates, which is the
quantity risk-parity and equal-risk-contribution allocations target.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray

_VolInputs = tuple[NDArray[np.float64], NDArray[np.float64], list[Any] | None]


def _as_arrays(
    weights: pd.Series | NDArray[np.float64],
    covariance: pd.DataFrame | NDArray[np.float64],
) -> _VolInputs:
    if isinstance(covariance, pd.DataFrame):
        cov = covariance.to_numpy(dtype=np.float64)
        index = list(covariance.columns)
        if isinstance(weights, pd.Series):
            w = weights.reindex(covariance.columns).fillna(0.0).to_numpy(dtype=np.float64)
        else:
            w = np.asarray(weights, dtype=np.float64)
    else:
        cov = np.asarray(covariance, dtype=np.float64)
        index = list(weights.index) if isinstance(weights, pd.Series) else None
        w = np.asarray(weights, dtype=np.float64)
    if w.shape[0] != cov.shape[0]:
        raise ValueError("weights and covariance dimensions do not match.")
    return w, cov, index


def portfolio_volatility(
    weights: pd.Series | NDArray[np.float64],
    covariance: pd.DataFrame | NDArray[np.float64],
) -> float:
    """Portfolio volatility ``sqrt(w' Sigma w)``."""
    w, cov, _ = _as_arrays(weights, covariance)
    return float(np.sqrt(max(w @ cov @ w, 0.0)))


def marginal_risk_contributions(
    weights: pd.Series | NDArray[np.float64],
    covariance: pd.DataFrame | NDArray[np.float64],
) -> pd.Series:
    """Marginal contribution to volatility, ``(Sigma w)_i / sigma_p``."""
    w, cov, index = _as_arrays(weights, covariance)
    sigma = float(np.sqrt(max(w @ cov @ w, 0.0)))
    marginal = np.zeros_like(w) if sigma <= 0.0 else (cov @ w) / sigma
    return pd.Series(marginal, index=index)


def component_risk_contributions(
    weights: pd.Series | NDArray[np.float64],
    covariance: pd.DataFrame | NDArray[np.float64],
) -> pd.Series:
    """Component contributions ``w_i * marginal_i`` that sum to ``sigma_p``."""
    w, _, _ = _as_arrays(weights, covariance)
    marginal = marginal_risk_contributions(weights, covariance)
    return pd.Series(w * marginal.to_numpy(), index=marginal.index)


def percentage_risk_contributions(
    weights: pd.Series | NDArray[np.float64],
    covariance: pd.DataFrame | NDArray[np.float64],
) -> pd.Series:
    """Component contributions normalised to sum to one."""
    component = component_risk_contributions(weights, covariance)
    total = float(component.sum())
    if total == 0.0:
        return component
    return component / total


def risk_concentration(
    weights: pd.Series | NDArray[np.float64],
    covariance: pd.DataFrame | NDArray[np.float64],
) -> float:
    """Herfindahl-Hirschman index of the percentage risk contributions.

    Ranges from ``1/N`` (perfect risk parity) to ``1`` (all risk in one asset).
    """
    percentage = percentage_risk_contributions(weights, covariance).to_numpy()
    return float(np.sum(percentage**2))

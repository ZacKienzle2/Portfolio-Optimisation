"""Risk-parity (equal-risk-contribution) allocation.

A risk-parity portfolio equalises each asset's contribution to total
volatility (optionally to a supplied risk budget). The weights solve the
convex programme

    min_{y > 0}  0.5 y' Sigma y - sum_i b_i log(y_i),    w = y / sum(y),

whose stationarity condition ``(Sigma y)_i = b_i / y_i`` is exactly the
equal-(budgeted-)risk-contribution condition. The problem is convex, so the
solution is unique and solver-independent.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy.optimize import minimize
from sklearn.covariance import ledoit_wolf


def _ledoit_wolf_cov(returns: pd.DataFrame) -> pd.DataFrame:
    cov, _ = ledoit_wolf(returns, assume_centered=False)
    return pd.DataFrame(cov, index=returns.columns, columns=returns.columns)


def risk_parity_weights(
    returns: pd.DataFrame,
    *,
    cov_matrix: pd.DataFrame | None = None,
    risk_budgets: NDArray[np.float64] | None = None,
) -> pd.Series:
    """Compute long-only risk-parity weights summing to one.

    Args:
        returns (pd.DataFrame): Historical asset returns; columns are tickers.
        cov_matrix (pd.DataFrame | None): Covariance to use. Defaults to
            Ledoit-Wolf shrinkage of ``returns``.
        risk_budgets (NDArray[float64] | None): Target risk contributions per
            asset. Defaults to equal budgets (``1/N``). Normalised internally.

    Returns:
        pd.Series: Long-only risk-parity weights summing to one, indexed by
        ticker.
    """
    cov_df = _ledoit_wolf_cov(returns) if cov_matrix is None else cov_matrix
    tickers = list(cov_df.columns)
    cov = cov_df.to_numpy(dtype=np.float64)
    n = cov.shape[0]

    if risk_budgets is None:
        budgets = np.full(n, 1.0 / n)
    else:
        budgets = np.asarray(risk_budgets, dtype=np.float64).ravel()
        if budgets.shape[0] != n or np.any(budgets <= 0.0):
            raise ValueError("risk_budgets must be positive and match the asset count.")
        budgets = budgets / budgets.sum()

    def objective(y: NDArray[np.float64]) -> float:
        return float(0.5 * y @ cov @ y - budgets @ np.log(y))

    def gradient(y: NDArray[np.float64]) -> NDArray[np.float64]:
        return cov @ y - budgets / y

    start = 1.0 / np.sqrt(np.clip(np.diag(cov), 1e-12, None))
    result = minimize(
        objective,
        start,
        jac=gradient,
        method="L-BFGS-B",
        bounds=[(1e-12, None)] * n,
    )
    weights = result.x / result.x.sum()
    return pd.Series(weights, index=tickers)


class RiskParityModel:
    """Object wrapper around :func:`risk_parity_weights`."""

    def __init__(
        self,
        returns: pd.DataFrame,
        *,
        cov_matrix: pd.DataFrame | None = None,
        risk_budgets: NDArray[np.float64] | None = None,
    ) -> None:
        self.returns = returns
        self.cov_matrix = cov_matrix
        self.risk_budgets = risk_budgets
        self.weights: pd.Series = pd.Series(dtype=np.float64)

    def optimise(self) -> pd.Series:
        """Solve the risk-parity programme and cache the weights."""
        self.weights = risk_parity_weights(
            self.returns,
            cov_matrix=self.cov_matrix,
            risk_budgets=self.risk_budgets,
        )
        return self.weights

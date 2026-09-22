"""Most-diversified portfolio of Choueifaty and Coignard.

The diversification ratio of a long-only portfolio is its weighted average
volatility over its volatility,

    DR(w) = w' sigma / sqrt(w' Sigma w),

at least one, with equality only for perfectly correlated holdings
(Choueifaty and Coignard, 2008). The most-diversified portfolio maximises it
over the long-only simplex. The ratio is invariant to scaling the weights, so
Choueifaty, Froidure and Reynier (2013) solve the quadratic
programme

    min  y' Sigma y   s.t.  sigma' y = 1,  y >= 0,    w = y / sum(y),

whose solution exists, and is unique for a definite covariance. Their core
property characterises it. Every asset held has the same correlation with the
portfolio, and every asset left out is at least as correlated with it.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from portfolio_optimisation.optim.constraints import DEFAULT_SOLVER, require_cvxpy
from portfolio_optimisation.optim.constraints import long_only as project_long_only
from portfolio_optimisation.optim.shrinkage import linear_shrinkage_covariance

if TYPE_CHECKING:
    from numpy.typing import NDArray


def diversification_ratio(weights: NDArray[np.float64], covariance: NDArray[np.float64]) -> float:
    """Weighted average volatility over portfolio volatility.

    Args:
        weights: Portfolio weights.
        covariance: Asset covariance.

    Returns:
        The diversification ratio.
    """
    volatilities = np.sqrt(np.diag(covariance))
    return float(weights @ volatilities / np.sqrt(weights @ covariance @ weights))


def max_diversification_weights(
    returns: pd.DataFrame,
    *,
    cov_matrix: pd.DataFrame | None = None,
    solver: str = DEFAULT_SOLVER,
) -> pd.Series:
    """Long-only weights of the most-diversified portfolio.

    Args:
        returns: Historical asset returns; columns are tickers.
        cov_matrix: Covariance to use. Defaults to Ledoit-Wolf shrinkage.
        solver: ``cvxpy`` solver name.

    Returns:
        Long-only weights summing to one, indexed by ticker.

    Raises:
        RuntimeError: If the solver does not report an optimal solution.
    """
    cp = require_cvxpy()
    cov_df = linear_shrinkage_covariance(returns) if cov_matrix is None else cov_matrix
    covariance = cov_df.to_numpy(dtype=np.float64)
    volatilities = np.sqrt(np.diag(covariance))
    y = cp.Variable(covariance.shape[0], nonneg=True)
    problem = cp.Problem(
        cp.Minimize(cp.quad_form(y, cp.psd_wrap(covariance))), [volatilities @ y == 1.0]
    )
    problem.solve(solver=solver)
    if problem.status not in {"optimal", "optimal_inaccurate"}:
        msg = f"most-diversified QP solver failed: status={problem.status}"
        raise RuntimeError(msg)
    return pd.Series(project_long_only(np.asarray(y.value)), index=list(cov_df.columns))

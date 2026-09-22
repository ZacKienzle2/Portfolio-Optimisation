"""Conditional Drawdown-at-Risk allocation.

For portfolio weight ``w``, cumulative log-return ``P_t = sum_{s <= t} r_s' w``,
and running maximum ``M_t = max_{s <= t} P_s``, define the drawdown
``D_t = M_t - P_t >= 0``. The alpha-CDaR is the expected drawdown beyond the
VaR threshold at level alpha. This is a coherent risk measure with the LP
formulation:

    min  zeta + 1 / (alpha T) * sum_t u_t
    s.t. u_t >= D_t - zeta,        u_t >= 0
         D_t >= M_t - P_t,         M_t >= P_s for all s <= t.

A standard simplification linearises ``M_t`` via a non-decreasing auxiliary
variable ``m_t`` with ``m_t >= m_{t-1}`` and ``m_t >= P_t``. We minimise CDaR
subject to budget, long-only and (optional) expected-return constraints.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from portfolio_optimisation.optim.constraints import PortfolioConstraints, require_cvxpy
from portfolio_optimisation.optim.constraints import long_only as project_long_only


def min_cdar_weights(
    returns: pd.DataFrame,
    *,
    alpha: float = 0.05,
    constraints: PortfolioConstraints | None = None,
    solver: str | None = None,
) -> pd.Series:
    """Solve the Chekhlov-Uryasev minimum-CDaR linear programme.

    Args:
        returns: Asset returns; rows are dates, columns assets.
        alpha: Tail level in (0, 1). The objective averages drawdowns in the worst
            ``alpha`` fraction of the cumulative path.
        constraints: Feasibility set shared with the other mean-risk allocators.
            Defaults to a long-only, fully-invested mandate.
        solver: Optional cvxpy solver name. Defaults to automatic selection.

    Returns:
        pd.Series: Optimal weights indexed by ticker.

    Raises:
        ValueError: If ``alpha`` lies outside ``(0, 1)``.
        RuntimeError: If the solver does not report an optimal solution.
    """
    if not 0.0 < alpha < 1.0:
        msg = "alpha must lie in (0, 1)."
        raise ValueError(msg)
    cp = require_cvxpy()
    spec = constraints if constraints is not None else PortfolioConstraints()
    tickers = list(returns.columns)
    r_arr = returns.to_numpy(dtype=np.float64)
    t_steps, n_assets = r_arr.shape

    w = cp.Variable(n_assets)
    peak = cp.Variable(t_steps)
    cumulative = cp.cumsum(r_arr @ w)
    problem_constraints = spec.build(cp, w, n_assets=n_assets, expected_returns=r_arr.mean(axis=0))
    problem_constraints += [peak >= cumulative, peak[1:] >= peak[:-1], peak[0] >= 0]
    problem = cp.Problem(cp.Minimize(cp.cvar(peak - cumulative, 1.0 - alpha)), problem_constraints)
    problem.solve(solver=solver)

    if problem.status not in {"optimal", "optimal_inaccurate"}:
        msg = f"CDaR LP solver failed: status={problem.status}"
        raise RuntimeError(msg)

    weights = np.asarray(w.value, dtype=np.float64)
    return pd.Series(project_long_only(weights) if spec.long_only else weights, index=tickers)


def cdar(returns: pd.Series, *, alpha: float = 0.05) -> float:
    """Compute the empirical conditional drawdown at risk for a return series.

    Useful for ex-post evaluation of any candidate allocation.
    """
    if not 0.0 < alpha < 1.0:
        msg = "alpha must lie in (0, 1)."
        raise ValueError(msg)
    arr = returns.to_numpy(dtype=np.float64)
    cumulative = np.cumsum(arr)
    running_max = np.maximum.accumulate(cumulative)
    drawdowns = running_max - cumulative
    threshold = float(np.quantile(drawdowns, 1.0 - alpha))
    tail = drawdowns[drawdowns >= threshold]
    if tail.size == 0:
        return threshold
    return float(tail.mean())

"""Conditional Drawdown-at-Risk allocation.

Chekhlov, A., Uryasev, S., Zabarankin, M. (2005).
    "Drawdown Measure in Portfolio Optimization." International Journal of
    Theoretical and Applied Finance 8(1):13-58.

For portfolio weight ``w``, cumulative log-PnL ``P_t = sum_{s <= t} r_s' w``,
and running maximum ``M_t = max_{s <= t} P_s``, define the drawdown
``D_t = M_t - P_t >= 0``. The alpha-CDaR is the expected drawdown beyond the
VaR threshold at level alpha. Chekhlov-Uryasev show it is a coherent risk
measure with the LP formulation:

    min  zeta + 1 / ((1-alpha) T) * sum_t u_t
    s.t. u_t >= D_t - zeta,        u_t >= 0
         D_t >= M_t - P_t,         M_t >= P_s for all s <= t.

A standard simplification linearises ``M_t`` via a non-decreasing auxiliary
variable ``m_t`` with ``m_t >= m_{t-1}`` and ``m_t >= P_t``. We minimise CDaR
subject to budget, long-only and (optional) expected-return constraints.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:  # pragma: no cover - typing only
    import cvxpy as cp


def _require_cvxpy() -> cp:
    try:
        import cvxpy as cp
    except ImportError as exc:  # pragma: no cover - exercised via [optim] extra
        raise ImportError(
            "min_cdar requires the [optim] extra: `uv sync --extra optim` or "
            "`pip install 'portfolio-optimisation[optim]'`."
        ) from exc
    return cp


def min_cdar_weights(
    returns: pd.DataFrame,
    *,
    alpha: float = 0.05,
    target_return: float | None = None,
    long_only: bool = True,
    solver: str | None = None,
) -> pd.Series:
    """Solve the Chekhlov-Uryasev minimum-CDaR LP.

    Args:
        returns (pd.DataFrame): Asset returns; rows are dates, columns assets.
        alpha (float): Tail level in (0, 1). The objective averages drawdowns
            beyond the ``(1 - alpha)``-quantile of the drawdown distribution.
        target_return (float | None): Optional lower bound on the in-sample
            mean portfolio return.
        long_only (bool): If True, enforce ``w >= 0`` and ``sum w == 1``;
            otherwise only ``sum w == 1`` (allow shorts).
        solver: cvxpy solver name. Defaults to None (auto-select; ECOS for LP).

    Returns:
        pd.Series: Optimal weights indexed by ticker.
    """
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must lie in (0, 1).")
    cp = _require_cvxpy()
    tickers = list(returns.columns)
    r_arr = returns.to_numpy(dtype=np.float64)
    t_steps, n_assets = r_arr.shape

    w = cp.Variable(n_assets)
    zeta = cp.Variable()
    u = cp.Variable(t_steps, nonneg=True)
    m = cp.Variable(t_steps)
    # Cumulative portfolio return P_t = sum_{s<=t} r_s' w.
    cumulative = cp.cumsum(r_arr @ w)

    constraints: list[cp.constraints.constraint.Constraint] = []
    constraints += [u >= (m - cumulative) - zeta]
    constraints += [m >= cumulative]
    constraints += [m[1:] >= m[:-1]]
    constraints += [m[0] >= 0]
    constraints += [cp.sum(w) == 1]
    if long_only:
        constraints += [w >= 0]
    if target_return is not None:
        mean_r = r_arr.mean(axis=0)
        constraints += [mean_r @ w >= target_return]

    objective = cp.Minimize(zeta + cp.sum(u) / ((1.0 - alpha) * t_steps))
    problem = cp.Problem(objective, constraints)
    problem.solve(solver=solver)

    if problem.status not in {"optimal", "optimal_inaccurate"}:
        raise RuntimeError(f"CDaR LP solver failed: status={problem.status}")

    weights = np.asarray(w.value, dtype=np.float64)
    weights = np.clip(weights, 0.0, None) if long_only else weights
    total = weights.sum()
    if total > 0:
        weights = weights / total
    return pd.Series(weights, index=tickers)


def cdar(returns: pd.Series, *, alpha: float = 0.05) -> float:
    """Compute the empirical conditional drawdown at risk for a return series.

    Useful for ex-post evaluation of any candidate allocation.
    """
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must lie in (0, 1).")
    arr = returns.to_numpy(dtype=np.float64)
    cumulative = np.cumsum(arr)
    running_max = np.maximum.accumulate(cumulative)
    drawdowns = running_max - cumulative
    threshold = float(np.quantile(drawdowns, 1.0 - alpha))
    tail = drawdowns[drawdowns >= threshold]
    if tail.size == 0:
        return threshold
    return float(tail.mean())

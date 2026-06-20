"""Second-order Stochastic Dominance (SSD) constrained portfolio optimisation.

Dentcheva, D., Ruszczynski, A. (2003).
    "Optimization with stochastic dominance constraints." SIAM Journal on
    Optimization 14(2):548-566.

A portfolio random variable ``X`` second-order stochastically dominates a
benchmark ``Y`` if and only if

    E[max(eta - X, 0)] <= E[max(eta - Y, 0)]    for all eta in R.

On a discrete return panel with T scenarios, Dentcheva-Ruszczynski show that
the continuum of constraints collapses to constraints evaluated at the
benchmark realisations: ``eta_i = Y_i, i = 1, ..., T``. The SSD-constrained
expected-return maximisation becomes the LP

    max  mu' w
    s.t. (1/T) sum_t u_{t,i} <= s_i(Y)            for i = 1, ..., T
         u_{t,i} >= eta_i - r_t' w                 t, i = 1, ..., T
         u_{t,i} >= 0                              t, i
         sum w = 1,   w >= 0 (long-only)

where ``s_i(Y) = (1/T) sum_t max(eta_i - Y_t, 0)`` is the empirical lower
partial moment of the benchmark at threshold ``eta_i``. This file exposes a
single :func:`ssd_constrained_weights` entry point.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _require_cvxpy():
    try:
        import cvxpy as cp
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "ssd_constrained_weights requires the [optim] extra: `uv sync --extra optim`."
        ) from exc
    return cp


def ssd_constrained_weights(
    returns: pd.DataFrame,
    benchmark_returns: pd.Series,
    *,
    long_only: bool = True,
    solver: str | None = None,
) -> pd.Series:
    """Solve max ``mu' w`` subject to SSD-dominance of the benchmark.

    Args:
        returns (pd.DataFrame): Asset returns; rows scenarios, columns assets.
        benchmark_returns (pd.Series): Benchmark scenario returns of length
            ``T`` (must match ``returns.shape[0]``).
        long_only (bool): If True, enforce ``w >= 0``.
        solver: Optional cvxpy solver name.

    Returns:
        pd.Series: Optimal weights indexed by ticker.
    """
    if benchmark_returns.shape[0] != returns.shape[0]:
        raise ValueError("benchmark_returns must have the same length as returns.")
    cp = _require_cvxpy()

    tickers = list(returns.columns)
    r = returns.to_numpy(dtype=np.float64)
    y = benchmark_returns.to_numpy(dtype=np.float64)
    t_steps, n_assets = r.shape

    # Benchmark lower-partial moments at thresholds eta_i = y_i.
    eta = y.copy()
    diff = eta[:, None] - y[None, :]
    s_benchmark = np.maximum(diff, 0.0).mean(axis=1)

    w = cp.Variable(n_assets)
    u = cp.Variable((t_steps, t_steps), nonneg=True)
    portfolio = r @ w

    constraints = [
        cp.sum(w) == 1,
        u >= eta[:, None] - portfolio[None, :],
        cp.sum(u, axis=1) / t_steps <= s_benchmark,
    ]
    if long_only:
        constraints += [w >= 0]

    mu = r.mean(axis=0)
    problem = cp.Problem(cp.Maximize(mu @ w), constraints)
    problem.solve(solver=solver)

    if problem.status not in {"optimal", "optimal_inaccurate"}:
        raise RuntimeError(f"SSD LP solver failed: status={problem.status}")

    weights = np.asarray(w.value, dtype=np.float64)
    if long_only:
        weights = np.clip(weights, 0.0, None)
        total = weights.sum()
        if total > 0:
            weights = weights / total
    return pd.Series(weights, index=tickers)


def ssd_dominates(portfolio_returns: pd.Series, benchmark_returns: pd.Series) -> bool:
    """Empirical test: does ``portfolio_returns`` SSD-dominate the benchmark?

    Checks the integrated lower partial moment criterion at every benchmark
    realisation. Returns True iff
    ``E[max(eta - X, 0)] <= E[max(eta - Y, 0)]`` at every ``eta in Y``.
    """
    x = portfolio_returns.to_numpy(dtype=np.float64)
    y = benchmark_returns.to_numpy(dtype=np.float64)
    eta = y
    lpm_x = np.maximum(eta[:, None] - x[None, :], 0.0).mean(axis=1)
    lpm_y = np.maximum(eta[:, None] - y[None, :], 0.0).mean(axis=1)
    return bool(np.all(lpm_x <= lpm_y + 1e-9))

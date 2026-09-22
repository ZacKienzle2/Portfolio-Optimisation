"""Second-order Stochastic Dominance (SSD) constrained portfolio optimisation.

A portfolio random variable ``X`` second-order stochastically dominates a
benchmark ``Y`` if and only if

    E[max(eta - X, 0)] <= E[max(eta - Y, 0)]    for all eta in R.

On a panel of ``T`` equally likely scenarios this is equivalent to dominance of
the tail sums: for every ``k``, the sum of the ``k`` worst portfolio outcomes is
at least the sum of the ``k`` worst benchmark outcomes. Each tail sum of the
portfolio is the minimum over ``k``-subsets ``J`` of ``sum_{t in J} r_t' w``, so
the SSD-constrained expected-return maximisation is the linear programme

    max  mu' w
    s.t. sum_{t in J} r_t' w >= sum_{i <= k} y_(i)    for every J with |J| = k
         sum w = 1,   w >= 0 (long-only)

with exponentially many constraints, of which only the binding few matter.
:func:`ssd_constrained_weights` solves it by cutting planes, the method of
Fabian, Mitra and Roman, adding for each violated ``k`` the cut whose subset is
the current portfolio's ``k`` worst scenarios. Each master problem has ``N``
variables, where the formulation with one auxiliary variable per scenario pair
has ``T^2``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from scipy.optimize import linprog

from portfolio_optimisation.optim.constraints import long_only as project_long_only

if TYPE_CHECKING:
    from numpy.typing import NDArray

_CUTS_PER_ROUND = 20
_TOLERANCE = 1e-7


def lower_partial_moments(
    sample: NDArray[np.float64], thresholds: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Lower partial moment ``E[max(eta - X, 0)]`` at each threshold.

    One sort and one cumulative sum of the sample serve every threshold, which
    ``searchsorted`` locates in the sorted outcomes.

    Args:
        sample: Scenario outcomes of ``X``.
        thresholds: Thresholds ``eta``.

    Returns:
        The lower partial moment at each threshold.
    """
    ordered = np.sort(sample)
    prefix = np.concatenate([[0.0], np.cumsum(ordered)])
    below = np.searchsorted(ordered, thresholds, side="right")
    return (thresholds * below - prefix[below]) / sample.size


def ssd_constrained_weights(
    returns: pd.DataFrame,
    benchmark_returns: pd.Series,
    *,
    long_only: bool = True,
    solver: str = "highs",
    max_iterations: int = 1000,
) -> pd.Series:
    """Solve max ``mu' w`` subject to SSD dominance of the benchmark.

    Each round adds the cuts of the ``_CUTS_PER_ROUND`` most violated tail sums
    and stops when none is violated. Adding every violated cut grew the master
    problem faster than it converged, five times slower on 2000 scenarios of 30
    assets. A long-only master problem is bounded by the simplex alone, while a
    long-short one starts from the single-scenario cuts, which bound it
    whenever the full programme is bounded. A tail sum counts as violated only
    beyond HiGHS's primal feasibility tolerance, ``_TOLERANCE``; below it the
    master solution already satisfies the cut to the solver's precision, and a
    tighter test re-adds the same cut indefinitely.

    Args:
        returns: Asset returns, rows scenarios and columns assets.
        benchmark_returns: Benchmark scenario returns, as many as ``returns`` has rows.
        long_only: Enforce ``w >= 0``.
        solver: ``scipy.optimize.linprog`` HiGHS method.
        max_iterations: Bound on the number of cutting-plane rounds.

    Returns:
        Optimal weights indexed by ticker.

    Raises:
        ValueError: If the benchmark length differs from the scenario count.
        RuntimeError: If a master problem fails or the cuts do not converge.
    """
    if benchmark_returns.shape[0] != returns.shape[0]:
        msg = "benchmark_returns must have the same length as returns."
        raise ValueError(msg)
    r = returns.to_numpy(dtype=np.float64)
    tails = np.cumsum(np.sort(benchmark_returns.to_numpy(dtype=np.float64)))
    tolerance = _TOLERANCE * (1.0 + np.abs(tails))
    n_assets = r.shape[1]

    cut_rows: list[NDArray[np.float64]] = [np.empty((0, n_assets)) if long_only else -r]
    cut_bounds: list[NDArray[np.float64]] = [
        np.empty(0) if long_only else np.full(r.shape[0], -tails[0])
    ]
    for _ in range(max_iterations):
        rows = np.vstack(cut_rows)
        result = linprog(
            -r.mean(axis=0),
            A_ub=rows if rows.size else None,
            b_ub=np.concatenate(cut_bounds) if rows.size else None,
            A_eq=np.ones((1, n_assets)),
            b_eq=[1.0],
            bounds=(0.0, None) if long_only else (None, None),
            method=solver,
        )
        if result.status != 0:
            msg = f"SSD master LP failed: {result.message}"
            raise RuntimeError(msg)
        order = np.argsort(r @ result.x)
        worst = np.cumsum(r[order], axis=0)
        shortfall = tails - tolerance - worst @ result.x
        violated = np.flatnonzero(shortfall > 0.0)
        if violated.size == 0:
            weights = project_long_only(result.x) if long_only else result.x
            return pd.Series(weights, index=returns.columns)
        violated = violated[np.argsort(-shortfall[violated])[:_CUTS_PER_ROUND]]
        cut_rows.append(-worst[violated])
        cut_bounds.append(-tails[violated])
    msg = f"SSD cutting planes did not converge in {max_iterations} rounds."
    raise RuntimeError(msg)


def ssd_dominates(portfolio_returns: pd.Series, benchmark_returns: pd.Series) -> bool:
    """Empirical test of whether ``portfolio_returns`` SSD-dominates the benchmark.

    Checks the lower partial moment criterion
    ``E[max(eta - X, 0)] <= E[max(eta - Y, 0)]`` at every benchmark realisation.

    Args:
        portfolio_returns: Candidate scenario returns.
        benchmark_returns: Benchmark scenario returns.

    Returns:
        Whether the criterion holds at every realisation.
    """
    x = portfolio_returns.to_numpy(dtype=np.float64)
    y = benchmark_returns.to_numpy(dtype=np.float64)
    return bool(np.all(lower_partial_moments(x, y) <= lower_partial_moments(y, y) + 1e-9))

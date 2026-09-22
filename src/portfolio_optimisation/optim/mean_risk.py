"""Mean-risk portfolio optimisation under coherent tail measures.

Two convex programmes minimise a coherent tail measure of the portfolio loss
``L_t = -(r_t' w)`` subject to a shared :class:`PortfolioConstraints` set:

* :func:`min_cvar_weights` minimises Conditional Value-at-Risk, the average
  of the worst ``alpha`` fraction of losses, through cvxpy's ``cvar`` atom,
  which canonicalises to the Rockafellar-Uryasev linear programme.

* :func:`min_evar_weights` minimises Entropic Value-at-Risk, the tightest
  coherent upper bound on CVaR, through the exponential-cone programme

      min   t - z * log(alpha T)
      s.t.  sum_t u_t <= z,   u_t >= z * exp((L_t - t) / z),   z >= 0,

  whose perspective constraint ``u_t >= z exp((L_t - t)/z)`` is the exponential
  cone. At the optimum the objective equals
  ``z log((1 / (alpha T)) sum_t exp(L_t / z))``, the empirical EVaR.

Both pair the tail objective with a minimum expected-return floor (via the
constraint set) to trace a mean-risk efficient frontier, and both require the
``[optim]`` extra for ``cvxpy``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import numpy as np
import pandas as pd

from portfolio_optimisation.optim.constraints import (
    DEFAULT_SOLVER,
    PortfolioConstraints,
    require_cvxpy,
)
from portfolio_optimisation.optim.constraints import long_only as project_long_only

if TYPE_CHECKING:
    from numpy.typing import NDArray

_EVAR_CONDITIONING_SCALE = 100.0


def _finalise(
    weights: NDArray[np.float64] | None, tickers: list[str], *, long_only: bool
) -> pd.Series:
    clean = np.asarray(weights, dtype=np.float64)
    return pd.Series(project_long_only(clean) if long_only else clean, index=tickers)


def min_cvar_weights(
    returns: pd.DataFrame,
    *,
    alpha: float = 0.05,
    constraints: PortfolioConstraints | None = None,
    solver: str = DEFAULT_SOLVER,
) -> pd.Series:
    """Solve the Rockafellar-Uryasev minimum-CVaR linear programme.

    Args:
        returns: Asset returns; rows are scenarios, columns assets.
        alpha: Tail level in (0, 1). The objective averages losses in the worst
            ``alpha`` fraction of scenarios.
        constraints: Feasibility set. Defaults to a long-only, fully-invested mandate.
        solver: ``cvxpy`` solver name.

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
    r = returns.to_numpy(dtype=np.float64)
    w = cp.Variable(r.shape[1])
    problem = cp.Problem(
        cp.Minimize(cp.cvar(-(r @ w), 1.0 - alpha)),
        spec.build(cp, w, n_assets=r.shape[1], expected_returns=r.mean(axis=0)),
    )
    problem.solve(solver=solver)

    if problem.status not in {"optimal", "optimal_inaccurate"}:
        msg = f"mean-CVaR LP solver failed: status={problem.status}"
        raise RuntimeError(msg)

    return _finalise(w.value, tickers, long_only=spec.long_only)


def min_evar_weights(
    returns: pd.DataFrame,
    *,
    alpha: float = 0.05,
    constraints: PortfolioConstraints | None = None,
    solver: str = DEFAULT_SOLVER,
) -> pd.Series:
    """Solve the minimum-EVaR exponential-cone programme.

    Args:
        returns: Asset returns; rows are scenarios, columns assets.
        alpha: Tail level in (0, 1).
        constraints: Feasibility set. Defaults to a long-only, fully-invested mandate.
        solver: ``cvxpy`` solver name.

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
    r = returns.to_numpy(dtype=np.float64)
    t_steps, n_assets = r.shape

    w = cp.Variable(n_assets)
    t = cp.Variable()
    z = cp.Variable(nonneg=True)
    u = cp.Variable(t_steps)
    # EVaR is positively homogeneous in the loss, so scaling the cone losses
    # improves conditioning without changing the optimal weights.
    scaled_losses = -(r @ w) * _EVAR_CONDITIONING_SCALE

    problem_constraints = spec.build(cp, w, n_assets=n_assets, expected_returns=r.mean(axis=0))
    problem_constraints += [
        cp.sum(u) <= z,
        cp.constraints.ExpCone(scaled_losses - t, z * np.ones(t_steps), u),
    ]

    objective = cp.Minimize(t - z * np.log(alpha * t_steps))
    problem = cp.Problem(objective, problem_constraints)
    problem.solve(solver=solver)

    if problem.status not in {"optimal", "optimal_inaccurate"}:
        msg = f"mean-EVaR cone solver failed: status={problem.status}"
        raise RuntimeError(msg)

    return _finalise(w.value, tickers, long_only=spec.long_only)


def mean_risk_weights(
    returns: pd.DataFrame,
    *,
    measure: Literal["cvar", "evar"] = "cvar",
    alpha: float = 0.05,
    constraints: PortfolioConstraints | None = None,
    solver: str = DEFAULT_SOLVER,
) -> pd.Series:
    """Dispatch to the minimum-CVaR or minimum-EVaR optimiser.

    Args:
        returns: Asset returns; rows scenarios, columns assets.
        measure: ``"cvar"`` for the Rockafellar-Uryasev LP or ``"evar"`` for the
            exponential-cone EVaR programme.
        alpha: Tail level in (0, 1).
        constraints: Shared feasibility set.
        solver: ``cvxpy`` solver name.

    Returns:
        pd.Series: Optimal weights indexed by ticker.

    Raises:
        ValueError: If ``measure`` is neither ``"cvar"`` nor ``"evar"``.
    """
    if measure == "cvar":
        return min_cvar_weights(returns, alpha=alpha, constraints=constraints, solver=solver)
    if measure == "evar":
        return min_evar_weights(returns, alpha=alpha, constraints=constraints, solver=solver)
    msg = "measure must be 'cvar' or 'evar'."
    raise ValueError(msg)


def efficient_frontier(
    expected_returns: NDArray[np.float64],
    covariance: NDArray[np.float64],
    *,
    points: int = 100,
    constraints: PortfolioConstraints | None = None,
    solver: str = DEFAULT_SOLVER,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Trace the mean-variance frontier from minimum variance to maximum return.

    The target return is a cvxpy parameter, so the quadratic programme is
    canonicalised once and re-solved for each target rather than rebuilt.

    Args:
        expected_returns: Expected return of each asset.
        covariance: Positive semi-definite covariance matrix.
        points: Number of frontier portfolios.
        constraints: Feasibility set. Defaults to long-only and fully invested.
        solver: ``cvxpy`` solver name.

    Returns:
        Volatility and expected return of each frontier portfolio.
    """
    cp = require_cvxpy()
    spec = constraints if constraints is not None else PortfolioConstraints()
    mu = np.asarray(expected_returns, dtype=np.float64)
    w = cp.Variable(mu.size)
    feasible = spec.build(cp, w, n_assets=mu.size)
    variance = cp.quad_form(w, cp.psd_wrap(np.asarray(covariance, dtype=np.float64)))

    highest = cp.Problem(cp.Maximize(mu @ w), feasible)
    highest.solve(solver=solver)
    target = cp.Parameter()
    frontier = cp.Problem(cp.Minimize(variance), [*feasible, mu @ w >= target])
    target.value = -np.inf
    frontier.solve(solver=solver)
    lowest = float(mu @ w.value)

    risks, returns = [], []
    for level in np.linspace(lowest, float(highest.value), points):
        target.value = level
        frontier.solve(solver=solver)
        if frontier.status in {"optimal", "optimal_inaccurate"}:
            risks.append(float(np.sqrt(max(variance.value, 0.0))))
            returns.append(float(mu @ w.value))
    return np.asarray(risks), np.asarray(returns)


class MeanRiskModel:
    """Object wrapper around the mean-risk optimisers."""

    def __init__(
        self,
        returns: pd.DataFrame,
        *,
        measure: Literal["cvar", "evar"] = "cvar",
        alpha: float = 0.05,
        constraints: PortfolioConstraints | None = None,
        solver: str = DEFAULT_SOLVER,
    ) -> None:
        self.returns = returns
        self.measure: Literal["cvar", "evar"] = measure
        self.alpha = alpha
        self.constraints = constraints
        self.solver = solver
        self.weights: pd.Series = pd.Series(dtype=np.float64)

    def optimise(self) -> pd.Series:
        """Solve the configured mean-risk programme and cache the weights."""
        self.weights = mean_risk_weights(
            self.returns,
            measure=self.measure,
            alpha=self.alpha,
            constraints=self.constraints,
            solver=self.solver,
        )
        return self.weights

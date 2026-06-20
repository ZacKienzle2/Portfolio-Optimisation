"""Mean-risk portfolio optimisation under coherent tail measures.

Two convex programmes minimise a coherent tail measure of the portfolio loss
``L_t = -(r_t' w)`` subject to a shared :class:`PortfolioConstraints` set:

* :func:`min_cvar_weights` minimises Conditional Value-at-Risk via the
  Rockafellar-Uryasev linear programme

      min   zeta + 1 / (alpha T) * sum_t u_t
      s.t.  u_t >= L_t - zeta,    u_t >= 0,

  where ``zeta`` recovers the Value-at-Risk and the objective recovers the
  average loss beyond it. The programme is an LP.

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

from typing import Literal

import numpy as np
import pandas as pd

from portfolio_optimisation.optim.constraints import PortfolioConstraints

_EVAR_CONDITIONING_SCALE = 100.0


def _require_cvxpy():
    try:
        import cvxpy as cp
    except ImportError as exc:  # pragma: no cover - exercised via [optim] extra
        raise ImportError(
            "mean-risk optimisation requires the [optim] extra: "
            "`uv sync --extra optim` or `pip install 'portfolio-optimisation[optim]'`."
        ) from exc
    return cp


def _finalise(weights: np.ndarray | None, tickers: list[str], *, long_only: bool) -> pd.Series:
    clean = np.asarray(weights, dtype=np.float64)
    if long_only:
        clean = np.clip(clean, 0.0, None)
        total = clean.sum()
        if total > 0:
            clean = clean / total
    return pd.Series(clean, index=tickers)


def min_cvar_weights(
    returns: pd.DataFrame,
    *,
    alpha: float = 0.05,
    constraints: PortfolioConstraints | None = None,
    solver: str | None = None,
) -> pd.Series:
    """Solve the Rockafellar-Uryasev minimum-CVaR linear programme.

    Args:
        returns (pd.DataFrame): Asset returns; rows are scenarios, columns
            assets.
        alpha (float): Tail level in (0, 1). The objective averages losses in
            the worst ``alpha`` fraction of scenarios.
        constraints (PortfolioConstraints | None): Feasibility set. Defaults to
            a long-only, fully-invested mandate.
        solver: Optional ``cvxpy`` solver name. Defaults to automatic selection.

    Returns:
        pd.Series: Optimal weights indexed by ticker.
    """
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must lie in (0, 1).")
    cp = _require_cvxpy()
    spec = constraints if constraints is not None else PortfolioConstraints()
    tickers = list(returns.columns)
    r = returns.to_numpy(dtype=np.float64)
    t_steps, n_assets = r.shape

    w = cp.Variable(n_assets)
    zeta = cp.Variable()
    u = cp.Variable(t_steps, nonneg=True)
    losses = -(r @ w)

    problem_constraints = spec.build(cp, w, n_assets=n_assets, expected_returns=r.mean(axis=0))
    problem_constraints += [u >= losses - zeta]

    objective = cp.Minimize(zeta + cp.sum(u) / (alpha * t_steps))
    problem = cp.Problem(objective, problem_constraints)
    problem.solve(solver=solver)

    if problem.status not in {"optimal", "optimal_inaccurate"}:
        raise RuntimeError(f"mean-CVaR LP solver failed: status={problem.status}")

    return _finalise(w.value, tickers, long_only=spec.long_only)


def min_evar_weights(
    returns: pd.DataFrame,
    *,
    alpha: float = 0.05,
    constraints: PortfolioConstraints | None = None,
    solver: str | None = None,
) -> pd.Series:
    """Solve the minimum-EVaR exponential-cone programme.

    Args:
        returns (pd.DataFrame): Asset returns; rows are scenarios, columns
            assets.
        alpha (float): Tail level in (0, 1).
        constraints (PortfolioConstraints | None): Feasibility set. Defaults to
            a long-only, fully-invested mandate.
        solver: Optional ``cvxpy`` solver name. Defaults to automatic selection
            of an exponential-cone-capable solver.

    Returns:
        pd.Series: Optimal weights indexed by ticker.
    """
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must lie in (0, 1).")
    cp = _require_cvxpy()
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
        raise RuntimeError(f"mean-EVaR cone solver failed: status={problem.status}")

    return _finalise(w.value, tickers, long_only=spec.long_only)


def mean_risk_weights(
    returns: pd.DataFrame,
    *,
    measure: Literal["cvar", "evar"] = "cvar",
    alpha: float = 0.05,
    constraints: PortfolioConstraints | None = None,
    solver: str | None = None,
) -> pd.Series:
    """Dispatch to the minimum-CVaR or minimum-EVaR optimiser.

    Args:
        returns (pd.DataFrame): Asset returns; rows scenarios, columns assets.
        measure: ``"cvar"`` for the Rockafellar-Uryasev LP or ``"evar"`` for the
            exponential-cone EVaR programme.
        alpha (float): Tail level in (0, 1).
        constraints (PortfolioConstraints | None): Shared feasibility set.
        solver: Optional ``cvxpy`` solver name.

    Returns:
        pd.Series: Optimal weights indexed by ticker.
    """
    if measure == "cvar":
        return min_cvar_weights(returns, alpha=alpha, constraints=constraints, solver=solver)
    if measure == "evar":
        return min_evar_weights(returns, alpha=alpha, constraints=constraints, solver=solver)
    raise ValueError("measure must be 'cvar' or 'evar'.")


class MeanRiskModel:
    """Object wrapper around the mean-risk optimisers."""

    def __init__(
        self,
        returns: pd.DataFrame,
        *,
        measure: Literal["cvar", "evar"] = "cvar",
        alpha: float = 0.05,
        constraints: PortfolioConstraints | None = None,
        solver: str | None = None,
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

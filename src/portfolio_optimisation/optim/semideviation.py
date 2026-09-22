"""Mean-semideviation allocation consistent with second-order dominance.

Mean-variance choices can be dominated in the second order, since variance
penalises gains as much as losses. Ogryczak and Ruszczynski (1999) show that
the mean-risk models built on the semideviations below the mean are
consistent with second-order stochastic dominance for bounded trade-offs. For
the portfolio return ``X = r' w`` with mean ``mu_X``, the absolute
semideviation and the standard semideviation are

    delta_X = E[(mu_X - X)^+],      sigma_X = E[((mu_X - X)^+)^2]^(1/2),

and a maximiser of ``mu_X - lambda delta_X`` (their Corollary 4) or of
``mu_X - lambda sigma_X`` (Corollary 8) with ``0 < lambda <= 1`` is efficient
under the second-order rules, apart from ties in mean and semideviation. The
bound cannot be raised for general distributions. On scenario returns the
first is the linear programme of the Konno-Yamazaki family, and the second a
second-order cone programme.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd

from portfolio_optimisation.optim.constraints import (
    DEFAULT_SOLVER,
    PortfolioConstraints,
    require_cvxpy,
)
from portfolio_optimisation.optim.constraints import long_only as project_long_only

Semideviation = Literal["absolute", "standard"]


def mean_semideviation_weights(
    returns: pd.DataFrame,
    *,
    risk_aversion: float = 1.0,
    measure: Semideviation = "absolute",
    constraints: PortfolioConstraints | None = None,
    solver: str = DEFAULT_SOLVER,
) -> pd.Series:
    """Maximise mean return less ``risk_aversion`` times a semideviation.

    The objective is positively homogeneous in the returns, so they are divided
    by their standard deviation before the solve. Daily returns otherwise put
    the objective near ``1e-4``, where the solver's absolute tolerance left one
    solve in a hundred short of the optimum by more than ``1e-6`` relative.

    Args:
        returns: Asset returns; rows are equally likely scenarios.
        risk_aversion: Trade-off ``lambda``. Values in ``(0, 1]`` give choices
            efficient under second-order stochastic dominance.
        measure: ``"absolute"`` or ``"standard"`` semideviation below the mean.
        constraints: Feasibility set. Defaults to a long-only, fully-invested
            mandate.
        solver: ``cvxpy`` solver name.

    Returns:
        Optimal weights indexed by ticker.

    Raises:
        ValueError: If ``risk_aversion`` is not positive.
        RuntimeError: If the solver does not report an optimal solution.
    """
    if risk_aversion <= 0.0:
        msg = "risk_aversion must be positive."
        raise ValueError(msg)
    cp = require_cvxpy()
    spec = constraints if constraints is not None else PortfolioConstraints()
    r = returns.to_numpy(dtype=np.float64)
    t_steps, n_assets = r.shape
    scaled = r / (float(r.std()) or 1.0)
    mu = scaled.mean(axis=0)
    w = cp.Variable(n_assets)
    shortfall = cp.pos(mu @ w - scaled @ w)
    risk = (
        cp.sum(shortfall) / t_steps
        if measure == "absolute"
        else cp.norm(shortfall, 2) / np.sqrt(t_steps)
    )
    problem = cp.Problem(
        cp.Maximize(mu @ w - risk_aversion * risk),
        spec.build(cp, w, n_assets=n_assets, expected_returns=r.mean(axis=0)),
    )
    problem.solve(solver=solver)
    if problem.status not in {"optimal", "optimal_inaccurate"}:
        msg = f"mean-semideviation solver failed: status={problem.status}"
        raise RuntimeError(msg)
    weights = np.asarray(w.value, dtype=np.float64)
    return pd.Series(
        project_long_only(weights) if spec.long_only else weights, index=list(returns.columns)
    )

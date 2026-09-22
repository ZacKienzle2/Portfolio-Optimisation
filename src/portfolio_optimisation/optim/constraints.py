"""Reusable linear constraint set for convex portfolio programmes.

A :class:`PortfolioConstraints` value object collects the linear feasibility
conditions shared by the mean-risk optimisers: a full-investment budget, sign
and box bounds, group (sector) exposure limits, a leverage cap for long-short
mandates, an L1 turnover budget against an incumbent portfolio, and a minimum
expected-return floor. Its :meth:`linear` method encodes them once, as SciPy's
two-sided ``LinearConstraint`` and ``Bounds`` over the weights and the slacks
that make each L1 budget linear. :meth:`build` renders that encoding as
``cvxpy`` constraints for the conic programmes, and the smooth EVaR programme
hands it to SLSQP directly, so every solver enforces the same feasible set.

The full-investment budget ``sum(w) == 1`` is always emitted. All remaining
conditions are linear, so they preserve the convexity (and hence the global
optimality) of any mean-risk programme that consumes them.

``DEFAULT_SOLVER`` names Clarabel, the solver cvxpy selects for these linear,
quadratic and exponential-cone programmes. Left unnamed, cvxpy retries the
import of every solver that is not installed on each solve, which cost 4 ms of
a 40 ms CVaR solve, and it hands quadratic programmes to OSQP, whose frontier
volatilities differed from Clarabel's by ``3e-5``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from scipy.optimize import Bounds, LinearConstraint

if TYPE_CHECKING:
    from types import ModuleType

    import cvxpy as cp
    from numpy.typing import ArrayLike, NDArray

DEFAULT_SOLVER = "CLARABEL"


def require_cvxpy() -> ModuleType:
    """Import ``cvxpy``, which the ``[optim]`` extra installs.

    Returns:
        The ``cvxpy`` module.

    Raises:
        ImportError: If the ``[optim]`` extra is not installed.
    """
    try:
        import cvxpy
    except ImportError as exc:  # pragma: no cover
        msg = (
            "convex optimisation requires the [optim] extra: `uv sync --extra optim` "
            "or `pip install 'portfolio-optimisation[optim]'`."
        )
        raise ImportError(msg) from exc
    return cvxpy


def long_only(weights: ArrayLike) -> NDArray[np.float64]:
    """Clip negative weights to zero and rescale onto the simplex.

    Solver output carries negative weights of the order of its tolerance, and a
    closed-form solution can carry genuine short positions. A vector with no
    positive entry falls back to equal weights. A stack of portfolios is
    projected along its last axis.

    Args:
        weights: Raw portfolio weights, one portfolio per trailing row.

    Returns:
        Non-negative weights that sum to one along the last axis.
    """
    clipped = np.clip(np.asarray(weights, dtype=np.float64), 0.0, None)
    total = clipped.sum(axis=-1, keepdims=True)
    positive = total > 0.0
    return np.where(positive, clipped / np.where(positive, total, 1.0), 1.0 / clipped.shape[-1])


def _as_vector(value: float | NDArray[np.float64], n_assets: int) -> NDArray[np.float64]:
    arr = np.broadcast_to(np.asarray(value, dtype=np.float64), (n_assets,))
    return np.array(arr, dtype=np.float64)


@dataclass(frozen=True)
class PortfolioConstraints:
    """Linear feasibility set for a long-only or long-short mandate.

    Attributes:
        long_only: If True, require ``w >= 0``. Ignored when an explicit ``min_weight``
            is supplied.
        min_weight: Lower bound per asset, scalar or length-``N`` vector. Overrides
            ``long_only`` when set.
        max_weight: Upper bound per asset, scalar or length-``N`` vector.
        max_leverage: Gross-exposure cap ``sum |w| <= L``. Only meaningful for
            long-short mandates; ``1.0`` enforces no leverage.
        group_matrix: ``(G, N)`` membership matrix; row ``g`` selects the assets in
            group ``g`` (typically 0/1 entries).
        group_min: Lower bound on each group exposure ``group_matrix @ w``; length
            ``G``.
        group_max: Upper bound on each group exposure; length ``G``.
        previous_weights: Incumbent weights for the turnover budget; length ``N``.
        max_turnover: L1 turnover cap ``sum |w - previous_weights| <= max_turnover``.
            Requires ``previous_weights``.
        target_return: Minimum portfolio expected return ``mu @ w >= target_return``.
            Requires expected returns at build time.
    """

    long_only: bool = True
    min_weight: float | NDArray[np.float64] | None = None
    max_weight: float | NDArray[np.float64] | None = None
    max_leverage: float | None = None
    group_matrix: NDArray[np.float64] | None = None
    group_min: NDArray[np.float64] | None = None
    group_max: NDArray[np.float64] | None = None
    previous_weights: NDArray[np.float64] | None = None
    max_turnover: float | None = None
    target_return: float | None = None

    def _l1_budgets(self, n_assets: int) -> list[tuple[NDArray[np.float64], float]]:
        budgets: list[tuple[NDArray[np.float64], float]] = []
        if self.max_leverage is not None:
            if self.max_leverage <= 0.0:
                msg = "max_leverage must be positive."
                raise ValueError(msg)
            budgets.append((np.zeros(n_assets), self.max_leverage))
        if self.max_turnover is not None:
            if self.previous_weights is None:
                msg = "max_turnover requires previous_weights."
                raise ValueError(msg)
            previous = np.asarray(self.previous_weights, dtype=np.float64).ravel()
            if previous.shape[0] != n_assets:
                msg = "previous_weights must match the asset count."
                raise ValueError(msg)
            budgets.append((previous, self.max_turnover))
        return budgets

    def linear(
        self, n_assets: int, *, expected_returns: NDArray[np.float64] | None = None
    ) -> tuple[LinearConstraint, Bounds]:
        """The feasible set as two-sided linear rows and bounds on ``x = [w, s]``.

        An L1 budget ``sum |w - c| <= b`` is linear once split: a slack block
        ``s >= |w - c|`` enters through the rows ``w - s <= c`` and
        ``-w - s <= -c`` with ``sum(s) <= b``. The leverage cap splits about
        zero and the turnover budget about the incumbent weights, each with its
        own block after the ``N`` weights.

        Args:
            n_assets: Number of assets ``N``, used to broadcast bounds.
            expected_returns: Mean returns required only when ``target_return`` is set.

        Returns:
            The rows ``lb <= A x <= ub`` and the bounds on ``x``.

        Raises:
            ValueError: If ``max_leverage`` is not positive, a group or turnover input
                has the wrong shape, or a condition lacks the input it requires.
        """
        budgets = self._l1_budgets(n_assets)
        width = n_assets * (1 + len(budgets))
        on_weights = np.eye(n_assets, width)
        rows = [np.ones((1, n_assets)) @ on_weights]
        lower = [np.ones(1)]
        upper = [np.ones(1)]

        if self.group_matrix is not None:
            groups = np.asarray(self.group_matrix, dtype=np.float64)
            if groups.ndim != 2 or groups.shape[1] != n_assets:
                msg = "group_matrix must have shape (n_groups, n_assets)."
                raise ValueError(msg)
            rows.append(groups @ on_weights)
            lower.append(
                _as_vector(-np.inf if self.group_min is None else self.group_min, len(groups))
            )
            upper.append(
                _as_vector(np.inf if self.group_max is None else self.group_max, len(groups))
            )

        if self.target_return is not None:
            if expected_returns is None:
                msg = "target_return requires expected_returns."
                raise ValueError(msg)
            mu = np.asarray(expected_returns, dtype=np.float64).ravel()
            rows.append(mu[None, :] @ on_weights)
            lower.append(np.array([self.target_return]))
            upper.append(np.array([np.inf]))

        for block, (centre, budget) in enumerate(budgets, start=1):
            on_slack = np.eye(n_assets, width, k=block * n_assets)
            rows += [
                on_weights - on_slack,
                -on_weights - on_slack,
                on_slack.sum(axis=0, keepdims=True),
            ]
            lower += [np.full(n_assets, -np.inf), np.full(n_assets, -np.inf), np.array([-np.inf])]
            upper += [centre, -centre, np.array([budget])]

        if self.min_weight is not None:
            floor = _as_vector(self.min_weight, n_assets)
        else:
            floor = np.zeros(n_assets) if self.long_only else np.full(n_assets, -np.inf)
        cap = np.inf if self.max_weight is None else self.max_weight
        slack = width - n_assets
        bounds = Bounds(
            np.concatenate([floor, np.zeros(slack)]),
            np.concatenate([_as_vector(cap, n_assets), np.full(slack, np.inf)]),
        )
        return LinearConstraint(
            np.vstack(rows), np.concatenate(lower), np.concatenate(upper)
        ), bounds

    def build(
        self,
        cp: ModuleType,
        w: cp.Variable,
        *,
        n_assets: int,
        expected_returns: NDArray[np.float64] | None = None,
    ) -> list[cp.constraints.constraint.Constraint]:
        """Render :meth:`linear` as ``cvxpy`` constraints on ``w`` and its slacks.

        Args:
            cp: The imported ``cvxpy`` module.
            w: The length-``N`` weight variable to constrain.
            n_assets: Number of assets ``N``, used to broadcast bounds.
            expected_returns: Mean returns required only when ``target_return`` is set.

        Returns:
            list: The ``cvxpy`` constraints implied by this value object.
        """
        rows, bounds = self.linear(n_assets, expected_returns=expected_returns)
        slack = bounds.lb.size - n_assets
        x = cp.hstack([w, cp.Variable(slack)]) if slack else w
        equal = rows.lb == rows.ub
        above = ~equal & np.isfinite(rows.ub)
        below = ~equal & np.isfinite(rows.lb)
        constraints: list[cp.constraints.constraint.Constraint] = [
            rows.A[equal] @ x == rows.ub[equal]
        ]
        if above.any():
            constraints.append(rows.A[above] @ x <= rows.ub[above])
        if below.any():
            constraints.append(rows.A[below] @ x >= rows.lb[below])
        floor = np.flatnonzero(np.isfinite(bounds.lb))
        cap = np.flatnonzero(np.isfinite(bounds.ub))
        if floor.size:
            constraints.append(x[floor] >= bounds.lb[floor])
        if cap.size:
            constraints.append(x[cap] <= bounds.ub[cap])
        return constraints

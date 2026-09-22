"""Reusable linear constraint set for convex portfolio programmes.

A :class:`PortfolioConstraints` value object collects the linear feasibility
conditions shared by the mean-risk optimisers: a full-investment budget, sign
and box bounds, group (sector) exposure limits, a leverage cap for long-short
mandates, an L1 turnover budget against an incumbent portfolio, and a minimum
expected-return floor. Its :meth:`build` method translates the value object into
a list of ``cvxpy`` constraints for a supplied weight variable, so every solver
in this layer enforces the same vocabulary without duplicating the encoding.

The full-investment budget ``sum(w) == 1`` is always emitted. All remaining
conditions are linear, so they preserve the convexity (and hence the global
optimality) of any mean-risk programme that consumes them.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from types import ModuleType

    import cvxpy as cp
    from numpy.typing import ArrayLike, NDArray


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

    def build(
        self,
        cp: ModuleType,
        w: cp.Variable,
        *,
        n_assets: int,
        expected_returns: NDArray[np.float64] | None = None,
    ) -> list[cp.constraints.constraint.Constraint]:
        """Translate the value object into ``cvxpy`` constraints.

        Args:
            cp: The imported ``cvxpy`` module.
            w: The length-``N`` weight variable to constrain.
            n_assets: Number of assets ``N``, used to broadcast bounds.
            expected_returns: Mean returns required only when ``target_return`` is set.

        Returns:
            list: The ``cvxpy`` constraints implied by this value object.

        Raises:
            ValueError: If ``max_leverage`` is not positive, a group or turnover input
                has the wrong shape, or a condition lacks the input it requires.
        """
        constraints: list[cp.constraints.constraint.Constraint] = [cp.sum(w) == 1]

        if self.min_weight is not None:
            constraints.append(w >= _as_vector(self.min_weight, n_assets))
        elif self.long_only:
            constraints.append(w >= 0)
        if self.max_weight is not None:
            constraints.append(w <= _as_vector(self.max_weight, n_assets))

        if self.max_leverage is not None:
            if self.max_leverage <= 0.0:
                msg = "max_leverage must be positive."
                raise ValueError(msg)
            constraints.append(cp.norm1(w) <= self.max_leverage)

        if self.group_matrix is not None:
            groups = np.asarray(self.group_matrix, dtype=np.float64)
            if groups.ndim != 2 or groups.shape[1] != n_assets:
                msg = "group_matrix must have shape (n_groups, n_assets)."
                raise ValueError(msg)
            exposure = groups @ w
            if self.group_max is not None:
                constraints.append(exposure <= np.asarray(self.group_max, dtype=np.float64))
            if self.group_min is not None:
                constraints.append(exposure >= np.asarray(self.group_min, dtype=np.float64))

        if self.max_turnover is not None:
            if self.previous_weights is None:
                msg = "max_turnover requires previous_weights."
                raise ValueError(msg)
            prev = np.asarray(self.previous_weights, dtype=np.float64).ravel()
            if prev.shape[0] != n_assets:
                msg = "previous_weights must match the asset count."
                raise ValueError(msg)
            constraints.append(cp.norm1(w - prev) <= self.max_turnover)

        if self.target_return is not None:
            if expected_returns is None:
                msg = "target_return requires expected_returns."
                raise ValueError(msg)
            mu = np.asarray(expected_returns, dtype=np.float64).ravel()
            constraints.append(mu @ w >= self.target_return)

        return constraints

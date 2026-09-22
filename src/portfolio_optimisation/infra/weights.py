"""Inverse-variance weighting and whole-share allocation."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import Bounds, LinearConstraint, milp


def inverse_variance_weights(cov_matrix: pd.DataFrame) -> pd.Series:
    """Calculate inverse-variance portfolio weights.

    Weights are inversely proportional to asset variance, the diagonal of the
    covariance matrix, which minimises portfolio variance when correlations are
    ignored.

    Args:
        cov_matrix: Covariance matrix of asset returns.

    Returns:
        Asset weights for the inverse-variance portfolio.
    """
    inverse = 1.0 / (np.diag(cov_matrix) + 1e-12)
    return pd.Series(inverse / inverse.sum(), index=cov_matrix.index)


def _whole_shares(weights: pd.Series, prices: pd.Series, budget: float) -> tuple[pd.Series, float]:
    """Integer share counts closest in L1 to ``weights * budget`` within the budget.

    The deviation plus leftover cash, ``sum_i |w_i V - p_i x_i| + (V - p' x)``,
    equals twice the total shortfall ``sum_i (w_i V - p_i x_i)^+`` for weights
    that sum to one, since a share bought above target adds as much deviation as
    it removes cash. The programme therefore minimises shortfall variables
    ``s_i >= w_i V - p_i x_i`` over integer counts within the budget, one row per
    asset where the absolute values needed two.

    The counts are posed as steps ``d_i`` from rounding every target down, so
    the shortfall rows read ``s_i >= r_i - p_i d_i`` with ``r_i`` the residual
    below one share, and the budget row spends the cash rounding down leaves.
    A step above one lowers no shortfall, which caps ``d_i`` at one. Dropping
    ``k`` shares adds ``k p_i`` of shortfall, and the optimal shortfall is at
    most the spare cash, the shortfall of rounding down, which bounds ``k``. The
    narrow integer box lets HiGHS finish at its root node.

    The residuals come from ``numpy.divmod``, whose remainder is exact and never
    negative, and the spare cash is their sum, which it equals for weights that
    sum to one. Flooring the quotient and subtracting the spend in floating
    point let a target that rounded onto a whole share leave the spare cash a
    fraction of a cent below zero, and the box empty.

    Every quantity is then of the order of one share price, and money is
    measured in units of the dearest share. Measured in units of the budget, a
    shortfall of a few cents fell below HiGHS's feasibility tolerance and a
    whole share could overspend unnoticed; in currency units a cheap asset held
    in hundreds of thousands of shares made HiGHS abandon the solve. The
    relative gap is closed to zero, which leaves HiGHS's absolute gap of
    ``1e-6`` of the dearest share to decide. The default relative gap admitted a
    shortfall a cent above the optimum, and closing it cost no extra nodes.
    """
    money = prices.to_numpy(dtype=np.float64)
    floor, remainder = np.divmod(weights.to_numpy(dtype=np.float64) * budget, money)
    unit = money.max()
    p = money / unit
    residual = remainder / unit
    spare = residual.sum()
    n = p.size
    rows = np.block([[np.diag(p), np.eye(n)], [p[None, :], np.zeros((1, n))]])
    result = milp(
        c=np.concatenate([np.zeros(n), np.ones(n)]),
        constraints=LinearConstraint(
            rows,
            np.concatenate([residual, [-np.inf]]),
            np.concatenate([np.full(n, np.inf), [spare]]),
        ),
        integrality=np.concatenate([np.ones(n), np.zeros(n)]),
        bounds=Bounds(
            np.concatenate([-np.minimum(floor, np.floor(spare / p)), np.zeros(n)]),
            np.concatenate([(remainder > 0.0).astype(np.float64), np.full(n, np.inf)]),
        ),
        options={"mip_rel_gap": 0.0},
    )
    if not result.success:
        msg = f"whole-share allocation failed: {result.message}"
        raise RuntimeError(msg)
    shares = (floor + np.rint(result.x[:n])).astype(np.int64)
    return pd.Series(shares, index=weights.index), float(budget - money @ shares)


def get_discrete_portfolio(
    weights: pd.Series, prices: pd.DataFrame, total_value: float = 1_000_000.0
) -> tuple[dict[str, int], float]:
    """Convert continuous weights to whole share counts.

    Long and short books are allocated separately, the short book sized by the
    gross short weight, and positions of zero shares are dropped.

    Args:
        weights: Target continuous weights.
        prices: Historical asset prices whose last row is used.
        total_value: Monetary value of the long book.

    Returns:
        Share count per ticker, negative for shorts, and the leftover cash.
    """
    latest = prices.iloc[-1].reindex(weights.index)
    longs = weights.clip(lower=0.0)
    shorts = (-weights).clip(lower=0.0)
    shares, leftover = _whole_shares(longs / longs.sum(), latest, total_value)
    if shorts.sum() > 0:
        short_shares, short_leftover = _whole_shares(
            shorts / shorts.sum(), latest, total_value * shorts.sum()
        )
        shares -= short_shares
        leftover += short_leftover
    return {str(k): int(v) for k, v in shares.items() if v != 0}, leftover

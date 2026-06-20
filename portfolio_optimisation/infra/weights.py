"""Weight-construction helpers for inverse-variance weighting and discrete
share allocation."""

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from pypfopt import discrete_allocation


def inverse_variance_weights(cov_matrix: pd.DataFrame) -> pd.Series:
    """Calculate inverse-variance portfolio weights.

    Weights are inversely proportional to asset variance (the diagonal of the
    covariance matrix), aiming to minimise portfolio variance ignoring returns.

    Args:
        cov_matrix (pd.DataFrame): Covariance matrix of asset returns.

    Returns:
        pd.Series: Asset weights for the inverse-variance portfolio.
    """
    variances: NDArray[np.float64] = np.diag(cov_matrix)
    inv_variances: NDArray[np.float64] = 1 / (variances + 1e-12)
    ivp_weights: NDArray[np.float64] = inv_variances / np.sum(inv_variances)
    return pd.Series(ivp_weights, index=cov_matrix.index)


def get_discrete_portfolio(
    weights: pd.Series, prices: pd.DataFrame, total_value: float = 1_000_000.0
) -> tuple[dict[str, int], float]:
    """Convert continuous weights to a discrete number of shares (LP).

    Args:
        weights (pd.Series): Target continuous weights.
        prices (pd.DataFrame): Historical asset prices (latest row used).
        total_value (float, optional): Total monetary value to allocate.

    Returns:
        Tuple[Dict[str, int], float]: {ticker: shares} and leftover cash.
    """
    latest_prices: pd.Series = prices.iloc[-1]
    da = discrete_allocation.DiscreteAllocation(
        weights=weights.to_dict(),
        latest_prices=latest_prices,
        total_portfolio_value=int(total_value),
    )
    allocation: dict[str, int]
    leftover: float
    allocation, leftover = da.lp_portfolio(verbose=False)
    return allocation, leftover

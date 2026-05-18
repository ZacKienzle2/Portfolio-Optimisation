from typing import Dict, Tuple

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from pypfopt import discrete_allocation


def inverseVarianceWeights(covMatrix: pd.DataFrame) -> pd.Series:
    """Calculate inverse-variance portfolio weights.

    Weights are inversely proportional to asset variance (the diagonal of the
    covariance matrix), aiming to minimise portfolio variance ignoring returns.

    Args:
        covMatrix (pd.DataFrame): Covariance matrix of asset returns.

    Returns:
        pd.Series: Asset weights for the inverse-variance portfolio.
    """
    variances: NDArray[np.float64] = np.diag(covMatrix)
    invVariances: NDArray[np.float64] = 1 / (variances + 1e-12)
    ivpWeights: NDArray[np.float64] = invVariances / np.sum(invVariances)
    return pd.Series(ivpWeights, index=covMatrix.index)


def get_discrete_portfolio(
    weights: pd.Series, prices: pd.DataFrame, totalValue: float = 1_000_000.0
) -> Tuple[Dict[str, int], float]:
    """Convert continuous weights to a discrete number of shares (LP).

    Args:
        weights (pd.Series): Target continuous weights.
        prices (pd.DataFrame): Historical asset prices (latest row used).
        totalValue (float, optional): Total monetary value to allocate.

    Returns:
        Tuple[Dict[str, int], float]: {ticker: shares} and leftover cash.
    """
    latestPrices: pd.Series = prices.iloc[-1]
    da = discrete_allocation.DiscreteAllocation(
        weights=weights.to_dict(),
        latest_prices=latestPrices,
        total_portfolio_value=int(totalValue),
    )
    allocation: Dict[str, int]
    leftover: float
    allocation, leftover = da.lp_portfolio(verbose=False)
    return allocation, leftover

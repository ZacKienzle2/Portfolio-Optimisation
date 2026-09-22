"""Hierarchical Risk Parity allocation."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from portfolio_optimisation.infra.weights import get_discrete_portfolio
from portfolio_optimisation.optim.clustering import (
    bisection,
    correlation_linkage,
    inverse_variance_weights,
    seriation,
)
from portfolio_optimisation.optim.shrinkage import linear_shrinkage_covariance

if TYPE_CHECKING:
    from numpy.typing import NDArray


def _cluster_variance(covariance: NDArray[np.float64]) -> float:
    """Variance of the inverse-variance portfolio of one cluster."""
    weights = inverse_variance_weights(covariance)
    return float(weights @ covariance @ weights)


def hrp_weights(covariance: NDArray[np.float64], order: NDArray[np.intp]) -> NDArray[np.float64]:
    """Recursive-bisection weights over a seriated asset order.

    Each split hands the left cluster ``V_R / (V_L + V_R)`` of the parent weight,
    where ``V`` is the variance of a cluster's inverse-variance portfolio.

    Args:
        covariance: Symmetric ``(N, N)`` covariance matrix.
        order: Seriated asset indices from the dendrogram.

    Returns:
        Weights aligned with ``order``, summing to one.
    """
    weights = np.ones(order.size, dtype=np.float64)
    for left, right in bisection(order.size):
        idx_left, idx_right = order[left], order[right]
        var_left = _cluster_variance(covariance[np.ix_(idx_left, idx_left)])
        var_right = _cluster_variance(covariance[np.ix_(idx_right, idx_right)])
        total = var_left + var_right
        alpha = var_right / total if total != 0 else 0.5
        weights[left] *= alpha
        weights[right] *= 1.0 - alpha
    return weights


class HRPModel:
    """Hierarchical Risk Parity portfolio model.

    Clusters assets on the correlation distance of a Ledoit-Wolf shrunk
    covariance, seriates the dendrogram and splits the seriated order by
    inverse cluster variance.

    Args:
        returns: Historical asset returns, one column per ticker.
        cov_matrix: Covariance to use instead of Ledoit-Wolf shrinkage, such as an
            RMT-denoised or conditional estimate.

    Attributes:
        returns: The returns the model was built from.
        cov_matrix: Covariance driving the clustering and the split.
        weights: HRP weights in seriated order, empty until ``optimize`` runs.
        ordered_tickers: Tickers in seriated order.
        linkage_matrix: Linkage matrix of the last ``optimize`` call.

    Raises:
        TypeError: If ``returns`` is not a DataFrame.
    """

    def __init__(self, returns: pd.DataFrame, *, cov_matrix: pd.DataFrame | None = None) -> None:
        if not isinstance(returns, pd.DataFrame):  # pyright: ignore[reportUnnecessaryIsInstance]
            msg = "Returns must be a pandas DataFrame."
            raise TypeError(msg)
        self.returns: pd.DataFrame = returns
        self.cov_matrix: pd.DataFrame = (
            cov_matrix if cov_matrix is not None else linear_shrinkage_covariance(returns)
        )
        self.weights: pd.Series = pd.Series(dtype=np.float64)
        self.ordered_tickers: list[str] = []
        self.linkage_matrix: NDArray[np.float64] | None = None

    def optimize(self, linkage_method: str = "ward") -> None:
        """Cluster, seriate and bisect to produce the HRP weights.

        Args:
            linkage_method: Linkage criterion passed to SciPy.
        """
        covariance = self.cov_matrix.to_numpy(dtype=np.float64)
        self.linkage_matrix = correlation_linkage(covariance, linkage_method)
        order = seriation(self.linkage_matrix)
        tickers = list(self.cov_matrix.columns)
        self.ordered_tickers = [tickers[i] for i in order]
        self.weights = pd.Series(hrp_weights(covariance, order), index=self.ordered_tickers)

    def clean_weights(self) -> pd.Series:
        """Return the HRP weights sorted by ticker.

        Returns:
            HRP weights indexed alphabetically by ticker.

        Raises:
            RuntimeError: If ``optimize`` has not been called.
        """
        if self.weights.empty:
            msg = "Optimisation must be run before accessing weights."
            raise RuntimeError(msg)
        return self.weights.sort_index()

    def get_discrete_allocation(
        self, prices: pd.DataFrame, total_portfolio_value: float
    ) -> tuple[dict[str, int], float]:
        """Convert the HRP weights to whole share counts.

        Args:
            prices: Historical prices whose last row is used.
            total_portfolio_value: Cash available to allocate.

        Returns:
            Share count per ticker and the leftover cash.
        """
        return get_discrete_portfolio(self.clean_weights(), prices, total_portfolio_value)

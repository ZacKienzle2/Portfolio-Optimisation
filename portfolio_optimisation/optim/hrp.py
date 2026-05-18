from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from pandas import Series
from pypfopt import discrete_allocation, risk_models
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import squareform
from sklearn.covariance import ledoit_wolf


class HRPModel:
    """Implements the Hierarchical Risk Parity (HRP) portfolio model.

    Calculates portfolio weights based on hierarchical clustering of asset
    returns, promoting diversification by correlation-based risk budgeting.
    Uses Ledoit-Wolf covariance shrinkage.
    """

    def __init__(self, returns: pd.DataFrame):
        """Initialise the HRP model.

        Args:
            returns (pd.DataFrame): Historical asset returns (columns=assets).

        Raises:
            TypeError: If returns is not a pandas DataFrame.
        """
        if not isinstance(returns, pd.DataFrame):
            raise TypeError("Returns must be a pandas DataFrame.")
        self.returns: pd.DataFrame = returns
        self.weights: Series = pd.Series(dtype=np.float64)
        self.ordered_tickers: list[str] = []
        self.cov_matrix: pd.DataFrame = self._calculate_covariance()
        self.linkage_matrix: NDArray[Any] | None = None

    def _calculate_covariance(self) -> pd.DataFrame:
        """Calculate the Ledoit-Wolf shrunk covariance matrix."""
        cov_matrix_internal, _ = ledoit_wolf(self.returns, assume_centered=False)
        return pd.DataFrame(
            cov_matrix_internal, index=self.returns.columns, columns=self.returns.columns
        )

    def _get_quasi_diag(self, linkage_matrix: NDArray[Any]) -> list[int]:
        """Sorts assets according to the quasi-diagonalisation algorithm."""
        num_items = linkage_matrix.shape[0] + 1
        sorted_index: list[float] = [linkage_matrix[-1, 0], linkage_matrix[-1, 1]]

        while len(sorted_index) < num_items:
            expand_cluster_idx = -1
            for i, item in enumerate(sorted_index):
                if item >= num_items:
                    expand_cluster_idx = i
                    break

            if expand_cluster_idx == -1:
                break

            cluster_val = sorted_index[expand_cluster_idx]
            link_row = int(cluster_val - num_items)
            item1 = linkage_matrix[link_row, 0]
            item2 = linkage_matrix[link_row, 1]

            sorted_index = [
                *sorted_index[:expand_cluster_idx],
                item1,
                item2,
                *sorted_index[expand_cluster_idx + 1 :],
            ]

        return [int(i) for i in sorted_index]

    def _get_cluster_var(self, cov: pd.DataFrame, cluster_items: list[str]) -> float:
        """Calculate the variance of an Inverse Variance Portfolio within a cluster."""
        cov_slice = cov.loc[cluster_items, cluster_items]
        inv_diag = 1 / np.diag(cov_slice)
        weights_internal = (inv_diag / inv_diag.sum()).reshape(-1, 1)
        cluster_var = (weights_internal.T @ cov_slice @ weights_internal).iloc[0, 0]
        return float(cluster_var)

    def _recursive_bisection(self, ordered_tickers: list[str]):
        """Recursively bisects the clusters and weights the sub-portfolios."""
        self.weights = pd.Series(1.0, index=ordered_tickers)
        cluster_items: list[list[str]] = [ordered_tickers]

        while len(cluster_items) > 0:
            next_clusters: list[list[str]] = []
            for cluster in cluster_items:
                if len(cluster) > 1:
                    mid = len(cluster) // 2
                    next_clusters.extend([cluster[:mid], cluster[mid:]])
            cluster_items = next_clusters

            for i in range(0, len(cluster_items), 2):
                cluster1 = cluster_items[i]
                cluster2 = cluster_items[i + 1]
                var1 = self._get_cluster_var(self.cov_matrix, cluster1)
                var2 = self._get_cluster_var(self.cov_matrix, cluster2)

                var_sum = var1 + var2
                alpha = 1 - var1 / var_sum if var_sum != 0 else 0.5

                self.weights.loc[cluster1] *= alpha
                self.weights.loc[cluster2] *= 1 - alpha

    def optimize(self, linkage_method: str = "ward"):
        """Perform the Hierarchical Risk Parity optimisation.

        Args:
            linkage_method (str, optional): Clustering method (e.g., 'ward',
                                           'single', 'complete'). Defaults to 'ward'.

        Raises:
            ValueError: If the linkage matrix computation fails.
        """
        corr = risk_models.cov_to_corr(self.cov_matrix)
        dist_matrix = np.sqrt((1 - corr.round(8).fillna(0)) / 2)
        condensed_dist = squareform(dist_matrix, checks=False)
        self.linkage_matrix = linkage(condensed_dist, method=linkage_method)

        if self.linkage_matrix is None:
            raise ValueError("Linkage matrix could not be computed.")

        sorted_indices = self._get_quasi_diag(self.linkage_matrix)
        self.ordered_tickers = list(self.cov_matrix.index[sorted_indices])
        self._recursive_bisection(self.ordered_tickers)

    def clean_weights(self) -> Series:
        """Returns the final HRP weights, sorted by ticker.

        Raises:
            RuntimeError: If `optimize()` has not been called.

        Returns:
            Series: The final HRP weights.
        """
        if self.weights.empty:
            raise RuntimeError("Optimisation must be run before accessing weights.")
        return self.weights.sort_index()

    def get_discrete_allocation(
        self, prices: pd.DataFrame, total_portfolio_value: float
    ) -> tuple[dict[str, int], float]:
        """Converts continuous HRP weights to a discrete share allocation.

        Args:
            prices (pd.DataFrame): Historical asset prices (latest row used).
            total_portfolio_value (float): Total cash available for allocation.

        Returns:
            Tuple[Dict[str, int], float]: Dictionary of {ticker: shares}
                                         and the leftover cash amount.
        """
        weights_cleaned = self.clean_weights()
        latest_prices = prices.iloc[-1]

        da = discrete_allocation.DiscreteAllocation(
            weights=weights_cleaned.to_dict(),
            latest_prices=latest_prices,
            total_portfolio_value=int(total_portfolio_value),
        )
        allocation: dict[str, int]
        leftover: float
        allocation, leftover = da.lp_portfolio(verbose=False)
        return allocation, leftover

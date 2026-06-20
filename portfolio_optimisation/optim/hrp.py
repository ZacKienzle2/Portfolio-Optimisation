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
    """Hierarchical Risk Parity portfolio model.

    Computes portfolio weights from a hierarchical clustering of asset
    returns with correlation-based risk budgeting. Uses Ledoit-Wolf
    covariance shrinkage. The recursive bisection step operates on a
    contiguous numpy view of the shrunk covariance so the inner loop
    avoids pandas `.loc` indexing overhead.
    """

    def __init__(
        self,
        returns: pd.DataFrame,
        *,
        cov_matrix: pd.DataFrame | None = None,
    ):
        """Initialise the model.

        Args:
            returns (pd.DataFrame): Historical asset returns, columns are tickers.
            cov_matrix (pd.DataFrame | None, optional): Pre-computed covariance
                matrix to use instead of the default Ledoit-Wolf shrinkage. Use
                this entry point to inject RMT-denoised, DCC-GARCH conditional,
                or any other externally estimated covariance.

        Raises:
            TypeError: If returns is not a pandas DataFrame.
        """
        if not isinstance(returns, pd.DataFrame):  # pyright: ignore[reportUnnecessaryIsInstance]
            raise TypeError("Returns must be a pandas DataFrame.")
        self.returns: pd.DataFrame = returns
        self.weights: Series = pd.Series(dtype=np.float64)
        self.ordered_tickers: list[str] = []
        self.cov_matrix: pd.DataFrame = (
            cov_matrix if cov_matrix is not None else self._calculate_covariance()
        )
        self.linkage_matrix: NDArray[Any] | None = None
        # Numpy view + ticker->row index map are rebuilt lazily inside
        # optimize() so the inner-loop hot path never touches pandas.
        self._cov_values: NDArray[np.float64] | None = None
        self._ticker_to_idx: dict[str, int] = {}

    def _calculate_covariance(self) -> pd.DataFrame:
        """Calculate the Ledoit-Wolf shrunk covariance matrix."""
        cov_matrix_internal, _ = ledoit_wolf(self.returns, assume_centered=False)
        return pd.DataFrame(
            cov_matrix_internal,
            index=self.returns.columns,
            columns=self.returns.columns,
        )

    def _get_quasi_diag(self, linkage_matrix: NDArray[Any]) -> list[int]:
        """Quasi-diagonalise asset ordering from the linkage matrix."""
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

    @staticmethod
    def _cluster_var_numpy(cov_values: NDArray[np.float64], idx: NDArray[np.intp]) -> float:
        """Inverse-variance portfolio variance for the index-selected sub-matrix.

        Numpy-only hot path: takes the dense covariance buffer + an integer
        ticker-index array and computes ``w' @ cov_sub @ w`` where ``w`` is
        proportional to ``1/diag(cov_sub)``. Avoids pandas slicing and the
        DataFrame matmul wrap.
        """
        cov_sub = cov_values[np.ix_(idx, idx)]
        inv_diag = 1.0 / np.diag(cov_sub)
        w = inv_diag / inv_diag.sum()
        return float(w @ cov_sub @ w)

    def _recursive_bisection_numpy(self, ordered_idx: NDArray[np.intp]) -> NDArray[np.float64]:
        """Recursive bisection over integer indices into the numpy cov view.

        Returns a 1D weight vector aligned with ``ordered_idx``.
        """
        assert self._cov_values is not None
        cov_values = self._cov_values
        n = ordered_idx.size
        weights = np.ones(n, dtype=np.float64)

        # Track clusters as (start, stop) half-open ranges within ordered_idx.
        cluster_ranges: list[tuple[int, int]] = [(0, n)]
        while cluster_ranges:
            next_ranges: list[tuple[int, int]] = []
            for start, stop in cluster_ranges:
                if stop - start > 1:
                    mid = start + (stop - start) // 2
                    next_ranges.append((start, mid))
                    next_ranges.append((mid, stop))
            cluster_ranges = next_ranges

            # Pair adjacent siblings and split weight by inverse cluster variance.
            for i in range(0, len(cluster_ranges), 2):
                left = cluster_ranges[i]
                right = cluster_ranges[i + 1]
                idx_left = ordered_idx[left[0] : left[1]]
                idx_right = ordered_idx[right[0] : right[1]]
                var_left = self._cluster_var_numpy(cov_values, idx_left)
                var_right = self._cluster_var_numpy(cov_values, idx_right)
                var_sum = var_left + var_right
                alpha = 1.0 - var_left / var_sum if var_sum != 0 else 0.5
                weights[left[0] : left[1]] *= alpha
                weights[right[0] : right[1]] *= 1.0 - alpha

        return weights

    def optimize(self, linkage_method: str = "ward"):
        """Run the HRP optimisation.

        Args:
            linkage_method (str, optional): Clustering method passed to
                ``scipy.cluster.hierarchy.linkage``. Defaults to ``"ward"``.

        Raises:
            ValueError: If the linkage matrix computation fails.
        """
        corr = risk_models.cov_to_corr(self.cov_matrix)
        dist_matrix = np.sqrt((1 - corr.round(8).fillna(0)) / 2)
        condensed_dist = squareform(dist_matrix, checks=False)
        self.linkage_matrix = linkage(condensed_dist, method=linkage_method)

        if self.linkage_matrix is None:
            raise ValueError("Linkage matrix could not be computed.")

        # Materialise a dense numpy view of the shrunk covariance and a
        # ticker -> row index map. Subsequent bisection touches only numpy.
        cov_values = np.ascontiguousarray(self.cov_matrix.to_numpy(dtype=np.float64))
        self._cov_values = cov_values
        cols = list(self.cov_matrix.columns)
        self._ticker_to_idx = {ticker: i for i, ticker in enumerate(cols)}

        sorted_indices = self._get_quasi_diag(self.linkage_matrix)
        self.ordered_tickers = [cols[i] for i in sorted_indices]

        ordered_idx = np.array(sorted_indices, dtype=np.intp)
        weight_values = self._recursive_bisection_numpy(ordered_idx)
        self.weights = pd.Series(weight_values, index=self.ordered_tickers)

    def clean_weights(self) -> Series:
        """Return the final HRP weights sorted by ticker.

        Raises:
            RuntimeError: If ``optimize()`` has not been called.

        Returns:
            Series: Final HRP weights, alphabetically indexed.
        """
        if self.weights.empty:
            raise RuntimeError("Optimisation must be run before accessing weights.")
        return self.weights.sort_index()

    def get_discrete_allocation(
        self, prices: pd.DataFrame, total_portfolio_value: float
    ) -> tuple[dict[str, int], float]:
        """Convert continuous HRP weights to a discrete share allocation.

        Args:
            prices (pd.DataFrame): Historical asset prices, latest row used.
            total_portfolio_value (float): Total cash available.

        Returns:
            tuple[dict[str, int], float]: Map of ticker -> share count and
            the leftover cash amount.
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

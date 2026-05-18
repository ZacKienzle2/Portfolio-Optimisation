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
        self.orderedTickers: list[str] = []
        self.covMatrix: pd.DataFrame = self._calculateCovariance()
        self.linkageMatrix: NDArray[Any] | None = None

    def _calculateCovariance(self) -> pd.DataFrame:
        """Calculate the Ledoit-Wolf shrunk covariance matrix."""
        covMatrixInternal, _ = ledoit_wolf(self.returns, assume_centered=False)
        return pd.DataFrame(
            covMatrixInternal, index=self.returns.columns, columns=self.returns.columns
        )

    def _getQuasiDiag(self, linkageMatrix: NDArray[Any]) -> list[int]:
        """Sorts assets according to the quasi-diagonalisation algorithm."""
        numItems = linkageMatrix.shape[0] + 1
        sortedIndex: list[float] = [linkageMatrix[-1, 0], linkageMatrix[-1, 1]]

        while len(sortedIndex) < numItems:
            expandClusterIdx = -1
            for i, item in enumerate(sortedIndex):
                if item >= numItems:
                    expandClusterIdx = i
                    break

            if expandClusterIdx == -1:
                break

            clusterVal = sortedIndex[expandClusterIdx]
            linkRow = int(clusterVal - numItems)
            item1 = linkageMatrix[linkRow, 0]
            item2 = linkageMatrix[linkRow, 1]

            sortedIndex = [
                *sortedIndex[:expandClusterIdx],
                item1,
                item2,
                *sortedIndex[expandClusterIdx + 1 :],
            ]

        return [int(i) for i in sortedIndex]

    def _getClusterVar(self, cov: pd.DataFrame, clusterItems: list[str]) -> float:
        """Calculate the variance of an Inverse Variance Portfolio within a cluster."""
        covSlice = cov.loc[clusterItems, clusterItems]
        invDiag = 1 / np.diag(covSlice)
        weightsInternal = (invDiag / invDiag.sum()).reshape(-1, 1)
        clusterVar = (weightsInternal.T @ covSlice @ weightsInternal).iloc[0, 0]
        return float(clusterVar)

    def _recursiveBisection(self, orderedTickers: list[str]):
        """Recursively bisects the clusters and weights the sub-portfolios."""
        self.weights = pd.Series(1.0, index=orderedTickers)
        clusterItems: list[list[str]] = [orderedTickers]

        while len(clusterItems) > 0:
            nextClusters: list[list[str]] = []
            for cluster in clusterItems:
                if len(cluster) > 1:
                    mid = len(cluster) // 2
                    nextClusters.extend([cluster[:mid], cluster[mid:]])
            clusterItems = nextClusters

            for i in range(0, len(clusterItems), 2):
                cluster1 = clusterItems[i]
                cluster2 = clusterItems[i + 1]
                var1 = self._getClusterVar(self.covMatrix, cluster1)
                var2 = self._getClusterVar(self.covMatrix, cluster2)

                varSum = var1 + var2
                alpha = 1 - var1 / varSum if varSum != 0 else 0.5

                self.weights.loc[cluster1] *= alpha
                self.weights.loc[cluster2] *= 1 - alpha

    def optimize(self, linkageMethod: str = "ward"):
        """Perform the Hierarchical Risk Parity optimisation.

        Args:
            linkageMethod (str, optional): Clustering method (e.g., 'ward',
                                           'single', 'complete'). Defaults to 'ward'.

        Raises:
            ValueError: If the linkage matrix computation fails.
        """
        corr = risk_models.cov_to_corr(self.covMatrix)
        distMatrix = np.sqrt((1 - corr.round(8).fillna(0)) / 2)
        condensedDist = squareform(distMatrix, checks=False)
        self.linkageMatrix = linkage(condensedDist, method=linkageMethod)

        if self.linkageMatrix is None:
            raise ValueError("Linkage matrix could not be computed.")

        sortedIndices = self._getQuasiDiag(self.linkageMatrix)
        self.orderedTickers = list(self.covMatrix.index[sortedIndices])
        self._recursiveBisection(self.orderedTickers)

    def cleanWeights(self) -> Series:
        """Returns the final HRP weights, sorted by ticker.

        Raises:
            RuntimeError: If `optimize()` has not been called.

        Returns:
            Series: The final HRP weights.
        """
        if self.weights.empty:
            raise RuntimeError("Optimisation must be run before accessing weights.")
        return self.weights.sort_index()

    def getDiscreteAllocation(
        self, prices: pd.DataFrame, totalPortfolioValue: float
    ) -> tuple[dict[str, int], float]:
        """Converts continuous HRP weights to a discrete share allocation.

        Args:
            prices (pd.DataFrame): Historical asset prices (latest row used).
            totalPortfolioValue (float): Total cash available for allocation.

        Returns:
            Tuple[Dict[str, int], float]: Dictionary of {ticker: shares}
                                         and the leftover cash amount.
        """
        weightsCleaned = self.cleanWeights()
        latestPrices = prices.iloc[-1]

        da = discrete_allocation.DiscreteAllocation(
            weights=weightsCleaned.to_dict(),
            latest_prices=latestPrices,
            total_portfolio_value=int(totalPortfolioValue),
        )
        allocation: dict[str, int]
        leftover: float
        allocation, leftover = da.lp_portfolio(verbose=False)
        return allocation, leftover

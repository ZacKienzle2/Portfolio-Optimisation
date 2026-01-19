# Portfolio_Theory/portfolio.py
from typing import Any, Optional, Dict, List, Tuple
from multiprocessing import Pool, cpu_count
import numpy as np
import pandas as pd
from pandas import Series
from pathlib import Path
from numpy.typing import NDArray
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import squareform
from sklearn.covariance import ledoit_wolf
from pypfopt import risk_models, discrete_allocation, expected_returns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from arch.bootstrap import StationaryBootstrap, optimal_block_length


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
        self.linkageMatrix: Optional[NDArray[Any]] = None

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

            sortedIndex = (
                sortedIndex[:expandClusterIdx]
                + [item1, item2]
                + sortedIndex[expandClusterIdx + 1 :]
            )

        return [int(i) for i in sortedIndex]

    def _getClusterVar(self, cov: pd.DataFrame, clusterItems: list[str]) -> float:
        """Calculate the variance of an Inverse Variance Portfolio within a cluster."""
        covSlice = cov.loc[clusterItems, clusterItems]
        invDiag = 1 / np.diag(covSlice)
        weightsInternal = (invDiag / invDiag.sum()).reshape(-1, 1)
        # Use .iloc[0, 0] for type stability
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

            # Apply risk parity allocation to the bisected clusters
            for i in range(0, len(clusterItems), 2):
                cluster1 = clusterItems[i]
                cluster2 = clusterItems[i + 1]
                var1 = self._getClusterVar(self.covMatrix, cluster1)
                var2 = self._getClusterVar(self.covMatrix, cluster2)

                # Calculate alpha for risk-based weighting
                varSum = var1 + var2
                alpha = 1 - var1 / varSum if varSum != 0 else 0.5

                self.weights.loc[cluster1] *= alpha
                self.weights.loc[cluster2] *= 1 - alpha

    def optimize(self, linkageMethod: str = "ward"):
        """Perform the Hierarchical Risk Parity optimisation.

        This involves:
        1. Calculating the distance matrix from the correlation.
        2. Performing hierarchical clustering (e.g., 'ward' linkage).
        3. Sorting assets using quasi-diagonalisation.
        4. Calculating final weights via recursive bisection.

        Args:
            linkageMethod (str, optional): Clustering method (e.g., 'ward',
                                           'single', 'complete'). Defaults to 'ward'.

        Raises:
            ValueError: If the linkage matrix computation fails.
        """
        corr = risk_models.cov_to_corr(self.covMatrix)
        # Use .round(8).fillna(0) for numerical stability and NaNs
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
    ) -> Tuple[Dict[str, int], float]:
        """Converts continuous HRP weights to a discrete share allocation.

        Uses linear programming to find the optimal integer number of shares
        to match the continuous weights given the latest prices.

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
        allocation: Dict[str, int]
        leftover: float
        allocation, leftover = da.lp_portfolio(verbose=False)
        return allocation, leftover


class HRPAnalyser:
    """Performs stability and performance analysis on the HRP model using bootstrap.

    This class runs a stationary bootstrap simulation to test the robustness of
    HRP portfolios constructed using various clustering linkage methods.
    """

    def __init__(self, returns: pd.DataFrame, riskFreeRate: float = 0.02):
        """Initialise the analyser.

        Args:
            returns (pd.DataFrame): Historical asset returns.
            riskFreeRate (float, optional): Annual risk-free rate for Sharpe
                                            ratio calculation. Defaults to 0.02.
        """
        self.returns: pd.DataFrame = returns
        self.riskFreeRate: float = riskFreeRate
        self.bootstrapResults: Optional[pd.DataFrame] = None
        self.cacheDir: Path = Path("cache")
        self.cacheDir.mkdir(exist_ok=True)

    def _getCachePath(self) -> Path:
        """Generates a unique cache path based on the tickers used."""
        tickerHash = abs(hash("".join(sorted(self.returns.columns))))
        return self.cacheDir / f"bootstrap_{tickerHash}.parquet"

    def runBootstrap(
        self,
        linkageMethods: Optional[List[str]] = None,
        reps: int = 500,
        verbose: bool = True,
        forceRecalculate: bool = False,
    ):
        """Run the stationary bootstrap analysis for multiple HRP linkage methods.

        Loads results from cache if available and `forceRecalculate` is False.
        Uses multiprocessing to speed up the numerous HRP optimisations.

        Args:
            linkageMethods (Optional[List[str]]): List of linkage methods
                                                  to test. Defaults to
                                                  ['ward', 'single', 'complete', 'average'].
            reps (int, optional): Number of bootstrap repetitions. Defaults to 500.
            verbose (bool, optional): Print status messages. Defaults to True.
            forceRecalculate (bool, optional): Ignore cache and run calculation.
                                               Defaults to False.
        """
        cachePath = self._getCachePath()
        if cachePath.exists() and not forceRecalculate:
            if verbose:
                print("Loading bootstrap results from cache...")
            self.bootstrapResults = pd.read_parquet(cachePath)
            if verbose:
                print("Bootstrap results loaded.")
            return

        if linkageMethods is None:
            linkageMethods = ["ward", "single", "complete", "average"]
        if verbose:
            print(f"Running bootstrap with {reps} reps...")

        allResults: List[pd.DataFrame] = []
        for method in linkageMethods:
            nCores = max(1, cpu_count() - 1)
            # Generate unique seeds for parallel processing
            seeds: NDArray[np.int32] = np.random.randint(
                0, 1_000_000, size=reps, dtype=np.int32
            )
            args = [(seed, method) for seed in seeds]
            resultsList: List[Dict[str, float]]

            with Pool(nCores) as pool:
                # Use starmap for multi-argument function
                resultsList = pool.starmap(self._bootstrapSingleMethod, args)

            methodDf = pd.DataFrame(resultsList)
            methodDf["linkage_method"] = method
            allResults.append(methodDf)

        self.bootstrapResults = pd.concat(allResults, ignore_index=True)
        self.bootstrapResults.to_parquet(cachePath)
        if verbose:
            print("Bootstrap complete and results saved to cache.")

    def _bootstrapSingleMethod(self, seed: int, linkageMethod: str) -> Dict[str, float]:
        """Performs a single HRP optimisation on a bootstrapped sample."""
        rs = np.random.default_rng(seed)

        # Calculate optimal block length based on squared returns (volatility clustering)
        blockLengths = optimal_block_length(self.returns**2)
        blockSize = int(blockLengths.iloc[:, 0].mean())
        bs = StationaryBootstrap(blockSize, self.returns, seed=rs)

        # Draw a single sample
        sampleData = next(iter(bs.bootstrap(1)))
        sampleReturns: pd.DataFrame = sampleData[0][0]

        # Ensure sampleReturns is DataFrame before calculation
        if isinstance(sampleReturns, np.ndarray):
            sampleReturns = pd.DataFrame(sampleReturns, columns=self.returns.columns)

        mu = expected_returns.mean_historical_return(
            sampleReturns, returns_data=True, frequency=252
        )
        sMatrix = self._calculateCovariance(sampleReturns)
        weightsResult = self._calculateHrpWeights(sMatrix, sampleReturns, linkageMethod)

        expReturn = (weightsResult * mu).sum()
        volatility = np.sqrt(weightsResult.T @ sMatrix @ weightsResult) * np.sqrt(252)
        sharpe = (
            (expReturn - self.riskFreeRate) / volatility if volatility > 1e-9 else 0.0
        )

        return {
            "exp_return": float(expReturn),
            "volatility": float(volatility),
            "sharpe_ratio": float(sharpe),
        }

    @staticmethod
    def _calculateCovariance(returns: pd.DataFrame) -> pd.DataFrame:
        """Calculate the Ledoit-Wolf shrunk covariance matrix from returns."""
        covMatrixInternal, _ = ledoit_wolf(returns, assume_centered=False)
        return pd.DataFrame(
            covMatrixInternal, index=returns.columns, columns=returns.columns
        )

    @staticmethod
    def _calculateHrpWeights(
        sMatrix: pd.DataFrame, returns: pd.DataFrame, linkage: str
    ) -> pd.Series:
        """Instantiate and run HRP model given inputs."""
        hrp = HRPModel(returns=returns)
        hrp.covMatrix = sMatrix  # Use provided shrunk covariance
        hrp.optimize(linkageMethod=linkage)
        return hrp.cleanWeights()

    def plotAssetPrices(self, prices: pd.DataFrame) -> None:
        """Plot the historical prices for each asset using Plotly subplots.

        Args:
            prices (pd.DataFrame): Asset price time series.
        """
        tickers = prices.columns
        numTickers = len(tickers)
        nRows = (numTickers + 1) // 2
        fig = make_subplots(rows=nRows, cols=2, subplot_titles=tickers)
        for i, ticker in enumerate(tickers):
            fig.add_trace(
                go.Scatter(x=prices.index, y=prices[ticker], name=ticker),
                row=i // 2 + 1,
                col=i % 2 + 1,
            )
        fig.update_layout(
            height=250 * nRows, title_text="Daily Prices", showlegend=False
        )
        fig.show()

    def plotPerformanceDistributions(self, metric: str = "sharpe_ratio") -> None:
        """Plot the distribution of a specified performance metric by linkage method.

        Uses a violin plot to show the full distribution of bootstrapped
        results for comparison.

        Args:
            metric (str, optional): Metric column name to plot (e.g., 'sharpe_ratio',
                                    'exp_return', 'volatility'). Defaults to 'sharpe_ratio'.

        Raises:
            RuntimeError: If `runBootstrap()` has not been called.
        """
        if self.bootstrapResults is None:
            raise RuntimeError("Run bootstrap analysis first.")

        metricTitle = metric.replace("_", " ").title()
        medianValues = (
            self.bootstrapResults.groupby("linkage_method")[metric]
            .median()
            .reset_index()
        )
        fig = px.violin(
            self.bootstrapResults,
            x="linkage_method",
            y=metric,
            color="linkage_method",
            title=f"Bootstrapped {metricTitle}s of HRP Portfolios",
            labels={"linkage_method": "Linkage Method", metric: metricTitle},
        )
        for _, row in medianValues.iterrows():
            value = row[metric]
            formattedValue = (
                f"{value * 100:.2f}%"
                if "return" in metric or "volatility" in metric
                else f"{value:.2f}"
            )
            fig.add_annotation(
                x=row["linkage_method"],
                y=value,
                text=f"Median:<br>{formattedValue}",
                showarrow=True,
                arrowhead=2,
                ax=0,
                ay=-40,
            )
        fig.update_layout(
            xaxis_title="Linkage Method",
            yaxis_title=metricTitle,
            legend_title="Linkage Method",
            legend=dict(
                orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5
            ),
        )
        fig.show()

    def plotRiskReturnProfiles(self) -> None:
        """Plot the risk-return scatter profiles grouped by linkage method.

        Generates a separate scatter plot for each linkage method in a
        multi-panel subplot, colouring points by the Sharpe Ratio.

        Raises:
            RuntimeError: If `runBootstrap()` has not been called.
        """
        if self.bootstrapResults is None:
            raise RuntimeError("Run bootstrap analysis first.")

        methods = sorted(self.bootstrapResults["linkage_method"].unique())
        nCols = 2
        nRows = (len(methods) + 1) // nCols
        fig = make_subplots(
            rows=nRows,
            cols=nCols,
            subplot_titles=methods,
            x_title="Annualised Volatility (Risk)",
            y_title="Annualised Expected Return",
        )
        shMin = self.bootstrapResults["sharpe_ratio"].min()
        shMax = self.bootstrapResults["sharpe_ratio"].max()

        for i, method in enumerate(methods):
            rowIdx, colIdx = i // nCols + 1, i % nCols + 1
            dfSubset = self.bootstrapResults[
                self.bootstrapResults["linkage_method"] == method
            ]
            fig.add_trace(
                go.Scatter(
                    x=dfSubset["volatility"],
                    y=dfSubset["exp_return"],
                    mode="markers",
                    name=method,
                    hovertext=dfSubset["sharpe_ratio"].round(2),
                    marker=dict(
                        color=dfSubset["sharpe_ratio"],
                        cmin=shMin,
                        cmax=shMax,
                        colorscale="Viridis",
                        showscale=(i == 0),
                        colorbar=dict(title="Sharpe Ratio") if i == 0 else None,
                    ),
                ),
                row=rowIdx,
                col=colIdx,
            )
        fig.update_layout(
            title="Risk-Return Profile by Linkage Method",
            showlegend=False,
            height=400 * nRows,
        )
        fig.show()

    def plotInvestmentGrowth(self, initialInvestment: float, years: int) -> None:
        """Plots the projected investment growth for each HRP method.

        Uses the median expected return from the bootstrap analysis to project
        the growth of an initial investment over a specified number of years.

        Args:
            initialInvestment (float): Starting value of the investment.
            years (int): Number of years for the projection.

        Raises:
            RuntimeError: If `runBootstrap()` has not been called.
        """
        if self.bootstrapResults is None:
            raise RuntimeError("Run bootstrap analysis first.")

        medianReturns = (
            self.bootstrapResults.groupby("linkage_method")["exp_return"]
            .median()
            .reset_index()
        )
        growthData = pd.DataFrame({"Year": range(1, years + 1)})

        for _, row in medianReturns.iterrows():
            rate = row["exp_return"]
            method = row["linkage_method"]
            growthData[method] = initialInvestment * ((1 + rate) ** growthData["Year"])

        growthDataMelted = growthData.melt(
            id_vars=["Year"], var_name="Linkage Method", value_name="Investment Value"
        )

        fig = px.line(
            growthDataMelted,
            x="Year",
            y="Investment Value",
            color="Linkage Method",
            title=(
                f"Projected Growth Over {years} Years "
                f"(Initial Investment: ${initialInvestment:,.0f})"
            ),
        )
        fig.update_layout(
            yaxis_title="Investment Value ($)",
            legend=dict(
                orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5
            ),
        )
        fig.show()

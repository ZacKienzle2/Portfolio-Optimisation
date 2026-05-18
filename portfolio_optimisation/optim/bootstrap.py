from multiprocessing import Pool, cpu_count
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from arch.bootstrap import StationaryBootstrap, optimal_block_length
from numpy.typing import NDArray
from plotly.subplots import make_subplots
from pypfopt import expected_returns
from sklearn.covariance import ledoit_wolf

from portfolio_optimisation.optim.hrp import HRPModel


class HRPAnalyser:
    """Performs stability and performance analysis on the HRP model using bootstrap.

    Runs a stationary bootstrap simulation to test the robustness of HRP
    portfolios constructed using various clustering linkage methods.
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
        self.bootstrapResults: pd.DataFrame | None = None
        self.cacheDir: Path = Path("cache")
        self.cacheDir.mkdir(exist_ok=True)
        self._blockSize: int = self._computeBlockSize(returns)

    @staticmethod
    def _computeBlockSize(returns: pd.DataFrame) -> int:
        """Optimal stationary-bootstrap block length on squared returns.

        Hoisted out of the per-iteration worker because it depends only on
        the full sample, not on the bootstrap seed.
        """
        blockLengths = optimal_block_length(returns**2)
        return int(blockLengths.iloc[:, 0].mean())

    def _getCachePath(self) -> Path:
        """Generates a unique cache path based on the tickers used."""
        tickerHash = abs(hash("".join(sorted(self.returns.columns))))
        return self.cacheDir / f"bootstrap_{tickerHash}.parquet"

    def runBootstrap(
        self,
        linkageMethods: list[str] | None = None,
        reps: int = 500,
        verbose: bool = True,
        forceRecalculate: bool = False,
    ):
        """Run the stationary bootstrap analysis for multiple HRP linkage methods.

        Loads results from cache if available and `forceRecalculate` is False.
        Uses multiprocessing to speed up the numerous HRP optimisations.

        Args:
            linkageMethods (Optional[List[str]]): List of linkage methods.
                                                  Defaults to ['ward', 'single', 'complete', 'average'].
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

        nCores = max(1, cpu_count() - 1)
        rng = np.random.default_rng()
        # One job-list spanning all (seed, method) pairs so a single worker
        # pool amortises spawn cost across every linkage method.
        jobs: list[tuple[int, str]] = []
        for method in linkageMethods:
            seeds: NDArray[np.int64] = rng.integers(
                0, 1_000_000, size=reps, dtype=np.int64
            )
            jobs.extend((int(seed), method) for seed in seeds)

        with Pool(nCores) as pool:
            results: list[dict[str, float]] = pool.starmap(
                self._bootstrapSingleMethod, jobs
            )

        methodColumn = [method for method in linkageMethods for _ in range(reps)]
        df = pd.DataFrame(results)
        df["linkage_method"] = methodColumn
        self.bootstrapResults = df

        self.bootstrapResults.to_parquet(cachePath)
        if verbose:
            print("Bootstrap complete and results saved to cache.")

    def _bootstrapSingleMethod(self, seed: int, linkageMethod: str) -> dict[str, float]:
        """Performs a single HRP optimisation on a bootstrapped sample."""
        rs = np.random.default_rng(seed)
        # _blockSize hoisted in __init__; reuse instead of recomputing per iter.
        bs = StationaryBootstrap(self._blockSize, self.returns, seed=rs)

        sampleData = next(iter(bs.bootstrap(1)))
        sampleReturns: pd.DataFrame = sampleData[0][0]

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
        hrp.covMatrix = sMatrix
        hrp.optimize(linkageMethod=linkage)
        return hrp.cleanWeights()

    def plotAssetPrices(self, prices: pd.DataFrame) -> None:
        """Plot the historical prices for each asset using Plotly subplots."""
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
        """Plot the distribution of a specified performance metric by linkage method."""
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
            legend={
                "orientation": "h",
                "yanchor": "bottom",
                "y": 1.02,
                "xanchor": "center",
                "x": 0.5,
            },
        )
        fig.show()

    def plotRiskReturnProfiles(self) -> None:
        """Plot the risk-return scatter profiles grouped by linkage method."""
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
                    marker={
                        "color": dfSubset["sharpe_ratio"],
                        "cmin": shMin,
                        "cmax": shMax,
                        "colorscale": "Viridis",
                        "showscale": (i == 0),
                        "colorbar": {"title": "Sharpe Ratio"} if i == 0 else None,
                    },
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
        """Project investment growth using median expected return per HRP method."""
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
            legend={
                "orientation": "h",
                "yanchor": "bottom",
                "y": 1.02,
                "xanchor": "center",
                "x": 0.5,
            },
        )
        fig.show()

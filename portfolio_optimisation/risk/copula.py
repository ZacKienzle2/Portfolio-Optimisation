from typing import Any, Dict, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.axes import Axes
from numpy.typing import NDArray
from scipy import stats
from scipy.optimize import OptimizeResult, minimize
from statsmodels.distributions.copula.api import StudentTCopula


class CopulaRiskAnalyser:
    """Performs multivariate risk simulation using a Student's t-copula.

    Models marginal returns and their dependence via a t-copula. Fits
    marginals (default: t-dist), estimates copula df (MLE) and correlation
    (Kendall's Tau), then simulates correlated returns.
    """

    def __init__(self, returns: pd.DataFrame, weights: pd.Series):
        """Initialise analyser with asset returns and portfolio weights.

        Args:
            returns (pd.DataFrame): Asset returns (columns=assets, index=time).
            weights (pd.Series): Portfolio weights corresponding to asset columns.
        """
        common_tickers = list(set(returns.columns) & set(weights.index))
        if not common_tickers:
            raise ValueError("Returns and weights must share common tickers.")

        self.returns: pd.DataFrame = returns[common_tickers]
        self.weights: pd.Series = weights.reindex(common_tickers).fillna(0.0)

        if self.returns.empty or self.weights.empty:
            raise ValueError("Filtered returns or weights are empty.")

        self.fitMarginals: Dict[str, Dict[str, Any]] = {}
        self.copula: Optional[Any] = None
        self.uniformReturns: Optional[pd.DataFrame] = None

    def fit_marginal_distributions(self, dist: Any = stats.t):
        """Fit a univariate distribution to each asset's returns series.

        Args:
            dist (Any): scipy.stats distribution object with fit/cdf/ppf.
        """
        if not all(hasattr(dist, method) for method in ["fit", "cdf", "ppf"]):
            raise TypeError("dist object must have fit, cdf, and ppf methods.")

        for ticker in self.returns.columns:
            seriesData = self.returns[ticker].dropna()
            if len(seriesData) < 10:
                print(f"Warning: Insufficient data for {ticker} marginal fit.")
                self.fitMarginals[ticker] = {"dist": None, "params": None}
                continue
            try:
                params: tuple[Any, ...] = dist.fit(seriesData.to_numpy())
                self.fitMarginals[ticker] = {"dist": dist, "params": params}
            except Exception as e:
                print(f"Warning: Failed marginal fit for {ticker}: {e}")
                self.fitMarginals[ticker] = {"dist": None, "params": None}

    def _estimate_copula_params(self) -> tuple[NDArray[np.float64], float]:
        """Estimate correlation matrix and degrees of freedom for the t-copula."""
        if self.uniformReturns is None:
            raise RuntimeError("Uniform returns required but not generated.")

        kendall_tau: pd.DataFrame = self.uniformReturns.corr("kendall")
        tau: NDArray[np.float64] = kendall_tau.to_numpy()
        corr: NDArray[np.float64] = np.sin(tau * np.pi / 2)
        corr = self._ensure_psd(corr)

        def log_likelihood(dfParam: float) -> float:
            """Calculate negative log-likelihood of t-copula given data."""
            if self.uniformReturns is None:
                raise RuntimeError("Internal Error: uniformReturns became None.")
            try:
                copulaInternal: Any = StudentTCopula(
                    corr=corr, df=dfParam, k_dim=self.returns.shape[1]
                )
                uniform_vals_np = self.uniformReturns.to_numpy()
                uniformValues = np.clip(uniform_vals_np, 1e-9, 1 - 1e-9)
                pdfValues: NDArray[np.float64] = copulaInternal.pdf(uniformValues)
                pdfValues = np.maximum(pdfValues, 1e-12)
                penalty = (
                    0.01 * (dfParam - 10) ** 2 if dfParam > 25 or dfParam < 3 else 0
                )
                return -np.sum(np.log(pdfValues)) + penalty
            except Exception as e:
                print(f"Warning: Error in likelihood calc for df={dfParam}: {e}")
                return np.inf

        result: OptimizeResult = minimize(
            log_likelihood,
            x0=np.array([10.0]),
            bounds=[(2.1, 30.0)],
            method="L-BFGS-B",
        )
        if not result.success:
            print(f"Warning: Copula df optimization issue: {result.message}")

        estimatedDf: float = float(result.x[0])
        return corr, estimatedDf

    def _ensure_psd(
        self, matrix: NDArray[np.float64], epsilon: float = 1e-8
    ) -> NDArray[np.float64]:
        """Ensure matrix positive semi-definiteness via eigenvalue adjustment."""
        eigenvalues, eigenvectors = np.linalg.eigh(matrix)
        eigenvalues[eigenvalues < epsilon] = epsilon
        psdMatrix = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T
        scale = np.sqrt(np.diag(psdMatrix))
        scale = np.where(scale < epsilon, 1.0, scale)
        psdMatrix = psdMatrix / np.outer(scale, scale)
        return np.clip(psdMatrix, -1.0, 1.0)

    def fit_copula(self):
        """Fit the Student's t-copula to transformed marginal returns."""
        if not self.fitMarginals:
            raise RuntimeError("Fit marginal distributions first.")

        uniformData = {}
        validTickers = []
        for ticker in self.returns.columns:
            marginalInfo = self.fitMarginals.get(ticker)
            if (
                marginalInfo
                and marginalInfo["dist"] is not None
                and marginalInfo["params"] is not None
            ):
                dist = marginalInfo["dist"]
                params = marginalInfo["params"]
                seriesData = self.returns[ticker].dropna()
                if not seriesData.empty:
                    try:
                        uniformData[ticker] = dist.cdf(seriesData, *params)
                        validTickers.append(ticker)
                    except Exception as e:
                        print(f"Warning: CDF transformation failed for {ticker}: {e}")
            else:
                print(f"Warning: Skipping {ticker} (no valid marginal fit).")

        if not validTickers:
            raise RuntimeError("No valid marginals; cannot fit copula.")

        self.returns = self.returns[validTickers]
        self.weights = self.weights.loc[validTickers]
        if not np.isclose(self.weights.sum(), 1.0):
            self.weights /= self.weights.sum()

        self.uniformReturns = pd.DataFrame(uniformData)
        self.uniformReturns.dropna(inplace=True)

        if (
            self.uniformReturns.empty
            or len(self.uniformReturns) < self.returns.shape[1] + 1
        ):
            raise RuntimeError("Insufficient valid uniform data; cannot fit copula.")

        corr, df = self._estimate_copula_params()
        self.copula = StudentTCopula(corr=corr, df=df, k_dim=self.returns.shape[1])

    def run_simulation(self, nSimulations: int = 10000) -> pd.DataFrame:
        """Generate correlated portfolio returns using the fitted copula.

        Args:
            nSimulations (int): Number of simulation paths.

        Returns:
            pd.DataFrame: Simulated portfolio returns ('simulated_returns').
        """
        if self.copula is None:
            raise RuntimeError("Fit the copula first.")

        simulatedUniform: NDArray[np.float64] = self.copula.rvs(nSimulations)
        simulatedUniform = np.clip(simulatedUniform, 1e-9, 1 - 1e-9)

        simulatedReturnsData = {}
        for i, ticker in enumerate(self.returns.columns):
            marginalInfo = self.fitMarginals.get(ticker)
            if (
                marginalInfo
                and marginalInfo["dist"] is not None
                and marginalInfo["params"] is not None
            ):
                dist = marginalInfo["dist"]
                params = marginalInfo["params"]
                try:
                    simulatedReturnsData[ticker] = dist.ppf(
                        simulatedUniform[:, i], *params
                    )
                except Exception as e:
                    print(f"Warning: PPF failed for {ticker}: {e}")
                    simulatedReturnsData[ticker] = np.full(nSimulations, np.nan)
            else:
                simulatedReturnsData[ticker] = np.full(nSimulations, np.nan)

        simulatedReturnsDf = pd.DataFrame(simulatedReturnsData).dropna()

        if simulatedReturnsDf.empty:
            print("Warning: Simulation yielded empty DataFrame after NaNs.")
            return pd.DataFrame({"simulated_returns": []})

        alignedWeights = self.weights.reindex(simulatedReturnsDf.columns).fillna(0.0)
        portfolioReturnsGenerated: pd.Series = simulatedReturnsDf.dot(alignedWeights)

        return pd.DataFrame({"simulated_returns": portfolioReturnsGenerated})

    def plot_marginal_fits(self):
        """Plot fitted marginal distributions against historical data."""
        if self.returns.empty or not self.fitMarginals:
            print("Return data or marginal fits missing for plotting.")
            return

        nAssets = len(self.returns.columns)
        if nAssets == 0:
            return
        nCols = 2
        nRows = (nAssets + nCols - 1) // nCols

        fig: plt.Figure
        axes: NDArray[Axes]
        fig, axes = plt.subplots(nRows, nCols, figsize=(14, nRows * 5), squeeze=False)
        fig.suptitle(
            "Fitted Marginal Distributions vs. Historical Returns", fontsize=16
        )
        axes_flat: NDArray[Axes] = axes.flatten()

        plotIndex = 0
        for ticker in self.returns.columns:
            ax: Axes = axes_flat[plotIndex]
            data = self.returns[ticker].dropna()

            if data.empty:
                ax.set_title(f"{ticker}: No Data")
                ax.set_xticks([])
                ax.set_yticks([])
                plotIndex += 1
                continue

            sns.histplot(x=data, bins=50, stat="density", ax=ax, label="Historical")

            marginalInfo = self.fitMarginals.get(ticker)
            if (
                marginalInfo
                and marginalInfo["dist"] is not None
                and marginalInfo["params"] is not None
            ):
                dist = marginalInfo["dist"]
                params = marginalInfo["params"]
                xMin, xMax = data.min(), data.max()
                if np.isclose(xMin, xMax):
                    xMin -= 0.1
                    xMax += 0.1
                xRange: NDArray[np.float64] = np.linspace(xMin, xMax, 200)
                try:
                    pdfValues = dist.pdf(xRange, *params)
                    ax.plot(xRange, pdfValues, "r-", lw=2, label="Fitted PDF")
                except Exception as e:
                    print(f"Warning: PDF plot failed for {ticker}: {e}")
                    ax.plot([], [], "r-", lw=2, label="Fit Error")
            else:
                ax.plot([], [], "r--", label="Fit Failed")

            ax.set_title(f"Fit for {ticker}")
            ax.legend()
            plotIndex += 1

        for j in range(plotIndex, len(axes_flat)):
            fig.delaxes(axes_flat[j])

        fig.tight_layout(rect=(0, 0.03, 1, 0.96))
        plt.show()

    def plot_copula_dependence(self):
        """Visualise pairwise dependence structure in uniform (copula) space."""
        if self.uniformReturns is None or self.uniformReturns.empty:
            print("Uniform returns unavailable/empty; cannot plot dependence.")
            return

        g: Any = sns.pairplot(
            self.uniformReturns,
            kind="scatter",
            diag_kind="kde",
            plot_kws={"alpha": 0.4},
        )
        if hasattr(g, "figure"):
            g.figure.suptitle("Pairwise Dependence Structure (Uniform Space)", y=1.02)
        plt.show()


def run_historical_simulation(
    returns: pd.DataFrame, weights: pd.Series, nSimulations: int = 10000
) -> pd.DataFrame:
    """Simulate portfolio returns by resampling historical daily returns.

    Performs Monte Carlo simulation by drawing samples with replacement
    from the observed historical portfolio return distribution.
    Assumes IID returns.

    Args:
        returns (pd.DataFrame): Historical asset returns.
        weights (pd.Series): Portfolio weights.
        nSimulations (int): Number of simulated days.

    Returns:
        pd.DataFrame: Simulated portfolio returns ('simulated_returns').
    """
    if returns.empty or weights.empty:
        return pd.DataFrame({"simulated_returns": []})

    alignedWeights = weights.reindex(returns.columns).fillna(0.0)
    historicalPortfolioReturns: pd.Series = returns.dot(alignedWeights)

    validHistoricalReturns = historicalPortfolioReturns.dropna()
    if validHistoricalReturns.empty:
        return pd.DataFrame({"simulated_returns": []})

    simulatedReturnsArray = np.random.choice(
        validHistoricalReturns.to_numpy(), size=nSimulations, replace=True
    )
    return pd.DataFrame({"simulated_returns": simulatedReturnsArray})

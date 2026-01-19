# Portfolio_Theory/risk.py
from typing import Any, Dict, Literal, Optional
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
import numpy as np
import pandas as pd
import seaborn as sns
from numpy.typing import NDArray
from scipy import stats
from scipy.optimize import minimize, OptimizeResult
from statsmodels.distributions.copula.api import StudentTCopula


def calculateRiskMetrics(
    simulatedReturns: pd.DataFrame,
    alpha: float = 0.05,
    method: Literal["empirical", "parametric"] = "empirical",
) -> Dict[str, float]:
    """Calculate Value at Risk (VaR) and Conditional VaR (CVaR).

    Computes VaR using either the empirical quantile of simulated returns
    or a parametric method assuming normality. CVaR is the mean of returns
    below the selected VaR threshold.

    Args:
        simulatedReturns (pd.DataFrame): DataFrame with 'simulated_returns'.
        alpha (float): Significance level for VaR (e.g., 0.05).
        method (Literal["empirical", "parametric"]): VaR calculation method.

    Returns:
        Dict[str, float]: VaR, CVaR, Empirical VaR, and Parametric VaR.
    """
    returnsSeries: pd.Series = simulatedReturns["simulated_returns"]
    returnsArray: NDArray[np.float64] = returnsSeries.to_numpy()

    if returnsArray.size == 0:
        return {
            "VaR": np.nan,
            "CVaR": np.nan,
            "Empirical VaR": np.nan,
            "Parametric VaR": np.nan,
        }

    empiricalVar: float = float(np.quantile(returnsArray, alpha))
    meanReturns: float = float(returnsArray.mean())
    stdDevReturns: float = float(returnsArray.std())

    parametricVar: float
    if stdDevReturns < 1e-12:
        parametricVar = meanReturns
    else:
        parametricVar = float(
            stats.norm.ppf(alpha, loc=meanReturns, scale=stdDevReturns)
        )

    selectedVar: float = parametricVar if method == "parametric" else empiricalVar

    tailReturns = returnsArray[returnsArray <= selectedVar]
    cvar: float = float(np.mean(tailReturns)) if tailReturns.size > 0 else selectedVar

    return {
        "VaR": selectedVar,
        "CVaR": cvar,
        "Empirical VaR": empiricalVar,
        "Parametric VaR": parametricVar,
    }


def calculatePerformanceMetrics(
    portfolioReturns: pd.Series, riskFreeRate: float = 0.02
) -> Dict[str, float]:
    """Calculate standard portfolio performance metrics.

    Computes annualised return, volatility, Sharpe ratio, Sortino ratio,
    and maximum drawdown from a series of returns.

    Args:
        portfolioReturns (pd.Series): Time series of portfolio returns.
        riskFreeRate (float): Annual risk-free rate.

    Returns:
        Dict[str, float]: Key performance metrics.
    """
    ANNUAL_FACTOR = 252
    SQRT_ANNUAL_FACTOR = np.sqrt(ANNUAL_FACTOR)

    if portfolioReturns.empty:
        return {
            "Annualised Return": np.nan,
            "Annualised Volatility": np.nan,
            "Sharpe Ratio": np.nan,
            "Sortino Ratio": np.nan,
            "Max Drawdown": np.nan,
        }

    annualisedReturn: float = float(portfolioReturns.mean()) * ANNUAL_FACTOR
    annualisedVolatility: float = float(portfolioReturns.std()) * SQRT_ANNUAL_FACTOR

    targetReturnDaily: float = (1 + riskFreeRate) ** (1 / ANNUAL_FACTOR) - 1
    downsideReturns: pd.Series = portfolioReturns.loc[
        portfolioReturns < targetReturnDaily
    ]
    downsideStd: float = (
        float(downsideReturns.std()) * SQRT_ANNUAL_FACTOR
        if not downsideReturns.empty and downsideReturns.std() > 0
        else 0.0
    )

    cumulativeReturns: pd.Series = (1 + portfolioReturns).cumprod()
    peak: pd.Series = cumulativeReturns.expanding(min_periods=1).max()
    drawdown: pd.Series = (cumulativeReturns - peak) / peak
    maxDrawdown: float = float(drawdown.min()) if not drawdown.empty else 0.0

    sharpeRatio: float = (
        (annualisedReturn - riskFreeRate) / annualisedVolatility
        if annualisedVolatility > 1e-9
        else 0.0
    )
    sortinoRatio: float = (
        (annualisedReturn - riskFreeRate) / downsideStd if downsideStd > 1e-9 else 0.0
    )

    return {
        "Annualised Return": annualisedReturn,
        "Annualised Volatility": annualisedVolatility,
        "Sharpe Ratio": sharpeRatio,
        "Sortino Ratio": sortinoRatio,
        "Max Drawdown": maxDrawdown,
    }


def plotSimulationResults(
    simulatedReturns: pd.DataFrame,
    riskMetrics: Dict[str, float],
):
    """Visualise the distribution of simulated returns with VaR/CVaR.

    Creates a histogram with density curve for simulated returns, overlaying
    VaR and CVaR lines. A rug plot below highlights tail events.

    Args:
        simulatedReturns (pd.DataFrame): DataFrame with 'simulated_returns'.
        riskMetrics (Dict[str, float]): Dictionary containing 'VaR' and 'CVaR'.
    """
    if simulatedReturns.empty or "simulated_returns" not in simulatedReturns.columns:
        print("Warning: Empty or invalid DataFrame provided for plotting.")
        return

    returnsSeries: pd.Series = simulatedReturns["simulated_returns"]
    varValue: Optional[float] = riskMetrics.get("VaR")
    cvarValue: Optional[float] = riskMetrics.get("CVaR")

    if varValue is None or cvarValue is None:
        print("Warning: Valid VaR or CVaR missing in riskMetrics for plotting.")
        return

    tailReturns: pd.Series = returnsSeries[returnsSeries <= varValue]

    fig: plt.Figure
    axHist: Axes
    axRug: Axes
    fig, (axHist, axRug) = plt.subplots(
        2,
        1,
        figsize=(12, 8),
        sharex=True,
        gridspec_kw={"height_ratios": [0.8, 0.2]},
    )

    plotTitle = "Distribution of Simulated Portfolio Returns"
    axHist = sns.histplot(
        x=returnsSeries,
        bins=70,
        stat="density",
        ax=axHist,
        label="Simulated Returns",
        alpha=0.7,
        color="skyblue",
        kde=False,
    )
    axHist.axvline(
        varValue,
        color="red",
        linestyle="--",
        lw=2,
        label=f"VaR (95%): {varValue:.2%}",
    )
    axHist.axvline(
        cvarValue,
        color="purple",
        linestyle="-",
        lw=2,
        label=f"CVaR (95%): {cvarValue:.2%}",
    )
    axHist.set_title(plotTitle)
    axHist.set_ylabel("Density")
    axHist.legend()
    axHist.grid(True, alpha=0.3)

    axRug = sns.rugplot(x=tailReturns, ax=axRug, color="darkviolet", height=0.5)
    axRug.axvline(cvarValue, color="purple", linestyle="-", lw=2)
    axRug.set_title("Individual Tail Events (Losses Beyond VaR)")
    axRug.set_xlabel("Daily Portfolio Return")
    axRug.set_yticks([])
    axRug.grid(True, axis="x", alpha=0.3)

    fig.tight_layout()
    plt.show()


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
        # Filter returns to only include columns present in weights.
        common_tickers = list(set(returns.columns) & set(weights.index))
        if not common_tickers:
            raise ValueError("Returns and weights must share common tickers.")

        self.returns: pd.DataFrame = returns[common_tickers]
        self.weights: pd.Series = weights.reindex(common_tickers).fillna(0.0)

        # Rerun validation using filtered data
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

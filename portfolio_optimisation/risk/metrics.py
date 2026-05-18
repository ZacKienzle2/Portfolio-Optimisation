from typing import Literal

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy import stats


def calculateRiskMetrics(
    simulatedReturns: pd.DataFrame,
    alpha: float = 0.05,
    method: Literal["empirical", "parametric"] = "empirical",
) -> dict[str, float]:
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
) -> dict[str, float]:
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

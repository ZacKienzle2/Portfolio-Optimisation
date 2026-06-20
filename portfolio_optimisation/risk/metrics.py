from typing import Literal

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy import stats

ANNUAL_FACTOR: int = 252
SQRT_ANNUAL_FACTOR: float = float(np.sqrt(ANNUAL_FACTOR))


def calculate_risk_metrics(
    simulated_returns: pd.DataFrame,
    alpha: float = 0.05,
    method: Literal["empirical", "parametric"] = "empirical",
) -> dict[str, float]:
    """Calculate Value at Risk (VaR) and Conditional VaR (CVaR).

    Computes VaR using either the empirical quantile of simulated returns
    or a parametric method assuming normality. CVaR is the mean of returns
    below the selected VaR threshold.

    Args:
        simulated_returns (pd.DataFrame): DataFrame with 'simulated_returns'.
        alpha (float): Significance level for VaR (e.g., 0.05).
        method (Literal["empirical", "parametric"]): VaR calculation method.

    Returns:
        Dict[str, float]: VaR, CVaR, Empirical VaR, and Parametric VaR.
    """
    returns_series: pd.Series = simulated_returns["simulated_returns"]
    returns_array: NDArray[np.float64] = returns_series.to_numpy()

    if returns_array.size == 0:
        return {
            "VaR": np.nan,
            "CVaR": np.nan,
            "Empirical VaR": np.nan,
            "Parametric VaR": np.nan,
        }

    empirical_var: float = float(np.quantile(returns_array, alpha))
    mean_returns: float = float(returns_array.mean())
    std_dev_returns: float = float(returns_array.std())

    parametric_var: float
    if std_dev_returns < 1e-12:
        parametric_var = mean_returns
    else:
        parametric_var = float(stats.norm.ppf(alpha, loc=mean_returns, scale=std_dev_returns))

    selected_var: float = parametric_var if method == "parametric" else empirical_var

    tail_returns = returns_array[returns_array <= selected_var]
    cvar: float = float(np.mean(tail_returns)) if tail_returns.size > 0 else selected_var

    return {
        "VaR": selected_var,
        "CVaR": cvar,
        "Empirical VaR": empirical_var,
        "Parametric VaR": parametric_var,
    }


def calculate_performance_metrics(
    portfolio_returns: pd.Series, risk_free_rate: float = 0.02
) -> dict[str, float]:
    """Calculate standard portfolio performance metrics.

    Computes annualised return, volatility, Sharpe ratio, Sortino ratio,
    and maximum drawdown from a series of returns.

    Args:
        portfolio_returns (pd.Series): Time series of portfolio returns.
        risk_free_rate (float): Annual risk-free rate.

    Returns:
        Dict[str, float]: Key performance metrics.
    """
    if portfolio_returns.empty:
        return {
            "Annualised Return": np.nan,
            "Annualised Volatility": np.nan,
            "Sharpe Ratio": np.nan,
            "Sortino Ratio": np.nan,
            "Max Drawdown": np.nan,
        }

    annualised_return: float = float(portfolio_returns.mean()) * ANNUAL_FACTOR
    annualised_volatility: float = float(portfolio_returns.std()) * SQRT_ANNUAL_FACTOR

    target_return_daily: float = (1 + risk_free_rate) ** (1 / ANNUAL_FACTOR) - 1
    downside_returns: pd.Series = portfolio_returns.loc[portfolio_returns < target_return_daily]
    downside_std: float = (
        float(downside_returns.std()) * SQRT_ANNUAL_FACTOR
        if not downside_returns.empty and downside_returns.std() > 0
        else 0.0
    )

    cumulative_returns: NDArray[np.float64] = (
        (1.0 + portfolio_returns).cumprod().to_numpy(dtype=np.float64)
    )
    peak: NDArray[np.float64] = np.maximum.accumulate(cumulative_returns)
    drawdown: NDArray[np.float64] = (cumulative_returns - peak) / peak
    max_drawdown: float = float(drawdown.min()) if drawdown.size > 0 else 0.0

    sharpe_ratio: float = (
        (annualised_return - risk_free_rate) / annualised_volatility
        if annualised_volatility > 1e-9
        else 0.0
    )
    sortino_ratio: float = (
        (annualised_return - risk_free_rate) / downside_std if downside_std > 1e-9 else 0.0
    )

    return {
        "Annualised Return": annualised_return,
        "Annualised Volatility": annualised_volatility,
        "Sharpe Ratio": sharpe_ratio,
        "Sortino Ratio": sortino_ratio,
        "Max Drawdown": max_drawdown,
    }

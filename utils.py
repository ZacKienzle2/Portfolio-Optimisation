# Portfolio_Theory/utils.py
from typing import Any, Dict, List, Tuple, Union, Optional
from pathlib import Path
import numpy as np
import pandas as pd
import yfinance as yf
from numpy.typing import NDArray
from rich.console import Console
from rich.table import Table
from pypfopt import discrete_allocation


def getData(
    tickers: List[str], startDate: str, console: Console
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Fetch and preprocess financial time series data.

    Downloads or loads cached adjusted closing prices using yfinance,
    aligns data to a common start date, handles missing values by dropping
    assets with incomplete data, and calculates daily percentage returns.
    Uses parquet caching for efficiency.

    Args:
        tickers (List[str]): List of asset symbols (e.g., ['AAPL', 'MSFT']).
        startDate (str): Start date for data retrieval (YYYY-MM-DD format).
        console (Console): Rich console object for displaying status messages.

    Raises:
        ValueError: If data download fails or results in empty data.
        TypeError: If downloaded data is not a DataFrame or Series.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: Cleaned prices DataFrame and
                                           Daily returns DataFrame.
    """
    cachePath = Path.cwd() / "Initial_Files" / "market_data.parquet"
    cachePath.parent.mkdir(exist_ok=True)

    pricesRaw: Union[pd.DataFrame, pd.Series, None]
    if cachePath.exists():
        console.print(f"[green]Loading cached data from {cachePath}...[/green]")
        pricesRaw = pd.read_parquet(cachePath, engine="pyarrow")
    else:
        console.print(
            f"[yellow]Fetching data for {len(tickers)} assets "
            f"from {startDate}...[/yellow]"
        )
        downloadedData: Any = yf.download(
            tickers, start=startDate, auto_adjust=False, progress=False
        )
        if (
            isinstance(downloadedData, pd.DataFrame)
            and "Adj Close" in downloadedData.columns
        ):
            pricesRaw = downloadedData["Adj Close"]
        elif isinstance(downloadedData, pd.Series):
            pricesRaw = downloadedData
        else:
            pricesRaw = None

        if pricesRaw is None or pricesRaw.empty:
            raise ValueError("Failed to download valid price data.")

        pricesRaw.to_parquet(cachePath, engine="pyarrow")
        console.print(f"[green]Saved new data cache to {cachePath}.[/green]")

    if pricesRaw is None:
        # This check is technically redundant due to the earlier
        # check, but it satisfies static analysis.
        raise ValueError("Price data is None after download/load.")

    pricesRawDf: pd.DataFrame
    if isinstance(pricesRaw, pd.Series):
        pricesRawDf = pricesRaw.to_frame(name=tickers[0])
    else:
        # Type is narrowed: Must be pd.DataFrame
        pricesRawDf = pricesRaw

    # Break up apply().max() for type clarity
    first_indices: pd.Series = pricesRawDf.apply(lambda col: col.first_valid_index())
    commonStart: pd.Timestamp = first_indices.max()
    prices: pd.DataFrame = pricesRawDf.loc[commonStart:].dropna(axis=1)
    returns: pd.DataFrame = prices.pct_change().dropna()

    console.print(
        f"[green]Analysis ready for {len(prices.columns)} "
        f"assets from {prices.index[0].date()}.[/green]\n"
    )
    return prices, returns


def generateFinalReport(console: Console, metricsData: Dict[str, Dict[str, float]]):
    """Display portfolio performance and risk metrics in a formatted table.

    Renders key metrics (return, volatility, Sharpe, Sortino, drawdown,
    VaR, CVaR) for multiple portfolios in a visually structured table
    using the Rich library for console output.

    Args:
        console (Console): Rich console object used for printing the table.
        metricsData (Dict[str, Dict[str, float]]): Nested dictionary where
            outer keys are portfolio names and inner keys are metric names
            (e.g., 'Sharpe Ratio') with their corresponding float values.
    """
    portfolioNames: List[str] = list(metricsData.keys())
    table = Table(
        show_header=True,
        header_style="bold cyan",
        title="\nPortfolio & Risk Analysis",
        title_style="bold magenta",
    )
    table.add_column("Metric", style="dim", width=25)
    for name in portfolioNames:
        table.add_column(name, justify="right")

    rows: List[Tuple[str, str, Optional[str]]] = [
        ("Annualised Return", ".2%", "green"),
        ("Annualised Volatility", ".2%", None),
        ("Sharpe Ratio", ".2f", "bold"),
        ("Sortino Ratio", ".2f", "bold"),
        ("Max Drawdown", ".2%", "red"),
        ("Daily VaR (95%)", ".2%", "red"),
        ("Daily CVaR (95%)", ".2%", "bold red"),
    ]
    keyMap: Dict[str, str] = {
        "Daily VaR (95%)": "VaR",
        "Daily CVaR (95%)": "CVaR",
    }

    for i, (metric, fmt, style) in enumerate(rows):
        if i in [2, 5]:
            table.add_section()
        key: str = keyMap.get(metric, metric)

        values: List[str] = []
        for p in portfolioNames:
            values.append(f"{metricsData[p][key]:{fmt}}")

        if style:
            table.add_row(metric, *values, style=style)
        else:
            table.add_row(metric, *values)

    console.print(table)


def inverseVarianceWeights(covMatrix: pd.DataFrame) -> pd.Series:
    """Calculate inverse variance portfolio weights.

    Computes portfolio weights which are inversely proportional to asset
    variance, derived from the diagonal elements of the covariance matrix.
    Aims to minimize portfolio variance without considering expected returns.

    Args:
        covMatrix (pd.DataFrame): Covariance matrix of asset returns.

    Returns:
        pd.Series: Asset weights for the inverse variance portfolio.
    """
    variances: NDArray[np.float64] = np.diag(covMatrix)
    invVariances: NDArray[np.float64] = 1 / (variances + 1e-12)
    ivpWeights: NDArray[np.float64] = invVariances / np.sum(invVariances)
    return pd.Series(ivpWeights, index=covMatrix.index)


def get_discrete_portfolio(
    weights: pd.Series, prices: pd.DataFrame, totalValue: float = 1_000_000.0
) -> Tuple[Dict[str, int], float]:
    """Convert continuous weights to a discrete number of shares.

    Uses linear programming (via PyPortfolioOpt) to find the optimal
    integer number of shares per asset that closely matches target
    continuous weights, given the latest asset prices and total
    portfolio value. Also returns leftover cash.

    Args:
        weights (pd.Series): Target continuous weights for each asset.
        prices (pd.DataFrame): DataFrame of historical asset prices
                               (latest row is used).
        totalValue (float, optional): Total monetary value to allocate.

    Returns:
        Tuple[Dict[str, int], float]: Dictionary mapping asset tickers to
                                      share counts, and the unallocated
                                      cash amount.
    """

    latestPrices: pd.Series = prices.iloc[-1]
    da = discrete_allocation.DiscreteAllocation(
        weights=weights.to_dict(),
        latest_prices=latestPrices,
        total_portfolio_value=int(totalValue),
    )
    allocation: Dict[str, int]
    leftover: float
    allocation, leftover = da.lp_portfolio(verbose=False)
    return allocation, leftover

from pathlib import Path
from typing import Any, List, Tuple, Union

import pandas as pd
import yfinance as yf
from rich.console import Console


def getData(
    tickers: List[str], startDate: str, console: Console
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Fetch and preprocess financial time series data.

    Downloads or loads cached adjusted closing prices using yfinance,
    aligns data to a common start date, drops assets with incomplete data,
    and calculates daily percentage returns. Uses parquet caching.

    Args:
        tickers (List[str]): Asset symbols.
        startDate (str): Start date (YYYY-MM-DD).
        console (Console): Rich console for status messages.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: Cleaned prices and daily returns.

    Raises:
        ValueError: If data download fails or yields empty data.
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
        raise ValueError("Price data is None after download/load.")

    pricesRawDf: pd.DataFrame
    if isinstance(pricesRaw, pd.Series):
        pricesRawDf = pricesRaw.to_frame(name=tickers[0])
    else:
        pricesRawDf = pricesRaw

    first_indices: pd.Series = pricesRawDf.apply(lambda col: col.first_valid_index())
    commonStart: pd.Timestamp = first_indices.max()
    prices: pd.DataFrame = pricesRawDf.loc[commonStart:].dropna(axis=1)
    returns: pd.DataFrame = prices.pct_change().dropna()

    console.print(
        f"[green]Analysis ready for {len(prices.columns)} "
        f"assets from {prices.index[0].date()}.[/green]\n"
    )
    return prices, returns

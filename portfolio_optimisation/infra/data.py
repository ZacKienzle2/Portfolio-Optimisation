from pathlib import Path
from typing import Any

import pandas as pd
import yfinance as yf
from rich.console import Console


def get_data(
    tickers: list[str], start_date: str, console: Console
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fetch and preprocess financial time series data.

    Downloads or loads cached adjusted closing prices using yfinance,
    aligns data to a common start date, drops assets with incomplete data,
    and calculates daily percentage returns. Uses parquet caching.

    Args:
        tickers (List[str]): Asset symbols.
        start_date (str): Start date (YYYY-MM-DD).
        console (Console): Rich console for status messages.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: Cleaned prices and daily returns.

    Raises:
        ValueError: If data download fails or yields empty data.
    """
    cache_path = Path.cwd() / "Initial_Files" / "market_data.parquet"
    cache_path.parent.mkdir(exist_ok=True)

    prices_raw: pd.DataFrame | pd.Series | None
    if cache_path.exists():
        console.print(f"[green]Loading cached data from {cache_path}...[/green]")
        prices_raw = pd.read_parquet(cache_path, engine="pyarrow")
    else:
        console.print(
            f"[yellow]Fetching data for {len(tickers)} assets from {start_date}...[/yellow]"
        )
        downloaded_data: Any = yf.download(
            tickers, start=start_date, auto_adjust=False, progress=False
        )
        if isinstance(downloaded_data, pd.DataFrame) and "Adj Close" in downloaded_data.columns:
            prices_raw = downloaded_data["Adj Close"]
        elif isinstance(downloaded_data, pd.Series):
            prices_raw = downloaded_data
        else:
            prices_raw = None

        if prices_raw is None or prices_raw.empty:
            raise ValueError("Failed to download valid price data.")

        prices_raw.to_parquet(cache_path, engine="pyarrow")
        console.print(f"[green]Saved new data cache to {cache_path}.[/green]")

    if prices_raw is None:
        raise ValueError("Price data is None after download/load.")

    prices_raw_df: pd.DataFrame
    if isinstance(prices_raw, pd.Series):
        prices_raw_df = prices_raw.to_frame(name=tickers[0])
    else:
        prices_raw_df = prices_raw

    first_indices: pd.Series = prices_raw_df.apply(lambda col: col.first_valid_index())
    common_start: pd.Timestamp = first_indices.max()
    prices: pd.DataFrame = prices_raw_df.loc[common_start:].dropna(axis=1)
    returns: pd.DataFrame = prices.pct_change().dropna()

    console.print(
        f"[green]Analysis ready for {len(prices.columns)} "
        f"assets from {prices.index[0].date()}.[/green]\n"
    )
    return prices, returns

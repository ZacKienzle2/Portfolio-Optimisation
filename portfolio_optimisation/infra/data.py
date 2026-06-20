"""Market-data acquisition: yfinance download with a request-validated cache.

Exposes the low-level helpers shared by the functional :func:`get_data` entry
point and :class:`portfolio_optimisation.infra.repositories.YfinanceParquetRepository`.

The parquet snapshot is validated against the request before use, so changing
the ticker universe or moving the start date earlier never silently returns a
stale snapshot - the previous implementation keyed the cache only on a fixed
path and would serve whatever was last written regardless of the request.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yfinance as yf
from rich.console import Console


def default_cache_path() -> Path:
    """Return the conventional on-disk snapshot location under the working dir."""
    return Path.cwd() / "Initial_Files" / "market_data.parquet"


def first_trading_day(frame: pd.DataFrame) -> str:
    """ISO date of the first row, robust to pandas index static typing."""
    return str(np.datetime_as_string(frame.index.to_numpy()[0], unit="D"))


def download_adj_close(tickers: list[str], start_date: str) -> pd.DataFrame:
    """Download adjusted close prices for ``tickers`` from ``start_date``.

    Args:
        tickers (list[str]): Asset symbols.
        start_date (str): ISO start date (YYYY-MM-DD).

    Returns:
        pd.DataFrame: Wide frame of adjusted close prices, one column per
        ticker.

    Raises:
        ValueError: If yfinance returns no usable price data for the request.
            The message includes the requested tickers and start date so the
            failure is diagnosable.
    """
    downloaded: Any = yf.download(
        tickers, start=start_date, auto_adjust=False, progress=False
    )
    prices_raw: pd.DataFrame | pd.Series | None
    if isinstance(downloaded, pd.DataFrame) and "Adj Close" in downloaded.columns:
        prices_raw = downloaded["Adj Close"]
    elif isinstance(downloaded, pd.Series):
        prices_raw = downloaded
    else:
        prices_raw = None

    if prices_raw is None or prices_raw.empty:
        raise ValueError(
            f"No price data returned for tickers={sorted(tickers)} "
            f"from {start_date}."
        )

    if isinstance(prices_raw, pd.Series):
        return prices_raw.to_frame(name=tickers[0])
    return prices_raw


def clean_prices(
    prices_raw: pd.DataFrame, tickers: list[str], start_date: str
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Align, trim and difference raw prices into a returns matrix.

    Restricts to the requested ``tickers`` that are present, trims to the latest
    common first-valid date at or after ``start_date``, drops assets that still
    carry gaps, and differences into daily simple returns.

    Args:
        prices_raw (pd.DataFrame): Wide adjusted-close prices.
        tickers (list[str]): Requested symbols; absent ones are ignored.
        start_date (str): ISO start date used as the lower window bound.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: Cleaned prices and daily returns.
    """
    requested = [ticker for ticker in tickers if ticker in prices_raw.columns]
    frame = prices_raw[requested] if requested else prices_raw
    frame = frame.loc[start_date:]

    first_indices: pd.Series = frame.apply(lambda col: col.first_valid_index())
    common_start: pd.Timestamp = first_indices.max()
    prices: pd.DataFrame = frame.loc[common_start:].dropna(axis=1)
    returns: pd.DataFrame = prices.pct_change(fill_method=None).dropna()
    return prices, returns


def cache_satisfies_request(
    cached: pd.DataFrame, tickers: list[str], start_date: str
) -> bool:
    """Return True when ``cached`` covers every requested ticker and the window.

    The cache is usable only if it contains all requested tickers and its first
    observation is at or before the requested ``start_date``; otherwise the
    request needs fresh data and the snapshot must be refetched.
    """
    if cached.empty or not set(tickers).issubset(cached.columns):
        return False
    cached_start = cached.index.to_numpy()
    return bool(cached_start.min() <= np.datetime64(start_date))


def get_data(
    tickers: list[str],
    start_date: str,
    console: Console,
    *,
    cache_path: Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fetch and preprocess financial time series data with a validated cache.

    Loads the parquet snapshot when it covers the request; otherwise downloads
    fresh data via yfinance, writes the snapshot through, then aligns to a
    common start date, drops assets with incomplete data, and computes daily
    returns.

    Args:
        tickers (list[str]): Asset symbols.
        start_date (str): Start date (YYYY-MM-DD).
        console (Console): Rich console for status messages.
        cache_path (Path | None): Snapshot location. Defaults to
            ``Initial_Files/market_data.parquet`` under the working directory.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: Cleaned prices and daily returns.

    Raises:
        ValueError: If the download fails or yields empty data.
    """
    path = cache_path if cache_path is not None else default_cache_path()
    path.parent.mkdir(parents=True, exist_ok=True)

    if path.exists():
        cached = pd.read_parquet(path, engine="pyarrow")
        if cache_satisfies_request(cached, tickers, start_date):
            console.print(f"[green]Loading cached data from {path}...[/green]")
            prices, returns = clean_prices(cached, tickers, start_date)
            console.print(
                f"[green]Analysis ready for {len(prices.columns)} "
                f"assets from {first_trading_day(prices)}.[/green]\n"
            )
            return prices, returns
        console.print(
            "[yellow]Cached snapshot does not cover the request; refetching...[/yellow]"
        )

    console.print(
        f"[yellow]Fetching data for {len(tickers)} assets from {start_date}...[/yellow]"
    )
    prices_raw = download_adj_close(tickers, start_date)
    prices_raw.to_parquet(path, engine="pyarrow")
    console.print(f"[green]Saved new data cache to {path}.[/green]")

    prices, returns = clean_prices(prices_raw, tickers, start_date)
    console.print(
        f"[green]Analysis ready for {len(prices.columns)} "
        f"assets from {pd.Timestamp(prices.index[0]).date()}.[/green]\n"
    )
    return prices, returns

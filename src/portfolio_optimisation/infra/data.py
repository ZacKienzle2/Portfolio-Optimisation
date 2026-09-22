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
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
import yfinance as yf

from portfolio_optimisation.infra.errors import MarketDataError
from portfolio_optimisation.infra.logging import get_logger

if TYPE_CHECKING:
    from rich.console import Console

logger = get_logger(__name__)


def default_cache_path() -> Path:
    """Return the conventional on-disk snapshot location under the working dir."""
    return Path.cwd() / "data" / "market_data.parquet"


def first_trading_day(frame: pd.DataFrame) -> str:
    """ISO date of the first row, robust to pandas index static typing."""
    return str(np.datetime_as_string(frame.index.to_numpy()[0], unit="D"))


def download_adj_close(tickers: list[str], start_date: str, *, attempts: int = 3) -> pd.DataFrame:
    """Download adjusted close prices for ``tickers`` from ``start_date``.

    Transient network errors are retried with exponential backoff by yfinance
    itself, through its ``retries`` setting. An empty response is a permanent
    failure and is not retried.

    Args:
        tickers: Asset symbols.
        start_date: ISO start date (YYYY-MM-DD).
        attempts: Maximum download attempts before giving up.

    Returns:
        pd.DataFrame: Wide frame of adjusted close prices, one column per
        ticker.

    Raises:
        MarketDataError: If yfinance returns no usable price data for the request.
            Carries the requested tickers and start date so the failure is diagnosable.
            Subclasses ValueError for compatibility.
    """
    yf.set_config(retries=attempts - 1)
    downloaded: Any = yf.download(tickers, start=start_date, auto_adjust=False, progress=False)
    prices_raw: pd.DataFrame | pd.Series | None
    if isinstance(downloaded, pd.DataFrame) and "Adj Close" in downloaded.columns:
        prices_raw = downloaded["Adj Close"]
    elif isinstance(downloaded, pd.Series):
        prices_raw = downloaded
    else:
        prices_raw = None

    if prices_raw is None or prices_raw.empty:
        raise MarketDataError(tickers, start_date)

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
        prices_raw: Wide adjusted-close prices.
        tickers: Requested symbols; absent ones are ignored.
        start_date: ISO start date used as the lower window bound.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: Cleaned prices and daily returns.
    """
    requested = [ticker for ticker in tickers if ticker in prices_raw.columns]
    frame = prices_raw[requested] if requested else prices_raw
    frame = frame.loc[start_date:]

    prices: pd.DataFrame = frame.loc[frame.notna().idxmax().max() :].dropna(axis=1)
    returns: pd.DataFrame = prices.pct_change(fill_method=None).dropna()
    return prices, returns


def cache_satisfies_request(cached: pd.DataFrame, tickers: list[str], start_date: str) -> bool:
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
    """Fetch and preprocess prices through the validated parquet cache.

    A functional front end to
    :class:`portfolio_optimisation.infra.repositories.YfinanceParquetRepository`,
    which owns the cache policy.

    Args:
        tickers: Asset symbols.
        start_date: Start date (YYYY-MM-DD).
        console: Rich console for status messages.
        cache_path: Snapshot location. Defaults to ``data/market_data.parquet`` under
            the working directory.

    Returns:
        Cleaned prices and daily returns.
    """
    from portfolio_optimisation.infra.repositories import YfinanceParquetRepository

    return YfinanceParquetRepository(cache_path=cache_path, console=console).load_prices(
        tickers, start_date
    )

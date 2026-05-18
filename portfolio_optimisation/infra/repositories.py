"""Concrete repository implementations + an in-memory UnitOfWork.

The yfinance + parquet implementation mirrors the legacy ``get_data``
function but exposes it behind :class:`MarketDataRepository`. Tests can
swap in :class:`FakeMarketDataRepository` for hermetic runs.
"""

from __future__ import annotations

from pathlib import Path
from types import TracebackType
from typing import Any

import pandas as pd
import yfinance as yf
from rich.console import Console

from portfolio_optimisation.domain.repositories import (
    MarketDataRepository,
    UnitOfWork,
)


class YfinanceParquetRepository(MarketDataRepository):
    """yfinance fetch with a parquet snapshot cache.

    Attributes:
        cache_path (Path): On-disk parquet snapshot. Read when present,
            written through after a successful network fetch.
        console (Console): Rich console used for status output.
    """

    def __init__(
        self,
        cache_path: Path | None = None,
        console: Console | None = None,
    ) -> None:
        self.cache_path: Path = (
            cache_path if cache_path is not None
            else Path.cwd() / "Initial_Files" / "market_data.parquet"
        )
        self.console: Console = console or Console()
        self.cache_path.parent.mkdir(exist_ok=True)

    def load_prices(
        self, tickers: list[str], start_date: str
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        prices_raw: pd.DataFrame | pd.Series | None
        if self.cache_path.exists():
            self.console.print(
                f"[green]Loading cached data from {self.cache_path}...[/green]"
            )
            prices_raw = pd.read_parquet(self.cache_path, engine="pyarrow")
        else:
            self.console.print(
                f"[yellow]Fetching data for {len(tickers)} assets "
                f"from {start_date}...[/yellow]"
            )
            downloaded: Any = yf.download(
                tickers, start=start_date, auto_adjust=False, progress=False
            )
            if isinstance(downloaded, pd.DataFrame) and "Adj Close" in downloaded.columns:
                prices_raw = downloaded["Adj Close"]
            elif isinstance(downloaded, pd.Series):
                prices_raw = downloaded
            else:
                prices_raw = None
            if prices_raw is None or prices_raw.empty:
                raise ValueError("Failed to download valid price data.")
            prices_raw.to_parquet(self.cache_path, engine="pyarrow")
            self.console.print(
                f"[green]Saved new data cache to {self.cache_path}.[/green]"
            )

        if prices_raw is None:
            raise ValueError("Price data is None after download/load.")

        prices_raw_df: pd.DataFrame = (
            prices_raw.to_frame(name=tickers[0])
            if isinstance(prices_raw, pd.Series)
            else prices_raw
        )

        first_indices: pd.Series = prices_raw_df.apply(
            lambda col: col.first_valid_index()
        )
        common_start: pd.Timestamp = first_indices.max()
        prices: pd.DataFrame = prices_raw_df.loc[common_start:].dropna(axis=1)
        returns: pd.DataFrame = prices.pct_change().dropna()

        self.console.print(
            f"[green]Analysis ready for {len(prices.columns)} "
            f"assets from {prices.index[0].date()}.[/green]\n"
        )
        return prices, returns


class FakeMarketDataRepository(MarketDataRepository):
    """In-memory repository for tests. No filesystem, no network."""

    def __init__(self, prices: pd.DataFrame) -> None:
        self._prices: pd.DataFrame = prices

    def load_prices(
        self, tickers: list[str], start_date: str  # noqa: ARG002
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        prices = self._prices[tickers]
        returns = prices.pct_change().dropna()
        return prices, returns


class InMemoryUnitOfWork(UnitOfWork):
    """Trivial UoW that carries a single MarketDataRepository.

    There is no persistent state to commit yet, but the structure is in
    place so future write-side aggregates (positions, allocations) get a
    natural place to live without breaking call sites.
    """

    def __init__(self, market_data: MarketDataRepository) -> None:
        self.market_data: MarketDataRepository = market_data
        self._committed: bool = False

    def __enter__(self) -> InMemoryUnitOfWork:
        self._committed = False
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        if exc_type is not None or not self._committed:
            self.rollback()

    def commit(self) -> None:
        self._committed = True

    def rollback(self) -> None:
        self._committed = False

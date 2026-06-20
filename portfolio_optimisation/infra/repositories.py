"""Concrete repository implementations + an in-memory UnitOfWork.

The yfinance + parquet implementation mirrors the legacy ``get_data``
function but exposes it behind :class:`MarketDataRepository`. Tests can
swap in :class:`FakeMarketDataRepository` for hermetic runs.
"""

from __future__ import annotations

from pathlib import Path
from types import TracebackType

import pandas as pd
from rich.console import Console

from portfolio_optimisation.domain.repositories import (
    MarketDataRepository,
    UnitOfWork,
)
from portfolio_optimisation.infra.data import (
    cache_satisfies_request,
    clean_prices,
    default_cache_path,
    download_adj_close,
    first_trading_day,
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
            cache_path if cache_path is not None else default_cache_path()
        )
        self.console: Console = console or Console()
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)

    def load_prices(
        self, tickers: list[str], start_date: str
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        if self.cache_path.exists():
            cached = pd.read_parquet(self.cache_path, engine="pyarrow")
            if cache_satisfies_request(cached, tickers, start_date):
                self.console.print(
                    f"[green]Loading cached data from {self.cache_path}...[/green]"
                )
                prices, returns = clean_prices(cached, tickers, start_date)
                self._announce(prices)
                return prices, returns
            self.console.print(
                "[yellow]Cached snapshot does not cover the request; "
                "refetching...[/yellow]"
            )

        self.console.print(
            f"[yellow]Fetching data for {len(tickers)} assets "
            f"from {start_date}...[/yellow]"
        )
        prices_raw = download_adj_close(tickers, start_date)
        prices_raw.to_parquet(self.cache_path, engine="pyarrow")
        self.console.print(
            f"[green]Saved new data cache to {self.cache_path}.[/green]"
        )
        prices, returns = clean_prices(prices_raw, tickers, start_date)
        self._announce(prices)
        return prices, returns

    def _announce(self, prices: pd.DataFrame) -> None:
        """Report the cleaned universe size and start date to the console."""
        self.console.print(
            f"[green]Analysis ready for {len(prices.columns)} "
            f"assets from {first_trading_day(prices)}.[/green]\n"
        )


class FakeMarketDataRepository(MarketDataRepository):
    """In-memory repository for tests. No filesystem, no network."""

    def __init__(self, prices: pd.DataFrame) -> None:
        self._prices: pd.DataFrame = prices

    def load_prices(
        self, tickers: list[str], start_date: str  # noqa: ARG002
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        prices = self._prices[tickers]
        returns = prices.pct_change(fill_method=None).dropna()
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

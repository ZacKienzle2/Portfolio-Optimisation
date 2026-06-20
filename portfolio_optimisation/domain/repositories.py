"""Repository and Unit-of-Work protocols.

These Protocols decouple analytical code from any specific data source
(yfinance, parquet cache, future SQL backend, mocked test double).
Implementations live under :mod:`portfolio_optimisation.infra.repositories`.
"""

from __future__ import annotations

from types import TracebackType
from typing import Protocol, runtime_checkable

import pandas as pd


@runtime_checkable
class MarketDataRepository(Protocol):
    """Read interface for price + return time series.

    Implementations may pull from an HTTP feed, a local parquet snapshot,
    a SQL backend, or a unit-test fake. Consumers see only this Protocol.
    """

    def load_prices(self, tickers: list[str], start_date: str) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Return (prices, returns) aligned to a common start date.

        Args:
            tickers (list[str]): Asset symbols.
            start_date (str): ISO-format start date (``YYYY-MM-DD``).

        Returns:
            tuple[pd.DataFrame, pd.DataFrame]: Cleaned price frame and
            daily-return frame. Assets with incomplete history dropped.
        """
        ...


@runtime_checkable
class UnitOfWork(Protocol):
    """Atomic-operation envelope.

    A ``UnitOfWork`` exposes the repositories used inside a single logical
    transaction. On context exit the implementation either commits (when
    :meth:`commit` was explicitly called) or rolls back (the default).
    Designed to follow the pattern in Architecture Patterns project conventions.
    """

    market_data: MarketDataRepository

    def __enter__(self) -> UnitOfWork: ...

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None: ...

    def commit(self) -> None: ...

    def rollback(self) -> None: ...

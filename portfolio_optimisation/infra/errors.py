"""Package-specific exceptions.

Typed errors carry the request context (which tickers, which window) so a
failure deep in the data layer is diagnosable at the call site instead of
surfacing as a bare ``ValueError`` with no provenance.
"""

from __future__ import annotations

from collections.abc import Iterable


class PortfolioError(Exception):
    """Base class for all package-specific errors."""


class MarketDataError(PortfolioError, ValueError):
    """Raised when market data cannot be acquired for a request.

    Subclasses ``ValueError`` for backward compatibility with callers that
    catch the broader type.

    Attributes:
        tickers (list[str]): The requested symbols.
        start_date (str): The requested ISO start date.
    """

    def __init__(
        self, tickers: Iterable[str], start_date: str, message: str | None = None
    ) -> None:
        self.tickers: list[str] = list(tickers)
        self.start_date: str = start_date
        detail = message or "no usable price data returned"
        super().__init__(
            f"Market data unavailable for tickers={sorted(self.tickers)} "
            f"from {start_date}: {detail}"
        )

"""Pure domain types and Protocol abstractions.

Anything in this package is framework-agnostic and contains no IO. Concrete
infrastructure (yfinance fetch, parquet IO) lives under
:mod:`portfolio_optimisation.infra` and is wired together by the service
layer at :mod:`portfolio_optimisation.services`.
"""

from portfolio_optimisation.domain.repositories import (
    MarketDataRepository,
    UnitOfWork,
)

__all__ = ["MarketDataRepository", "UnitOfWork"]

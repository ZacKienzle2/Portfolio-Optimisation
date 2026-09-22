"""Pure domain types and Protocol abstractions.

Anything in this package is framework-agnostic and contains no IO. Concrete
infrastructure (yfinance fetch, parquet IO) lives under
:mod:`portfolio_optimisation.infra` and is wired together by the service
layer at :mod:`portfolio_optimisation.services`.
"""

import lazy_loader

__getattr__, __dir__, __all__ = lazy_loader.attach_stub(__name__, __file__)

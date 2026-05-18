"""Deprecation shim. Import from ``portfolio_optimisation.viz`` instead."""

from __future__ import annotations

import warnings

from portfolio_optimisation.viz import PortfolioVisualiser

warnings.warn(
    "`plotting` is a transitional shim. Import from `portfolio_optimisation.viz` instead.",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = ["PortfolioVisualiser"]

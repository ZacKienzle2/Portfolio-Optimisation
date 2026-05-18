"""Deprecation shim. Import from ``markets.viz`` instead."""

from __future__ import annotations

import warnings

from markets.viz import PortfolioVisualiser

warnings.warn(
    "`plotting` is a transitional shim. Import from `markets.viz` instead.",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = ["PortfolioVisualiser"]

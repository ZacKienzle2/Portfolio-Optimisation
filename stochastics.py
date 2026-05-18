"""Deprecation shim. Import from ``portfolio_optimisation.sde`` instead."""

from __future__ import annotations

import warnings

from portfolio_optimisation.sde import SDEFitter

warnings.warn(
    "`stochastics` is a transitional shim. Import from `portfolio_optimisation.sde` instead.",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = ["SDEFitter"]

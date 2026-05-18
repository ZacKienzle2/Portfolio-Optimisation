"""Deprecation shim. Import from ``markets.sde`` instead."""

from __future__ import annotations

import warnings

from markets.sde import SDEFitter

warnings.warn(
    "`stochastics` is a transitional shim. Import from `markets.sde` instead.",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = ["SDEFitter"]

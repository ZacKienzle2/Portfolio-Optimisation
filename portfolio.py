"""Deprecation shim. Import from ``markets.optim`` instead."""

from __future__ import annotations

import warnings

from markets.optim import HRPAnalyser, HRPModel

warnings.warn(
    "`portfolio` is a transitional shim. Import from `markets.optim` instead.",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = ["HRPAnalyser", "HRPModel"]

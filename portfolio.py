"""Deprecation shim. Import from ``portfolio_optimisation.optim`` instead."""

from __future__ import annotations

import warnings

from portfolio_optimisation.optim import HRPAnalyser, HRPModel

warnings.warn(
    "`portfolio` is a transitional shim. Import from `portfolio_optimisation.optim` instead.",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = ["HRPAnalyser", "HRPModel"]

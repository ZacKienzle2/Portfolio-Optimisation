"""Deprecation shim. Import from ``portfolio_optimisation.econometrics`` instead."""

from __future__ import annotations

import warnings

from portfolio_optimisation.econometrics import Econometrics

warnings.warn(
    "`econometrics` is a transitional shim. Import from `portfolio_optimisation.econometrics` instead.",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = ["Econometrics"]

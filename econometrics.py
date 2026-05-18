"""Deprecation shim. Import from ``markets.econometrics`` instead."""

from __future__ import annotations

import warnings

from markets.econometrics import Econometrics

warnings.warn(
    "`econometrics` is a transitional shim. Import from `markets.econometrics` instead.",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = ["Econometrics"]

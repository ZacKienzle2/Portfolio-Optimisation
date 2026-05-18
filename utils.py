"""Deprecation shim. Import from ``markets.infra`` instead."""

from __future__ import annotations

import warnings

from markets.infra import (
    generateFinalReport,
    getData,
    get_discrete_portfolio,
    inverseVarianceWeights,
)

warnings.warn(
    "`utils` is a transitional shim. Import from `markets.infra` instead.",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = [
    "generateFinalReport",
    "getData",
    "get_discrete_portfolio",
    "inverseVarianceWeights",
]

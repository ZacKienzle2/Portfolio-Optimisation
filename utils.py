"""Deprecation shim. Import from ``portfolio_optimisation.infra`` instead."""

from __future__ import annotations

import warnings

from portfolio_optimisation.infra import (
    generateFinalReport,
    getData,
    get_discrete_portfolio,
    inverseVarianceWeights,
)

warnings.warn(
    "`utils` is a transitional shim. Import from `portfolio_optimisation.infra` instead.",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = [
    "generateFinalReport",
    "getData",
    "get_discrete_portfolio",
    "inverseVarianceWeights",
]

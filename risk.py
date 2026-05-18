"""Deprecation shim. Import from ``portfolio_optimisation.risk`` instead."""

from __future__ import annotations

import warnings

from portfolio_optimisation.risk import (
    CopulaRiskAnalyser,
    calculatePerformanceMetrics,
    calculateRiskMetrics,
    plotSimulationResults,
    run_historical_simulation,
)

warnings.warn(
    "`risk` is a transitional shim. Import from `portfolio_optimisation.risk` instead.",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = [
    "CopulaRiskAnalyser",
    "calculatePerformanceMetrics",
    "calculateRiskMetrics",
    "plotSimulationResults",
    "run_historical_simulation",
]

"""Quant infrastructure for portfolio construction, risk and time series analysis.

Subpackages:
    domain        Pure types and value objects.
    risk          VaR, CVaR, copula simulation, performance metrics.
    optim         Portfolio construction (HRP, bootstrap robustness).
    econometrics  Stationarity, normality, autocorrelation, ARCH, structural tests.
    sde           Maximum-likelihood SDE parameter fitting (GBM, OU).
    viz           Plotting helpers for portfolios and clustering.
    infra         Data loaders, reporting, weight utilities.
"""

from importlib.metadata import PackageNotFoundError, version

from portfolio_optimisation.config import Settings, load_settings

try:
    __version__ = version("portfolio-optimisation")
except PackageNotFoundError:  # pragma: no cover - source tree without install
    __version__ = "0.0.0"

__all__ = ["Settings", "__version__", "load_settings"]

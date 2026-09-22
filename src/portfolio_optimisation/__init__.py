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

from portfolio_optimisation.config import Settings, load_settings

__all__ = ["Settings", "__version__", "load_settings"]

__version__: str


def __getattr__(name: str) -> str:
    """Resolve ``__version__`` from the installed metadata on first access.

    Reading package metadata imports ``importlib.metadata`` and scans the
    environment, which cost about 100 ms of every command-line start, so it
    happens only when the version is asked for.

    Args:
        name: Attribute being looked up.

    Returns:
        The installed distribution version.

    Raises:
        AttributeError: For any other attribute.
    """
    if name != "__version__":
        msg = f"module {__name__!r} has no attribute {name!r}"
        raise AttributeError(msg)
    from importlib.metadata import version

    return version("portfolio-optimisation")

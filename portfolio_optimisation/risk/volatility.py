"""Conditional-volatility models and GARCH-based VaR/ES forecasts.

Fits GARCH-family models to a return series and turns the filtered conditional
volatility into one-step Value-at-Risk and Expected Shortfall forecasts. The
forecasts are returned in return space (negative for a long position) so they
feed directly into the coverage tests in
:mod:`portfolio_optimisation.risk.backtesting`.

Returns are rescaled by 100 before fitting (the conventional percent scale)
to keep the optimiser well conditioned, then the conditional volatility and
mean are scaled back to the original units.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd
from arch import arch_model
from numpy.typing import NDArray
from scipy.stats import norm
from scipy.stats import t as student_t

VolModel = Literal["GARCH", "EGARCH", "GJR"]
Distribution = Literal["normal", "t"]

_SCALE = 100.0


def _index_of(returns: pd.Series | NDArray[np.float64]) -> pd.Index | None:
    return returns.index if isinstance(returns, pd.Series) else None


def fit_garch(
    returns: pd.Series | NDArray[np.float64],
    *,
    vol: VolModel = "GARCH",
    p: int = 1,
    q: int = 1,
    o: int = 0,
    dist: Distribution = "t",
):
    """Fit a GARCH-family model to a return series.

    Args:
        returns (pd.Series | NDArray[float64]): Return series.
        vol (VolModel): Volatility process: standard ``"GARCH"``, ``"EGARCH"``,
            or ``"GJR"`` (GJR-GARCH with a leverage term).
        p (int): Number of lagged volatility (ARCH) terms.
        q (int): Number of lagged variance (GARCH) terms.
        o (int): Number of asymmetry terms (forced to at least 1 for ``"GJR"``).
        dist (Distribution): Innovation distribution, ``"normal"`` or ``"t"``.

    Returns:
        The fitted ``arch`` results object.
    """
    scaled = np.asarray(returns, dtype=np.float64).ravel() * _SCALE
    if vol == "EGARCH":
        model = arch_model(
            scaled, mean="Constant", vol="EGARCH", p=p, o=o, q=q, dist=dist, rescale=False
        )
    else:
        asymmetry = max(o, 1) if vol == "GJR" else o
        model = arch_model(
            scaled, mean="Constant", vol="GARCH", p=p, o=asymmetry, q=q, dist=dist, rescale=False
        )
    return model.fit(disp="off")


def conditional_volatility(
    returns: pd.Series | NDArray[np.float64],
    *,
    vol: VolModel = "GARCH",
    p: int = 1,
    q: int = 1,
    o: int = 0,
    dist: Distribution = "t",
) -> pd.Series:
    """Return the filtered conditional volatility in the original return units.

    Args:
        returns (pd.Series | NDArray[float64]): Return series.
        vol, p, q, o, dist: Passed through to :func:`fit_garch`.

    Returns:
        pd.Series: Per-period conditional volatility, indexed like ``returns``.
    """
    result = fit_garch(returns, vol=vol, p=p, q=q, o=o, dist=dist)
    sigma = np.asarray(result.conditional_volatility, dtype=np.float64) / _SCALE
    return pd.Series(sigma, index=_index_of(returns))


def _standardised_quantile_and_es(
    alpha: float, dist: Distribution, result: object
) -> tuple[float, float]:
    if dist == "normal":
        z = float(norm.ppf(alpha))
        return z, -float(norm.pdf(z)) / alpha
    nu = float(result.params["nu"])  # type: ignore[attr-defined]
    unit_scale = np.sqrt((nu - 2.0) / nu)
    raw_quantile = float(student_t.ppf(alpha, nu))
    quantile = raw_quantile * unit_scale
    raw_es = -(student_t.pdf(raw_quantile, nu) * (nu + raw_quantile**2) / (nu - 1.0)) / alpha
    return quantile, raw_es * unit_scale


def garch_var_es(
    returns: pd.Series | NDArray[np.float64],
    *,
    alpha: float = 0.05,
    vol: VolModel = "GARCH",
    p: int = 1,
    q: int = 1,
    o: int = 0,
    dist: Distribution = "t",
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """One-step conditional VaR and Expected Shortfall from a GARCH fit.

    The forecasts combine the fitted constant mean with the filtered
    conditional volatility scaled by the standardised quantile and tail mean of
    the innovation distribution, both in return space (negative for losses).

    Args:
        returns (pd.Series | NDArray[float64]): Return series.
        alpha (float): Tail level in (0, 1).
        vol, p, q, o, dist: Passed through to :func:`fit_garch`.

    Returns:
        tuple[NDArray[float64], NDArray[float64]]: VaR and ES arrays aligned to
        ``returns``, suitable for the coverage backtests.
    """
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must lie in (0, 1).")
    result = fit_garch(returns, vol=vol, p=p, q=q, o=o, dist=dist)
    sigma = np.asarray(result.conditional_volatility, dtype=np.float64) / _SCALE
    mu = float(result.params.get("mu", 0.0)) / _SCALE
    quantile, es_factor = _standardised_quantile_and_es(alpha, dist, result)
    var = mu + sigma * quantile
    es = mu + sigma * es_factor
    return var, es

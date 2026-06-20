"""Coherent risk measures beyond VaR and CVaR.

Implements:

* :func:`entropic_value_at_risk` (EVaR). Coherent and computationally
  tractable: ``EVaR_alpha(L) = inf_{z>0} z^-1 log(M_L(z)/alpha)`` with ``M_L``
  the moment-generating function of the loss random variable. We minimise over
  ``z > 0`` against the empirical MGF.

* :func:`spectral_risk_measure`. Returns ``M_phi(L) = int_0^1 phi(p) F_L^-1(p) dp``
  for an admissible weight function ``phi: [0,1] -> R_+`` that is
  non-decreasing and integrates to 1.

* :func:`wang_transform_risk`. Distortion risk ``int_0^1 F_L^-1(u) dg(u)``
  with ``g(u) = Phi(Phi^-1(u) + lambda)``.

All three accept a 1D loss array (positive = loss) or a returns Series (in
which case losses are ``-returns``). They are coherent: monotone, sub-additive,
positively-homogeneous, and translation-invariant.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Literal

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy.optimize import minimize_scalar
from scipy.stats import norm


def _as_losses(
    values: pd.Series | NDArray[np.float64], *, kind: Literal["loss", "return"]
) -> NDArray[np.float64]:
    arr = np.asarray(values, dtype=np.float64).ravel()
    return arr if kind == "loss" else -arr


def entropic_value_at_risk(
    values: pd.Series | NDArray[np.float64],
    *,
    alpha: float = 0.05,
    kind: Literal["loss", "return"] = "return",
    z_bounds: tuple[float, float] = (1e-6, 1e3),
) -> float:
    """Empirical Entropic Value-at-Risk.

    ``EVaR_alpha(L) = inf_{z>0} z^-1 log(M_L(z) / alpha)``.

    Args:
        values: Loss or return sample.
        alpha (float): Tail level in (0, 1).
        kind: ``"loss"`` if values are positive-loss, ``"return"`` otherwise.
        z_bounds: (lo, hi) bounds for the scalar minimisation over ``z > 0``.

    Returns:
        float: EVaR in the same units as the input (positive = loss).
    """
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must lie in (0, 1).")
    losses = _as_losses(values, kind=kind)
    if losses.size == 0:
        return float("nan")
    n = float(losses.size)
    log_alpha = np.log(alpha)

    def objective(z: float) -> float:
        # Use log-sum-exp for numerical stability: M(z) = (1/n) sum exp(z L_i).
        zl = z * losses
        m = zl.max()
        log_mgf = m + np.log(np.exp(zl - m).sum() / n)
        return float((log_mgf - log_alpha) / z)

    result = minimize_scalar(objective, bounds=z_bounds, method="bounded")
    return float(result.fun)


def spectral_risk_measure(
    values: pd.Series | NDArray[np.float64],
    phi: Callable[[NDArray[np.float64]], NDArray[np.float64]],
    *,
    kind: Literal["loss", "return"] = "return",
    points: int = 1000,
) -> float:
    """Empirical spectral risk ``int_0^1 phi(p) F_L^-1(p) dp``.

    Discretises the unit interval with ``points`` nodes and evaluates the loss
    quantile and weight function at each node.
    """
    losses = _as_losses(values, kind=kind)
    grid = np.linspace(1.0 / (2 * points), 1.0 - 1.0 / (2 * points), points)
    weights = np.asarray(phi(grid), dtype=np.float64)
    if (weights < 0).any():
        raise ValueError("phi must be non-negative.")
    weights = weights / weights.sum()
    quantiles = np.quantile(losses, grid)
    return float(weights @ quantiles)


def exponential_spectrum(
    absolute_risk_aversion: float,
) -> Callable[[NDArray[np.float64]], NDArray[np.float64]]:
    """Exponential spectrum ``phi(u) = (1/k) exp(-(1-u)/k)`` for ``k > 0``.

    Concentrates weight on the upper tail of the loss distribution; larger
    ``absolute_risk_aversion`` (k) flattens the spectrum towards the mean.
    """
    if absolute_risk_aversion <= 0.0:
        raise ValueError("absolute_risk_aversion must be positive.")

    k = absolute_risk_aversion

    def phi(u: NDArray[np.float64]) -> NDArray[np.float64]:
        return np.exp(-(1.0 - u) / k) / k

    return phi


def power_spectrum(gamma: float) -> Callable[[NDArray[np.float64]], NDArray[np.float64]]:
    """Power spectrum ``phi(u) = gamma * u^(gamma - 1)`` for ``gamma > 1``."""
    if gamma <= 1.0:
        raise ValueError("gamma must exceed 1 for the spectrum to be admissible.")

    def phi(u: NDArray[np.float64]) -> NDArray[np.float64]:
        return gamma * u ** (gamma - 1.0)

    return phi


def wang_transform_risk(
    values: pd.Series | NDArray[np.float64],
    lam: float,
    *,
    kind: Literal["loss", "return"] = "return",
    points: int = 1000,
) -> float:
    """Wang-transform distortion risk.

    ``rho_lam(L) = int_0^1 F_L^-1(u) dg(u)``, where the distortion
    ``g(u) = Phi(Phi^-1(u) - lambda)`` reweights so a positive ``lambda``
    inflates the upper tail of the loss distribution. ``lambda = 0`` recovers
    the empirical mean loss. Wang 2000, "A class of distortion operators for
    pricing financial and insurance risks", J. of Risk and Insurance.
    """
    losses = _as_losses(values, kind=kind)
    edges = np.linspace(0.0, 1.0, points + 1)
    g = norm.cdf(norm.ppf(np.clip(edges, 1e-9, 1.0 - 1e-9)) - lam)
    dg = np.diff(g)
    mids = 0.5 * (edges[:-1] + edges[1:])
    quantiles = np.quantile(losses, mids)
    return float(quantiles @ dg)

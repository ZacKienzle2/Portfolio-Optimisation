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

from typing import TYPE_CHECKING, Literal

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import logsumexp
from scipy.stats import norm

if TYPE_CHECKING:
    from collections.abc import Callable

    import pandas as pd
    from numpy.typing import NDArray


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
) -> float:
    """Empirical Entropic Value-at-Risk.

    ``EVaR_alpha(L) = inf_{t>0} t log(M_L(1/t) / alpha)``, whose limit as
    ``t -> 0`` is the largest loss. Ahmadi-Javid (2012, Proposition 3.2) bounds
    it between the mean and that largest loss, and Jensen's inequality bounds the
    objective below by ``mean + t log(1/alpha)``, so the infimum lies at
    ``t <= (max - mean) / log(1/alpha)``. The measure is translation invariant
    and positively homogeneous, so the losses are centred on their mean and
    scaled by ``max - mean`` and the search runs over ``(0, 1/log(1/alpha)]``
    whatever the units of the sample.

    Args:
        values: Loss or return sample.
        alpha: Tail level in (0, 1).
        kind: ``"loss"`` if values are positive-loss, ``"return"`` otherwise.

    Returns:
        float: EVaR in the same units as the input (positive = loss).

    Raises:
        ValueError: If ``alpha`` lies outside ``(0, 1)``.
    """
    if not 0.0 < alpha < 1.0:
        msg = "alpha must lie in (0, 1)."
        raise ValueError(msg)
    losses = _as_losses(values, kind=kind)
    if losses.size == 0:
        return float("nan")
    centre = float(losses.mean())
    spread = float(losses.max()) - centre
    if spread <= 0.0:
        return centre
    standardised = (losses - centre) / spread
    log_scale = np.log(alpha * losses.size)

    def objective(t: float) -> float:
        return float(t * (logsumexp(standardised / t) - log_scale))

    result = minimize_scalar(
        objective, bounds=(0.0, -1.0 / np.log(alpha)), method="bounded", options={"xatol": 1e-10}
    )
    return centre + spread * min(float(result.fun), 1.0)


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
        msg = "phi must be non-negative."
        raise ValueError(msg)
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
        msg = "absolute_risk_aversion must be positive."
        raise ValueError(msg)

    k = absolute_risk_aversion

    def phi(u: NDArray[np.float64]) -> NDArray[np.float64]:
        return np.exp(-(1.0 - u) / k) / k

    return phi


def power_spectrum(gamma: float) -> Callable[[NDArray[np.float64]], NDArray[np.float64]]:
    """Power spectrum ``phi(u) = gamma * u^(gamma - 1)`` for ``gamma > 1``."""
    if gamma <= 1.0:
        msg = "gamma must exceed 1 for the spectrum to be admissible."
        raise ValueError(msg)

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

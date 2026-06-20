"""Extreme Value Theory tail risk: peaks-over-threshold and the Hill estimator.

Empirical and Gaussian VaR understate deep-tail risk because they are dominated
by the bulk of the distribution. Extreme Value Theory models only the tail: the
peaks-over-threshold method fits a Generalised Pareto Distribution to the
exceedances above a high threshold, giving VaR and Expected Shortfall estimates
that extrapolate beyond the observed sample.

Conventions follow the rest of the risk package: inputs are returns by default
(losses are ``-returns``), and the VaR/ES estimates are returned in return space
(negative for a long position).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy.stats import genpareto

Kind = Literal["loss", "return"]


def _as_losses(values: pd.Series | NDArray[np.float64], kind: Kind) -> NDArray[np.float64]:
    arr = np.asarray(values, dtype=np.float64).ravel()
    return arr if kind == "loss" else -arr


@dataclass
class GeneralisedParetoFit:
    """Fitted peaks-over-threshold model.

    Attributes:
        shape (float): GPD shape parameter ``xi`` (tail index).
        scale (float): GPD scale parameter ``beta``.
        threshold (float): Loss threshold ``u`` above which the tail is modelled.
        exceedance_rate (float): Fraction of observations above the threshold.
        n_exceedances (int): Number of exceedances used in the fit.
    """

    shape: float
    scale: float
    threshold: float
    exceedance_rate: float
    n_exceedances: int


def fit_peaks_over_threshold(
    values: pd.Series | NDArray[np.float64],
    *,
    threshold_quantile: float = 0.95,
    kind: Kind = "return",
) -> GeneralisedParetoFit:
    """Fit a Generalised Pareto Distribution to threshold exceedances.

    Args:
        values (pd.Series | NDArray[float64]): Returns or losses.
        threshold_quantile (float): Quantile of the loss distribution used as
            the threshold ``u`` (for example 0.95).
        kind (Kind): ``"return"`` if values are returns, else ``"loss"``.

    Returns:
        GeneralisedParetoFit: Fitted shape, scale, threshold and exceedance rate.

    Raises:
        ValueError: If there are too few exceedances to fit (fewer than 10).
    """
    losses = _as_losses(values, kind)
    threshold = float(np.quantile(losses, threshold_quantile))
    exceedances = losses[losses > threshold] - threshold
    if exceedances.size < 10:
        raise ValueError("Too few exceedances above the threshold to fit a GPD.")
    shape, _, scale = genpareto.fit(exceedances, floc=0.0)
    return GeneralisedParetoFit(
        shape=float(shape),
        scale=float(scale),
        threshold=threshold,
        exceedance_rate=exceedances.size / losses.size,
        n_exceedances=int(exceedances.size),
    )


def _pot_var_loss(fit: GeneralisedParetoFit, alpha: float) -> float:
    if abs(fit.shape) < 1e-6:
        return fit.threshold + fit.scale * np.log(fit.exceedance_rate / alpha)
    ratio = (alpha / fit.exceedance_rate) ** (-fit.shape)
    return fit.threshold + (fit.scale / fit.shape) * (ratio - 1.0)


def _pot_es_loss(fit: GeneralisedParetoFit, alpha: float) -> float:
    var_loss = _pot_var_loss(fit, alpha)
    if abs(fit.shape) < 1e-6:
        return var_loss + fit.scale
    if fit.shape >= 1.0:
        return float("inf")
    return (var_loss + fit.scale - fit.shape * fit.threshold) / (1.0 - fit.shape)


def evt_value_at_risk(
    values: pd.Series | NDArray[np.float64],
    *,
    alpha: float = 0.01,
    threshold_quantile: float = 0.95,
    kind: Kind = "return",
) -> float:
    """Peaks-over-threshold Value-at-Risk in return space (negative for losses).

    Args:
        values (pd.Series | NDArray[float64]): Returns or losses.
        alpha (float): Tail probability (for example 0.01).
        threshold_quantile (float): Threshold quantile for the GPD fit.
        kind (Kind): ``"return"`` or ``"loss"``.

    Returns:
        float: The VaR at level ``alpha`` in return space.
    """
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must lie in (0, 1).")
    fit = fit_peaks_over_threshold(values, threshold_quantile=threshold_quantile, kind=kind)
    return -_pot_var_loss(fit, alpha)


def evt_expected_shortfall(
    values: pd.Series | NDArray[np.float64],
    *,
    alpha: float = 0.01,
    threshold_quantile: float = 0.95,
    kind: Kind = "return",
) -> float:
    """Peaks-over-threshold Expected Shortfall in return space.

    Args:
        values (pd.Series | NDArray[float64]): Returns or losses.
        alpha (float): Tail probability (for example 0.01).
        threshold_quantile (float): Threshold quantile for the GPD fit.
        kind (Kind): ``"return"`` or ``"loss"``.

    Returns:
        float: The ES at level ``alpha`` in return space.
    """
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must lie in (0, 1).")
    fit = fit_peaks_over_threshold(values, threshold_quantile=threshold_quantile, kind=kind)
    return -_pot_es_loss(fit, alpha)


def hill_estimator(
    values: pd.Series | NDArray[np.float64],
    *,
    k: int,
    kind: Kind = "return",
) -> float:
    """Hill estimator of the tail index from the ``k`` largest losses.

    Args:
        values (pd.Series | NDArray[float64]): Returns or losses.
        k (int): Number of upper order statistics to use.
        kind (Kind): ``"return"`` or ``"loss"``.

    Returns:
        float: The Hill tail-index estimate (larger means a heavier tail).

    Raises:
        ValueError: If ``k`` is not in ``[1, n_positive_losses - 1]``.
    """
    losses = _as_losses(values, kind)
    positive = losses[losses > 0.0]
    ordered = np.sort(positive)[::-1]
    if not 1 <= k < ordered.size:
        raise ValueError("k must lie in [1, number of positive losses - 1].")
    top = ordered[:k]
    return float(np.mean(np.log(top) - np.log(ordered[k])))

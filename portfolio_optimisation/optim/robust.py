"""Robust and resampled portfolio optimisation.

Plug-in mean-variance optimisation is notoriously fragile: small changes in the
estimated mean or covariance produce large, concentrated swings in the optimal
weights. Two complementary remedies are provided:

* :func:`resampled_weights` - Michaud resampling. Repeatedly draw samples from
  the estimated distribution, re-optimise on each, and average the resulting
  portfolios. The averaging diversifies away estimation noise.
* :func:`robust_mean_variance_weights` - a box-uncertainty robust mean-variance
  portfolio that shrinks each expected return by a multiple of its standard
  error before optimising, producing a more conservative allocation.

Both return long-only weights on the simplex, indexed by ticker.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd
from numpy.typing import NDArray

Objective = Literal["min_variance", "max_sharpe"]


def _solve(cov: NDArray[np.float64], target: NDArray[np.float64]) -> NDArray[np.float64]:
    try:
        raw = np.linalg.solve(cov, target)
    except np.linalg.LinAlgError:
        raw = np.linalg.lstsq(cov, target, rcond=None)[0]
    return raw


def _long_only(weights: NDArray[np.float64]) -> NDArray[np.float64]:
    clipped = np.clip(weights, 0.0, None)
    total = clipped.sum()
    if total <= 0.0:
        return np.full_like(clipped, 1.0 / clipped.size)
    return clipped / total


def _optimal_weights(
    mu: NDArray[np.float64], cov: NDArray[np.float64], objective: Objective
) -> NDArray[np.float64]:
    if objective == "min_variance":
        target = np.ones(cov.shape[0])
    elif objective == "max_sharpe":
        target = mu
    else:
        raise ValueError("objective must be 'min_variance' or 'max_sharpe'.")
    return _long_only(_solve(cov, target))


def resampled_weights(
    returns: pd.DataFrame,
    *,
    objective: Objective = "max_sharpe",
    n_resamples: int = 500,
    seed: int | None = None,
) -> pd.Series:
    """Michaud resampled long-only weights.

    Draws ``n_resamples`` parametric samples from the estimated normal model,
    re-optimises on each, and averages the long-only portfolios to diversify
    away estimation error.

    Args:
        returns (pd.DataFrame): Historical asset returns; columns are tickers.
        objective (Objective): ``"min_variance"`` or ``"max_sharpe"``.
        n_resamples (int): Number of resampling draws.
        seed (int | None): Seed for reproducibility.

    Returns:
        pd.Series: Long-only resampled weights summing to one, indexed by ticker.
    """
    tickers = list(returns.columns)
    mu = returns.mean().to_numpy(dtype=np.float64)
    cov = returns.cov().to_numpy(dtype=np.float64)
    n_obs = returns.shape[0]
    rng = np.random.default_rng(seed)

    accumulated = np.zeros(len(tickers), dtype=np.float64)
    for _ in range(n_resamples):
        sample = rng.multivariate_normal(mu, cov, size=n_obs)
        sample_mu = sample.mean(axis=0)
        sample_cov = np.cov(sample, rowvar=False, ddof=1)
        accumulated += _optimal_weights(sample_mu, sample_cov, objective)

    averaged = accumulated / n_resamples
    return pd.Series(averaged / averaged.sum(), index=tickers)


def robust_mean_variance_weights(
    returns: pd.DataFrame,
    *,
    uncertainty: float = 1.0,
    n_obs: int | None = None,
) -> pd.Series:
    """Box-uncertainty robust mean-variance long-only weights.

    Each expected return is shrunk by ``uncertainty`` standard errors before a
    tangency optimisation, which guards against optimistic mean estimates. With
    ``uncertainty = 0`` this reduces to the plug-in tangency portfolio.

    Args:
        returns (pd.DataFrame): Historical asset returns; columns are tickers.
        uncertainty (float): Number of standard errors to subtract from each
            expected return (the box-uncertainty radius).
        n_obs (int | None): Observation count for the standard error; defaults
            to the sample length.

    Returns:
        pd.Series: Long-only robust weights summing to one, indexed by ticker.
    """
    tickers = list(returns.columns)
    mu = returns.mean().to_numpy(dtype=np.float64)
    cov = returns.cov().to_numpy(dtype=np.float64)
    sample_size = n_obs if n_obs is not None else returns.shape[0]

    standard_error = np.sqrt(np.clip(np.diag(cov), 0.0, None) / sample_size)
    robust_mu = mu - uncertainty * standard_error
    weights = _long_only(_solve(cov, robust_mu))
    return pd.Series(weights, index=tickers)

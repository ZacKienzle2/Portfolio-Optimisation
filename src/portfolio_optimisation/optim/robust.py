"""Robust and resampled portfolio optimisation.

Plug-in mean-variance optimisation is fragile: small changes in the estimated
mean or covariance produce large, concentrated swings in the optimal weights.
Two complementary remedies are provided.

* :func:`resampled_weights` averages the optimal portfolios of many resampled
  estimates, which diversifies away estimation noise.
* :func:`robust_mean_variance_weights` shrinks each expected return by a
  multiple of its standard error before optimising, which gives a more
  conservative allocation.

Both return long-only weights on the simplex, indexed by ticker.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import numpy as np
import pandas as pd
from scipy import linalg
from scipy.stats import wishart

from portfolio_optimisation.optim.constraints import long_only

if TYPE_CHECKING:
    from numpy.typing import NDArray

Objective = Literal["min_variance", "max_sharpe"]


def _solve(covariance: NDArray[np.float64], target: NDArray[np.float64]) -> NDArray[np.float64]:
    """Solve ``covariance @ x = target`` by Cholesky, or least squares if singular."""
    try:
        return linalg.solve(covariance, target, assume_a="pos")
    except linalg.LinAlgError:
        return np.linalg.lstsq(covariance, target, rcond=None)[0]


def minimum_variance_weights(covariance: NDArray[np.float64]) -> NDArray[np.float64]:
    """Long-only projection of the closed-form minimum-variance portfolio.

    Args:
        covariance: Symmetric positive definite covariance matrix.

    Returns:
        Non-negative weights that sum to one.
    """
    return long_only(_solve(covariance, np.ones(covariance.shape[0])))


def resampled_weights(
    returns: pd.DataFrame,
    *,
    objective: Objective = "max_sharpe",
    n_resamples: int = 500,
    seed: int | None = None,
) -> pd.Series:
    """Michaud resampled long-only weights.

    Each resample stands for a Gaussian sample of the original length drawn from
    the estimated model. Under normality the sample mean and the sample
    covariance of such a draw are independent, the mean normal with covariance
    ``Sigma / T`` and ``(T - 1)`` times the covariance Wishart with ``T - 1``
    degrees of freedom, so both are drawn directly rather than by simulating
    ``T`` returns. Every resample is then optimised in one batched solve.

    Args:
        returns: Historical asset returns, one column per ticker.
        objective: ``"min_variance"`` or ``"max_sharpe"``.
        n_resamples: Number of resampled estimates.
        seed: Seed for reproducibility.

    Returns:
        Long-only resampled weights summing to one, indexed by ticker.

    Raises:
        ValueError: If ``objective`` is unknown or the sample has no more observations
            than assets.
    """
    if objective not in {"min_variance", "max_sharpe"}:
        msg = "objective must be 'min_variance' or 'max_sharpe'."
        raise ValueError(msg)
    n_obs, n_assets = returns.shape
    if n_obs <= n_assets:
        msg = "resampling needs more observations than assets."
        raise ValueError(msg)
    mean = returns.mean().to_numpy(dtype=np.float64)
    covariance = returns.cov().to_numpy(dtype=np.float64)
    rng = np.random.default_rng(seed)

    dof = n_obs - 1
    covariances = wishart(df=dof, scale=covariance / dof).rvs(size=n_resamples, random_state=rng)
    covariances = np.reshape(covariances, (n_resamples, n_assets, n_assets))
    if objective == "max_sharpe":
        targets = rng.multivariate_normal(mean, covariance / n_obs, size=n_resamples)
    else:
        targets = np.ones((n_resamples, n_assets))

    raw = np.linalg.solve(covariances, targets[..., None])[..., 0]
    averaged = long_only(raw).mean(axis=0)
    return pd.Series(averaged / averaged.sum(), index=returns.columns)


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
        returns: Historical asset returns, one column per ticker.
        uncertainty: Number of standard errors subtracted from each expected return, the
            radius of the box.
        n_obs: Observation count for the standard error. Defaults to the sample length.

    Returns:
        Long-only robust weights summing to one, indexed by ticker.
    """
    mean = returns.mean().to_numpy(dtype=np.float64)
    covariance = returns.cov().to_numpy(dtype=np.float64)
    sample_size = n_obs if n_obs is not None else returns.shape[0]
    standard_error = np.sqrt(np.clip(np.diag(covariance), 0.0, None) / sample_size)
    weights = long_only(_solve(covariance, mean - uncertainty * standard_error))
    return pd.Series(weights, index=returns.columns)

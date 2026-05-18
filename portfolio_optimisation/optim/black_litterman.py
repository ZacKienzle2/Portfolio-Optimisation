"""Black-Litterman Bayesian asset allocation with user-supplied views.

Black, F., Litterman, R. (1992).
    "Global Portfolio Optimization." Financial Analysts Journal 48(5):28-43.
He, G., Litterman, R. (1999).
    "The Intuition Behind Black-Litterman Model Portfolios." Goldman Sachs.
Idzorek, T. (2004).
    "A step-by-step guide to the Black-Litterman model."

The model treats the equilibrium-implied excess returns ``Pi`` as a prior on
expected returns and updates it with subjective views. Letting ``w_market`` be
the equilibrium weights (here, the HRP allocation by default) and ``delta`` the
investor's risk-aversion coefficient, reverse optimisation gives

    Pi = delta * Sigma * w_market.

Views are encoded by a pick matrix ``P`` (k x N), a view-return vector ``Q``
(k,) and a diagonal uncertainty matrix ``Omega`` (k x k). The Bayesian
posterior mean and covariance are

    mu_BL    = inv((tau Sigma)^-1 + P' Omega^-1 P)
                @ ((tau Sigma)^-1 Pi + P' Omega^-1 Q)
    Sigma_BL = Sigma + inv((tau Sigma)^-1 + P' Omega^-1 P).

Optimal weights are then ``w = (1 / delta) * inv(Sigma_BL) * mu_BL``. When the
caller supplies no views the function reduces to the equilibrium weights.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from numpy.typing import NDArray


@dataclass
class BlackLittermanResult:
    """Container for Black-Litterman posterior diagnostics."""

    weights: pd.Series
    posterior_mean: pd.Series
    posterior_covariance: pd.DataFrame
    implied_equilibrium_returns: pd.Series


def implied_equilibrium_returns(
    covariance: pd.DataFrame, market_weights: pd.Series, *, risk_aversion: float = 2.5
) -> pd.Series:
    """Reverse-optimisation prior: ``Pi = delta * Sigma * w_market``."""
    tickers = list(covariance.columns)
    sigma = covariance.to_numpy(dtype=np.float64)
    w = market_weights.reindex(tickers).fillna(0.0).to_numpy(dtype=np.float64)
    pi = risk_aversion * sigma @ w
    return pd.Series(pi, index=tickers)


def _idzorek_omega(
    tau: float, sigma: NDArray[np.float64], p: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Diagonal Omega aligned with the variance of each view portfolio."""
    return np.diag(np.diag(p @ (tau * sigma) @ p.T))


def black_litterman_weights(
    covariance: pd.DataFrame,
    market_weights: pd.Series,
    *,
    views_matrix: NDArray[np.float64] | None = None,
    views_returns: NDArray[np.float64] | None = None,
    views_uncertainty: NDArray[np.float64] | None = None,
    tau: float = 0.05,
    risk_aversion: float = 2.5,
    long_only: bool = True,
) -> BlackLittermanResult:
    """Run Black-Litterman with an HRP (or any) equilibrium prior.

    Args:
        covariance (pd.DataFrame): Asset covariance matrix.
        market_weights (pd.Series): Equilibrium weights serving as the prior.
            Typical choices include HRP weights, market-cap weights, or an
            ERC allocation.
        views_matrix (NDArray[float64] | None): k x N pick matrix ``P``. If
            None, the function returns the equilibrium weights.
        views_returns (NDArray[float64] | None): View-return vector ``Q``
            (length k). Must accompany ``views_matrix``.
        views_uncertainty (NDArray[float64] | None): k x k diagonal matrix
            ``Omega``. Defaults to Idzorek's ``diag(diag(P tau Sigma P'))``.
        tau (float): Scaling of the prior covariance, typically in [0.01, 0.1].
        risk_aversion (float): Investor risk aversion ``delta``.
        long_only (bool): If True, clip negative weights to zero before
            renormalising to the simplex.

    Returns:
        BlackLittermanResult: Posterior weights, mean, covariance and the
        equilibrium prior used for reference.
    """
    if not tau > 0.0:
        raise ValueError("tau must be positive.")
    if risk_aversion <= 0.0:
        raise ValueError("risk_aversion must be positive.")

    tickers = list(covariance.columns)
    sigma = covariance.to_numpy(dtype=np.float64)
    pi = implied_equilibrium_returns(
        covariance, market_weights, risk_aversion=risk_aversion
    )
    pi_arr = pi.to_numpy(dtype=np.float64)

    if views_matrix is None or views_returns is None:
        weights = market_weights.reindex(tickers).fillna(0.0).to_numpy(dtype=np.float64)
        if long_only:
            weights = np.clip(weights, 0.0, None)
        weights = weights / max(weights.sum(), 1e-12)
        return BlackLittermanResult(
            weights=pd.Series(weights, index=tickers),
            posterior_mean=pi,
            posterior_covariance=covariance.copy(),
            implied_equilibrium_returns=pi,
        )

    p = np.asarray(views_matrix, dtype=np.float64)
    q = np.asarray(views_returns, dtype=np.float64).ravel()
    if p.shape[1] != len(tickers):
        raise ValueError("views_matrix column count must match covariance dimension.")
    if q.shape[0] != p.shape[0]:
        raise ValueError("views_returns length must match views_matrix row count.")

    omega = (
        np.asarray(views_uncertainty, dtype=np.float64)
        if views_uncertainty is not None
        else _idzorek_omega(tau, sigma, p)
    )
    if omega.shape != (p.shape[0], p.shape[0]):
        raise ValueError("views_uncertainty must be a k x k matrix matching P.")

    tau_sigma_inv = np.linalg.inv(tau * sigma)
    omega_inv = np.linalg.inv(omega)
    posterior_precision = tau_sigma_inv + p.T @ omega_inv @ p
    posterior_cov_excess = np.linalg.inv(posterior_precision)
    mu_bl = posterior_cov_excess @ (tau_sigma_inv @ pi_arr + p.T @ omega_inv @ q)
    sigma_bl = sigma + posterior_cov_excess

    weights = (1.0 / risk_aversion) * np.linalg.solve(sigma_bl, mu_bl)
    if long_only:
        weights = np.clip(weights, 0.0, None)
    total = weights.sum()
    if total > 0:
        weights = weights / total

    return BlackLittermanResult(
        weights=pd.Series(weights, index=tickers),
        posterior_mean=pd.Series(mu_bl, index=tickers),
        posterior_covariance=pd.DataFrame(sigma_bl, index=tickers, columns=tickers),
        implied_equilibrium_returns=pi,
    )

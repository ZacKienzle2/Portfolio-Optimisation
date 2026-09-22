"""Black-Litterman Bayesian asset allocation with user-supplied views.

The model treats the equilibrium-implied excess returns ``Pi`` as a prior on
expected returns and updates it with subjective views. Letting ``w_market`` be
the equilibrium weights (here, the HRP allocation by default) and ``delta`` the
investor's risk-aversion coefficient, reverse optimisation gives

    Pi = delta * Sigma * w_market.

Views are encoded by a pick matrix ``P`` (k x N), a view-return vector ``Q``
(k,) and a diagonal uncertainty matrix ``Omega`` (k x k). The Bayesian
posterior mean and covariance, written in the form that inverts only the
k x k matrix ``A = P tau Sigma P' + Omega``, are

    mu_BL    = Pi + tau Sigma P' A^-1 (Q - P Pi)
    Sigma_BL = Sigma + tau Sigma - tau Sigma P' A^-1 P tau Sigma.

By the Woodbury identity this equals the precision-weighted form, costs
``O(N^2 k)`` rather than three ``O(N^3)`` inversions, and stays defined when a
view is held with certainty (a zero row of ``Omega``).

Optimal weights are then ``w = (1 / delta) * inv(Sigma_BL) * mu_BL``. When the
caller supplies no views the function reduces to the equilibrium weights.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from scipy import linalg

from portfolio_optimisation.optim import constraints

if TYPE_CHECKING:
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


def _proportional_omega(
    tau: float, sigma: NDArray[np.float64], p: NDArray[np.float64]
) -> NDArray[np.float64]:
    """He-Litterman Omega, the prior variance of each view portfolio."""
    return np.diag(np.diag(p @ (tau * sigma) @ p.T))


def black_litterman_posterior(
    sigma: NDArray[np.float64],
    pi: NDArray[np.float64],
    p: NDArray[np.float64],
    q: NDArray[np.float64],
    omega: NDArray[np.float64],
    tau: float,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Posterior mean and return covariance through one ``k x k`` solve.

    The mean and the posterior parameter covariance share the factor
    ``tau Sigma P' A^-1`` with ``A = P tau Sigma P' + Omega``, so both come from
    one Cholesky solve of ``A`` against the view residual and ``P tau Sigma``.

    Args:
        sigma: Asset covariance.
        pi: Equilibrium returns.
        p: Pick matrix, one view per row.
        q: View returns.
        omega: View uncertainty.
        tau: Scale of the prior covariance.

    Returns:
        Posterior mean and posterior return covariance.
    """
    tau_sigma = tau * sigma
    shared = tau_sigma @ p.T
    solved = linalg.solve(
        p @ shared + omega, np.column_stack([q - p @ pi, shared.T]), assume_a="pos"
    )
    return pi + shared @ solved[:, 0], sigma + tau_sigma - shared @ solved[:, 1:]


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
        covariance: Asset covariance matrix.
        market_weights: Equilibrium weights serving as the prior. Typical choices
            include HRP weights, market-cap weights, or an ERC allocation.
        views_matrix: k x N pick matrix ``P``. If None, the function returns the
            equilibrium weights.
        views_returns: View-return vector ``Q`` (length k). Must accompany
            ``views_matrix``.
        views_uncertainty: k x k diagonal matrix ``Omega``. Defaults to the He-Litterman
            ``diag(diag(P tau Sigma P'))``.
        tau: Scaling of the prior covariance, typically in [0.01, 0.1].
        risk_aversion: Investor risk aversion ``delta``.
        long_only: If True, clip negative weights to zero before renormalising to the
            simplex.

    Returns:
        BlackLittermanResult: Posterior weights, mean, covariance and the
        equilibrium prior used for reference.

    Raises:
        ValueError: If ``tau`` or ``risk_aversion`` is not positive, or a view input
            does not conform with the covariance.
    """
    if not tau > 0.0:
        msg = "tau must be positive."
        raise ValueError(msg)
    if risk_aversion <= 0.0:
        msg = "risk_aversion must be positive."
        raise ValueError(msg)

    tickers = list(covariance.columns)
    sigma = covariance.to_numpy(dtype=np.float64)
    pi = implied_equilibrium_returns(covariance, market_weights, risk_aversion=risk_aversion)
    pi_arr = pi.to_numpy(dtype=np.float64)

    if views_matrix is None or views_returns is None:
        prior = market_weights.reindex(tickers).fillna(0.0).to_numpy(dtype=np.float64)
        weights = constraints.long_only(prior) if long_only else prior / prior.sum()
        return BlackLittermanResult(
            weights=pd.Series(weights, index=tickers),
            posterior_mean=pi,
            posterior_covariance=covariance.copy(),
            implied_equilibrium_returns=pi,
        )

    p = np.asarray(views_matrix, dtype=np.float64)
    q = np.asarray(views_returns, dtype=np.float64).ravel()
    if p.shape[1] != len(tickers):
        msg = "views_matrix column count must match covariance dimension."
        raise ValueError(msg)
    if q.shape[0] != p.shape[0]:
        msg = "views_returns length must match views_matrix row count."
        raise ValueError(msg)

    omega = (
        np.asarray(views_uncertainty, dtype=np.float64)
        if views_uncertainty is not None
        else _proportional_omega(tau, sigma, p)
    )
    if omega.shape != (p.shape[0], p.shape[0]):
        msg = "views_uncertainty must be a k x k matrix matching P."
        raise ValueError(msg)

    mu_bl, sigma_bl = black_litterman_posterior(sigma, pi_arr, p, q, omega, tau)
    raw = linalg.solve(sigma_bl, mu_bl, assume_a="pos") / risk_aversion
    weights = constraints.long_only(raw) if long_only else raw / raw.sum()

    return BlackLittermanResult(
        weights=pd.Series(weights, index=tickers),
        posterior_mean=pd.Series(mu_bl, index=tickers),
        posterior_covariance=pd.DataFrame(sigma_bl, index=tickers, columns=tickers),
        implied_equilibrium_returns=pi,
    )

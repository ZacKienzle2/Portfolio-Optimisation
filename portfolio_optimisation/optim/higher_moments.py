"""Polynomial Goal Programming over the first four portfolio moments.

Lai, T.-Y. (1991).
    "Portfolio Selection with Skewness: A Multiple-Objective Approach."
    Review of Quantitative Finance and Accounting 1(3):293-305.

The portfolio's first four central moments at weight ``w`` are

    mu(w)    = w' mu_vec,
    sig2(w)  = w' Sigma w,
    skew(w)  = w' M3 (w kron w)   / sig2(w)^(3/2),
    kurt(w)  = w' M4 (w kron w kron w) / sig2(w)^2,

with the co-skewness tensor ``M3`` (N x N^2) and co-kurtosis tensor ``M4``
(N x N^3) computed empirically from returns. Investors typically prefer
high mean and skewness, low variance and kurtosis. PGP balances these
goals by first solving four single-objective sub-problems for the individual
optima ``mu*, sig2*, skew*, kurt*`` and then minimising

    G(w) = ((1 - mu(w)/mu*)^alpha
          + (sig2(w)/sig2* - 1)^beta
          + (1 - skew(w)/skew*)^gamma
          + (kurt(w)/kurt* - 1)^delta)

subject to the long-only simplex. Exponents alpha, beta, gamma, delta
encode investor preferences and default to 1.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy.optimize import minimize


@dataclass
class HigherMomentResult:
    """Container with PGP weights plus achieved single-objective optima."""

    weights: pd.Series
    achieved_mean: float
    achieved_variance: float
    achieved_skewness: float
    achieved_kurtosis: float
    mean_star: float
    variance_star: float
    skewness_star: float
    kurtosis_star: float


def coskewness_tensor(returns: pd.DataFrame) -> NDArray[np.float64]:
    """Empirical co-skewness M3 of shape (N, N*N).

    ``M3[i, j*N + k] = (1/T) sum_t (r_t,i - mu_i)(r_t,j - mu_j)(r_t,k - mu_k)``.
    """
    centred = (returns - returns.mean()).to_numpy(dtype=np.float64)
    t, n = centred.shape
    return (centred.T @ (centred[:, :, None] * centred[:, None, :]).reshape(t, n * n)) / t


def cokurtosis_tensor(returns: pd.DataFrame) -> NDArray[np.float64]:
    """Empirical co-kurtosis M4 of shape (N, N^3)."""
    centred = (returns - returns.mean()).to_numpy(dtype=np.float64)
    t, n = centred.shape
    outer = (centred[:, :, None, None] * centred[:, None, :, None] * centred[:, None, None, :])
    return (centred.T @ outer.reshape(t, n * n * n)) / t


def _portfolio_moments(
    w: NDArray[np.float64],
    mu: NDArray[np.float64],
    sigma: NDArray[np.float64],
    m3: NDArray[np.float64],
    m4: NDArray[np.float64],
) -> tuple[float, float, float, float]:
    """Return (mean, variance, skewness, kurtosis) for weights w."""
    mean = float(w @ mu)
    var = float(w @ sigma @ w)
    w_kron_w = np.kron(w, w)
    third = float(w @ m3 @ w_kron_w)
    fourth = float(w @ m4 @ np.kron(w, w_kron_w))
    if var <= 0:
        return mean, var, 0.0, 0.0
    skew = third / var**1.5
    kurt = fourth / var**2
    return mean, var, skew, kurt


def _solve_single_objective(
    objective: Callable[[NDArray[np.float64]], float],
    n_assets: int,
    minimise: bool,
) -> tuple[NDArray[np.float64], float]:
    """Long-only simplex sub-problem solver."""
    x0 = np.full(n_assets, 1.0 / n_assets)
    bounds = [(0.0, 1.0)] * n_assets
    constraints = ({"type": "eq", "fun": lambda w: np.sum(w) - 1.0},)
    sign = 1.0 if minimise else -1.0

    def wrapped(w: NDArray[np.float64]) -> float:
        return sign * objective(w)

    result = minimize(
        wrapped,
        x0,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"ftol": 1e-9, "maxiter": 250},
    )
    return result.x, sign * float(result.fun)


def pgp_higher_moment_weights(
    returns: pd.DataFrame,
    *,
    alpha: float = 1.0,
    beta: float = 1.0,
    gamma: float = 1.0,
    delta: float = 1.0,
) -> HigherMomentResult:
    """Solve the Lai (1991) PGP four-moment portfolio.

    Args:
        returns (pd.DataFrame): Asset returns.
        alpha, beta, gamma, delta (float): Investor preference exponents on
            mean shortfall, variance excess, negative-skew penalty and
            kurtosis excess respectively.

    Returns:
        HigherMomentResult: Posterior weights plus the realised and
        individually-optimal moments used as goal references.
    """
    tickers = list(returns.columns)
    mu = returns.mean().to_numpy(dtype=np.float64)
    sigma = returns.cov().to_numpy(dtype=np.float64)
    m3 = coskewness_tensor(returns)
    m4 = cokurtosis_tensor(returns)
    n_assets = len(tickers)

    def mean_obj(w: NDArray[np.float64]) -> float:
        return float(w @ mu)

    def var_obj(w: NDArray[np.float64]) -> float:
        return float(w @ sigma @ w)

    def skew_obj(w: NDArray[np.float64]) -> float:
        return _portfolio_moments(w, mu, sigma, m3, m4)[2]

    def kurt_obj(w: NDArray[np.float64]) -> float:
        return _portfolio_moments(w, mu, sigma, m3, m4)[3]

    _, mean_star = _solve_single_objective(mean_obj, n_assets, minimise=False)
    _, var_star = _solve_single_objective(var_obj, n_assets, minimise=True)
    _, skew_star = _solve_single_objective(skew_obj, n_assets, minimise=False)
    _, kurt_star = _solve_single_objective(kurt_obj, n_assets, minimise=True)

    def safe_div(num: float, den: float) -> float:
        return num / den if abs(den) > 1e-12 else 0.0

    def pgp_objective(w: NDArray[np.float64]) -> float:
        mean, var, skew, kurt = _portfolio_moments(w, mu, sigma, m3, m4)
        return (
            max(0.0, 1.0 - safe_div(mean, mean_star)) ** alpha
            + max(0.0, safe_div(var, var_star) - 1.0) ** beta
            + max(0.0, 1.0 - safe_div(skew, skew_star)) ** gamma
            + max(0.0, safe_div(kurt, kurt_star) - 1.0) ** delta
        )

    w_opt, _ = _solve_single_objective(pgp_objective, n_assets, minimise=True)
    w_opt = np.clip(w_opt, 0.0, None)
    w_opt /= max(w_opt.sum(), 1e-12)
    mean, var, skew, kurt = _portfolio_moments(w_opt, mu, sigma, m3, m4)

    return HigherMomentResult(
        weights=pd.Series(w_opt, index=tickers),
        achieved_mean=mean,
        achieved_variance=var,
        achieved_skewness=skew,
        achieved_kurtosis=kurt,
        mean_star=mean_star,
        variance_star=var_star,
        skewness_star=skew_star,
        kurtosis_star=kurt_star,
    )

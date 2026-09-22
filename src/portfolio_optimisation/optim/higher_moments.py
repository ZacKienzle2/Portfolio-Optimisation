"""Polynomial Goal Programming over the first four portfolio moments.

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

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from portfolio_optimisation.optim.constraints import long_only

if TYPE_CHECKING:
    from collections.abc import Callable

    from numpy.typing import NDArray


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


def _centred(returns: pd.DataFrame) -> NDArray[np.float64]:
    values = returns.to_numpy(dtype=np.float64)
    return values - values.mean(axis=0)


def _pair_products(centred: NDArray[np.float64]) -> NDArray[np.float64]:
    """Row-wise Kronecker square ``c_t kron c_t`` as a ``(T, N^2)`` matrix."""
    t, n = centred.shape
    return (centred[:, :, None] * centred[:, None, :]).reshape(t, n * n)


def coskewness_tensor(returns: pd.DataFrame) -> NDArray[np.float64]:
    """Empirical co-skewness M3 of shape (N, N*N).

    ``M3[i, j*N + k] = (1/T) sum_t (r_t,i - mu_i)(r_t,j - mu_j)(r_t,k - mu_k)``,
    computed as one matrix product of the centred returns with their row-wise
    Kronecker square.

    Args:
        returns: Asset returns, one column per asset.

    Returns:
        The flattened co-skewness tensor.
    """
    centred = _centred(returns)
    return centred.T @ _pair_products(centred) / centred.shape[0]


def cokurtosis_tensor(returns: pd.DataFrame) -> NDArray[np.float64]:
    """Empirical co-kurtosis M4 of shape (N, N^3).

    Computed as the Gram matrix of the row-wise Kronecker square, one matrix
    product. The dense tensor holds ``N^4`` entries, so memory and work grow as
    ``O(N^4)``; the allocator below never forms it.

    Args:
        returns: Asset returns, one column per asset.

    Returns:
        The flattened co-kurtosis tensor.
    """
    centred = _centred(returns)
    pairs = _pair_products(centred)
    n = centred.shape[1]
    return (pairs.T @ pairs / centred.shape[0]).reshape(n, n**3)


def portfolio_third_and_fourth_moments(
    weights: NDArray[np.float64], centred: NDArray[np.float64]
) -> tuple[float, float]:
    """Third and fourth central moments of the portfolio return.

    ``w' M3 (w kron w)`` is the mean of the cubed centred portfolio return and
    ``w' M4 (w kron w kron w)`` the mean of its fourth power, so both cost
    ``O(TN)`` from the return series instead of ``O(N^4)`` from the tensors.

    Args:
        weights: Portfolio weights.
        centred: Demeaned returns, one column per asset.

    Returns:
        The third and fourth central moments.
    """
    portfolio = centred @ weights
    squared = portfolio * portfolio
    return float(np.mean(squared * portfolio)), float(np.mean(squared * squared))


def _portfolio_moments(
    w: NDArray[np.float64],
    mu: NDArray[np.float64],
    sigma: NDArray[np.float64],
    centred: NDArray[np.float64],
) -> tuple[float, float, float, float]:
    """Return (mean, variance, skewness, kurtosis) for weights ``w``."""
    mean = float(w @ mu)
    var = float(w @ sigma @ w)
    if var <= 0:
        return mean, var, 0.0, 0.0
    third, fourth = portfolio_third_and_fourth_moments(w, centred)
    return mean, var, third / var**1.5, fourth / var**2


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
    """Solve the PGP four-moment portfolio.

    The skewness and kurtosis objectives are evaluated from the portfolio return
    series, so the co-moment tensors are never formed and the universe size is
    bounded by the optimiser rather than by ``O(N^4)`` memory.

    Args:
        returns: Asset returns.
        alpha: Preference exponent on the mean-shortfall goal.
        beta: Preference exponent on the variance-excess goal.
        gamma: Preference exponent on the negative-skewness goal.
        delta: Preference exponent on the kurtosis-excess goal.

    Returns:
        Weights plus the realised and individually optimal moments used as goal
        references.
    """
    tickers = list(returns.columns)
    n_assets = len(tickers)
    mu = returns.mean().to_numpy(dtype=np.float64)
    sigma = returns.cov().to_numpy(dtype=np.float64)
    centred = _centred(returns)

    def mean_obj(w: NDArray[np.float64]) -> float:
        return float(w @ mu)

    def var_obj(w: NDArray[np.float64]) -> float:
        return float(w @ sigma @ w)

    def skew_obj(w: NDArray[np.float64]) -> float:
        return _portfolio_moments(w, mu, sigma, centred)[2]

    def kurt_obj(w: NDArray[np.float64]) -> float:
        return _portfolio_moments(w, mu, sigma, centred)[3]

    _, mean_star = _solve_single_objective(mean_obj, n_assets, minimise=False)
    _, var_star = _solve_single_objective(var_obj, n_assets, minimise=True)
    _, skew_star = _solve_single_objective(skew_obj, n_assets, minimise=False)
    _, kurt_star = _solve_single_objective(kurt_obj, n_assets, minimise=True)

    def safe_div(num: float, den: float) -> float:
        return num / den if abs(den) > 1e-12 else 0.0

    def pgp_objective(w: NDArray[np.float64]) -> float:
        mean, var, skew, kurt = _portfolio_moments(w, mu, sigma, centred)
        return (
            max(0.0, 1.0 - safe_div(mean, mean_star)) ** alpha
            + max(0.0, safe_div(var, var_star) - 1.0) ** beta
            + max(0.0, 1.0 - safe_div(skew, skew_star)) ** gamma
            + max(0.0, safe_div(kurt, kurt_star) - 1.0) ** delta
        )

    w_opt, _ = _solve_single_objective(pgp_objective, n_assets, minimise=True)
    w_opt = long_only(w_opt)
    mean, var, skew, kurt = _portfolio_moments(w_opt, mu, sigma, centred)

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

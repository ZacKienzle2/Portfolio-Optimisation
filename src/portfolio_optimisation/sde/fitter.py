"""Maximum-likelihood fitting of GBM and Ornstein-Uhlenbeck parameters per asset.

Both models have Gaussian transitions, so both likelihoods are maximised in
closed form rather than by a bounded numerical search.

Geometric Brownian motion ``dS = mu S dt + sigma S dW`` has independent
log-returns ``N((mu - sigma^2 / 2) dt, sigma^2 dt)``, whose maximum-likelihood
mean and variance are the sample moments.

The Ornstein-Uhlenbeck process ``dX = kappa (alpha - X) dt + sigma dW`` sampled
at step ``dt`` is the AR(1) recursion
``X_t = alpha (1 - b) + b X_(t-1) + e_t`` with ``b = exp(-kappa dt)`` and
``Var(e_t) = sigma^2 (1 - b^2) / (2 kappa)``. Conditional on the first
observation, Tang and Chen (2009, eq. 2.5) give the maximum-likelihood
estimators from the least-squares fit of that recursion,

    kappa = -log(b) / dt,   alpha = c / (1 - b),   sigma^2 = 2 kappa s^2 / (1 - b^2),

with ``c`` the intercept and ``s^2`` the residual variance over ``n``, which
``statsmodels.tsa.ar_model.AutoReg`` computes. Their Theorem 3.1.1 gives the
bias ``E[kappa] - kappa = (5/2 + e^(kappa dt) + e^(2 kappa dt) / 2) / (n dt)``,
of the order of ``4 / T`` for a span of ``T`` years, which ``bias_correction``
subtracts at the estimate.

The numerical search this replaced bounded ``sigma`` at 5, a volatility of
returns, while the process is fitted to price levels, where ``sigma`` is in
price units per root year and sat on the bound.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from scipy.stats import norm
from statsmodels.tools.eval_measures import aic, bic
from statsmodels.tsa.ar_model import AutoReg

if TYPE_CHECKING:
    from numpy.typing import NDArray


def _information(log_likelihood: float, n_obs: int, n_params: int) -> dict[str, float]:
    return {
        "Log-Likelihood": log_likelihood,
        "AIC": float(aic(log_likelihood, n_obs, n_params)),
        "BIC": float(bic(log_likelihood, n_obs, n_params)),
    }


def gbm_estimates(prices: NDArray[np.float64], dt: float) -> dict[str, float]:
    """Maximum-likelihood GBM drift and volatility of one price series.

    Args:
        prices: Positive prices in time order.
        dt: Time step between observations in years.

    Returns:
        ``mu``, ``sigma`` and the log-likelihood of the prices with AIC and BIC.
    """
    log_returns = np.diff(np.log(prices))
    location, scale = log_returns.mean(), log_returns.std()
    sigma = scale / np.sqrt(dt)
    log_likelihood = float(
        norm.logpdf(log_returns, location, scale).sum() - np.log(prices[1:]).sum()
    )
    return {
        "mu": float(location / dt + sigma**2 / 2.0),
        "sigma": float(sigma),
        **_information(log_likelihood, log_returns.size, 2),
    }


def ou_estimates(
    levels: NDArray[np.float64], dt: float, *, bias_correction: bool = False
) -> dict[str, float]:
    """Maximum-likelihood Ornstein-Uhlenbeck parameters of one series.

    Args:
        levels: Observations in time order.
        dt: Time step between observations in years.
        bias_correction: Subtract the first-order bias of ``kappa`` from
            Theorem 3.1.1 of Tang and Chen (2009).

    Returns:
        ``kappa``, ``mu`` (the long-run level ``alpha``), ``sigma`` and the
        conditional log-likelihood with AIC and BIC. A slope of one or more
        admits no mean reversion and gives a non-positive ``kappa``.
    """
    fit = AutoReg(levels, lags=1, trend="c").fit()
    intercept, slope = fit.params
    kappa = -np.log(slope) / dt
    n_obs = int(fit.nobs)
    if bias_correction:
        kappa -= (2.5 + np.exp(kappa * dt) + np.exp(2.0 * kappa * dt) / 2.0) / (n_obs * dt)
    sigma = np.sqrt(-2.0 * np.log(slope) / dt * fit.sigma2 / (1.0 - slope**2))
    return {
        "kappa": float(kappa),
        "mu": float(intercept / (1.0 - slope)),
        "sigma": float(sigma),
        **_information(float(fit.llf), n_obs, 3),
    }


class SDEFitter:
    """Estimates GBM and Ornstein-Uhlenbeck parameters for each price series.

    Args:
        prices_df: Asset price time series.
        dt: Time step, 1/252 for daily data.
    """

    def __init__(self, prices_df: pd.DataFrame, dt: float = 1 / 252) -> None:
        self.prices_df: pd.DataFrame = prices_df
        self.dt: float = dt
        self.gbm_results: pd.DataFrame | None = None
        self.ou_results: pd.DataFrame | None = None

    def _per_column(self, estimates: dict[str, dict[str, float]]) -> pd.DataFrame:
        return pd.DataFrame.from_dict(estimates, orient="index").rename_axis("Ticker")

    def fit_gbm(self) -> pd.DataFrame:
        """Fit GBM to each asset price series.

        Returns:
            ``mu``, ``sigma``, log-likelihood, AIC and BIC indexed by ticker.
        """
        self.gbm_results = self._per_column(
            {
                str(ticker): gbm_estimates(
                    self.prices_df[ticker].to_numpy(dtype=np.float64), self.dt
                )
                for ticker in self.prices_df.columns
            }
        )
        return self.gbm_results

    def fit_ou(self, *, bias_correction: bool = False) -> pd.DataFrame:
        """Fit an Ornstein-Uhlenbeck process to each asset price series.

        Args:
            bias_correction: Subtract the first-order bias of ``kappa``.

        Returns:
            ``kappa``, ``mu``, ``sigma``, log-likelihood, AIC and BIC indexed by
            ticker.
        """
        self.ou_results = self._per_column(
            {
                str(ticker): ou_estimates(
                    self.prices_df[ticker].to_numpy(dtype=np.float64),
                    self.dt,
                    bias_correction=bias_correction,
                )
                for ticker in self.prices_df.columns
            }
        )
        return self.ou_results

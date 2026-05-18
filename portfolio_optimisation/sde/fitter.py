import contextlib
import os
from multiprocessing import Pool, cpu_count
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from pymle.core.TransitionDensity import ExactDensity
from pymle.fit.AnalyticalMLE import AnalyticalMLE
from pymle.fit.Minimizer import ScipyMinimizer
from pymle.models import GeometricBM, OrnsteinUhlenbeck


def _fit_one_gbm(
    args: tuple[str, NDArray[np.float64], list[tuple[float, float]], float, NDArray[np.float64]],
) -> dict[str, float | str]:
    """Worker for parallel GBM MLE. Top-level so the Pool can pickle it."""
    ticker, series, bounds, dt, init = args
    mle = AnalyticalMLE(
        series,
        bounds,
        dt,
        ExactDensity(GeometricBM()),
        minimizer=ScipyMinimizer(),
    )
    est: Any
    with open(os.devnull, "w") as fnull, contextlib.redirect_stdout(fnull):
        est = mle.estimate_params(init)
    return {
        "Ticker": ticker,
        "mu": est.params[0],
        "sigma": est.params[1],
        "Log-Likelihood": est.log_like,
        "AIC": est.aic,
        "BIC": est.bic,
    }


def _fit_one_ou(
    args: tuple[str, NDArray[np.float64], list[tuple[float, float]], float, NDArray[np.float64]],
) -> dict[str, float | str]:
    """Worker for parallel OU MLE. Top-level so the Pool can pickle it."""
    ticker, series, bounds, dt, init = args
    mle = AnalyticalMLE(
        series,
        bounds,
        dt,
        ExactDensity(OrnsteinUhlenbeck()),
        minimizer=ScipyMinimizer(),
    )
    est: Any
    with open(os.devnull, "w") as fnull, contextlib.redirect_stdout(fnull):
        est = mle.estimate_params(init)
    return {
        "Ticker": ticker,
        "kappa": est.params[0],
        "mu": est.params[1],
        "sigma": est.params[2],
        "Log-Likelihood": est.log_like,
        "AIC": est.aic,
        "BIC": est.bic,
    }


class SDEFitter:
    """Estimates parameters for Stochastic Differential Equations (SDEs).

    Fits price data to common SDE models, including Geometric Brownian Motion
    (GBM) and Ornstein-Uhlenbeck (OU), using Maximum Likelihood Estimation (MLE)
    based on exact transition densities. Per-column fits run in a worker pool
    when more than one column is present.
    """

    def __init__(self, prices_df: pd.DataFrame, dt: float = 1 / 252):
        """Initialise the fitter.

        Args:
            prices_df (pd.DataFrame): Asset price time series.
            dt (float, optional): Time step. Defaults to 1/252 for daily data.
        """
        self.prices_df: pd.DataFrame = prices_df
        self.dt: float = dt
        self.gbm_results: pd.DataFrame | None = None
        self.ou_results: pd.DataFrame | None = None

    def _resolve_workers(self) -> int:
        """Worker count: at most one per column, never more than cpu_count-1."""
        return max(1, min(cpu_count() - 1, len(self.prices_df.columns)))

    def fit_gbm(
        self,
        param_bounds: list[tuple[float, float]] | None = None,
        initial_guess: NDArray[np.float64] | None = None,
    ) -> pd.DataFrame:
        """Fit GBM to each asset price series via analytical MLE."""
        if initial_guess is None:
            initial_guess = np.array([0.01, 0.2])
        if param_bounds is None:
            param_bounds = [(-1.0, 1.0), (1e-5, 5.0)]

        jobs = [
            (ticker, self.prices_df[ticker].values, param_bounds, self.dt, initial_guess)
            for ticker in self.prices_df.columns
        ]

        n_workers = self._resolve_workers()
        if n_workers <= 1:
            results = [_fit_one_gbm(job) for job in jobs]
        else:
            with Pool(n_workers) as pool:
                results = pool.map(_fit_one_gbm, jobs)

        self.gbm_results = pd.DataFrame(results).set_index("Ticker")
        return self.gbm_results

    def fit_ou(
        self,
        param_bounds: list[tuple[float, float]] | None = None,
        initial_guess: NDArray[np.float64] | None = None,
    ) -> pd.DataFrame:
        """Fit Ornstein-Uhlenbeck to each asset price series via analytical MLE."""
        if initial_guess is None:
            initial_mean = self.prices_df.mean().mean() if not self.prices_df.empty else 1.0
            initial_guess = np.array([1.0, initial_mean, 0.2])

        if param_bounds is None:
            min_val = self.prices_df.min().min() if not self.prices_df.empty else 0.0
            max_val = self.prices_df.max().max() if not self.prices_df.empty else 1000.0
            param_bounds = [(1e-5, 20.0), (min_val, max_val), (1e-5, 5.0)]

        jobs = [
            (ticker, self.prices_df[ticker].values, param_bounds, self.dt, initial_guess)
            for ticker in self.prices_df.columns
        ]

        n_workers = self._resolve_workers()
        if n_workers <= 1:
            results = [_fit_one_ou(job) for job in jobs]
        else:
            with Pool(n_workers) as pool:
                results = pool.map(_fit_one_ou, jobs)

        self.ou_results = pd.DataFrame(results).set_index("Ticker")
        return self.ou_results

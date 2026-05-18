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

    def __init__(self, pricesDf: pd.DataFrame, dt: float = 1 / 252):
        """Initialise the fitter.

        Args:
            pricesDf (pd.DataFrame): Asset price time series.
            dt (float, optional): Time step. Defaults to 1/252 for daily data.
        """
        self.pricesDf: pd.DataFrame = pricesDf
        self.dt: float = dt
        self.gbmResults: pd.DataFrame | None = None
        self.ouResults: pd.DataFrame | None = None

    def _resolveWorkers(self) -> int:
        """Worker count: at most one per column, never more than cpu_count-1."""
        return max(1, min(cpu_count() - 1, len(self.pricesDf.columns)))

    def fitGbm(
        self,
        paramBounds: list[tuple[float, float]] | None = None,
        initialGuess: NDArray[np.float64] | None = None,
    ) -> pd.DataFrame:
        """Fit GBM to each asset price series via analytical MLE."""
        if initialGuess is None:
            initialGuess = np.array([0.01, 0.2])
        if paramBounds is None:
            paramBounds = [(-1.0, 1.0), (1e-5, 5.0)]

        jobs = [
            (ticker, self.pricesDf[ticker].values, paramBounds, self.dt, initialGuess)
            for ticker in self.pricesDf.columns
        ]

        nWorkers = self._resolveWorkers()
        if nWorkers <= 1:
            results = [_fit_one_gbm(job) for job in jobs]
        else:
            with Pool(nWorkers) as pool:
                results = pool.map(_fit_one_gbm, jobs)

        self.gbmResults = pd.DataFrame(results).set_index("Ticker")
        return self.gbmResults

    def fitOu(
        self,
        paramBounds: list[tuple[float, float]] | None = None,
        initialGuess: NDArray[np.float64] | None = None,
    ) -> pd.DataFrame:
        """Fit Ornstein-Uhlenbeck to each asset price series via analytical MLE."""
        if initialGuess is None:
            initialMean = (
                self.pricesDf.mean().mean() if not self.pricesDf.empty else 1.0
            )
            initialGuess = np.array([1.0, initialMean, 0.2])

        if paramBounds is None:
            minVal = self.pricesDf.min().min() if not self.pricesDf.empty else 0.0
            maxVal = self.pricesDf.max().max() if not self.pricesDf.empty else 1000.0
            paramBounds = [(1e-5, 20.0), (minVal, maxVal), (1e-5, 5.0)]

        jobs = [
            (ticker, self.pricesDf[ticker].values, paramBounds, self.dt, initialGuess)
            for ticker in self.pricesDf.columns
        ]

        nWorkers = self._resolveWorkers()
        if nWorkers <= 1:
            results = [_fit_one_ou(job) for job in jobs]
        else:
            with Pool(nWorkers) as pool:
                results = pool.map(_fit_one_ou, jobs)

        self.ouResults = pd.DataFrame(results).set_index("Ticker")
        return self.ouResults

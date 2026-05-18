import contextlib
import os
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from pymle.core.TransitionDensity import ExactDensity
from pymle.fit.AnalyticalMLE import AnalyticalMLE
from pymle.fit.Minimizer import ScipyMinimizer
from pymle.models import GeometricBM, OrnsteinUhlenbeck


class SDEFitter:
    """Estimates parameters for Stochastic Differential Equations (SDEs).

    Fits price data to common SDE models, including Geometric Brownian Motion
    (GBM) and Ornstein-Uhlenbeck (OU), using Maximum Likelihood Estimation (MLE)
    based on exact transition densities.
    """

    def __init__(self, pricesDf: pd.DataFrame, dt: float = 1 / 252):
        """Initialise the fitter with asset prices and time step.

        Args:
            pricesDf (pd.DataFrame): Asset price time series.
            dt (float, optional): Time step (e.g., 1/252 for daily). Defaults to 1/252.
        """
        self.pricesDf: pd.DataFrame = pricesDf
        self.dt: float = dt
        self.gbmResults: pd.DataFrame | None = None
        self.ouResults: pd.DataFrame | None = None

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

        results = []
        for ticker in self.pricesDf.columns:
            seriesData: NDArray[np.float64] = self.pricesDf[ticker].values

            mle = AnalyticalMLE(
                seriesData,
                paramBounds,
                self.dt,
                ExactDensity(GeometricBM()),
                minimizer=ScipyMinimizer(),
            )

            est: Any
            with open(os.devnull, "w") as fnull, contextlib.redirect_stdout(fnull):
                est = mle.estimate_params(initialGuess)

            results.append(
                {
                    "Ticker": ticker,
                    "mu": est.params[0],
                    "sigma": est.params[1],
                    "Log-Likelihood": est.log_like,
                    "AIC": est.aic,
                    "BIC": est.bic,
                }
            )

        self.gbmResults = pd.DataFrame(results).set_index("Ticker")
        if self.gbmResults is None:
            raise RuntimeError("Failed to generate GBM results DataFrame.")
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

        results = []
        for ticker in self.pricesDf.columns:
            seriesData: NDArray[np.float64] = self.pricesDf[ticker].values

            mle = AnalyticalMLE(
                seriesData,
                paramBounds,
                self.dt,
                ExactDensity(OrnsteinUhlenbeck()),
                minimizer=ScipyMinimizer(),
            )
            est: Any
            with open(os.devnull, "w") as fnull, contextlib.redirect_stdout(fnull):
                est = mle.estimate_params(initialGuess)

            results.append(
                {
                    "Ticker": ticker,
                    "kappa": est.params[0],
                    "mu": est.params[1],
                    "sigma": est.params[2],
                    "Log-Likelihood": est.log_like,
                    "AIC": est.aic,
                    "BIC": est.bic,
                }
            )

        self.ouResults = pd.DataFrame(results).set_index("Ticker")
        if self.ouResults is None:
            raise RuntimeError("Failed to generate OU results DataFrame.")
        return self.ouResults

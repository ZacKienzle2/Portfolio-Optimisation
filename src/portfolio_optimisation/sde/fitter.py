"""Maximum-likelihood fitting of GBM and Ornstein-Uhlenbeck parameters per asset."""

from __future__ import annotations

import contextlib
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
from pymle.core.TransitionDensity import ExactDensity
from pymle.fit.AnalyticalMLE import AnalyticalMLE
from pymle.fit.Minimizer import ScipyMinimizer
from pymle.models import GeometricBM, OrnsteinUhlenbeck

if TYPE_CHECKING:
    from numpy.typing import NDArray


def _fit_columns(
    prices: pd.DataFrame,
    model: Any,
    names: tuple[str, ...],
    bounds: list[tuple[float, float]],
    initial_guess: NDArray[np.float64],
    dt: float,
) -> pd.DataFrame:
    """Fit one pymle model to every column by exact-density maximum likelihood.

    The fits run in this process. Each takes a fraction of a second, while a
    spawned worker spends seconds importing the numerical stack before its
    first fit. pymle prints its progress, which is discarded.

    Args:
        prices: Price series, one column per asset.
        model: pymle model instance defining the transition density.
        names: Parameter names in the model's order.
        bounds: Box bounds for each parameter.
        initial_guess: Starting point for the optimiser.
        dt: Time step between observations.

    Returns:
        Estimated parameters with log-likelihood, AIC and BIC, indexed by ticker.
    """
    density = ExactDensity(model)
    rows: dict[str, dict[str, float]] = {}
    for ticker in prices.columns:
        mle = AnalyticalMLE(
            prices[ticker].to_numpy(), bounds, dt, density, minimizer=ScipyMinimizer()
        )
        with contextlib.redirect_stdout(None):
            estimate = mle.estimate_params(initial_guess)
        rows[str(ticker)] = {
            **dict(zip(names, estimate.params, strict=True)),
            "Log-Likelihood": estimate.log_like,
            "AIC": estimate.aic,
            "BIC": estimate.bic,
        }
    return pd.DataFrame.from_dict(rows, orient="index").rename_axis("Ticker")


class SDEFitter:
    """Estimates GBM and Ornstein-Uhlenbeck parameters for each price series.

    Fits by maximum likelihood on the exact transition density of each model.

    Args:
        prices_df: Asset price time series.
        dt: Time step, 1/252 for daily data.
    """

    def __init__(self, prices_df: pd.DataFrame, dt: float = 1 / 252) -> None:
        self.prices_df: pd.DataFrame = prices_df
        self.dt: float = dt
        self.gbm_results: pd.DataFrame | None = None
        self.ou_results: pd.DataFrame | None = None

    def fit_gbm(
        self,
        param_bounds: list[tuple[float, float]] | None = None,
        initial_guess: NDArray[np.float64] | None = None,
    ) -> pd.DataFrame:
        """Fit GBM to each asset price series.

        Args:
            param_bounds: Bounds for ``(mu, sigma)``.
            initial_guess: Starting ``(mu, sigma)``.

        Returns:
            Estimates indexed by ticker.
        """
        self.gbm_results = _fit_columns(
            self.prices_df,
            GeometricBM(),
            ("mu", "sigma"),
            param_bounds if param_bounds is not None else [(-1.0, 1.0), (1e-5, 5.0)],
            initial_guess if initial_guess is not None else np.array([0.01, 0.2]),
            self.dt,
        )
        return self.gbm_results

    def fit_ou(
        self,
        param_bounds: list[tuple[float, float]] | None = None,
        initial_guess: NDArray[np.float64] | None = None,
    ) -> pd.DataFrame:
        """Fit an Ornstein-Uhlenbeck process to each asset price series.

        Args:
            param_bounds: Bounds for ``(kappa, mu, sigma)``. Defaults span the observed
                price range for ``mu``.
            initial_guess: Starting ``(kappa, mu, sigma)``.

        Returns:
            Estimates indexed by ticker.
        """
        prices = self.prices_df
        if initial_guess is None:
            initial_guess = np.array([1.0, prices.mean().mean() if not prices.empty else 1.0, 0.2])
        if param_bounds is None:
            low = prices.min().min() if not prices.empty else 0.0
            high = prices.max().max() if not prices.empty else 1000.0
            param_bounds = [(1e-5, 20.0), (low, high), (1e-5, 5.0)]
        self.ou_results = _fit_columns(
            prices,
            OrnsteinUhlenbeck(),
            ("kappa", "mu", "sigma"),
            param_bounds,
            initial_guess,
            self.dt,
        )
        return self.ou_results

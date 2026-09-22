"""Copula-based simulation of joint return scenarios for portfolio risk."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import minimize_scalar
from statsmodels.distributions.copula.api import StudentTCopula
from statsmodels.stats.correlation_tools import corr_clipped

from portfolio_optimisation.infra.logging import get_logger

if TYPE_CHECKING:
    from numpy.typing import NDArray

logger = get_logger(__name__)

_FIT_ERRORS = (ValueError, RuntimeError, FloatingPointError)
_UNIT_CLIP = 1e-9


def _start(dist: Any, data: NDArray[np.float64]) -> tuple[tuple[float, ...], dict[str, float]]:
    """Moment-matched starting point for a Student-t fit, none for other families.

    Starting Nelder-Mead at five degrees of freedom, the median and the scale
    that matches the sample variance reaches the same optimum as SciPy's
    default start in fewer than half the likelihood evaluations.
    """
    if dist is not stats.t:
        return (), {}
    df = 5.0
    return (df,), {
        "loc": float(np.median(data)),
        "scale": float(data.std() * np.sqrt((df - 2.0) / df)),
    }


class CopulaRiskAnalyser:
    """Multivariate risk simulation with a Student-t copula.

    Fits a univariate marginal to each asset, estimates the copula correlation
    by inverting Kendall's tau and its degrees of freedom by maximum
    likelihood, then simulates joint returns through the fitted marginals.

    Args:
        returns: Asset returns, one column per asset.
        weights: Portfolio weights indexed by ticker.

    Raises:
        ValueError: If the returns and weights share no ticker.
    """

    def __init__(self, returns: pd.DataFrame, weights: pd.Series) -> None:
        tickers = [ticker for ticker in returns.columns if ticker in weights.index]
        if not tickers:
            msg = "Returns and weights must share common tickers."
            raise ValueError(msg)
        self.returns: pd.DataFrame = returns[tickers]
        self.weights: pd.Series = weights.reindex(tickers).fillna(0.0)
        self.fit_marginals_: dict[str, dict[str, Any]] = {}
        self.copula: Any | None = None
        self.uniform_returns: pd.DataFrame | None = None

    def _fitted(self) -> list[str]:
        return [
            ticker
            for ticker in self.returns.columns
            if self.fit_marginals_.get(ticker, {}).get("params") is not None
        ]

    def _params(self, tickers: list[str]) -> list[NDArray[np.float64]]:
        """Marginal parameters as one row vector per parameter, for broadcasting."""
        table = np.array([self.fit_marginals_[ticker]["params"] for ticker in tickers])
        return [table[None, :, j] for j in range(table.shape[1])]

    def fit_marginal_distributions(self, dist: Any = stats.t) -> None:
        """Fit a univariate distribution to each asset's return series.

        Args:
            dist: SciPy continuous distribution with ``fit``, ``cdf`` and ``ppf``.

        Raises:
            TypeError: If ``dist`` lacks one of those methods.
        """
        if not all(hasattr(dist, method) for method in ("fit", "cdf", "ppf")):
            msg = "dist object must have fit, cdf, and ppf methods."
            raise TypeError(msg)
        for ticker in self.returns.columns:
            data = self.returns[ticker].dropna().to_numpy(dtype=np.float64)
            self.fit_marginals_[ticker] = {"dist": None, "params": None}
            if data.size < 10:
                logger.warning("Insufficient data for %s marginal fit.", ticker)
                continue
            args, kwargs = _start(dist, data)
            try:
                self.fit_marginals_[ticker] = {
                    "dist": dist,
                    "params": dist.fit(data, *args, **kwargs),
                }
            except _FIT_ERRORS as error:
                logger.warning("Failed marginal fit for %s: %s", ticker, error)

    def _estimate_copula_params(self) -> tuple[NDArray[np.float64], float]:
        """Kendall-tau correlation and maximum-likelihood degrees of freedom.

        Returns:
            The positive semi-definite copula correlation and degrees of freedom.

        Raises:
            RuntimeError: If no uniform returns are available.
        """
        if self.uniform_returns is None:
            msg = "Uniform returns required but not generated."
            raise RuntimeError(msg)
        k_dim = self.uniform_returns.shape[1]
        uniforms = np.clip(self.uniform_returns.to_numpy(), _UNIT_CLIP, 1.0 - _UNIT_CLIP)
        corr = corr_clipped(StudentTCopula(k_dim=k_dim).fit_corr_param(uniforms), threshold=1e-8)

        def negative_log_likelihood(df: float) -> float:
            logpdf = StudentTCopula(corr=corr, df=df, k_dim=k_dim).logpdf(uniforms)
            penalty = 0.01 * (df - 10.0) ** 2 if df > 25.0 or df < 3.0 else 0.0
            return float(-np.sum(np.maximum(logpdf, np.log(1e-12))) + penalty)

        result = minimize_scalar(negative_log_likelihood, bounds=(2.1, 30.0), method="bounded")
        if not result.success:
            logger.warning("Copula df optimization issue: %s", result.message)
        return corr, float(result.x)

    def fit_copula(self) -> None:
        """Fit the Student-t copula to the probability-integral-transformed returns.

        Raises:
            RuntimeError: If no marginal fitted or too few observations remain.
        """
        if not self.fit_marginals_:
            msg = "Fit marginal distributions first."
            raise RuntimeError(msg)
        tickers = self._fitted()
        if not tickers:
            msg = "No valid marginals; cannot fit copula."
            raise RuntimeError(msg)
        dist = self.fit_marginals_[tickers[0]]["dist"]
        returns = self.returns[tickers].dropna()
        self.uniform_returns = pd.DataFrame(
            dist.cdf(returns.to_numpy(), *self._params(tickers)),
            index=returns.index,
            columns=tickers,
        )
        self.returns = self.returns[tickers]
        self.weights = self.weights.loc[tickers]
        if not np.isclose(self.weights.sum(), 1.0):
            self.weights /= self.weights.sum()
        if len(self.uniform_returns) < len(tickers) + 1:
            msg = "Insufficient valid uniform data; cannot fit copula."
            raise RuntimeError(msg)
        corr, df = self._estimate_copula_params()
        self.copula = StudentTCopula(corr=corr, df=df, k_dim=len(tickers))

    def run_simulation(
        self, n_simulations: int = 10000, *, seed: int | None = None
    ) -> pd.DataFrame:
        """Simulate portfolio returns through the fitted copula and marginals.

        Args:
            n_simulations: Number of simulated scenarios.
            seed: Seed forwarded to the copula sampler.

        Returns:
            Simulated portfolio returns in column ``simulated_returns``.

        Raises:
            RuntimeError: If the copula has not been fitted.
        """
        if self.copula is None:
            msg = "Fit the copula first."
            raise RuntimeError(msg)
        tickers = list(self.returns.columns)
        dist = self.fit_marginals_[tickers[0]]["dist"]
        uniforms = np.clip(
            self.copula.rvs(n_simulations, random_state=seed), _UNIT_CLIP, 1.0 - _UNIT_CLIP
        )
        scenarios = dist.ppf(uniforms, *self._params(tickers))
        scenarios = scenarios[np.isfinite(scenarios).all(axis=1)]
        return pd.DataFrame({"simulated_returns": scenarios @ self.weights.to_numpy()})

    def plot_marginal_fits(self) -> None:
        """Plot fitted marginal densities against the return histograms."""
        import matplotlib.pyplot as plt
        import seaborn as sns

        if self.returns.empty or not self.fit_marginals_:
            logger.warning("Return data or marginal fits missing for plotting.")
            return
        n_assets = len(self.returns.columns)
        n_rows = (n_assets + 1) // 2
        fig, axes = plt.subplots(n_rows, 2, figsize=(14, n_rows * 5), squeeze=False)
        fig.suptitle("Fitted Marginal Distributions vs. Historical Returns", fontsize=16)
        axes_flat = axes.flatten()
        for ax, ticker in zip(axes_flat, self.returns.columns, strict=False):
            data = self.returns[ticker].dropna()
            sns.histplot(x=data, bins=50, stat="density", ax=ax, label="Historical")
            marginal = self.fit_marginals_.get(ticker, {})
            if marginal.get("params") is not None:
                grid = np.linspace(data.min() - 1e-3, data.max() + 1e-3, 200)
                ax.plot(
                    grid,
                    marginal["dist"].pdf(grid, *marginal["params"]),
                    "r-",
                    lw=2,
                    label="Fitted PDF",
                )
            ax.set_title(f"Fit for {ticker}")
            ax.legend()
        for ax in axes_flat[n_assets:]:
            fig.delaxes(ax)
        fig.tight_layout(rect=(0, 0.03, 1, 0.96))
        plt.show()

    def plot_copula_dependence(self) -> None:
        """Plot pairwise dependence in the uniform copula space."""
        import matplotlib.pyplot as plt
        import seaborn as sns

        if self.uniform_returns is None or self.uniform_returns.empty:
            logger.warning("Uniform returns unavailable/empty; cannot plot dependence.")
            return
        grid = sns.pairplot(
            self.uniform_returns, kind="scatter", diag_kind="kde", plot_kws={"alpha": 0.4}
        )
        grid.figure.suptitle("Pairwise Dependence Structure (Uniform Space)", y=1.02)
        plt.show()


def run_historical_simulation(
    returns: pd.DataFrame,
    weights: pd.Series,
    n_simulations: int = 10000,
    *,
    seed: int | None = None,
) -> pd.DataFrame:
    """Simulate portfolio returns by resampling historical portfolio returns.

    Draws with replacement from the observed portfolio return distribution,
    which assumes independent and identically distributed returns.

    Args:
        returns: Historical asset returns.
        weights: Portfolio weights.
        n_simulations: Number of simulated days.
        seed: Seed for the resampling generator.

    Returns:
        Simulated portfolio returns in column ``simulated_returns``.
    """
    if returns.empty or weights.empty:
        return pd.DataFrame({"simulated_returns": []})
    history = returns.dot(weights.reindex(returns.columns).fillna(0.0)).dropna()
    if history.empty:
        return pd.DataFrame({"simulated_returns": []})
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {"simulated_returns": rng.choice(history.to_numpy(), size=n_simulations, replace=True)}
    )

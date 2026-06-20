from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.axes import Axes
from numpy.typing import NDArray
from scipy import stats
from scipy.optimize import OptimizeResult, minimize
from statsmodels.distributions.copula.api import StudentTCopula

from portfolio_optimisation.infra.logging import get_logger

logger = get_logger(__name__)


class CopulaRiskAnalyser:
    """Performs multivariate risk simulation using a Student's t-copula.

    Models marginal returns and their dependence via a t-copula. Fits
    marginals (default: t-dist), estimates copula df (MLE) and correlation
    (Kendall's Tau), then simulates correlated returns.
    """

    def __init__(self, returns: pd.DataFrame, weights: pd.Series):
        """Initialise analyser with asset returns and portfolio weights.

        Args:
            returns (pd.DataFrame): Asset returns (columns=assets, index=time).
            weights (pd.Series): Portfolio weights corresponding to asset columns.
        """
        common_tickers = list(set(returns.columns) & set(weights.index))
        if not common_tickers:
            raise ValueError("Returns and weights must share common tickers.")

        self.returns: pd.DataFrame = returns[common_tickers]
        self.weights: pd.Series = weights.reindex(common_tickers).fillna(0.0)

        if self.returns.empty or self.weights.empty:
            raise ValueError("Filtered returns or weights are empty.")

        self.fit_marginals_: dict[str, dict[str, Any]] = {}
        self.copula: Any | None = None
        self.uniform_returns: pd.DataFrame | None = None

    def fit_marginal_distributions(self, dist: Any = stats.t):
        """Fit a univariate distribution to each asset's returns series.

        Args:
            dist (Any): scipy.stats distribution object with fit/cdf/ppf.
        """
        if not all(hasattr(dist, method) for method in ["fit", "cdf", "ppf"]):
            raise TypeError("dist object must have fit, cdf, and ppf methods.")

        for ticker in self.returns.columns:
            series_data = self.returns[ticker].dropna()
            if len(series_data) < 10:
                logger.warning(f"Insufficient data for {ticker} marginal fit.")
                self.fit_marginals_[ticker] = {"dist": None, "params": None}
                continue
            try:
                params: tuple[Any, ...] = dist.fit(series_data.to_numpy())
                self.fit_marginals_[ticker] = {"dist": dist, "params": params}
            except Exception as e:
                logger.warning(f"Failed marginal fit for {ticker}: {e}")
                self.fit_marginals_[ticker] = {"dist": None, "params": None}

    def _estimate_copula_params(self) -> tuple[NDArray[np.float64], float]:
        """Estimate correlation matrix and degrees of freedom for the t-copula."""
        if self.uniform_returns is None:
            raise RuntimeError("Uniform returns required but not generated.")

        kendall_tau: pd.DataFrame = self.uniform_returns.corr("kendall")
        tau: NDArray[np.float64] = kendall_tau.to_numpy()
        corr: NDArray[np.float64] = np.sin(tau * np.pi / 2)
        corr = self._ensure_psd(corr)

        def log_likelihood(df_param: float) -> float:
            """Calculate negative log-likelihood of t-copula given data."""
            if self.uniform_returns is None:
                raise RuntimeError("Internal Error: uniform_returns became None.")
            try:
                copula_internal: Any = StudentTCopula(
                    corr=corr, df=df_param, k_dim=self.returns.shape[1]
                )
                uniform_vals_np = self.uniform_returns.to_numpy()
                uniform_values = np.clip(uniform_vals_np, 1e-9, 1 - 1e-9)
                pdf_values: NDArray[np.float64] = copula_internal.pdf(uniform_values)
                pdf_values = np.maximum(pdf_values, 1e-12)
                penalty = 0.01 * (df_param - 10) ** 2 if df_param > 25 or df_param < 3 else 0
                return -np.sum(np.log(pdf_values)) + penalty
            except Exception as e:
                logger.warning(f"Error in likelihood calc for df={df_param}: {e}")
                return np.inf

        result: OptimizeResult = minimize(
            log_likelihood,
            x0=np.array([10.0]),
            bounds=[(2.1, 30.0)],
            method="L-BFGS-B",
        )
        if not result.success:
            logger.warning(f"Copula df optimization issue: {result.message}")

        estimated_df: float = float(result.x[0])
        return corr, estimated_df

    def _ensure_psd(
        self, matrix: NDArray[np.float64], epsilon: float = 1e-8
    ) -> NDArray[np.float64]:
        """Ensure matrix positive semi-definiteness via eigenvalue adjustment."""
        eigenvalues, eigenvectors = np.linalg.eigh(matrix)
        eigenvalues[eigenvalues < epsilon] = epsilon
        psd_matrix = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T
        scale = np.sqrt(np.diag(psd_matrix))
        scale = np.where(scale < epsilon, 1.0, scale)
        psd_matrix = psd_matrix / np.outer(scale, scale)
        return np.clip(psd_matrix, -1.0, 1.0)

    def fit_copula(self):
        """Fit the Student's t-copula to transformed marginal returns."""
        if not self.fit_marginals_:
            raise RuntimeError("Fit marginal distributions first.")

        uniform_data = {}
        valid_tickers = []
        for ticker in self.returns.columns:
            marginal_info = self.fit_marginals_.get(ticker)
            if (
                marginal_info
                and marginal_info["dist"] is not None
                and marginal_info["params"] is not None
            ):
                dist = marginal_info["dist"]
                params = marginal_info["params"]
                series_data = self.returns[ticker].dropna()
                if not series_data.empty:
                    try:
                        uniform_data[ticker] = dist.cdf(series_data, *params)
                        valid_tickers.append(ticker)
                    except Exception as e:
                        logger.warning(f"CDF transformation failed for {ticker}: {e}")
            else:
                logger.warning(f"Skipping {ticker} (no valid marginal fit).")

        if not valid_tickers:
            raise RuntimeError("No valid marginals; cannot fit copula.")

        self.returns = self.returns[valid_tickers]
        self.weights = self.weights.loc[valid_tickers]
        if not np.isclose(self.weights.sum(), 1.0):
            self.weights /= self.weights.sum()

        self.uniform_returns = pd.DataFrame(uniform_data)
        self.uniform_returns.dropna(inplace=True)

        if self.uniform_returns.empty or len(self.uniform_returns) < self.returns.shape[1] + 1:
            raise RuntimeError("Insufficient valid uniform data; cannot fit copula.")

        corr, df = self._estimate_copula_params()
        self.copula = StudentTCopula(corr=corr, df=df, k_dim=self.returns.shape[1])

    def run_simulation(
        self, n_simulations: int = 10000, *, seed: int | None = None
    ) -> pd.DataFrame:
        """Generate correlated portfolio returns using the fitted copula.

        Args:
            n_simulations (int): Number of simulation paths.
            seed (int | None): Seed forwarded to the copula sampler for
                reproducible draws. Defaults to None (non-deterministic).

        Returns:
            pd.DataFrame: Simulated portfolio returns ('simulated_returns').
        """
        if self.copula is None:
            raise RuntimeError("Fit the copula first.")

        simulated_uniform: NDArray[np.float64] = self.copula.rvs(
            n_simulations, random_state=seed
        )
        np.clip(simulated_uniform, 1e-9, 1 - 1e-9, out=simulated_uniform)

        # Pre-allocate the inverse-CDF matrix once and fill column-wise so the
        # downstream DataFrame is constructed from a contiguous buffer instead
        # of dict-of-arrays.
        tickers = list(self.returns.columns)
        n_tickers = len(tickers)
        ppf_matrix = np.empty((n_simulations, n_tickers), dtype=np.float64)

        valid_mask = np.zeros(n_tickers, dtype=bool)
        for i, ticker in enumerate(tickers):
            marginal_info = self.fit_marginals_.get(ticker)
            if (
                marginal_info
                and marginal_info["dist"] is not None
                and marginal_info["params"] is not None
            ):
                dist = marginal_info["dist"]
                params = marginal_info["params"]
                try:
                    ppf_matrix[:, i] = dist.ppf(simulated_uniform[:, i], *params)
                    valid_mask[i] = True
                except Exception as e:
                    logger.warning(f"PPF failed for {ticker}: {e}")
                    ppf_matrix[:, i] = np.nan
            else:
                ppf_matrix[:, i] = np.nan

        if not valid_mask.any():
            logger.warning("Simulation yielded empty DataFrame after NaNs.")
            return pd.DataFrame({"simulated_returns": []})

        valid_tickers = [t for t, ok in zip(tickers, valid_mask, strict=True) if ok]
        simulated_returns_df = pd.DataFrame(
            ppf_matrix[:, valid_mask], columns=valid_tickers
        ).dropna()

        if simulated_returns_df.empty:
            logger.warning("Simulation yielded empty DataFrame after NaNs.")
            return pd.DataFrame({"simulated_returns": []})

        aligned_weights = self.weights.reindex(simulated_returns_df.columns).fillna(0.0)
        portfolio_returns_generated: pd.Series = simulated_returns_df.dot(aligned_weights)

        return pd.DataFrame({"simulated_returns": portfolio_returns_generated})

    def plot_marginal_fits(self):
        """Plot fitted marginal distributions against historical data."""
        if self.returns.empty or not self.fit_marginals_:
            logger.warning("Return data or marginal fits missing for plotting.")
            return

        n_assets = len(self.returns.columns)
        if n_assets == 0:
            return
        n_cols = 2
        n_rows = (n_assets + n_cols - 1) // n_cols

        fig: plt.Figure
        axes: NDArray[Axes]
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, n_rows * 5), squeeze=False)
        fig.suptitle("Fitted Marginal Distributions vs. Historical Returns", fontsize=16)
        axes_flat: NDArray[Axes] = axes.flatten()

        plot_index = 0
        for ticker in self.returns.columns:
            ax: Axes = axes_flat[plot_index]
            data = self.returns[ticker].dropna()

            if data.empty:
                ax.set_title(f"{ticker}: No Data")
                ax.set_xticks([])
                ax.set_yticks([])
                plot_index += 1
                continue

            sns.histplot(x=data, bins=50, stat="density", ax=ax, label="Historical")

            marginal_info = self.fit_marginals_.get(ticker)
            if (
                marginal_info
                and marginal_info["dist"] is not None
                and marginal_info["params"] is not None
            ):
                dist = marginal_info["dist"]
                params = marginal_info["params"]
                x_min, x_max = data.min(), data.max()
                if np.isclose(x_min, x_max):
                    x_min -= 0.1
                    x_max += 0.1
                x_range: NDArray[np.float64] = np.linspace(x_min, x_max, 200)
                try:
                    pdf_values = dist.pdf(x_range, *params)
                    ax.plot(x_range, pdf_values, "r-", lw=2, label="Fitted PDF")
                except Exception as e:
                    logger.warning(f"PDF plot failed for {ticker}: {e}")
                    ax.plot([], [], "r-", lw=2, label="Fit Error")
            else:
                ax.plot([], [], "r--", label="Fit Failed")

            ax.set_title(f"Fit for {ticker}")
            ax.legend()
            plot_index += 1

        for j in range(plot_index, len(axes_flat)):
            fig.delaxes(axes_flat[j])

        fig.tight_layout(rect=(0, 0.03, 1, 0.96))
        plt.show()

    def plot_copula_dependence(self):
        """Visualise pairwise dependence structure in uniform (copula) space."""
        if self.uniform_returns is None or self.uniform_returns.empty:
            logger.warning("Uniform returns unavailable/empty; cannot plot dependence.")
            return

        g: Any = sns.pairplot(
            self.uniform_returns,
            kind="scatter",
            diag_kind="kde",
            plot_kws={"alpha": 0.4},
        )
        if hasattr(g, "figure"):
            g.figure.suptitle("Pairwise Dependence Structure (Uniform Space)", y=1.02)
        plt.show()


def run_historical_simulation(
    returns: pd.DataFrame,
    weights: pd.Series,
    n_simulations: int = 10000,
    *,
    seed: int | None = None,
) -> pd.DataFrame:
    """Simulate portfolio returns by resampling historical daily returns.

    Performs Monte Carlo simulation by drawing samples with replacement
    from the observed historical portfolio return distribution.
    Assumes IID returns.

    Args:
        returns (pd.DataFrame): Historical asset returns.
        weights (pd.Series): Portfolio weights.
        n_simulations (int): Number of simulated days.
        seed (int | None): Seed for the resampling generator. Pass an int for
            reproducible draws. Defaults to None (non-deterministic).

    Returns:
        pd.DataFrame: Simulated portfolio returns ('simulated_returns').
    """
    if returns.empty or weights.empty:
        return pd.DataFrame({"simulated_returns": []})

    aligned_weights = weights.reindex(returns.columns).fillna(0.0)
    historical_portfolio_returns: pd.Series = returns.dot(aligned_weights)

    valid_historical_returns = historical_portfolio_returns.dropna()
    if valid_historical_returns.empty:
        return pd.DataFrame({"simulated_returns": []})

    rng = np.random.default_rng(seed)
    simulated_returns_array = rng.choice(
        valid_historical_returns.to_numpy(), size=n_simulations, replace=True
    )
    return pd.DataFrame({"simulated_returns": simulated_returns_array})

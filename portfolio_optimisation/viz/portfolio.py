from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from numpy.typing import NDArray
from pypfopt import EfficientFrontier
from pypfopt import plotting as pypfopt_plotting
from scipy.cluster.hierarchy import dendrogram

from portfolio_optimisation.optim.hrp import HRPModel


class PortfolioVisualiser:
    """Visualise portfolio allocation and risk characteristics.

    Encapsulates plotting logic for comparing different portfolio
    strategies, including efficient frontiers, weight allocations, and
    Hierarchical Risk Parity (HRP) clustering.
    """

    def __init__(
        self,
        mu: pd.Series | NDArray[np.float64],
        s: pd.DataFrame | NDArray[np.float64],
        all_weights: dict[str, pd.Series | dict[str, float]],
    ):
        """Initialise the visualiser.

        Args:
            mu: Expected annual returns.
            s: Annual covariance matrix.
            all_weights: Map of portfolio name -> weights (Series or dict).
        """
        self.mu = mu
        self.s = s
        self.all_weights = all_weights

    def plot_efficient_frontier(self):
        """Plot the efficient frontier and overlay portfolio risk/return points."""
        ef = EfficientFrontier(self.mu, self.s)
        _fig: Figure
        ax: Axes
        _fig, ax = plt.subplots(figsize=(10, 6))
        pypfopt_plotting.plot_efficient_frontier(ef, ax=ax, show_assets=False)

        for name, weights_data in self.all_weights.items():
            weights: pd.Series
            if isinstance(weights_data, dict):
                weights = pd.Series(weights_data)
            elif isinstance(weights_data, pd.Series):
                weights = weights_data
            else:
                continue

            ret = (weights * self.mu).sum()
            vol = np.sqrt(np.dot(weights.T, np.dot(self.s, weights)))
            ax.scatter(vol, ret, marker="x", s=120, label=name)

        ax.set_title("Efficient Frontier with Portfolio Comparisons")
        ax.legend()
        plt.tight_layout()
        plt.show()

    def plot_comparative_weights(self):
        """Bar chart comparing asset weights across portfolios."""
        weights_df = pd.DataFrame(self.all_weights)
        ax: Axes = weights_df.plot(kind="bar", figsize=(12, 7))
        ax.set_title("Portfolio Weight Comparison")
        ax.set_ylabel("Weight")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

    def plot_dendrogram(self, hrp_model: HRPModel, **kwargs: Any):
        """Plot the dendrogram from the HRPModel's hierarchical clustering."""
        if hrp_model.linkage_matrix is None:
            raise RuntimeError("HRP model must be optimized first.")

        fig_size = kwargs.get("figsize", (10, 7))
        _fig: Figure
        ax: Axes
        _fig, ax = plt.subplots(figsize=fig_size)
        ax.set_title("Hierarchical Clustering Dendrogram")

        labels = hrp_model.returns.columns.tolist()

        dendrogram(hrp_model.linkage_matrix, labels=labels, ax=ax)
        ax.set_ylabel("Distance")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.show()

    def plot_correlation_matrix(self, hrp_model: HRPModel, ordered: bool = False, **kwargs: Any):
        """Heatmap of the asset correlation matrix (raw or quasi-diagonalised)."""
        corr_matrix = pd.DataFrame(
            np.corrcoef(hrp_model.cov_matrix, rowvar=False),
            columns=hrp_model.returns.columns,
            index=hrp_model.returns.columns,
        )
        title = "Correlation Matrix"

        if ordered:
            if not hrp_model.ordered_tickers:
                raise RuntimeError("HRP model must be optimized to get ordered tickers.")
            corr_matrix = corr_matrix.loc[hrp_model.ordered_tickers, hrp_model.ordered_tickers]
            title = "Quasi-Diagonalised Correlation Matrix"

        fig_size = kwargs.get("figsize", (8, 8))
        fig: Figure
        ax: Axes
        fig, ax = plt.subplots(figsize=fig_size)
        cmap = kwargs.get("cmap", "coolwarm")
        im = ax.imshow(corr_matrix, cmap=cmap)

        ax.set_title(title)
        fig.colorbar(im)

        ax.set_xticks(np.arange(len(corr_matrix)), labels=corr_matrix.columns.tolist())
        ax.set_yticks(np.arange(len(corr_matrix)), labels=corr_matrix.index.tolist())
        plt.setp(
            ax.get_xticklabels(),
            rotation=90,
            ha="right",
            rotation_mode="anchor",
        )
        fig.tight_layout()
        plt.show()

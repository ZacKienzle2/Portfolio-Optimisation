"""Matplotlib visualisations for portfolio allocation and clustering.

Every method applies the shared project style, returns the ``(Figure, Axes)``
pair for testing and composition, and optionally writes the figure to disk.
No method calls ``plt.show``; rendering is the caller's choice.
"""

from __future__ import annotations

from pathlib import Path
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
from portfolio_optimisation.viz.style import (
    DIVERGING_CMAP,
    configure_style,
    save_figure,
)


class PortfolioVisualiser:
    """Visualise portfolio allocation and risk characteristics.

    Encapsulates plotting logic for comparing different portfolio strategies,
    including efficient frontiers, weight allocations, and Hierarchical Risk
    Parity (HRP) clustering.
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

    def plot_efficient_frontier(
        self, *, ax: Axes | None = None, save_path: str | Path | None = None
    ) -> tuple[Figure, Axes]:
        """Plot the efficient frontier and overlay portfolio risk/return points."""
        configure_style()
        fig, axis = _figure(ax)
        ef = EfficientFrontier(self.mu, self.s)
        pypfopt_plotting.plot_efficient_frontier(ef, ax=axis, show_assets=False)

        for name, weights_data in self.all_weights.items():
            weights: pd.Series = (
                pd.Series(weights_data) if isinstance(weights_data, dict) else weights_data
            )
            ret = (weights * self.mu).sum()
            vol = np.sqrt(np.dot(weights.T, np.dot(self.s, weights)))
            axis.scatter(vol, ret, marker="X", s=130, edgecolor="white", zorder=5, label=name)

        axis.set_title("Efficient Frontier with Portfolio Comparisons")
        axis.set_xlabel("Annualised Volatility")
        axis.set_ylabel("Annualised Return")
        axis.legend()
        _maybe_save(fig, save_path)
        return fig, axis

    def plot_comparative_weights(
        self, *, ax: Axes | None = None, save_path: str | Path | None = None
    ) -> tuple[Figure, Axes]:
        """Bar chart comparing asset weights across portfolios."""
        configure_style()
        fig, axis = _figure(ax)
        weights_df = pd.DataFrame(self.all_weights)
        weights_df.plot(kind="bar", ax=axis, width=0.8)
        axis.set_title("Portfolio Weight Comparison")
        axis.set_ylabel("Weight")
        axis.set_xlabel("Asset")
        axis.tick_params(axis="x", rotation=45)
        _maybe_save(fig, save_path)
        return fig, axis

    def plot_dendrogram(
        self,
        hrp_model: HRPModel,
        *,
        ax: Axes | None = None,
        save_path: str | Path | None = None,
        **kwargs: Any,
    ) -> tuple[Figure, Axes]:
        """Plot the dendrogram from the HRPModel's hierarchical clustering."""
        if hrp_model.linkage_matrix is None:
            raise RuntimeError("HRP model must be optimized first.")
        configure_style()
        fig, axis = _figure(ax, figsize=kwargs.get("figsize", (10, 6)))
        axis.set_title("Hierarchical Clustering Dendrogram")

        labels = hrp_model.returns.columns.tolist()
        dendrogram(hrp_model.linkage_matrix, labels=labels, ax=axis, color_threshold=0.0)
        axis.set_ylabel("Cluster Distance")
        axis.tick_params(axis="x", rotation=45)
        _maybe_save(fig, save_path)
        return fig, axis

    def plot_correlation_matrix(
        self,
        hrp_model: HRPModel,
        ordered: bool = False,
        *,
        ax: Axes | None = None,
        save_path: str | Path | None = None,
        **kwargs: Any,
    ) -> tuple[Figure, Axes]:
        """Heatmap of the asset correlation matrix (raw or quasi-diagonalised)."""
        configure_style()
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

        fig, axis = _figure(ax, figsize=kwargs.get("figsize", (7.5, 6.5)))
        cmap = kwargs.get("cmap", DIVERGING_CMAP)
        image = axis.imshow(corr_matrix, cmap=cmap, vmin=-1.0, vmax=1.0)

        labels = corr_matrix.columns.tolist()
        axis.set_xticks(np.arange(len(labels)), labels=labels)
        axis.set_yticks(np.arange(len(labels)), labels=labels)
        plt.setp(axis.get_xticklabels(), rotation=90, ha="center")

        if len(labels) <= 15:
            for i in range(len(labels)):
                for j in range(len(labels)):
                    value = corr_matrix.iloc[i, j]
                    axis.text(
                        j,
                        i,
                        f"{value:.2f}",
                        ha="center",
                        va="center",
                        fontsize=7.5,
                        color="white" if abs(value) > 0.5 else "black",
                    )

        axis.set_title(title)
        fig.colorbar(image, ax=axis, fraction=0.046, pad=0.04, label="Correlation")
        _maybe_save(fig, save_path)
        return fig, axis


def _figure(
    ax: Axes | None, *, figsize: tuple[float, float] | None = None
) -> tuple[Figure, Axes]:
    """Return ``(fig, ax)``, creating a styled figure when ``ax`` is None."""
    if ax is not None:
        return ax.get_figure(), ax  # type: ignore[return-value]
    fig, axis = plt.subplots(figsize=figsize)
    return fig, axis


def _maybe_save(fig: Figure, save_path: str | Path | None) -> None:
    if save_path is not None:
        save_figure(fig, save_path)

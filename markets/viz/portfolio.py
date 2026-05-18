from typing import Any, Dict, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from numpy.typing import NDArray
from pypfopt import EfficientFrontier
from pypfopt import plotting as pypfopt_plotting
from scipy.cluster.hierarchy import dendrogram

from markets.optim.hrp import HRPModel


class PortfolioVisualiser:
    """Visualise portfolio allocation and risk characteristics.

    Encapsulates plotting logic for comparing different portfolio
    strategies, including efficient frontiers, weight allocations, and
    Hierarchical Risk Parity (HRP) clustering.
    """

    def __init__(
        self,
        mu: Union[pd.Series, NDArray[np.float64]],
        s: Union[pd.DataFrame, NDArray[np.float64]],
        allWeights: Dict[str, Union[pd.Series, Dict[str, float]]],
    ):
        """Initialise the visualiser.

        Args:
            mu: Expected annual returns.
            s: Annual covariance matrix.
            allWeights: Map of portfolio name -> weights (Series or dict).
        """
        self.mu = mu
        self.s = s
        self.allWeights = allWeights

    def plotEfficientFrontier(self):
        """Plot the efficient frontier and overlay portfolio risk/return points."""
        ef = EfficientFrontier(self.mu, self.s)
        fig: Figure
        ax: Axes
        fig, ax = plt.subplots(figsize=(10, 6))
        pypfopt_plotting.plot_efficient_frontier(ef, ax=ax, show_assets=False)

        for name, weightsData in self.allWeights.items():
            weights: pd.Series
            if isinstance(weightsData, dict):
                weights = pd.Series(weightsData)
            elif isinstance(weightsData, pd.Series):
                weights = weightsData
            else:
                continue

            ret = (weights * self.mu).sum()
            vol = np.sqrt(np.dot(weights.T, np.dot(self.s, weights)))
            ax.scatter(vol, ret, marker="x", s=120, label=name)

        ax.set_title("Efficient Frontier with Portfolio Comparisons")
        ax.legend()
        plt.tight_layout()
        plt.show()

    def plotComparativeWeights(self):
        """Bar chart comparing asset weights across portfolios."""
        weightsDf = pd.DataFrame(self.allWeights)
        ax: Axes = weightsDf.plot(kind="bar", figsize=(12, 7))
        ax.set_title("Portfolio Weight Comparison")
        ax.set_ylabel("Weight")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

    def plotDendrogram(self, hrpModel: HRPModel, **kwargs: Any):
        """Plot the dendrogram from the HRPModel's hierarchical clustering."""
        if hrpModel.linkageMatrix is None:
            raise RuntimeError("HRP model must be optimized first.")

        figSize = kwargs.get("figsize", (10, 7))
        fig: Figure
        ax: Axes
        fig, ax = plt.subplots(figsize=figSize)
        ax.set_title("Hierarchical Clustering Dendrogram")

        labels = hrpModel.returns.columns.tolist()

        dendrogram(hrpModel.linkageMatrix, labels=labels, ax=ax)
        ax.set_ylabel("Distance")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.show()

    def plotCorrelationMatrix(
        self, hrpModel: HRPModel, ordered: bool = False, **kwargs: Any
    ):
        """Heatmap of the asset correlation matrix (raw or quasi-diagonalised)."""
        corrMatrix = pd.DataFrame(
            np.corrcoef(hrpModel.covMatrix, rowvar=False),
            columns=hrpModel.returns.columns,
            index=hrpModel.returns.columns,
        )
        title = "Correlation Matrix"

        if ordered:
            if not hrpModel.orderedTickers:
                raise RuntimeError(
                    "HRP model must be optimized to get ordered tickers."
                )
            corrMatrix = corrMatrix.loc[
                hrpModel.orderedTickers, hrpModel.orderedTickers
            ]
            title = "Quasi-Diagonalised Correlation Matrix"

        figSize = kwargs.get("figsize", (8, 8))
        fig: Figure
        ax: Axes
        fig, ax = plt.subplots(figsize=figSize)
        cmap = kwargs.get("cmap", "coolwarm")
        im = ax.imshow(corrMatrix, cmap=cmap)

        ax.set_title(title)
        fig.colorbar(im)

        ax.set_xticks(np.arange(len(corrMatrix)), labels=corrMatrix.columns.tolist())
        ax.set_yticks(np.arange(len(corrMatrix)), labels=corrMatrix.index.tolist())
        plt.setp(
            ax.get_xticklabels(),
            rotation=90,
            ha="right",
            rotation_mode="anchor",
        )
        fig.tight_layout()
        plt.show()

# Portfolio_Theory/plotting.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from numpy.typing import NDArray
from typing import Dict, Any, Union
from pypfopt import EfficientFrontier, plotting as pypfopt_plotting  # type: ignore
from scipy.cluster.hierarchy import dendrogram  # type: ignore

# The HRPModel class is needed for type hinting
from portfolio import HRPModel


class PortfolioVisualiser:
    """Provides methods to visualise portfolio allocation and risk characteristics.

    This class encapsulates plotting logic for comparing different portfolio
    strategies, including efficient frontiers, weight allocations, and
    Hierarchical Risk Parity (HRP) clustering.
    """

    def __init__(
        self,
        mu: Union[pd.Series, NDArray[np.float64]],
        s: Union[pd.DataFrame, NDArray[np.float64]],
        allWeights: Dict[str, Union[pd.Series, Dict[str, float]]],
    ):
        """Initialise the visualiser with expected returns, covariance, and weights.

        Args:
            mu (Union[pd.Series, NDArray[np.float64]]): Expected annual returns.
            s (Union[pd.DataFrame, NDArray[np.float64]]): Annual covariance matrix.
            allWeights (Dict[str, Union[pd.Series, Dict[str, float]]]):
                A dictionary mapping portfolio names (str) to their
                corresponding weights (as a Series or dict).
        """
        self.mu = mu
        self.s = s
        self.allWeights = allWeights

    def plotEfficientFrontier(self):
        """Plot the efficient frontier and overlay portfolio risk/return points.

        Generates the Markowitz efficient frontier based on the provided
        mu and S, then scatters the risk/return points for all portfolios
        stored in `self.allWeights` on the same axes for comparison.
        """
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
                continue  # Skip unrecognised type

            ret = (weights * self.mu).sum()
            vol = np.sqrt(np.dot(weights.T, np.dot(self.s, weights)))
            ax.scatter(vol, ret, marker="x", s=120, label=name)

        ax.set_title("Efficient Frontier with Portfolio Comparisons")
        ax.legend()
        plt.tight_layout()
        plt.show()

    def plotComparativeWeights(self):
        """Generate a bar chart comparing asset weights across portfolios.

        Creates a grouped bar chart where each group represents an asset,
        and bars within the group show the weight allocated by each
        portfolio strategy in `self.allWeights`.
        """
        weightsDf = pd.DataFrame(self.allWeights)
        ax: Axes = weightsDf.plot(kind="bar", figsize=(12, 7))
        ax.set_title("Portfolio Weight Comparison")
        ax.set_ylabel("Weight")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

    def plotDendrogram(self, hrpModel: HRPModel, **kwargs: Any):
        """Plot the dendrogram from the HRPModel's hierarchical clustering.

        Visualises the asset cluster hierarchy derived from the HRP
        optimisation process.

        Args:
            hrpModel (HRPModel): An *optimised* HRPModel instance containing
                                 the linkage matrix.
            **kwargs (Any): Additional keyword arguments passed to
                            `plt.subplots` (e.g., `figsize`).

        Raises:
            RuntimeError: If the HRP model has not been optimised
                          (i.e., `linkageMatrix` is None).
        """
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
        """Plot the asset correlation matrix as a heatmap.

        Can display the standard correlation matrix or the quasi-diagonalised
        matrix resulting from the HRP clustering.

        Args:
            hrpModel (HRPModel): An HRPModel instance containing the covariance
                                 matrix and (if ordered) ordered tickers.
            ordered (bool, optional): If True, plot the quasi-diagonalised
                                      matrix. Defaults to False.
            **kwargs (Any): Additional keyword arguments (e.g., `figsize`, `cmap`).

        Raises:
            RuntimeError: If `ordered` is True but the model has not been
                          optimised to produce ordered tickers.
        """
        # Calculate correlation from the HRP model's covariance matrix
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

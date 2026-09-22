"""Hierarchical Equal Risk Contribution allocation.

HERC keeps the HRP correlation-distance dendrogram but replaces the inverse
variance split with an equal risk contribution split at each cluster boundary.
The split between two child clusters with risks ``sigma_L`` and ``sigma_R``
allocates

    alpha = sigma_R / (sigma_L + sigma_R)

so the contributions ``alpha * sigma_L`` and ``(1 - alpha) * sigma_R`` are
equal. Within each cluster the weights are inverse-variance as in HRP. The
risk is the cluster volatility by default, or its expected shortfall at level
``cvar_alpha`` when ``risk_measure="cvar"``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import numpy as np
import pandas as pd

from portfolio_optimisation.optim.clustering import (
    LinkageMethod,
    bisection,
    correlation_linkage,
    inverse_variance_weights,
    seriation,
)
from portfolio_optimisation.optim.shrinkage import linear_shrinkage_covariance

if TYPE_CHECKING:
    from numpy.typing import NDArray

RiskMeasure = Literal["variance", "cvar"]


def _cluster_risk(
    risk_measure: RiskMeasure,
    covariance: NDArray[np.float64],
    returns: NDArray[np.float64],
    cvar_alpha: float,
) -> float:
    """Risk of a cluster's inverse-variance portfolio, which drives the split."""
    weights = inverse_variance_weights(covariance)
    if risk_measure == "variance":
        return float(np.sqrt(weights @ covariance @ weights))
    portfolio = returns @ weights
    var = np.quantile(portfolio, cvar_alpha)
    return float(-portfolio[portfolio <= var].mean())


def herc_weights(
    returns: pd.DataFrame,
    *,
    cov_matrix: pd.DataFrame | None = None,
    linkage_method: LinkageMethod = "ward",
    risk_measure: RiskMeasure = "variance",
    cvar_alpha: float = 0.05,
) -> pd.Series:
    """Compute HERC long-only weights summing to one.

    Args:
        returns: Historical asset returns.
        cov_matrix: Covariance driving the distance matrix and the in-cluster
            inverse-variance weights. Defaults to Ledoit-Wolf shrinkage.
        linkage_method: SciPy linkage criterion.
        risk_measure: ``"variance"`` for cluster volatility or ``"cvar"``.
        cvar_alpha: Tail level when ``risk_measure="cvar"``.

    Returns:
        Long-only HERC weights indexed by ticker.
    """
    cov_df = linear_shrinkage_covariance(returns) if cov_matrix is None else cov_matrix
    tickers = list(cov_df.columns)
    covariance = cov_df.to_numpy(dtype=np.float64)
    observations = returns[tickers].to_numpy(dtype=np.float64)
    order = seriation(correlation_linkage(covariance, linkage_method))

    weights = np.ones(order.size, dtype=np.float64)
    for left, right in bisection(order.size):
        idx_left, idx_right = order[left], order[right]
        sigma_left = _cluster_risk(
            risk_measure,
            covariance[np.ix_(idx_left, idx_left)],
            observations[:, idx_left],
            cvar_alpha,
        )
        sigma_right = _cluster_risk(
            risk_measure,
            covariance[np.ix_(idx_right, idx_right)],
            observations[:, idx_right],
            cvar_alpha,
        )
        total = sigma_left + sigma_right
        alpha = sigma_right / total if total > 0 else 0.5
        weights[left] *= alpha
        weights[right] *= 1.0 - alpha

    return pd.Series(weights, index=[tickers[i] for i in order]).reindex(tickers)


class HERCModel:
    """Object wrapper around :func:`herc_weights`.

    Args:
        returns: Historical asset returns.
        cov_matrix: Covariance to use instead of Ledoit-Wolf shrinkage.
        linkage_method: SciPy linkage criterion.
        risk_measure: ``"variance"`` or ``"cvar"``.
        cvar_alpha: Tail level when ``risk_measure="cvar"``.
    """

    def __init__(
        self,
        returns: pd.DataFrame,
        *,
        cov_matrix: pd.DataFrame | None = None,
        linkage_method: LinkageMethod = "ward",
        risk_measure: RiskMeasure = "variance",
        cvar_alpha: float = 0.05,
    ) -> None:
        self.returns = returns
        self.cov_matrix = cov_matrix
        self.linkage_method: LinkageMethod = linkage_method
        self.risk_measure: RiskMeasure = risk_measure
        self.cvar_alpha = cvar_alpha
        self.weights: pd.Series = pd.Series(dtype=np.float64)

    def optimise(self) -> pd.Series:
        """Solve the HERC programme and cache the resulting weights.

        Returns:
            Long-only HERC weights indexed by ticker.
        """
        self.weights = herc_weights(
            self.returns,
            cov_matrix=self.cov_matrix,
            linkage_method=self.linkage_method,
            risk_measure=self.risk_measure,
            cvar_alpha=self.cvar_alpha,
        )
        return self.weights

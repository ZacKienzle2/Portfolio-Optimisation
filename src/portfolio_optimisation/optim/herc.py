"""Hierarchical Equal Risk Contribution allocation.

HERC (Raffinot, 2018) proceeds in four steps. It clusters the assets on the
correlation distance, chooses the number of clusters ``K`` by the gap statistic
of Tibshirani, Walther and Hastie (2001), divides capital top-down along the
dendrogram's own splits until the ``K`` clusters are reached, and holds each
cluster by naive risk parity. A split between two child clusters of risk
``sigma_L`` and ``sigma_R`` allocates

    alpha = sigma_R / (sigma_L + sigma_R)

to the left child, so the two contributions ``alpha sigma_L`` and
``(1 - alpha) sigma_R`` are equal. Raffinot prints ``RC_1 / (RC_1 + RC_2)``,
which would give the riskier cluster more capital and contradict the equal
contribution the step is named for, so the allocation follows the condition.
Naive risk parity weights each asset by the reciprocal of its own risk, the
equal-risk-contribution portfolio when correlations within a cluster are
ignored. A cluster's risk is that of its naive risk parity portfolio, its
volatility or, with ``risk_measure="cvar"``, its expected shortfall.

The dendrogram fixes where capital divides, where recursive bisection at the
midpoint of the seriated order cuts clusters in two by count alone.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import to_tree

from portfolio_optimisation.optim.clustering import (
    LinkageMethod,
    correlation_embedding,
    correlation_linkage,
    gap_statistic,
)
from portfolio_optimisation.optim.shrinkage import linear_shrinkage_covariance

if TYPE_CHECKING:
    from numpy.typing import NDArray

RiskMeasure = Literal["variance", "cvar"]


def _expected_shortfall(returns: NDArray[np.float64], alpha: float) -> NDArray[np.float64]:
    """Mean of the worst ``ceil(alpha T)`` returns of each column, as a positive loss."""
    tail = max(1, int(np.ceil(alpha * returns.shape[0])))
    return -np.sort(returns, axis=0)[:tail].mean(axis=0)


def _portfolio_risk(
    risk_measure: RiskMeasure,
    members: NDArray[np.intp],
    weights: NDArray[np.float64],
    covariance: NDArray[np.float64],
    returns: NDArray[np.float64],
    cvar_alpha: float,
) -> float:
    """Risk of the portfolio holding ``weights`` in the assets ``members``."""
    if risk_measure == "variance":
        return float(np.sqrt(weights @ covariance[np.ix_(members, members)] @ weights))
    return float(_expected_shortfall(returns[:, members] @ weights[:, None], cvar_alpha)[0])


def herc_weights(
    returns: pd.DataFrame,
    *,
    cov_matrix: pd.DataFrame | None = None,
    linkage_method: LinkageMethod = "ward",
    risk_measure: RiskMeasure = "variance",
    cvar_alpha: float = 0.05,
    max_clusters: int = 10,
    n_references: int = 20,
    seed: int | None = 0,
) -> pd.Series:
    """Compute HERC long-only weights summing to one.

    Args:
        returns: Historical asset returns.
        cov_matrix: Covariance driving the distance matrix and the volatilities.
            Defaults to Ledoit-Wolf shrinkage.
        linkage_method: SciPy linkage criterion.
        risk_measure: ``"variance"`` for volatility or ``"cvar"`` for expected
            shortfall.
        cvar_alpha: Tail level when ``risk_measure="cvar"``.
        max_clusters: Largest number of clusters the gap statistic considers.
        n_references: Reference samples behind the gap statistic.
        seed: Seed for the reference samples, which makes the weights
            reproducible.

    Returns:
        Long-only HERC weights indexed by ticker.
    """
    cov_df = linear_shrinkage_covariance(returns) if cov_matrix is None else cov_matrix
    tickers = list(cov_df.columns)
    covariance = cov_df.to_numpy(dtype=np.float64)
    observations = returns[tickers].to_numpy(dtype=np.float64)
    linkage_matrix = correlation_linkage(covariance, linkage_method)
    n_clusters = gap_statistic(
        correlation_embedding(covariance),
        linkage_matrix,
        linkage_method,
        max_clusters=max_clusters,
        n_references=n_references,
        seed=seed,
    )

    if risk_measure == "variance":
        asset_risk = np.sqrt(np.clip(np.diag(covariance), 1e-24, None))
    else:
        asset_risk = np.clip(_expected_shortfall(observations, cvar_alpha), 1e-12, None)
    inverse_risk = 1.0 / asset_risk

    _, nodes = to_tree(linkage_matrix, rd=True)
    first_split = len(nodes) - n_clusters + 1
    splits = linkage_matrix[linkage_matrix.shape[0] - n_clusters + 1 :, :2].astype(np.intp)
    clusters = [i for i in splits.ravel() if i < first_split] or [len(nodes) - 1]
    leaves_of = {
        i: np.asarray(nodes[i].pre_order(), dtype=np.intp) for i in (*splits.ravel(), *clusters)
    }
    weights = inverse_risk.copy()
    for cluster in clusters:
        weights[leaves_of[cluster]] /= inverse_risk[leaves_of[cluster]].sum()

    for left_id, right_id in splits:
        left, right = leaves_of[left_id], leaves_of[right_id]
        risk_left, risk_right = (
            _portfolio_risk(
                risk_measure,
                side,
                inverse_risk[side] / inverse_risk[side].sum(),
                covariance,
                observations,
                cvar_alpha,
            )
            for side in (left, right)
        )
        total = risk_left + risk_right
        share = risk_right / total if total > 0.0 else 0.5
        weights[left] *= share
        weights[right] *= 1.0 - share

    return pd.Series(weights, index=tickers)


class HERCModel:
    """Object wrapper around :func:`herc_weights`.

    Args:
        returns: Historical asset returns.
        cov_matrix: Covariance to use instead of Ledoit-Wolf shrinkage.
        linkage_method: SciPy linkage criterion.
        risk_measure: ``"variance"`` or ``"cvar"``.
        cvar_alpha: Tail level when ``risk_measure="cvar"``.
        seed: Seed for the gap statistic's reference samples.
    """

    def __init__(
        self,
        returns: pd.DataFrame,
        *,
        cov_matrix: pd.DataFrame | None = None,
        linkage_method: LinkageMethod = "ward",
        risk_measure: RiskMeasure = "variance",
        cvar_alpha: float = 0.05,
        seed: int | None = 0,
    ) -> None:
        self.returns = returns
        self.cov_matrix = cov_matrix
        self.linkage_method: LinkageMethod = linkage_method
        self.risk_measure: RiskMeasure = risk_measure
        self.cvar_alpha = cvar_alpha
        self.seed = seed
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
            seed=self.seed,
        )
        return self.weights

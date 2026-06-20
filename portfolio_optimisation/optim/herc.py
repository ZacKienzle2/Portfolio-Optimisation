"""Hierarchical Equal Risk Contribution allocation.

HERC keeps the HRP correlation-distance dendrogram but replaces the recursive
inverse-variance bisection with a *top-down Equal Risk Contribution* split at
each cluster boundary. The ERC split between two child clusters with standard
deviations sigma_L and sigma_R allocates

    alpha = sigma_R / (sigma_L + sigma_R)

so the resulting cluster contributions ``alpha * sigma_L`` and
``(1 - alpha) * sigma_R`` are equal. Within each leaf cluster, weights are
inverse-variance just like HRP. The risk metric is configurable via the
``risk_measure`` parameter; the default ``"variance"`` uses the baseline
approach, ``"cvar"`` switches to expected shortfall at level ``alpha=0.05``.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import squareform
from sklearn.covariance import ledoit_wolf

from portfolio_optimisation.optim.hrp import HRPModel

RiskMeasure = Literal["variance", "cvar"]


def _ledoit_wolf_cov(returns: pd.DataFrame) -> pd.DataFrame:
    cov, _ = ledoit_wolf(returns, assume_centered=False)
    return pd.DataFrame(cov, index=returns.columns, columns=returns.columns)


def _cluster_risk(
    risk_measure: RiskMeasure,
    weights: NDArray[np.float64],
    cov_sub: NDArray[np.float64],
    returns_sub: NDArray[np.float64],
    cvar_alpha: float,
) -> float:
    """Cluster-level risk used to drive the ERC split."""
    if risk_measure == "variance":
        return float(np.sqrt(weights @ cov_sub @ weights))
    portfolio = returns_sub @ weights
    var = float(np.quantile(portfolio, cvar_alpha))
    tail = portfolio[portfolio <= var]
    if tail.size == 0:
        return float(-var)
    return float(-tail.mean())


def _inv_var_weights(cov: NDArray[np.float64]) -> NDArray[np.float64]:
    """Inverse-variance weights inside a leaf cluster."""
    inv = 1.0 / np.clip(np.diag(cov), 1e-12, None)
    return inv / inv.sum()


def herc_weights(
    returns: pd.DataFrame,
    *,
    cov_matrix: pd.DataFrame | None = None,
    linkage_method: Literal["ward", "single", "complete", "average"] = "ward",
    risk_measure: RiskMeasure = "variance",
    cvar_alpha: float = 0.05,
) -> pd.Series:
    """Compute HERC long-only weights summing to 1.

    Args:
        returns (pd.DataFrame): Historical asset returns.
        cov_matrix (pd.DataFrame | None, optional): Covariance to drive the
            distance matrix and intra-leaf inverse-variance allocation.
            Defaults to Ledoit-Wolf shrinkage.
        linkage_method: scipy linkage method.
        risk_measure: ``"variance"`` (Raffinot baseline) or ``"cvar"``.
        cvar_alpha (float): Tail level when ``risk_measure="cvar"``.

    Returns:
        pd.Series: Long-only HERC weights summing to 1, indexed by ticker.
    """
    cov_df = _ledoit_wolf_cov(returns) if cov_matrix is None else cov_matrix
    tickers = list(cov_df.columns)
    cov = cov_df.to_numpy(dtype=np.float64)
    returns_arr = returns[tickers].to_numpy(dtype=np.float64)

    std = np.sqrt(np.diag(cov))
    std = np.where(std < 1e-12, 1.0, std)
    corr = cov / np.outer(std, std)
    distance = np.sqrt(np.clip((1.0 - corr) / 2.0, 0.0, 1.0))
    condensed = squareform(distance, checks=False)
    z = linkage(condensed, method=linkage_method)

    # Reuse HRP's quasi-diagonalisation to obtain a leaf ordering.
    helper = HRPModel(returns, cov_matrix=cov_df)
    helper.linkage_matrix = z
    sorted_indices = helper._get_quasi_diag(z)  # pyright: ignore[reportPrivateUsage]
    ordered_idx = np.array(sorted_indices, dtype=np.intp)
    n = ordered_idx.size
    weights = np.ones(n, dtype=np.float64)

    cluster_ranges: list[tuple[int, int]] = [(0, n)]
    while cluster_ranges:
        next_ranges: list[tuple[int, int]] = []
        for start, stop in cluster_ranges:
            if stop - start > 1:
                mid = start + (stop - start) // 2
                next_ranges.append((start, mid))
                next_ranges.append((mid, stop))
        cluster_ranges = next_ranges

        for i in range(0, len(cluster_ranges), 2):
            left = cluster_ranges[i]
            right = cluster_ranges[i + 1]
            idx_l = ordered_idx[left[0] : left[1]]
            idx_r = ordered_idx[right[0] : right[1]]
            cov_l = cov[np.ix_(idx_l, idx_l)]
            cov_r = cov[np.ix_(idx_r, idx_r)]
            w_l = _inv_var_weights(cov_l)
            w_r = _inv_var_weights(cov_r)
            sigma_l = _cluster_risk(risk_measure, w_l, cov_l, returns_arr[:, idx_l], cvar_alpha)
            sigma_r = _cluster_risk(risk_measure, w_r, cov_r, returns_arr[:, idx_r], cvar_alpha)
            total = sigma_l + sigma_r
            alpha = sigma_r / total if total > 0 else 0.5
            weights[left[0] : left[1]] *= alpha
            weights[right[0] : right[1]] *= 1.0 - alpha

    ordered_tickers = [tickers[i] for i in sorted_indices]
    series = pd.Series(weights, index=ordered_tickers)
    return series.reindex(tickers)


class HERCModel:
    """Object wrapper around :func:`herc_weights`."""

    def __init__(
        self,
        returns: pd.DataFrame,
        *,
        cov_matrix: pd.DataFrame | None = None,
        linkage_method: Literal["ward", "single", "complete", "average"] = "ward",
        risk_measure: RiskMeasure = "variance",
        cvar_alpha: float = 0.05,
    ) -> None:
        self.returns = returns
        self.cov_matrix = cov_matrix
        self.linkage_method = linkage_method
        self.risk_measure = risk_measure
        self.cvar_alpha = cvar_alpha
        self.weights: pd.Series = pd.Series(dtype=np.float64)

    def optimise(self) -> pd.Series:
        self.weights = herc_weights(
            self.returns,
            cov_matrix=self.cov_matrix,
            linkage_method=self.linkage_method,
            risk_measure=self.risk_measure,
            cvar_alpha=self.cvar_alpha,
        )
        return self.weights

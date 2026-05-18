"""Nested Clustered Optimisation.

Lopez de Prado (2016, 2020): instead of running Markowitz mean-variance on the
full universe (which collapses under estimation error), partition the universe
into hierarchical clusters, run intra-cluster MVO on each leaf, aggregate the
cluster returns, then run a final inter-cluster MVO on the cluster portfolios.

Algorithm (NCO):
    1.  Hierarchical clustering of the correlation distance d_ij = sqrt((1-r_ij)/2).
    2.  Cut the dendrogram into K flat clusters.
    3.  For each cluster c, solve  w_c = argmin w' Sigma_c w  with  1' w = 1.
    4.  Project cluster returns r_c = R_c w_c.
    5.  Build the reduced covariance Sigma_red of cluster returns and solve
        the inter-cluster MVO w_red on Sigma_red.
    6.  Final weight for asset i in cluster c is w_intra(i) * w_red(c).

This file exposes a thin :class:`NCOOptimiser` plus a functional ``nco_weights``
entry point. By default the covariance comes from :mod:`sklearn.covariance`
Ledoit-Wolf shrinkage; pass an RMT-denoised covariance via ``cov_matrix`` to
combine NCO with the upstream denoising step.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform
from sklearn.covariance import ledoit_wolf


def _ledoit_wolf_cov(returns: pd.DataFrame) -> pd.DataFrame:
    cov, _ = ledoit_wolf(returns, assume_centered=False)
    return pd.DataFrame(cov, index=returns.columns, columns=returns.columns)


def _min_var_weights(cov: NDArray[np.float64]) -> NDArray[np.float64]:
    """Closed-form minimum-variance weights for the long-only sub-problem.

    Uses ``w = Sigma^-1 1 / (1' Sigma^-1 1)`` then clips negatives to zero and
    renormalises so the result is feasible for the typical no-short constraint.
    Sufficient for NCO's leaf-level allocation; downstream code can extend to
    return-aware MVO by accepting an ``mu`` vector.
    """
    ones = np.ones(cov.shape[0])
    try:
        inv_sigma_ones = np.linalg.solve(cov, ones)
    except np.linalg.LinAlgError:
        inv_sigma_ones = np.linalg.lstsq(cov, ones, rcond=None)[0]
    weights = inv_sigma_ones / inv_sigma_ones.sum()
    weights = np.clip(weights, 0.0, None)
    s = weights.sum()
    if s <= 0:
        return np.ones_like(weights) / weights.size
    return weights / s


def _cluster_labels(
    cov: NDArray[np.float64], n_clusters: int, linkage_method: str
) -> NDArray[np.intp]:
    """Hierarchical clustering on the correlation-distance matrix."""
    std = np.sqrt(np.diag(cov))
    std = np.where(std < 1e-12, 1.0, std)
    corr = cov / np.outer(std, std)
    distance = np.sqrt(np.clip((1.0 - corr) / 2.0, 0.0, 1.0))
    condensed = squareform(distance, checks=False)
    z = linkage(condensed, method=linkage_method)
    labels = fcluster(z, t=n_clusters, criterion="maxclust")
    return np.asarray(labels, dtype=np.intp)


def nco_weights(
    returns: pd.DataFrame,
    *,
    n_clusters: int | None = None,
    cov_matrix: pd.DataFrame | None = None,
    linkage_method: Literal["ward", "single", "complete", "average"] = "ward",
) -> pd.Series:
    """Compute Nested Clustered Optimisation weights.

    Args:
        returns (pd.DataFrame): Historical asset returns; columns are tickers.
        n_clusters (int | None, optional): Flat-cluster count for fcluster's
            ``maxclust`` criterion. Defaults to ``round(sqrt(N))``.
        cov_matrix (pd.DataFrame | None, optional): Externally estimated
            covariance (e.g. RMT-denoised). Defaults to Ledoit-Wolf shrinkage.
        linkage_method: scipy linkage method.

    Returns:
        pd.Series: Long-only NCO weights summing to 1, indexed by ticker.
    """
    cov_df = _ledoit_wolf_cov(returns) if cov_matrix is None else cov_matrix
    tickers = list(cov_df.columns)
    cov = cov_df.to_numpy(dtype=np.float64)
    n = cov.shape[0]

    if n_clusters is None:
        n_clusters = max(1, round(float(np.sqrt(n))))
    n_clusters = max(1, min(n_clusters, n))

    labels = _cluster_labels(cov, n_clusters, linkage_method)

    final = np.zeros(n)
    intra_returns: list[pd.Series] = []
    cluster_ids = np.unique(labels)
    cluster_index_map: dict[int, NDArray[np.intp]] = {}

    for cid in cluster_ids:
        idx = np.where(labels == cid)[0].astype(np.intp)
        cluster_index_map[int(cid)] = idx
        cov_c = cov[np.ix_(idx, idx)]
        w_c = _min_var_weights(cov_c)
        final[idx] = w_c
        cluster_returns = returns.iloc[:, idx].to_numpy(dtype=np.float64) @ w_c
        intra_returns.append(
            pd.Series(cluster_returns, index=returns.index, name=f"c{int(cid)}")
        )

    reduced = pd.concat(intra_returns, axis=1)
    reduced_cov = _ledoit_wolf_cov(reduced)
    w_red = _min_var_weights(reduced_cov.to_numpy(dtype=np.float64))

    for k, cid in enumerate(cluster_ids):
        idx = cluster_index_map[int(cid)]
        final[idx] *= w_red[k]

    total = final.sum()
    if total > 0:
        final /= total
    return pd.Series(final, index=tickers)


class NCOOptimiser:
    """Object wrapper for repeated NCO allocations against the same data."""

    def __init__(
        self,
        returns: pd.DataFrame,
        *,
        cov_matrix: pd.DataFrame | None = None,
        linkage_method: Literal["ward", "single", "complete", "average"] = "ward",
    ) -> None:
        self.returns: pd.DataFrame = returns
        self.cov_matrix: pd.DataFrame = (
            cov_matrix if cov_matrix is not None else _ledoit_wolf_cov(returns)
        )
        self.linkage_method = linkage_method
        self.weights: pd.Series = pd.Series(dtype=np.float64)

    def optimise(self, *, n_clusters: int | None = None) -> pd.Series:
        """Run NCO and cache the resulting weights on ``self.weights``."""
        self.weights = nco_weights(
            self.returns,
            n_clusters=n_clusters,
            cov_matrix=self.cov_matrix,
            linkage_method=self.linkage_method,
        )
        return self.weights

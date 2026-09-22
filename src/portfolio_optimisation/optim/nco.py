"""Nested Clustered Optimisation.

Instead of running Markowitz mean-variance on the full universe (which collapses
under estimation error), partition the universe into hierarchical clusters, run
intra-cluster MVO on each leaf, aggregate the cluster returns, then run a final
inter-cluster MVO on the cluster portfolios.

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

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster

from portfolio_optimisation.optim.clustering import LinkageMethod, correlation_linkage
from portfolio_optimisation.optim.robust import minimum_variance_weights
from portfolio_optimisation.optim.shrinkage import linear_shrinkage_covariance


def nco_weights(
    returns: pd.DataFrame,
    *,
    n_clusters: int | None = None,
    cov_matrix: pd.DataFrame | None = None,
    linkage_method: LinkageMethod = "ward",
) -> pd.Series:
    """Compute Nested Clustered Optimisation weights.

    Args:
        returns: Historical asset returns, one column per ticker.
        n_clusters: Flat-cluster count for the ``maxclust`` criterion. Defaults to
            ``round(sqrt(N))``.
        cov_matrix: Externally estimated covariance, for example RMT-denoised. Defaults
            to Ledoit-Wolf shrinkage.
        linkage_method: SciPy linkage criterion.

    Returns:
        Long-only NCO weights summing to one, indexed by ticker.
    """
    cov_df = linear_shrinkage_covariance(returns) if cov_matrix is None else cov_matrix
    tickers = list(cov_df.columns)
    covariance = cov_df.to_numpy(dtype=np.float64)
    observations = returns[tickers].to_numpy(dtype=np.float64)
    n_assets = covariance.shape[0]
    if n_clusters is None:
        n_clusters = round(float(np.sqrt(n_assets)))
    n_clusters = max(1, min(n_clusters, n_assets))

    labels = fcluster(
        correlation_linkage(covariance, linkage_method), t=n_clusters, criterion="maxclust"
    )
    members = [np.flatnonzero(labels == label) for label in np.unique(labels)]
    intra = np.zeros(n_assets)
    for idx in members:
        intra[idx] = minimum_variance_weights(covariance[np.ix_(idx, idx)])

    cluster_returns = pd.DataFrame(
        np.column_stack([observations[:, idx] @ intra[idx] for idx in members])
    )
    inter = minimum_variance_weights(
        linear_shrinkage_covariance(cluster_returns).to_numpy(dtype=np.float64)
    )
    final = intra.copy()
    for weight, idx in zip(inter, members, strict=True):
        final[idx] *= weight
    return pd.Series(final / final.sum(), index=tickers)


class NCOOptimiser:
    """Object wrapper for repeated NCO allocations against the same data.

    Args:
        returns: Historical asset returns, one column per ticker.
        cov_matrix: Covariance to use instead of Ledoit-Wolf shrinkage.
        linkage_method: SciPy linkage criterion.
    """

    def __init__(
        self,
        returns: pd.DataFrame,
        *,
        cov_matrix: pd.DataFrame | None = None,
        linkage_method: LinkageMethod = "ward",
    ) -> None:
        self.returns: pd.DataFrame = returns
        self.cov_matrix: pd.DataFrame = (
            cov_matrix if cov_matrix is not None else linear_shrinkage_covariance(returns)
        )
        self.linkage_method: LinkageMethod = linkage_method
        self.weights: pd.Series = pd.Series(dtype=np.float64)

    def optimise(self, *, n_clusters: int | None = None) -> pd.Series:
        """Run NCO and cache the resulting weights on ``self.weights``.

        Args:
            n_clusters: Flat-cluster count. Defaults to ``round(sqrt(N))``.

        Returns:
            Long-only NCO weights indexed by ticker.
        """
        self.weights = nco_weights(
            self.returns,
            n_clusters=n_clusters,
            cov_matrix=self.cov_matrix,
            linkage_method=self.linkage_method,
        )
        return self.weights

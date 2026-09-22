"""Hierarchical clustering primitives shared by the tree-based allocators.

HRP, HERC and NCO all cluster assets on the correlation distance
``d_ij = sqrt((1 - rho_ij) / 2)``. HRP seriates the dendrogram leaves and splits
the seriated order recursively; HERC cuts the dendrogram at the number of
clusters the gap statistic of Tibshirani, Walther and Hastie (2001) selects.
The linkage, the cuts, the leaf order and the correlation come from SciPy and
statsmodels.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import numpy as np
from scipy.cluster.hierarchy import fcluster, leaves_list, linkage
from scipy.spatial.distance import pdist, squareform
from statsmodels.stats.moment_helpers import cov2corr

if TYPE_CHECKING:
    from collections.abc import Iterator

    from numpy.typing import NDArray

LinkageMethod = Literal["ward", "single", "complete", "average"]


def correlation_linkage(
    covariance: NDArray[np.float64], method: str = "ward"
) -> NDArray[np.float64]:
    """Agglomerate assets on the correlation distance of a covariance matrix.

    A zero-variance asset has an undefined correlation, which is read as zero so
    it sits at distance ``sqrt(1 / 2)`` from every other asset.

    Args:
        covariance: Symmetric ``(N, N)`` covariance matrix.
        method: Linkage criterion understood by ``scipy.cluster.hierarchy.linkage``.

    Returns:
        The ``(N - 1, 4)`` linkage matrix.
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        correlation = np.nan_to_num(cov2corr(np.asarray(covariance, dtype=np.float64)))
    distance = np.sqrt(np.clip((1.0 - correlation) / 2.0, 0.0, 1.0))
    return linkage(squareform(distance, checks=False), method=method)


def correlation_embedding(covariance: NDArray[np.float64]) -> NDArray[np.float64]:
    """Points whose Euclidean distances are the correlation distances.

    Classical multidimensional scaling of the correlation matrix ``C = Q L Q'``
    gives ``X = Q L^(1/2)``, whose rows satisfy ``|x_i - x_j|^2 = 2 (1 - rho_ij)``,
    four times the squared distance :func:`correlation_linkage` clusters on. Any
    of the linkage criteria here is equivariant to that scale, so clustering
    these points reproduces the same dendrogram.

    Args:
        covariance: Symmetric ``(N, N)`` covariance matrix.

    Returns:
        One row per asset, ``(N, N)``.
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        correlation = np.nan_to_num(cov2corr(np.asarray(covariance, dtype=np.float64)))
    np.fill_diagonal(correlation, 1.0)
    eigenvalues, eigenvectors = np.linalg.eigh(correlation)
    return eigenvectors * np.sqrt(np.clip(eigenvalues, 0.0, None))


def dendrogram_cuts(linkage_matrix: NDArray[np.float64], n_cuts: int) -> NDArray[np.intp]:
    """Labels of the dendrogram cut into ``1, ..., n_cuts`` clusters, one column each.

    SciPy's compiled ``fcluster`` cuts by merge height. For the monotone
    criteria here that is the cut into ``k`` clusters unless two merges tie in
    height, where it may return fewer. ``cut_tree`` cuts by merge order and is
    exact under ties, but walks the tree in Python and took three quarters of
    the gap statistic's time at two hundred assets.

    Args:
        linkage_matrix: SciPy linkage matrix.
        n_cuts: Largest number of clusters.

    Returns:
        Zero-based labels, one column per number of clusters.
    """
    return np.column_stack(
        [fcluster(linkage_matrix, k, criterion="maxclust") - 1 for k in range(1, n_cuts + 1)]
    )


def within_cluster_dispersion(
    points: NDArray[np.float64], labels: NDArray[np.intp]
) -> NDArray[np.float64]:
    """Pooled within-cluster sum of squares ``W_k`` for every column of cut labels.

    Column ``j`` of ``labels`` assigns the rows of ``points`` to at most
    ``j + 1`` clusters numbered from zero. For squared Euclidean distances
    Tibshirani, Walther and Hastie's ``W_k = sum_r D_r / (2 n_r)`` equals the
    total sum of squares less ``sum_r |sum of cluster r|^2 / n_r``, and every
    cut is summed by one matrix product over the stacked cluster indicators.

    Args:
        points: One row per item.
        labels: Cluster labels, one column per cut.

    Returns:
        ``W_k`` for each column.
    """
    n_cuts = labels.shape[1]
    sizes = np.arange(1, n_cuts + 1)
    offsets = np.concatenate([[0], np.cumsum(sizes)[:-1]])
    indicators = np.zeros((points.shape[0], sizes.sum()))
    rows = np.repeat(np.arange(points.shape[0]), n_cuts)
    indicators[rows, (labels + offsets).ravel()] = 1.0
    sums = indicators.T @ points
    counts = np.maximum(indicators.sum(axis=0), 1.0)
    between = np.add.reduceat(np.einsum("kp,kp->k", sums, sums) / counts, offsets)
    return np.einsum("ip,ip->", points, points) - between


def gap_statistic(
    points: NDArray[np.float64],
    linkage_matrix: NDArray[np.float64],
    method: str,
    *,
    max_clusters: int = 10,
    n_references: int = 20,
    seed: int | None = None,
) -> int:
    """Number of clusters by the gap statistic of Tibshirani, Walther and Hastie.

    ``Gap(k) = E*[log W_k] - log W_k`` compares the within-cluster dispersion of
    the dendrogram cut into ``k`` clusters with its expectation under a
    reference distribution without clusters, estimated from ``n_references``
    samples. The reference is uniform over the box aligned with the principal
    components of the centred points, their prescription (b), which they found
    the best of the reference choices. Rotating back to the original axes leaves
    the distances unchanged, so each sample is clustered in principal-component
    coordinates. The estimate is the smallest ``k`` with
    ``Gap(k) >= Gap(k + 1) - s_(k + 1)``, where
    ``s_k = sd_k sqrt(1 + 1 / B)`` carries the simulation error.

    Args:
        points: One row per item, clustered by Euclidean distance.
        linkage_matrix: Linkage of ``points`` under ``method``.
        method: Linkage criterion used for the reference samples.
        max_clusters: Largest number of clusters considered, at most one less than
            the number of items, where every item is its own cluster and ``W_k``
            vanishes.
        n_references: Number of reference samples ``B``.
        seed: Seed for the reference samples.

    Returns:
        The estimated number of clusters.
    """
    n_items = points.shape[0]
    n_cuts = min(max_clusters, n_items - 1)
    if n_cuts < 2:
        return 1
    cuts = np.arange(1, n_cuts + 1)
    observed = np.log(within_cluster_dispersion(points, dendrogram_cuts(linkage_matrix, n_cuts)))

    centred = points - points.mean(axis=0)
    left, singular, _ = np.linalg.svd(centred, full_matrices=False)
    scores = left * singular
    low, high = scores.min(axis=0), scores.max(axis=0)
    rng = np.random.default_rng(seed)
    reference = np.empty((n_references, n_cuts))
    for b in range(n_references):
        sample = rng.uniform(low, high, size=scores.shape)
        tree = linkage(pdist(sample), method=method)
        reference[b] = np.log(within_cluster_dispersion(sample, dendrogram_cuts(tree, n_cuts)))

    gap = reference.mean(axis=0) - observed
    tolerance = reference.std(axis=0) * np.sqrt(1.0 + 1.0 / n_references)
    accepted = np.flatnonzero(gap[:-1] >= gap[1:] - tolerance[1:])
    return int(cuts[accepted[0]]) if accepted.size else int(cuts[-1])


def seriation(linkage_matrix: NDArray[np.float64]) -> NDArray[np.intp]:
    """Order the leaves of a dendrogram left to right.

    This is the quasi-diagonalisation step of HRP, which places correlated
    assets next to each other.

    Args:
        linkage_matrix: Linkage matrix from :func:`correlation_linkage`.

    Returns:
        Leaf indices in dendrogram order.
    """
    return leaves_list(linkage_matrix).astype(np.intp)


def bisection(n_items: int) -> Iterator[tuple[slice, slice]]:
    """Yield the sibling halves of a seriated order, one tree level at a time.

    Each cluster of the previous level is cut at its midpoint, so a caller that
    rescales the weights of each pair in turn performs the top-down recursive
    bisection of HRP.

    Args:
        n_items: Length of the seriated order.

    Yields:
        ``(left, right)`` slices into the seriated order.
    """
    clusters = [(0, n_items)]
    while clusters:
        clusters = [
            half
            for start, stop in clusters
            if stop - start > 1
            for half in ((start, (start + stop) // 2), ((start + stop) // 2, stop))
        ]
        for left, right in zip(clusters[::2], clusters[1::2], strict=True):
            yield slice(*left), slice(*right)


def inverse_variance_weights(covariance: NDArray[np.float64]) -> NDArray[np.float64]:
    """Weights proportional to the reciprocal of each asset's variance.

    Args:
        covariance: Symmetric covariance matrix of a cluster.

    Returns:
        Weights that sum to one.
    """
    inverse = 1.0 / np.clip(np.diag(covariance), 1e-12, None)
    return inverse / inverse.sum()

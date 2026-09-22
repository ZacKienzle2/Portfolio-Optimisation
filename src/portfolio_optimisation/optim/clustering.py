"""Hierarchical clustering primitives shared by the tree-based allocators.

HRP, HERC and NCO all cluster assets on the correlation distance
``d_ij = sqrt((1 - rho_ij) / 2)``, seriate the dendrogram leaves so correlated
assets sit next to each other, and split the seriated order recursively. The
linkage, the leaf order and the correlation come from SciPy and statsmodels,
so this module only composes them.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import numpy as np
from scipy.cluster.hierarchy import leaves_list, linkage
from scipy.spatial.distance import squareform
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

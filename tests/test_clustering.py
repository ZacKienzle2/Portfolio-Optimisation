"""Clustering primitives: seriation, the correlation embedding and the gap statistic.

The seriation test checks SciPy against the list splice it replaced.

``hypothesis write --equivalent`` drafted this test. Its strategy drew arbitrary
float arrays, which are not linkage matrices, so the linkage is built here from
random points, and its ``==`` is ambiguous for arrays, so the comparison is
element-wise.
"""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import pdist, squareform

from portfolio_optimisation import baselines
from portfolio_optimisation.optim.clustering import (
    bisection,
    correlation_embedding,
    correlation_linkage,
    dendrogram_cuts,
    gap_statistic,
    seriation,
    within_cluster_dispersion,
)

_POINTS = st.integers(min_value=2, max_value=40).flatmap(
    lambda n: arrays(
        np.float64,
        (n, 3),
        elements=st.floats(-100.0, 100.0, allow_nan=False, allow_infinity=False),
    )
)


@given(points=_POINTS, method=st.sampled_from(["ward", "single", "complete", "average"]))
def test_equivalent_quasi_diagonal_seriation(points: np.ndarray, method: str) -> None:
    linkage_matrix = linkage(pdist(points), method=method)
    np.testing.assert_array_equal(
        baselines.quasi_diagonal(linkage_matrix=linkage_matrix),
        seriation(linkage_matrix=linkage_matrix),
    )


@given(n_items=st.integers(min_value=1, max_value=200))
def test_bisection_partitions_every_cluster_into_its_halves(n_items: int) -> None:
    splits = list(bisection(n_items))
    assert len(splits) == n_items - 1
    for left, right in splits:
        assert left.stop == right.start
        assert 0 <= (right.stop - right.start) - (left.stop - left.start) <= 1


_COVARIANCES = st.tuples(st.integers(3, 25), st.integers(0, 2**32 - 1)).map(
    lambda size_seed: np.cov(
        np.random.default_rng(size_seed[1]).standard_normal((size_seed[0] + 5, size_seed[0])),
        rowvar=False,
    )
)


@given(covariance=_COVARIANCES, method=st.sampled_from(["ward", "single", "complete", "average"]))
def test_embedding_reproduces_the_correlation_distance_dendrogram(
    covariance: np.ndarray, method: str
) -> None:
    points = correlation_embedding(covariance)
    scale = np.sqrt(np.outer(np.diag(covariance), np.diag(covariance)))
    correlation = (covariance / scale)[np.triu_indices(covariance.shape[0], 1)]
    np.testing.assert_allclose(pdist(points) ** 2, 2.0 * (1.0 - correlation), atol=1e-9)
    np.testing.assert_array_equal(
        correlation_linkage(covariance, method)[:, :2], linkage(pdist(points), method)[:, :2]
    )


@given(points=_POINTS, method=st.sampled_from(["ward", "single", "complete", "average"]))
def test_within_dispersion_matches_the_pairwise_definition(points: np.ndarray, method: str) -> None:
    n_cuts = points.shape[0]
    labels = dendrogram_cuts(linkage(pdist(points), method), n_cuts)
    squared = squareform(pdist(points)) ** 2
    expected = [
        sum(
            squared[np.ix_(labels[:, k] == r, labels[:, k] == r)].sum()
            / (2 * (labels[:, k] == r).sum())
            for r in np.unique(labels[:, k])
        )
        for k in range(n_cuts)
    ]
    np.testing.assert_allclose(within_cluster_dispersion(points, labels), expected, atol=1e-6)


def _block_returns(n_blocks: int, per_block: int = 6, periods: int = 1000) -> np.ndarray:
    rng = np.random.default_rng(n_blocks)
    factors = rng.standard_normal((periods, n_blocks))
    loadings = np.repeat(np.eye(n_blocks), per_block, axis=1) * 0.7
    noise = rng.standard_normal((periods, n_blocks * per_block)) * np.sqrt(1.0 - 0.7**2)
    return factors @ loadings + noise


@pytest.mark.parametrize("n_blocks", [1, 2, 3, 5])
def test_gap_statistic_recovers_planted_blocks(n_blocks: int) -> None:
    covariance = np.cov(_block_returns(n_blocks), rowvar=False)
    tree = correlation_linkage(covariance, "average")
    estimate = gap_statistic(correlation_embedding(covariance), tree, "average", seed=0)
    assert estimate == n_blocks

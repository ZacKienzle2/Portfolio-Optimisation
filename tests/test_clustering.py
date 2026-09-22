"""Equivalence of the SciPy seriation with the list splice it replaced.

``hypothesis write --equivalent`` drafted this test. Its strategy drew arbitrary
float arrays, which are not linkage matrices, so the linkage is built here from
random points, and its ``==`` is ambiguous for arrays, so the comparison is
element-wise.
"""

from __future__ import annotations

import numpy as np
from hypothesis import given
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import pdist

from portfolio_optimisation import baselines
from portfolio_optimisation.optim.clustering import bisection, seriation

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

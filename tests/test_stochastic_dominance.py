"""Tests for second-order stochastic dominance constraints."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

from portfolio_optimisation import baselines
from portfolio_optimisation.optim import ssd_constrained_weights, ssd_dominates
from portfolio_optimisation.optim.stochastic_dominance import lower_partial_moments


def _returns_and_benchmark(seed: int = 19, t: int = 200, n: int = 4):
    rng = np.random.default_rng(seed)
    means = rng.uniform(0.0002, 0.0008, size=n)
    vols = rng.uniform(0.005, 0.015, size=n)
    r = rng.normal(loc=means, scale=vols, size=(t, n))
    returns = pd.DataFrame(r, columns=[f"A{i}" for i in range(n)])
    benchmark = returns.mean(axis=1)
    return returns, benchmark


def test_ssd_solution_satisfies_dominance() -> None:
    returns, benchmark = _returns_and_benchmark()
    weights = ssd_constrained_weights(returns, benchmark)
    portfolio = returns @ weights
    # The LP solution should empirically SSD-dominate (allow small slack).
    assert ssd_dominates(portfolio, benchmark)


def test_ssd_solution_is_long_only_simplex() -> None:
    returns, benchmark = _returns_and_benchmark()
    weights = ssd_constrained_weights(returns, benchmark)
    assert np.isclose(weights.sum(), 1.0, atol=1e-6)
    assert (weights >= -1e-9).all()


def test_ssd_dominates_self_is_true() -> None:
    _, benchmark = _returns_and_benchmark()
    assert ssd_dominates(benchmark, benchmark)


def test_ssd_dominates_inferior_is_false() -> None:
    rng = np.random.default_rng(0)
    benchmark = pd.Series(rng.normal(0.001, 0.01, size=200))
    inferior = pd.Series(rng.normal(-0.001, 0.02, size=200))
    assert not ssd_dominates(inferior, benchmark)


def test_ssd_rejects_length_mismatch() -> None:
    returns, _ = _returns_and_benchmark()
    short_bench = pd.Series(np.zeros(50))
    import pytest

    with pytest.raises(ValueError, match="same length"):
        ssd_constrained_weights(returns, short_bench)


_OUTCOMES = st.integers(min_value=1, max_value=200).flatmap(
    lambda n: arrays(np.float64, n, elements=st.floats(-1e3, 1e3, allow_nan=False))
)


@given(sample=_OUTCOMES, thresholds=_OUTCOMES)
def test_equivalent_lower_partial_moments_lower_partial_moments(
    sample: np.ndarray, thresholds: np.ndarray
) -> None:
    np.testing.assert_allclose(
        baselines.lower_partial_moments(sample=sample, thresholds=thresholds),
        lower_partial_moments(sample=sample, thresholds=thresholds),
        rtol=1e-9,
        atol=1e-6,
    )


# Returns are drawn in whole basis points. The cutting planes hold each tail sum
# to the LP feasibility tolerance of 1e-7, so a panel whose entries differ by
# less than that has several optima that are equally feasible to the solvers.
_BASIS_POINT_RETURNS = st.integers(-500, 500).map(lambda bp: bp / 10_000)


@settings(max_examples=20, deadline=None)
@given(
    panel=st.tuples(st.integers(10, 40), st.integers(2, 5)).flatmap(
        lambda shape: arrays(np.float64, shape, elements=_BASIS_POINT_RETURNS)
    )
)
def test_cutting_planes_reach_the_pairwise_programme_optimum(panel: np.ndarray) -> None:
    returns = pd.DataFrame(panel)
    benchmark = returns.mean(axis=1)
    weights = ssd_constrained_weights(returns, benchmark)
    expected = baselines.ssd_expected_return(panel, benchmark.to_numpy())
    assert float(panel.mean(axis=0) @ weights.to_numpy()) == pytest.approx(expected, abs=1e-7)

"""Tests for the Student-t copula analyser."""

from __future__ import annotations

import numpy as np
import pandas as pd
from hypothesis import given
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays
from statsmodels.stats.correlation_tools import corr_clipped

from portfolio_optimisation import baselines
from portfolio_optimisation.risk.copula import CopulaRiskAnalyser


@st.composite
def _unit_diagonal_symmetric(draw: st.DrawFn) -> np.ndarray:
    n = draw(st.integers(2, 8))
    upper = np.triu(draw(arrays(np.float64, (n, n), elements=st.floats(-1.0, 1.0))), 1)
    return upper + upper.T + np.eye(n)


@given(corr=_unit_diagonal_symmetric())
def test_equivalent_clipped_correlation_corr_clipped(corr: np.ndarray) -> None:
    np.testing.assert_allclose(
        baselines.clipped_correlation(corr=corr, threshold=1e-8),
        corr_clipped(corr=corr, threshold=1e-8),
        atol=1e-9,
    )


def test_analyser_keeps_the_column_order_of_the_returns() -> None:
    rng = np.random.default_rng(0)
    columns = [f"A{i}" for i in range(12)]
    returns = pd.DataFrame(rng.standard_t(5, size=(300, 12)) * 0.01, columns=columns)
    weights = pd.Series(1.0 / 12, index=columns[::-1])
    assert list(CopulaRiskAnalyser(returns, weights).returns.columns) == columns

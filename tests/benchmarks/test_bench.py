"""Micro-benchmarks for the performance-sensitive numerical paths.

These run once and untimed in the default suite (``--benchmark-disable``) to
confirm they execute; run ``pytest --benchmark-enable --benchmark-only
tests/benchmarks`` for timing. They guard against performance regressions in the
allocation, denoising and higher-moment routines.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from portfolio_optimisation.optim.denoise import denoise_correlation
from portfolio_optimisation.optim.higher_moments import cokurtosis_tensor
from portfolio_optimisation.optim.hrp import HRPModel


def _returns(seed: int = 0, n: int = 750, k: int = 20) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    data = rng.normal(0.0005, 0.012, size=(n, k))
    return pd.DataFrame(data, columns=[f"A{i}" for i in range(k)])


def test_benchmark_hrp(benchmark: Any) -> None:
    returns = _returns()

    def run() -> pd.Series:
        model = HRPModel(returns)
        model.optimize()
        return model.clean_weights()

    weights = benchmark(run)
    assert abs(float(weights.sum()) - 1.0) < 1e-8


def test_benchmark_denoise(benchmark: Any) -> None:
    returns = _returns()
    correlation = returns.corr().to_numpy(dtype=np.float64)
    q = returns.shape[0] / returns.shape[1]
    result = benchmark(denoise_correlation, correlation, q)
    assert np.asarray(result).shape == correlation.shape


def test_benchmark_cokurtosis(benchmark: Any) -> None:
    returns = _returns(k=12)
    result = benchmark(cokurtosis_tensor, returns)
    assert result.shape == (12, 12**3)

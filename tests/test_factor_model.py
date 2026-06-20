"""Tests for the factor-model covariance estimators."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from portfolio_optimisation.optim import (
    factor_model_covariance,
    risk_parity_weights,
    statistical_factor_covariance,
)


def _returns(seed: int = 0, n: int = 300, k: int = 6) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    factors = rng.normal(0.0, 0.01, size=(n, 2))
    loadings = rng.normal(0.0, 1.0, size=(2, k))
    idiosyncratic = rng.normal(0.0, 0.004, size=(n, k))
    data = factors @ loadings + idiosyncratic
    return pd.DataFrame(data, columns=[f"A{i}" for i in range(k)])


def _min_eigenvalue(frame: pd.DataFrame) -> float:
    return float(np.linalg.eigvalsh(frame.to_numpy()).min())


def test_statistical_factor_covariance_is_psd_and_symmetric() -> None:
    cov = statistical_factor_covariance(_returns(), n_factors=2)
    assert cov.shape == (6, 6)
    assert np.allclose(cov.to_numpy(), cov.to_numpy().T)
    assert _min_eigenvalue(cov) > -1e-8


def test_statistical_factor_covariance_preserves_diagonal() -> None:
    returns = _returns()
    cov = statistical_factor_covariance(returns, n_factors=2)
    assert np.allclose(np.diag(cov.to_numpy()), returns.var(ddof=1).to_numpy(), atol=1e-12)


def test_full_rank_recovers_sample_covariance() -> None:
    returns = _returns(k=4)
    cov = statistical_factor_covariance(returns, n_factors=4)
    assert np.allclose(cov.to_numpy(), returns.cov().to_numpy(), atol=1e-10)


def test_invalid_n_factors_raises() -> None:
    with pytest.raises(ValueError, match="n_factors"):
        statistical_factor_covariance(_returns(), n_factors=0)


def test_factor_model_covariance_is_psd() -> None:
    returns = _returns()
    rng = np.random.default_rng(1)
    factors = pd.DataFrame(
        rng.normal(0.0, 0.01, size=(returns.shape[0], 2)),
        index=returns.index,
        columns=["F1", "F2"],
    )
    cov = factor_model_covariance(returns, factors)
    assert cov.shape == (6, 6)
    assert _min_eigenvalue(cov) > -1e-8


def test_factor_covariance_feeds_allocator() -> None:
    returns = _returns()
    cov = statistical_factor_covariance(returns, n_factors=2)
    weights = risk_parity_weights(returns, cov_matrix=cov)
    assert abs(float(weights.sum()) - 1.0) < 1e-8

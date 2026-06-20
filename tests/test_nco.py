"""Tests for Nested Clustered Optimisation."""

from __future__ import annotations

import numpy as np
import pandas as pd

from portfolio_optimisation.optim import (
    NCOOptimiser,
    denoise_covariance,
    nco_weights,
)


def _block_returns(seed: int = 11, t: int = 800, n: int = 12, blocks: int = 3) -> pd.DataFrame:
    """Block-structured returns: assets within a block share a factor."""
    rng = np.random.default_rng(seed)
    market = rng.normal(0.0, 0.005, size=t)
    block_size = n // blocks
    cols: list[np.ndarray] = []
    for _ in range(blocks):
        factor = rng.normal(0.0, 0.01, size=t)
        for _ in range(block_size):
            idio = rng.normal(0.0, 0.003, size=t)
            cols.append(0.4 * market + 0.6 * factor + idio)
    data = np.column_stack(cols)
    return pd.DataFrame(data, columns=[f"A{i}" for i in range(data.shape[1])])


def test_nco_returns_long_only_simplex() -> None:
    returns = _block_returns()
    weights = nco_weights(returns)
    assert (weights >= -1e-12).all()
    assert np.isclose(weights.sum(), 1.0, atol=1e-9)


def test_nco_respects_n_clusters_setting() -> None:
    returns = _block_returns()
    weights = nco_weights(returns, n_clusters=4)
    assert (weights >= -1e-12).all()
    assert np.isclose(weights.sum(), 1.0, atol=1e-9)


def test_nco_optimiser_caches_weights() -> None:
    returns = _block_returns()
    nco = NCOOptimiser(returns)
    w1 = nco.optimise(n_clusters=3)
    assert np.allclose(w1.to_numpy(), nco.weights.to_numpy())


def test_nco_accepts_denoised_covariance() -> None:
    returns = _block_returns()
    cov = denoise_covariance(returns.cov(), q=returns.shape[0] / returns.shape[1], detone=False)
    weights = nco_weights(returns, cov_matrix=cov, n_clusters=3)
    assert np.isclose(weights.sum(), 1.0, atol=1e-9)
    assert (weights >= -1e-12).all()


def test_nco_handles_singleton_clusters() -> None:
    returns = _block_returns(n=4, blocks=4)
    weights = nco_weights(returns, n_clusters=4)
    assert np.isclose(weights.sum(), 1.0, atol=1e-9)

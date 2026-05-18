"""Tests for RMT denoising and detoning of sample correlations."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from portfolio_optimisation.optim import (
    HRPModel,
    denoise_correlation,
    denoise_covariance,
    detone_correlation,
)


def _gaussian_returns(seed: int = 7, t: int = 800, n: int = 20) -> pd.DataFrame:
    """Pure white-noise returns: sample correlation eigenvalues are entirely noise."""
    rng = np.random.default_rng(seed)
    data = rng.normal(0.0, 0.01, size=(t, n))
    return pd.DataFrame(data, columns=[f"A{i}" for i in range(n)])


def test_denoised_correlation_keeps_unit_diagonal() -> None:
    returns = _gaussian_returns()
    corr = returns.corr().to_numpy()
    out = denoise_correlation(corr, q=returns.shape[0] / returns.shape[1])
    assert isinstance(out, np.ndarray)
    np.testing.assert_allclose(np.diag(out), 1.0, atol=1e-8)


def test_denoised_correlation_collapses_pure_noise() -> None:
    """White-noise eigenvalues should collapse to a single noise mean."""
    returns = _gaussian_returns()
    corr = returns.corr().to_numpy()
    q = returns.shape[0] / returns.shape[1]
    out = denoise_correlation(corr, q=q)
    eig = np.linalg.eigvalsh(np.asarray(out))
    # Almost all eigenvalues should equal the average noise eigenvalue.
    counts = np.unique(np.round(eig, 6), return_counts=True)[1]
    assert counts.max() >= len(eig) - 2


def test_detoned_correlation_zeroes_market_component() -> None:
    """Top eigenvector of the detoned matrix should be orthogonal to the
    original top eigenvector (the removed market mode)."""
    returns = _gaussian_returns()
    corr = returns.corr().to_numpy()
    _, eigvecs_before = np.linalg.eigh(corr)
    market_mode = eigvecs_before[:, -1]
    out = np.asarray(detone_correlation(corr, n_market_components=1))
    np.testing.assert_allclose(np.diag(out), 1.0, atol=1e-8)
    # The detoned matrix should annihilate the original market mode.
    residual = float(np.linalg.norm(out @ market_mode))
    # Allow for rescaling-induced noise but require near-orthogonality.
    assert residual < 0.5


def test_denoise_correlation_dataframe_preserves_index() -> None:
    returns = _gaussian_returns()
    corr = returns.corr()
    out = denoise_correlation(corr, q=returns.shape[0] / returns.shape[1])
    assert isinstance(out, pd.DataFrame)
    assert list(out.index) == list(corr.index)
    assert list(out.columns) == list(corr.columns)


def test_denoise_covariance_preserves_variance_scale() -> None:
    returns = _gaussian_returns()
    cov = returns.cov()
    out = denoise_covariance(cov, q=returns.shape[0] / returns.shape[1])
    np.testing.assert_allclose(np.diag(out.to_numpy()), np.diag(cov.to_numpy()), rtol=1e-6)


def test_hrp_accepts_external_covariance() -> None:
    returns = _gaussian_returns()
    cov_denoised = denoise_covariance(
        returns.cov(), q=returns.shape[0] / returns.shape[1], detone=True
    )
    model = HRPModel(returns, cov_matrix=cov_denoised)
    model.optimize(linkage_method="ward")
    weights = model.clean_weights()
    assert np.isclose(weights.sum(), 1.0)
    assert (weights > 0).all()


def test_denoise_rejects_q_le_one() -> None:
    with pytest.raises(ValueError, match="q = T/N must be greater than 1"):
        denoise_correlation(np.eye(5), q=0.5)

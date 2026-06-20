"""Tests for covariance shrinkage estimators."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd
import pytest

from portfolio_optimisation.optim import (
    linear_shrinkage_covariance,
    nonlinear_shrinkage_covariance,
    oas_covariance,
)

ESTIMATORS = [
    linear_shrinkage_covariance,
    oas_covariance,
    nonlinear_shrinkage_covariance,
]


def _returns(seed: int, t: int, p: int) -> tuple[pd.DataFrame, np.ndarray]:
    rng = np.random.default_rng(seed)
    rotation = np.linalg.qr(rng.standard_normal((p, p)))[0]
    sigma = rotation @ np.diag(np.linspace(1.0, 20.0, p)) @ rotation.T
    data = rng.standard_normal((t, p)) @ np.linalg.cholesky(sigma).T
    frame = pd.DataFrame(data, columns=[f"A{i}" for i in range(p)])
    return frame, sigma


@pytest.mark.parametrize("estimator", ESTIMATORS)
def test_estimator_is_psd_symmetric_and_indexed(
    estimator: Callable[[pd.DataFrame], pd.DataFrame],
) -> None:
    returns, _ = _returns(0, 120, 8)
    cov = estimator(returns)
    assert list(cov.columns) == list(returns.columns)
    assert list(cov.index) == list(returns.columns)
    arr = cov.to_numpy(dtype=np.float64)
    assert np.allclose(arr, arr.T, atol=1e-10)
    assert float(np.linalg.eigvalsh(arr).min()) > -1e-10


def test_nonlinear_recovers_sample_spectrum_at_large_n() -> None:
    returns, _ = _returns(1, 200_000, 4)
    nl = np.linalg.eigvalsh(nonlinear_shrinkage_covariance(returns).to_numpy())
    sample = np.linalg.eigvalsh(np.cov(returns.to_numpy(dtype=np.float64), rowvar=False))
    assert np.allclose(np.sort(nl), np.sort(sample), rtol=0.02)


def test_nonlinear_beats_sample_covariance_in_frobenius() -> None:
    loss_sample = loss_nl = 0.0
    for seed in range(8):
        returns, sigma = _returns(seed + 10, 60, 25)
        sample = np.cov(returns.to_numpy(dtype=np.float64), rowvar=False)
        nl = nonlinear_shrinkage_covariance(returns).to_numpy(dtype=np.float64)
        loss_sample += float(np.linalg.norm(sample - sigma, "fro"))
        loss_nl += float(np.linalg.norm(nl - sigma, "fro"))
    assert loss_nl < loss_sample


def test_nonlinear_handles_more_assets_than_observations() -> None:
    returns, _ = _returns(3, 25, 40)
    cov = nonlinear_shrinkage_covariance(returns).to_numpy(dtype=np.float64)
    assert cov.shape == (40, 40)
    assert float(np.linalg.eigvalsh(cov).min()) > -1e-10


def test_nonlinear_rejects_single_observation() -> None:
    returns = pd.DataFrame({"A": [0.1], "B": [0.2]})
    with pytest.raises(ValueError, match="observations"):
        nonlinear_shrinkage_covariance(returns)

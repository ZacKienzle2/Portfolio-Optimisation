"""Covariance shrinkage estimators.

Three estimators stabilise the sample covariance when the number of assets is
not small relative to the sample length. :func:`linear_shrinkage_covariance`
and :func:`oas_covariance` apply the Ledoit-Wolf and Oracle-Approximating
single-target linear shrinkages. :func:`nonlinear_shrinkage_covariance`
implements the analytical nonlinear shrinkage of Ledoit and Wolf (2020), which
keeps the sample eigenvectors and applies a separate optimal shrinkage to each
sample eigenvalue through a kernel estimate of the limiting spectral density.

Every estimator takes a returns frame (rows are observations, columns assets)
and returns a covariance frame indexed and ordered by the input columns.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.covariance import ledoit_wolf, oas


def _frame(matrix: NDArray[np.float64], columns: pd.Index) -> pd.DataFrame:
    return pd.DataFrame(matrix, index=columns, columns=columns)


def linear_shrinkage_covariance(returns: pd.DataFrame) -> pd.DataFrame:
    """Ledoit-Wolf linear shrinkage toward a scaled-identity target.

    Args:
        returns (pd.DataFrame): Asset returns; rows observations, columns assets.

    Returns:
        pd.DataFrame: Shrunk covariance indexed by ticker.
    """
    cov, _ = ledoit_wolf(returns.to_numpy(dtype=np.float64), assume_centered=False)
    return _frame(cov, returns.columns)


def oas_covariance(returns: pd.DataFrame) -> pd.DataFrame:
    """Oracle-Approximating Shrinkage covariance.

    The shrinkage intensity is chosen to minimise the expected quadratic loss
    under normality, which converges faster than the Ledoit-Wolf intensity when
    the returns are close to Gaussian.

    Args:
        returns (pd.DataFrame): Asset returns; rows observations, columns assets.

    Returns:
        pd.DataFrame: Shrunk covariance indexed by ticker.
    """
    cov, _ = oas(returns.to_numpy(dtype=np.float64), assume_centered=False)
    return _frame(cov, returns.columns)


def _epanechnikov_density_and_hilbert(
    eigenvalues: NDArray[np.float64], n_obs: int
) -> tuple[NDArray[np.float64], NDArray[np.float64], float]:
    bandwidth = n_obs ** (-1.0 / 3.0)
    size = eigenvalues.size
    left = eigenvalues[:, None]
    right = np.broadcast_to(eigenvalues, (size, size))
    scale = bandwidth * right
    x = (left - right) / scale

    sqrt5 = np.sqrt(5.0)
    density_kernel = (3.0 / (4.0 * sqrt5)) / scale * np.maximum(1.0 - x**2 / 5.0, 0.0)
    density = density_kernel.mean(axis=1)

    with np.errstate(divide="ignore", invalid="ignore"):
        hilbert_kernel = (
            (-3.0 / (10.0 * np.pi)) * x
            + (3.0 / (4.0 * sqrt5 * np.pi))
            * (1.0 - x**2 / 5.0)
            * np.log(np.abs((sqrt5 - x) / (sqrt5 + x)))
        ) / scale
    boundary = np.abs(x) == sqrt5
    hilbert_kernel[boundary] = ((-3.0 / (10.0 * np.pi)) * x[boundary]) / scale[boundary]
    hilbert = hilbert_kernel.mean(axis=1)
    return density, hilbert, bandwidth


def nonlinear_shrinkage_covariance(returns: pd.DataFrame) -> pd.DataFrame:
    """Analytical nonlinear shrinkage covariance of Ledoit and Wolf (2020).

    The sample covariance is eigendecomposed, each eigenvalue is shrunk by the
    analytical optimal amount derived from a kernel estimate of the sample
    spectral density and its Hilbert transform, and the matrix is rebuilt from
    the unchanged sample eigenvectors.

    Args:
        returns (pd.DataFrame): Asset returns; rows observations, columns assets.
            At least two observations are required.

    Returns:
        pd.DataFrame: Shrunk covariance indexed by ticker.
    """
    observations = returns.to_numpy(dtype=np.float64)
    n_raw, n_assets = observations.shape
    if n_raw < 2:
        raise ValueError("nonlinear shrinkage requires at least two observations.")

    centred = observations - observations.mean(axis=0)
    n_obs = n_raw - 1
    sample = (centred.T @ centred) / n_obs
    sample = (sample + sample.T) / 2.0

    raw_eigenvalues, eigenvectors = np.linalg.eigh(sample)
    order = np.argsort(raw_eigenvalues)
    raw_eigenvalues = raw_eigenvalues[order]
    eigenvectors = eigenvectors[:, order]

    keep = max(0, n_assets - n_obs)
    eigenvalues = raw_eigenvalues[keep:]
    eigenvalues = np.clip(eigenvalues, np.finfo(np.float64).tiny, None)

    density, hilbert, bandwidth = _epanechnikov_density_and_hilbert(eigenvalues, n_obs)
    ratio = n_assets / n_obs

    if n_assets <= n_obs:
        denominator = (np.pi * ratio * eigenvalues * density) ** 2 + (
            1.0 - ratio - np.pi * ratio * eigenvalues * hilbert
        ) ** 2
        shrunk = eigenvalues / denominator
    else:
        sqrt5 = np.sqrt(5.0)
        hilbert_zero = (
            (1.0 / np.pi)
            * (
                3.0 / (10.0 * bandwidth**2)
                + 3.0
                / (4.0 * sqrt5 * bandwidth)
                * (1.0 - 1.0 / (5.0 * bandwidth**2))
                * np.log((1.0 + sqrt5 * bandwidth) / (1.0 - sqrt5 * bandwidth))
            )
            * np.mean(1.0 / eigenvalues)
        )
        shrunk_zero = 1.0 / (np.pi * (n_assets - n_obs) / n_obs * hilbert_zero)
        shrunk_positive = eigenvalues / (np.pi**2 * eigenvalues**2 * (density**2 + hilbert**2))
        shrunk = np.concatenate([np.full(keep, shrunk_zero), shrunk_positive])

    rebuilt = eigenvectors @ np.diag(shrunk) @ eigenvectors.T
    rebuilt = (rebuilt + rebuilt.T) / 2.0
    return _frame(rebuilt, returns.columns)

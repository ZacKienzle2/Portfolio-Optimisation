"""Random Matrix Theory denoising and detoning of sample correlation matrices.

References:
    Marchenko, V.A., Pastur, L.A. (1967). Distribution of eigenvalues for some
        sets of random matrices. Math USSR-Sbornik 1:457-483.
    Lopez de Prado, M. (2020). Machine Learning for Asset Managers.
        Cambridge Elements in Quantitative Finance.

The Marchenko-Pastur (MP) theorem characterises the eigenvalue distribution of
the sample correlation of T iid N(0, 1) observations on N variables. Sample
eigenvalues that fall inside the MP support are statistically indistinguishable
from noise; replacing them with their mean while leaving signal eigenvalues
untouched yields a denoised correlation. Detoning additionally subtracts the
top (market) eigen-component to leave only the cross-sectional structure.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy.optimize import minimize_scalar


def _mp_pdf(var: float, q: float, points: int = 1000) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Marchenko-Pastur density evaluated on the support [lam_minus, lam_plus]."""
    lam_minus = var * (1.0 - np.sqrt(1.0 / q)) ** 2
    lam_plus = var * (1.0 + np.sqrt(1.0 / q)) ** 2
    grid = np.linspace(lam_minus, lam_plus, points)
    pdf = (q / (2.0 * np.pi * var * grid)) * np.sqrt(
        np.clip((lam_plus - grid) * (grid - lam_minus), 0.0, None)
    )
    return grid, pdf


def _kde_eval(eigenvalues: NDArray[np.float64], grid: NDArray[np.float64], bandwidth: float) -> NDArray[np.float64]:
    """Gaussian KDE of the empirical eigenvalue distribution on ``grid``."""
    eigenvalues = eigenvalues.reshape(-1, 1)
    grid_col = grid.reshape(1, -1)
    kernels = np.exp(-0.5 * ((grid_col - eigenvalues) / bandwidth) ** 2) / (
        bandwidth * np.sqrt(2.0 * np.pi)
    )
    return kernels.mean(axis=0)


def _fit_marchenko_pastur_variance(
    eigenvalues: NDArray[np.float64], q: float, bandwidth: float = 0.01
) -> float:
    """Estimate the noise variance ``sigma^2`` by fitting the MP density to data.

    Solves ``min_sigma  sum((kde(lambda) - mp_pdf(lambda; sigma, q))^2)`` on a
    grid spanning the empirical eigenvalue range.
    """

    def loss(var: float) -> float:
        grid, pdf_mp = _mp_pdf(var, q)
        pdf_emp = _kde_eval(eigenvalues, grid, bandwidth)
        return float(np.sum((pdf_emp - pdf_mp) ** 2))

    result = minimize_scalar(loss, bounds=(1e-5, 1.0 - 1e-5), method="bounded")
    return float(result.x)


def denoise_correlation(
    correlation: NDArray[np.float64] | pd.DataFrame,
    q: float,
    *,
    bandwidth: float = 0.01,
) -> NDArray[np.float64] | pd.DataFrame:
    """Denoise a sample correlation matrix by MP eigenvalue averaging.

    Args:
        correlation (NDArray[np.float64] | pd.DataFrame): Sample correlation.
        q (float): Ratio T/N where T is observations and N is variables.
            Must satisfy ``q > 1`` for the MP fit to be well-posed.
        bandwidth (float, optional): Gaussian KDE bandwidth on the eigenvalue
            density used to fit the MP variance. Defaults to ``0.01``.

    Returns:
        Same type as ``correlation``. Signal eigenvalues are preserved; noise
        eigenvalues (those inside the MP support) are replaced with their
        average; eigenvectors are unchanged so the matrix remains symmetric
        with unit diagonal.
    """
    if q <= 1.0:
        raise ValueError("q = T/N must be greater than 1 for MP denoising.")

    if isinstance(correlation, pd.DataFrame):
        index = correlation.index
        columns = correlation.columns
        values = correlation.to_numpy(dtype=np.float64)
    else:
        index = None
        columns = None
        values = np.asarray(correlation, dtype=np.float64)

    eigenvalues, eigenvectors = np.linalg.eigh(values)
    sigma2 = _fit_marchenko_pastur_variance(eigenvalues, q, bandwidth=bandwidth)
    lam_plus = sigma2 * (1.0 + np.sqrt(1.0 / q)) ** 2

    noise_mask = eigenvalues < lam_plus
    if noise_mask.any():
        noise_mean = float(eigenvalues[noise_mask].mean())
        shrunk = eigenvalues.copy()
        shrunk[noise_mask] = noise_mean
    else:
        shrunk = eigenvalues

    denoised = eigenvectors @ np.diag(shrunk) @ eigenvectors.T
    # Rescale to unit diagonal so we still have a valid correlation.
    diag = np.sqrt(np.diag(denoised))
    diag = np.where(diag < 1e-12, 1.0, diag)
    denoised = denoised / np.outer(diag, diag)

    if index is not None and columns is not None:
        return pd.DataFrame(denoised, index=index, columns=columns)
    return denoised


def detone_correlation(
    correlation: NDArray[np.float64] | pd.DataFrame,
    *,
    n_market_components: int = 1,
) -> NDArray[np.float64] | pd.DataFrame:
    """Remove the top ``n_market_components`` eigen-components.

    The largest eigenvalue of a return-correlation matrix is typically the
    market mode. Subtracting it leaves the cross-sectional residual structure,
    which is what hierarchical clustering algorithms should consume.
    """
    if isinstance(correlation, pd.DataFrame):
        index = correlation.index
        columns = correlation.columns
        values = correlation.to_numpy(dtype=np.float64)
    else:
        index = None
        columns = None
        values = np.asarray(correlation, dtype=np.float64)

    eigenvalues, eigenvectors = np.linalg.eigh(values)
    n = values.shape[0]
    if not 0 < n_market_components < n:
        raise ValueError("n_market_components must be in (0, N).")

    market = np.zeros_like(values)
    for k in range(1, n_market_components + 1):
        idx = n - k
        market += eigenvalues[idx] * np.outer(eigenvectors[:, idx], eigenvectors[:, idx])

    detoned = values - market
    diag = np.sqrt(np.diag(detoned))
    diag = np.where(diag < 1e-12, 1.0, diag)
    detoned = detoned / np.outer(diag, diag)

    if index is not None and columns is not None:
        return pd.DataFrame(detoned, index=index, columns=columns)
    return detoned


def denoise_covariance(
    covariance: pd.DataFrame, q: float, *, detone: bool = False
) -> pd.DataFrame:
    """Apply MP denoising (and optional detoning) on the cov via its correlation.

    Args:
        covariance (pd.DataFrame): Sample covariance.
        q (float): T/N.
        detone (bool, optional): Subtract the top market eigen-component after
            denoising. Defaults to False.

    Returns:
        pd.DataFrame: Denoised covariance preserving variance scaling.
    """
    std = np.sqrt(np.diag(covariance.to_numpy(dtype=np.float64)))
    std = np.where(std < 1e-12, 1.0, std)
    corr = covariance.to_numpy(dtype=np.float64) / np.outer(std, std)

    corr_denoised = denoise_correlation(corr, q=q)
    if isinstance(corr_denoised, pd.DataFrame):
        corr_arr = corr_denoised.to_numpy(dtype=np.float64)
    else:
        corr_arr = np.asarray(corr_denoised, dtype=np.float64)

    if detone:
        corr_arr = np.asarray(detone_correlation(corr_arr), dtype=np.float64)

    cov_denoised = corr_arr * np.outer(std, std)
    return pd.DataFrame(
        cov_denoised, index=covariance.index, columns=covariance.columns
    )

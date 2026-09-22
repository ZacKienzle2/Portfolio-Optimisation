"""Random Matrix Theory denoising and detoning of sample correlation matrices.

The Marchenko-Pastur theorem characterises the eigenvalue distribution of the
sample correlation of T iid N(0, 1) observations on N variables. Sample
eigenvalues that fall inside the MP support are statistically indistinguishable
from noise; replacing them with their mean while leaving signal eigenvalues
untouched yields a denoised correlation. Detoning additionally subtracts the
top (market) eigen-component to leave only the cross-sectional structure.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from statsmodels.nonparametric.kde import KDEUnivariate
from statsmodels.stats.moment_helpers import corr2cov, cov2corr

if TYPE_CHECKING:
    from numpy.typing import NDArray

_BINS_PER_BANDWIDTH = 32


def _mp_pdf(
    var: float, q: float, points: int = 1000
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Marchenko-Pastur density evaluated on the support [lam_minus, lam_plus]."""
    lam_minus = var * (1.0 - np.sqrt(1.0 / q)) ** 2
    lam_plus = var * (1.0 + np.sqrt(1.0 / q)) ** 2
    grid = np.linspace(lam_minus, lam_plus, points)
    pdf = (q / (2.0 * np.pi * var * grid)) * np.sqrt(
        np.clip((lam_plus - grid) * (grid - lam_minus), 0.0, None)
    )
    return grid, pdf


def marchenko_pastur_variance(
    eigenvalues: NDArray[np.float64], q: float, bandwidth: float = 0.01
) -> float:
    """Estimate the noise variance ``sigma^2`` by fitting the MP density to data.

    Solves ``min_sigma  sum((kde(lambda) - mp_pdf(lambda; sigma, q))^2)`` on a
    grid spanning the MP support. The empirical density is a Gaussian kernel
    estimate computed once by the binned fast Fourier transform of Silverman
    (1982), which statsmodels implements, and each candidate grid reads it by
    linear interpolation. Evaluating the kernel sum afresh on each of the
    twenty-odd candidate grids was nine tenths of the cost. The error of linear
    binning falls with the square of the bin width over the bandwidth (Hall and
    Wand, 1996), so the grid holds ``_BINS_PER_BANDWIDTH`` bins per bandwidth
    across the data, which kept the density within ``1e-4`` of the exact sum.

    Args:
        eigenvalues: Eigenvalues of a sample correlation matrix.
        q: Ratio ``T/N`` of observations to variables.
        bandwidth: Gaussian kernel bandwidth on the eigenvalue density.

    Returns:
        The fitted noise variance ``sigma^2``.
    """
    span = np.ptp(eigenvalues) + 6.0 * bandwidth
    density = KDEUnivariate(eigenvalues)
    density.fit(
        kernel="gau",
        bw=bandwidth,
        fft=True,
        gridsize=int(2 ** np.ceil(np.log2(_BINS_PER_BANDWIDTH * span / bandwidth))),
    )

    def loss(var: float) -> float:
        grid, pdf_mp = _mp_pdf(var, q)
        pdf_emp = np.interp(grid, density.support, density.density, left=0.0, right=0.0)
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
        correlation: Sample correlation.
        q: Ratio T/N where T is observations and N is variables. Must satisfy ``q > 1``
            for the MP fit to be well-posed.
        bandwidth: Gaussian KDE bandwidth on the eigenvalue density used to fit the MP
            variance. Defaults to ``0.01``.

    Returns:
        Same type as ``correlation``. Signal eigenvalues are preserved; noise
        eigenvalues (those inside the MP support) are replaced with their
        average; eigenvectors are unchanged so the matrix remains symmetric
        with unit diagonal.

    Raises:
        ValueError: If the sample has no more observations than assets.
    """
    if q <= 1.0:
        msg = "q = T/N must be greater than 1 for MP denoising."
        raise ValueError(msg)

    if isinstance(correlation, pd.DataFrame):
        index = correlation.index
        columns = correlation.columns
        values = correlation.to_numpy(dtype=np.float64)
    else:
        index = None
        columns = None
        values = np.asarray(correlation, dtype=np.float64)

    eigenvalues, eigenvectors = np.linalg.eigh(values)
    sigma2 = marchenko_pastur_variance(eigenvalues, q, bandwidth=bandwidth)
    lam_plus = sigma2 * (1.0 + np.sqrt(1.0 / q)) ** 2

    noise_mask = eigenvalues < lam_plus
    if noise_mask.any():
        noise_mean = float(eigenvalues[noise_mask].mean())
        shrunk = eigenvalues.copy()
        shrunk[noise_mask] = noise_mean
    else:
        shrunk = eigenvalues

    denoised = cov2corr((eigenvectors * shrunk) @ eigenvectors.T)

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
        msg = "n_market_components must be in (0, N)."
        raise ValueError(msg)

    top = slice(n - n_market_components, n)
    top_vectors = eigenvectors[:, top]
    top_values = eigenvalues[top]
    market = (top_vectors * top_values) @ top_vectors.T

    detoned = cov2corr(values - market)

    if index is not None and columns is not None:
        return pd.DataFrame(detoned, index=index, columns=columns)
    return detoned


def denoise_covariance(covariance: pd.DataFrame, q: float, *, detone: bool = False) -> pd.DataFrame:
    """Apply MP denoising (and optional detoning) on the cov via its correlation.

    Args:
        covariance: Sample covariance.
        q: T/N.
        detone: Subtract the top market eigen-component after denoising. Defaults to
            False.

    Returns:
        pd.DataFrame: Denoised covariance preserving variance scaling.
    """
    corr, std = cov2corr(covariance.to_numpy(dtype=np.float64), return_std=True)
    corr_denoised = denoise_correlation(corr, q=q)
    if isinstance(corr_denoised, pd.DataFrame):
        corr_arr = corr_denoised.to_numpy(dtype=np.float64)
    else:
        corr_arr = np.asarray(corr_denoised, dtype=np.float64)

    if detone:
        corr_arr = np.asarray(detone_correlation(corr_arr), dtype=np.float64)

    cov_denoised = corr2cov(corr_arr, std)
    return pd.DataFrame(cov_denoised, index=covariance.index, columns=covariance.columns)

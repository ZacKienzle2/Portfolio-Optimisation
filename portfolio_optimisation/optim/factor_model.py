"""Factor-model covariance estimators.

Estimation error in the full sample covariance grows with the number of assets
and destabilises mean-variance and hierarchical allocators. Factor models
reduce that error by expressing the covariance through a small number of common
factors plus idiosyncratic noise:

    Sigma = B Cov(F) B' + diag(idiosyncratic variances),

where ``B`` are factor loadings. Two estimators are provided:

* :func:`statistical_factor_covariance` - latent factors from the leading
  principal components of the sample covariance (no external data needed).
* :func:`factor_model_covariance` - explicit factors (for example Fama-French
  return series) regressed onto the assets.

Both return a covariance ``pd.DataFrame`` that plugs directly into the
allocators via their ``cov_matrix`` argument.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def statistical_factor_covariance(returns: pd.DataFrame, n_factors: int) -> pd.DataFrame:
    """Principal-component factor covariance.

    Keeps the leading ``n_factors`` eigen-components of the sample covariance as
    common factors and restores the original diagonal as idiosyncratic
    variance, yielding a positive semi-definite low-rank-plus-diagonal estimate.

    Args:
        returns (pd.DataFrame): Asset returns; columns are tickers.
        n_factors (int): Number of principal-component factors to retain.

    Returns:
        pd.DataFrame: Factor-model covariance, indexed and columned by ticker.

    Raises:
        ValueError: If ``n_factors`` is not in ``[1, n_assets]``.
    """
    cov = returns.cov()
    n_assets = cov.shape[0]
    if not 1 <= n_factors <= n_assets:
        raise ValueError("n_factors must lie in [1, n_assets].")

    sample = cov.to_numpy(dtype=np.float64)
    eigenvalues, eigenvectors = np.linalg.eigh(sample)
    order = np.argsort(eigenvalues)[::-1][:n_factors]
    loadings = eigenvectors[:, order]
    weights = np.clip(eigenvalues[order], 0.0, None)
    common = (loadings * weights) @ loadings.T
    idiosyncratic = np.clip(np.diag(sample) - np.diag(common), 0.0, None)
    estimate = common + np.diag(idiosyncratic)
    return pd.DataFrame(estimate, index=cov.index, columns=cov.columns)


def factor_model_covariance(returns: pd.DataFrame, factors: pd.DataFrame) -> pd.DataFrame:
    """Explicit-factor covariance from a time-series regression.

    Regresses each asset on the supplied factor returns (with an intercept) and
    combines the loadings, factor covariance and residual variances into a
    covariance estimate.

    Args:
        returns (pd.DataFrame): Asset returns; columns are tickers.
        factors (pd.DataFrame): Factor returns aligned on the same index; one
            column per factor (for example market, size, value).

    Returns:
        pd.DataFrame: Factor-model covariance, indexed and columned by ticker.

    Raises:
        ValueError: If the aligned sample is empty.
    """
    common_index = returns.index.intersection(factors.index)
    if len(common_index) == 0:
        raise ValueError("returns and factors share no common observations.")

    asset_values = returns.loc[common_index].to_numpy(dtype=np.float64)
    factor_values = factors.loc[common_index].to_numpy(dtype=np.float64)
    design = np.column_stack([np.ones(factor_values.shape[0]), factor_values])

    coefficients, *_ = np.linalg.lstsq(design, asset_values, rcond=None)
    loadings = coefficients[1:, :].T
    residuals = asset_values - design @ coefficients
    idiosyncratic = residuals.var(axis=0, ddof=1)
    factor_cov = np.atleast_2d(np.cov(factor_values, rowvar=False, ddof=1))

    common = loadings @ factor_cov @ loadings.T
    estimate = common + np.diag(idiosyncratic)
    return pd.DataFrame(estimate, index=returns.columns, columns=returns.columns)

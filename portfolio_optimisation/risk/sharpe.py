"""Probabilistic Sharpe Ratio, Deflated Sharpe Ratio and bootstrap CIs.

The classical Sharpe ratio assumes IID normal returns; both assumptions are
violated by financial data. The Probabilistic Sharpe Ratio (PSR) returns the
probability that the true Sharpe exceeds a benchmark, correcting for skewness
and kurtosis via a higher-moment-aware standard error:

    sigma_SR = sqrt((1 - gamma_3 SR_hat + ((gamma_4 - 1) / 4) SR_hat^2) / (T - 1))
    PSR(SR*) = Phi((SR_hat - SR*) / sigma_SR).

The Deflated Sharpe Ratio additionally accounts for selection bias when
``N`` candidate strategies have been backtested:

    SR_0 = sqrt(Var(SR_estimates)) * ((1 - gamma_em) Phi^-1(1 - 1/N)
            + gamma_em Phi^-1(1 - 1/(N e))),
    DSR = PSR(SR_0),

with ``gamma_em`` the Euler-Mascheroni constant.

Stationary bootstrap CIs preserve temporal dependence in the returns. The
optimal block length is taken from ``arch.bootstrap``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd
from arch.bootstrap import StationaryBootstrap, optimal_block_length
from numpy.typing import NDArray
from scipy.stats import norm

EULER_MASCHERONI: float = 0.5772156649015329


@dataclass
class SharpeStatistics:
    """Sample Sharpe ratio plus its higher-moment-adjusted distribution stats."""

    sharpe: float
    n_observations: int
    skewness: float
    kurtosis: float
    standard_error: float


def _sharpe_stats(
    returns: pd.Series | NDArray[np.float64], risk_free_rate: float
) -> SharpeStatistics:
    arr = np.asarray(returns, dtype=np.float64).ravel()
    if arr.size < 2:
        raise ValueError("Need at least 2 observations to compute a Sharpe.")
    excess = arr - risk_free_rate
    mean = float(excess.mean())
    std = float(excess.std(ddof=1))
    if std == 0.0:
        raise ValueError("Zero variance: Sharpe is undefined.")
    sharpe = mean / std
    centred = (arr - arr.mean()) / arr.std(ddof=1)
    skew = float((centred**3).mean())
    kurt = float((centred**4).mean())  # raw (non-excess) kurtosis
    t = arr.size
    se = math.sqrt(max((1.0 - skew * sharpe + ((kurt - 1.0) / 4.0) * sharpe**2), 1e-12) / (t - 1))
    return SharpeStatistics(
        sharpe=sharpe,
        n_observations=t,
        skewness=skew,
        kurtosis=kurt,
        standard_error=se,
    )


def probabilistic_sharpe_ratio(
    returns: pd.Series | NDArray[np.float64],
    *,
    benchmark_sharpe: float = 0.0,
    risk_free_rate: float = 0.0,
) -> float:
    """PSR(SR*) = probability that the true Sharpe exceeds ``benchmark_sharpe``."""
    stats = _sharpe_stats(returns, risk_free_rate)
    return float(norm.cdf((stats.sharpe - benchmark_sharpe) / stats.standard_error))


def deflated_sharpe_ratio(
    returns: pd.Series | NDArray[np.float64],
    *,
    candidate_sharpes: NDArray[np.float64] | None = None,
    n_trials: int | None = None,
    risk_free_rate: float = 0.0,
) -> float:
    """Selection-bias-corrected DSR.

    Supply either the realised candidate Sharpe ratios (``candidate_sharpes``)
    to estimate their variance directly, or ``n_trials`` with an implicit
    Sharpe variance equal to ``1`` (the Bailey-Lopez de Prado default for the
    null-hypothesis case).
    """
    stats = _sharpe_stats(returns, risk_free_rate)
    if candidate_sharpes is not None:
        candidates = np.asarray(candidate_sharpes, dtype=np.float64).ravel()
        sigma_sr = float(np.std(candidates, ddof=1))
        n = candidates.size
    elif n_trials is not None and n_trials > 0:
        sigma_sr = 1.0
        n = n_trials
    else:
        raise ValueError("Provide either candidate_sharpes or n_trials.")
    if n < 2:
        raise ValueError("Need at least 2 trials.")

    z1 = norm.ppf(1.0 - 1.0 / n)
    z2 = norm.ppf(1.0 - 1.0 / (n * math.e))
    sr0 = sigma_sr * ((1.0 - EULER_MASCHERONI) * z1 + EULER_MASCHERONI * z2)
    psr = float(norm.cdf((stats.sharpe - sr0) / stats.standard_error))
    return psr


def stationary_bootstrap_sharpe_ci(
    returns: pd.Series | NDArray[np.float64],
    *,
    n_resamples: int = 1000,
    confidence: float = 0.95,
    risk_free_rate: float = 0.0,
    seed: int | None = None,
) -> tuple[float, float, NDArray[np.float64]]:
    """Politis-Romano stationary bootstrap percentile CI for the Sharpe ratio.

    Returns:
        (lower, upper, samples) tuple. ``samples`` is the bootstrap-resampled
        Sharpe distribution for downstream histogram / BCa adjustments.
    """
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must lie in (0, 1).")
    arr = np.asarray(returns, dtype=np.float64).ravel()
    if arr.size < 4:
        raise ValueError("Need at least 4 observations for the bootstrap CI.")

    raw_block = int(optimal_block_length(arr**2).iloc[0, 0])
    block_size = int(np.clip(raw_block, 1, max(1, arr.size // 2)))
    rng = np.random.default_rng(seed)
    bs = StationaryBootstrap(block_size, arr, seed=rng)
    samples = np.empty(n_resamples, dtype=np.float64)
    for i, data in enumerate(bs.bootstrap(n_resamples)):
        resample = data[0][0]
        excess = resample - risk_free_rate
        std = excess.std(ddof=1)
        samples[i] = (excess.mean() / std) if std > 0 else 0.0

    half = (1.0 - confidence) / 2.0
    lower = float(np.quantile(samples, half))
    upper = float(np.quantile(samples, 1.0 - half))
    return lower, upper, samples

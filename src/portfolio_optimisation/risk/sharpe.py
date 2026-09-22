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
from typing import TYPE_CHECKING

import numpy as np
from arch.bootstrap import StationaryBootstrap, optimal_block_length
from scipy import stats
from scipy.stats import norm

if TYPE_CHECKING:
    import pandas as pd
    from numpy.typing import NDArray

EULER_MASCHERONI: float = 0.5772156649015329


@dataclass
class SharpeStatistics:
    """Sample Sharpe ratio plus its higher-moment-adjusted distribution stats."""

    sharpe: float
    n_observations: int
    skewness: float
    kurtosis: float
    standard_error: float


def _sample_sharpe(excess: NDArray[np.float64]) -> float:
    """Sharpe ratio of one excess-return sample, zero when it has no spread."""
    std = excess.std(ddof=1)
    return float(excess.mean() / std) if std > 0 else 0.0


def _sharpe_stats(
    returns: pd.Series | NDArray[np.float64], risk_free_rate: float
) -> SharpeStatistics:
    arr = np.asarray(returns, dtype=np.float64).ravel()
    if arr.size < 2:
        msg = "Need at least 2 observations to compute a Sharpe."
        raise ValueError(msg)
    excess = arr - risk_free_rate
    mean = float(excess.mean())
    std = float(excess.std(ddof=1))
    if std == 0.0:
        msg = "Zero variance: Sharpe is undefined."
        raise ValueError(msg)
    sharpe = mean / std
    skew = float(stats.skew(arr))
    kurt = float(stats.kurtosis(arr, fisher=False))
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
        msg = "Provide either candidate_sharpes or n_trials."
        raise ValueError(msg)
    if n < 2:
        msg = "Need at least 2 trials."
        raise ValueError(msg)

    z1 = norm.ppf(1.0 - 1.0 / n)
    z2 = norm.ppf(1.0 - 1.0 / (n * math.e))
    sr0 = sigma_sr * ((1.0 - EULER_MASCHERONI) * z1 + EULER_MASCHERONI * z2)
    return float(norm.cdf((stats.sharpe - sr0) / stats.standard_error))


def stationary_bootstrap_sharpe_ci(
    returns: pd.Series | NDArray[np.float64],
    *,
    n_resamples: int = 1000,
    confidence: float = 0.95,
    risk_free_rate: float = 0.0,
    seed: int | None = None,
) -> tuple[float, float, NDArray[np.float64]]:
    """Politis-Romano stationary bootstrap percentile CI for the Sharpe ratio.

    The mean block length is the Politis-White estimate on the squared returns.

    Args:
        returns: Per-period returns.
        n_resamples: Number of bootstrap replications.
        confidence: Two-sided coverage of the percentile interval.
        risk_free_rate: Per-period risk-free rate subtracted before resampling.
        seed: Seed for the bootstrap generator.

    Returns:
        (lower, upper, samples) tuple. ``samples`` is the bootstrap-resampled
        Sharpe distribution for downstream histogram / BCa adjustments.

    Raises:
        ValueError: If ``confidence`` lies outside ``(0, 1)`` or fewer than four
            observations are supplied.
    """
    if not 0.0 < confidence < 1.0:
        msg = "confidence must lie in (0, 1)."
        raise ValueError(msg)
    arr = np.asarray(returns, dtype=np.float64).ravel()
    if arr.size < 4:
        msg = "Need at least 4 observations for the bootstrap CI."
        raise ValueError(msg)

    raw_block = int(optimal_block_length(arr**2).iloc[0, 0])
    block_size = int(np.clip(raw_block, 1, max(1, arr.size // 2)))
    bootstrap = StationaryBootstrap(
        block_size, arr - risk_free_rate, seed=np.random.default_rng(seed)
    )
    samples = bootstrap.apply(_sample_sharpe, n_resamples)[:, 0]
    half = (1.0 - confidence) / 2.0
    lower, upper = np.quantile(samples, [half, 1.0 - half])
    return float(lower), float(upper), samples

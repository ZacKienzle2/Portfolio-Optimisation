"""Statistical backtests for Value-at-Risk and Expected Shortfall models.

Implements the standard coverage tests used to validate a risk model against
realised returns:

* :func:`kupiec_pof_test` - Proportion-of-failures likelihood ratio for
  unconditional coverage.
* :func:`christoffersen_independence_test` - Markov test that violations are
  serially independent.
* :func:`christoffersen_conditional_coverage_test` - the joint test combining
  correct coverage and independence.
* :func:`acerbi_szekely_z2` - Test 2 statistic for the adequacy of an Expected
  Shortfall forecast.

Conventions: ``returns`` are realised returns and ``var_forecasts`` /
``es_forecasts`` are stated in the same (return) space, so both forecasts are
negative for a long position and a violation occurs when
``returns < var_forecasts``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.special import xlogy
from scipy.stats import chi2


@dataclass
class CoverageTestResult:
    """Outcome of a likelihood-ratio coverage test.

    Attributes:
        statistic (float): Likelihood-ratio test statistic.
        p_value (float): Upper-tail chi-squared p-value.
        degrees_of_freedom (int): Chi-squared degrees of freedom.
        violations (int): Number of VaR violations observed.
        observations (int): Sample size.
        reject (bool): Whether the null is rejected at the chosen significance.
    """

    statistic: float
    p_value: float
    degrees_of_freedom: int
    violations: int
    observations: int
    reject: bool


def _as_violations(
    returns: NDArray[np.float64], var_forecasts: NDArray[np.float64]
) -> NDArray[np.bool_]:
    realised = np.asarray(returns, dtype=np.float64).ravel()
    forecast = np.asarray(var_forecasts, dtype=np.float64).ravel()
    if realised.shape != forecast.shape:
        raise ValueError("returns and var_forecasts must have the same length.")
    if realised.size == 0:
        raise ValueError("Need at least one observation.")
    return realised < forecast


def kupiec_pof_test(
    returns: NDArray[np.float64],
    var_forecasts: NDArray[np.float64],
    *,
    alpha: float = 0.05,
    significance: float = 0.05,
) -> CoverageTestResult:
    """Kupiec proportion-of-failures test for unconditional coverage.

    Args:
        returns (NDArray[float64]): Realised returns.
        var_forecasts (NDArray[float64]): VaR forecast per period (return space).
        alpha (float): Target violation probability (the VaR tail level).
        significance (float): Test significance for the reject decision.

    Returns:
        CoverageTestResult: The likelihood-ratio statistic, chi-squared(1)
        p-value and reject decision.
    """
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must lie in (0, 1).")
    violations = _as_violations(returns, var_forecasts)
    n = violations.size
    failures = int(violations.sum())
    rate = failures / n

    ll_null = xlogy(failures, alpha) + xlogy(n - failures, 1.0 - alpha)
    ll_alt = xlogy(failures, rate) + xlogy(n - failures, 1.0 - rate)
    statistic = float(-2.0 * (ll_null - ll_alt))
    p_value = float(chi2.sf(statistic, df=1))
    return CoverageTestResult(
        statistic=statistic,
        p_value=p_value,
        degrees_of_freedom=1,
        violations=failures,
        observations=n,
        reject=p_value < significance,
    )


def _transition_counts(violations: NDArray[np.bool_]) -> tuple[int, int, int, int]:
    previous = violations[:-1].astype(np.int64)
    current = violations[1:].astype(np.int64)
    n00 = int(np.sum((previous == 0) & (current == 0)))
    n01 = int(np.sum((previous == 0) & (current == 1)))
    n10 = int(np.sum((previous == 1) & (current == 0)))
    n11 = int(np.sum((previous == 1) & (current == 1)))
    return n00, n01, n10, n11


def _independence_statistic(violations: NDArray[np.bool_]) -> float:
    n00, n01, n10, n11 = _transition_counts(violations)
    pi01 = n01 / (n00 + n01) if (n00 + n01) > 0 else 0.0
    pi11 = n11 / (n10 + n11) if (n10 + n11) > 0 else 0.0
    pi = (n01 + n11) / (n00 + n01 + n10 + n11)

    ll_restricted = xlogy(n00 + n10, 1.0 - pi) + xlogy(n01 + n11, pi)
    ll_unrestricted = (
        xlogy(n00, 1.0 - pi01) + xlogy(n01, pi01) + xlogy(n10, 1.0 - pi11) + xlogy(n11, pi11)
    )
    return float(-2.0 * (ll_restricted - ll_unrestricted))


def christoffersen_independence_test(
    returns: NDArray[np.float64],
    var_forecasts: NDArray[np.float64],
    *,
    significance: float = 0.05,
) -> CoverageTestResult:
    """Christoffersen Markov test that VaR violations are serially independent."""
    violations = _as_violations(returns, var_forecasts)
    statistic = _independence_statistic(violations)
    p_value = float(chi2.sf(statistic, df=1))
    return CoverageTestResult(
        statistic=statistic,
        p_value=p_value,
        degrees_of_freedom=1,
        violations=int(violations.sum()),
        observations=violations.size,
        reject=p_value < significance,
    )


def christoffersen_conditional_coverage_test(
    returns: NDArray[np.float64],
    var_forecasts: NDArray[np.float64],
    *,
    alpha: float = 0.05,
    significance: float = 0.05,
) -> CoverageTestResult:
    """Joint test of correct unconditional coverage and independence.

    The statistic is the sum of the Kupiec and Christoffersen independence
    statistics and is chi-squared(2) under the null.
    """
    pof = kupiec_pof_test(returns, var_forecasts, alpha=alpha, significance=significance)
    violations = _as_violations(returns, var_forecasts)
    statistic = pof.statistic + _independence_statistic(violations)
    p_value = float(chi2.sf(statistic, df=2))
    return CoverageTestResult(
        statistic=statistic,
        p_value=p_value,
        degrees_of_freedom=2,
        violations=pof.violations,
        observations=pof.observations,
        reject=p_value < significance,
    )


def acerbi_szekely_z2(
    returns: NDArray[np.float64],
    var_forecasts: NDArray[np.float64],
    es_forecasts: NDArray[np.float64],
    *,
    alpha: float = 0.05,
) -> float:
    """Acerbi-Szekely Test 2 statistic for Expected Shortfall adequacy.

    The statistic has expectation zero when the ES forecast is correct; a
    materially positive value indicates the model underestimates the tail loss,
    a negative value that it overestimates it. Significance is assessed by
    simulation under the forecasting model.

    Args:
        returns (NDArray[float64]): Realised returns.
        var_forecasts (NDArray[float64]): VaR forecast per period (return space).
        es_forecasts (NDArray[float64]): ES forecast per period (return space);
            must be non-zero.
        alpha (float): Tail level used for the forecasts.

    Returns:
        float: The Z2 statistic.
    """
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must lie in (0, 1).")
    realised = np.asarray(returns, dtype=np.float64).ravel()
    es = np.asarray(es_forecasts, dtype=np.float64).ravel()
    violations = _as_violations(returns, var_forecasts)
    if es.shape != realised.shape:
        raise ValueError("es_forecasts must match returns in length.")
    if np.any(es == 0.0):
        raise ValueError("es_forecasts must be non-zero.")
    contributions = np.where(violations, realised / es, 0.0)
    return float(contributions.sum() / (realised.size * alpha) - 1.0)

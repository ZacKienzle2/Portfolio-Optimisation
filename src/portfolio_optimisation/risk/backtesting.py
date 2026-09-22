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
from typing import TYPE_CHECKING

import numpy as np
from scipy.special import xlogy
from scipy.stats import chi2

if TYPE_CHECKING:
    from numpy.typing import NDArray


@dataclass
class CoverageTestResult:
    """Outcome of a likelihood-ratio coverage test.

    Attributes:
        statistic: Likelihood-ratio test statistic.
        p_value: Upper-tail chi-squared p-value.
        degrees_of_freedom: Chi-squared degrees of freedom.
        violations: Number of VaR violations observed.
        observations: Sample size.
        reject: Whether the null is rejected at the chosen significance.
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
        msg = "returns and var_forecasts must have the same length."
        raise ValueError(msg)
    if realised.size == 0:
        msg = "Need at least one observation."
        raise ValueError(msg)
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
        returns: Realised returns.
        var_forecasts: VaR forecast per period (return space).
        alpha: Target violation probability (the VaR tail level).
        significance: Test significance for the reject decision.

    Returns:
        CoverageTestResult: The likelihood-ratio statistic, chi-squared(1)
        p-value and reject decision.

    Raises:
        ValueError: If ``alpha`` lies outside ``(0, 1)``.
    """
    if not 0.0 < alpha < 1.0:
        msg = "alpha must lie in (0, 1)."
        raise ValueError(msg)
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


def transition_counts(violations: NDArray[np.bool_]) -> tuple[int, int, int, int]:
    """Count the four transitions between consecutive violation states.

    The counts are a 2 x 2 contingency table, fixed by one joint count and its
    two margins, ``n11 = #(previous & current)``, ``n10 = #previous - n11`` and
    ``n01 = #current - n11``, with ``n00`` the remainder. Each is a
    ``count_nonzero`` over boolean arrays of one byte per period. Encoding the
    pairs as ``2 * previous + current`` for ``numpy.bincount`` moved eight bytes
    per period through three temporaries and took forty times as long, and
    ``scipy.stats.contingency.crosstab`` sorts its input and was slower still.

    Args:
        violations: Violation indicator per period.

    Returns:
        ``(n00, n01, n10, n11)``, where ``nij`` counts a state ``i`` followed by
        a state ``j``.
    """
    previous, current = violations[:-1], violations[1:]
    n11 = np.count_nonzero(previous & current)
    n10 = np.count_nonzero(previous) - n11
    n01 = np.count_nonzero(current) - n11
    return int(previous.size - n11 - n10 - n01), int(n01), int(n10), int(n11)


def _independence_statistic(violations: NDArray[np.bool_]) -> float:
    n00, n01, n10, n11 = transition_counts(violations)
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
        returns: Realised returns.
        var_forecasts: VaR forecast per period (return space).
        es_forecasts: ES forecast per period (return space); must be non-zero.
        alpha: Tail level used for the forecasts.

    Returns:
        float: The Z2 statistic.

    Raises:
        ValueError: If ``alpha`` lies outside ``(0, 1)``, or ``es_forecasts`` differs in
            length from ``returns`` or has a zero entry.
    """
    if not 0.0 < alpha < 1.0:
        msg = "alpha must lie in (0, 1)."
        raise ValueError(msg)
    realised = np.asarray(returns, dtype=np.float64).ravel()
    es = np.asarray(es_forecasts, dtype=np.float64).ravel()
    violations = _as_violations(returns, var_forecasts)
    if es.shape != realised.shape:
        msg = "es_forecasts must match returns in length."
        raise ValueError(msg)
    if np.any(es == 0.0):
        msg = "es_forecasts must be non-zero."
        raise ValueError(msg)
    contributions = np.where(violations, realised / es, 0.0)
    return float(contributions.sum() / (realised.size * alpha) - 1.0)

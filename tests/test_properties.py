"""Property-based invariants for the allocation and risk primitives.

Uses Hypothesis to assert structural guarantees over a wide range of generated
inputs: long-only weights form a simplex, denoising preserves the algebraic
properties of a correlation matrix, and Conditional VaR never sits above VaR.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from hypothesis import HealthCheck, assume, given, settings
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

from portfolio_optimisation.infra.weights import inverse_variance_weights
from portfolio_optimisation.optim.denoise import denoise_correlation
from portfolio_optimisation.optim.herc import herc_weights
from portfolio_optimisation.optim.hrp import HRPModel
from portfolio_optimisation.optim.nco import nco_weights
from portfolio_optimisation.risk.metrics import calculate_risk_metrics

_SETTINGS = settings(
    deadline=None,
    max_examples=30,
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.filter_too_much],
)


@st.composite
def returns_frames(
    draw: st.DrawFn,
    *,
    min_assets: int = 3,
    max_assets: int = 6,
    min_obs: int = 90,
    max_obs: int = 160,
) -> pd.DataFrame:
    n_assets = draw(st.integers(min_assets, max_assets))
    n_obs = draw(st.integers(min_obs, max_obs))
    data = draw(
        arrays(
            dtype=np.float64,
            shape=(n_obs, n_assets),
            elements=st.floats(-0.08, 0.08, allow_nan=False, allow_infinity=False, width=64),
        )
    )
    frame = pd.DataFrame(data, columns=[f"A{i}" for i in range(n_assets)])
    assume(bool(np.all(frame.std(axis=0).to_numpy() > 1e-4)))
    return frame


def _assert_simplex(weights: pd.Series, columns: list[str]) -> None:
    assert abs(float(weights.sum()) - 1.0) < 1e-8
    assert bool((weights.to_numpy() >= -1e-12).all())
    assert set(weights.index) == set(columns)


@_SETTINGS
@given(returns=returns_frames())
def test_hrp_weights_form_a_simplex(returns: pd.DataFrame) -> None:
    model = HRPModel(returns)
    model.optimize()
    _assert_simplex(model.clean_weights(), list(returns.columns))


@_SETTINGS
@given(returns=returns_frames())
def test_herc_weights_form_a_simplex(returns: pd.DataFrame) -> None:
    _assert_simplex(herc_weights(returns), list(returns.columns))


@_SETTINGS
@given(returns=returns_frames())
def test_nco_weights_form_a_simplex(returns: pd.DataFrame) -> None:
    _assert_simplex(nco_weights(returns), list(returns.columns))


@_SETTINGS
@given(returns=returns_frames())
def test_inverse_variance_weights_form_a_simplex(returns: pd.DataFrame) -> None:
    _assert_simplex(inverse_variance_weights(returns.cov()), list(returns.columns))


@_SETTINGS
@given(returns=returns_frames())
def test_denoise_preserves_correlation_properties(returns: pd.DataFrame) -> None:
    correlation = returns.corr().to_numpy(dtype=np.float64)
    q = returns.shape[0] / returns.shape[1]
    denoised = np.asarray(denoise_correlation(correlation, q=q), dtype=np.float64)
    assert np.allclose(np.diag(denoised), 1.0, atol=1e-8)
    assert np.allclose(denoised, denoised.T, atol=1e-8)
    assert float(np.linalg.eigvalsh(denoised).min()) > -1e-8


@_SETTINGS
@given(
    sample=arrays(
        dtype=np.float64,
        shape=st.integers(60, 400),
        elements=st.floats(-0.2, 0.2, allow_nan=False, allow_infinity=False, width=64),
    )
)
def test_cvar_never_exceeds_var(sample: np.ndarray) -> None:
    assume(float(np.std(sample)) > 1e-6)
    metrics = calculate_risk_metrics(
        pd.DataFrame({"simulated_returns": sample}), alpha=0.05, method="empirical"
    )
    assert metrics["CVaR"] <= metrics["VaR"] + 1e-12

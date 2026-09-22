"""Tests for mean-semideviation allocation."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

from portfolio_optimisation.optim import mean_semideviation_weights, ssd_dominates

if TYPE_CHECKING:
    from portfolio_optimisation.optim.semideviation import Semideviation


def _returns(seed: int = 23, t: int = 120, n: int = 5) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    means = rng.uniform(-0.001, 0.002, size=n)
    vols = rng.uniform(0.005, 0.03, size=n)
    skew = rng.exponential(scale=vols, size=(t, n)) - vols
    return pd.DataFrame(
        means + rng.normal(scale=vols, size=(t, n)) + skew,
        columns=[f"A{i}" for i in range(n)],
    )


@settings(max_examples=40, deadline=None)
@given(
    measure=st.sampled_from(["absolute", "standard"]),
    risk_aversion=st.floats(0.05, 1.0),
    mix=arrays(np.float64, 5, elements=st.floats(0.0, 1.0)).filter(lambda m: m.sum() > 0.0),
)
def test_optimum_is_not_dominated_in_the_second_order(
    measure: Semideviation, risk_aversion: float, mix: np.ndarray
) -> None:
    returns = _returns()
    optimum = returns @ mean_semideviation_weights(
        returns, risk_aversion=risk_aversion, measure=measure
    )
    rival = returns @ pd.Series(mix / mix.sum(), index=returns.columns)
    assert not (ssd_dominates(rival, optimum) and not ssd_dominates(optimum, rival))


def test_standard_semideviation_prefers_the_asset_without_the_long_left_tail() -> None:
    rng = np.random.default_rng(5)
    draws = rng.exponential(size=(500, 2)) - 1.0
    returns = pd.DataFrame(0.01 * draws * [1.0, -1.0], columns=["right", "left"])
    weights = mean_semideviation_weights(returns, measure="standard")
    assert weights["right"] > weights["left"]


def test_small_trade_off_holds_the_highest_mean_asset() -> None:
    returns = _returns()
    weights = mean_semideviation_weights(returns, risk_aversion=1e-3)
    assert weights.idxmax() == returns.mean().idxmax()
    assert weights.max() == pytest.approx(1.0, abs=1e-6)


def test_rejects_non_positive_trade_off() -> None:
    with pytest.raises(ValueError, match="risk_aversion"):
        mean_semideviation_weights(_returns(), risk_aversion=0.0)

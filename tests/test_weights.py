"""Tests for whole-share allocation.

The programme is solved in floating point, so the budget and the optimality
comparison hold to the solver's tolerance, taken here as one part per million
of the budget. The deviation programme the rewrite replaced measures money in
units of the budget, where a shortfall of a few cents is below the solver's
feasibility tolerance, so it can overspend or stop short. The rewrite is
compared with it only where its answer is within budget, and must do at least
as well there.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from hypothesis import assume, given, settings
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

from portfolio_optimisation import baselines
from portfolio_optimisation.infra.weights import get_discrete_portfolio


@st.composite
def _book(draw: st.DrawFn) -> tuple[pd.Series, pd.DataFrame]:
    n = draw(st.integers(1, 12))
    tickers = [f"T{i}" for i in range(n)]
    raw = draw(arrays(np.float64, n, elements=st.floats(0.01, 1.0)))
    prices = draw(arrays(np.float64, n, elements=st.floats(1.0, 900.0)))
    return pd.Series(raw / raw.sum(), index=tickers), pd.DataFrame([prices], columns=tickers)


@settings(deadline=None)
@given(book=_book(), budget=st.floats(1_000.0, 1_000_000.0))
def test_allocation_spends_no_more_than_the_budget(
    book: tuple[pd.Series, pd.DataFrame], budget: float
) -> None:
    weights, prices = book
    shares, leftover = get_discrete_portfolio(weights, prices, budget)
    spent = sum(count * prices.iloc[-1][ticker] for ticker, count in shares.items())
    assert all(count > 0 for count in shares.values())
    assert leftover >= -1e-6 * budget
    assert abs(spent + leftover - budget) < 1e-6 * budget


@settings(deadline=None)
@given(book=_book(), budget=st.floats(1_000.0, 1_000_000.0))
def test_allocation_error_is_no_worse_than_rounding_down(
    book: tuple[pd.Series, pd.DataFrame], budget: float
) -> None:
    weights, prices = book
    latest = prices.iloc[-1]
    shares, leftover = get_discrete_portfolio(weights, prices, budget)
    target = weights * budget
    floor = np.floor(target / latest)
    chosen = pd.Series(shares, dtype=np.float64).reindex(weights.index, fill_value=0.0)
    optimum = float((target - latest * chosen).abs().sum() + leftover)
    rounded = float((target - latest * floor).abs().sum() + budget - latest @ floor)
    assert optimum <= rounded + 1e-6 * budget


def _deviation(targets: np.ndarray, prices: np.ndarray, shares: np.ndarray) -> float:
    return float(np.abs(targets - prices * shares).sum() + 1.0 - prices @ shares)


@settings(deadline=None)
@given(book=_book(), budget=st.floats(1_000.0, 1_000_000.0))
def test_shortfall_programme_is_no_worse_than_deviation_programme(
    book: tuple[pd.Series, pd.DataFrame], budget: float
) -> None:
    weights, prices = book
    shares, _ = get_discrete_portfolio(weights, prices, budget)
    chosen = pd.Series(shares, dtype=np.float64).reindex(weights.index, fill_value=0.0)
    targets = weights.to_numpy()
    unit_prices = prices.iloc[-1].to_numpy() / budget
    reference = baselines.whole_shares(targets, unit_prices)
    assume(unit_prices @ reference <= 1.0)
    assert _deviation(targets, unit_prices, chosen.to_numpy()) <= (
        _deviation(targets, unit_prices, reference) + 1e-6
    )

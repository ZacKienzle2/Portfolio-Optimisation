"""Reproducibility guarantees for the stochastic paths.

Every Monte Carlo entry point must be deterministic given a seed, so that
research results and the golden-master pipeline can be replicated exactly.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from portfolio_optimisation.infra.repositories import (
    FakeMarketDataRepository,
    InMemoryUnitOfWork,
)
from portfolio_optimisation.risk.copula import (
    CopulaRiskAnalyser,
    run_historical_simulation,
)
from portfolio_optimisation.risk.sharpe import stationary_bootstrap_sharpe_ci
from portfolio_optimisation.services import PortfolioPipeline


def _make_prices(seed: int = 7, n: int = 300, k: int = 6) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    daily_returns = rng.normal(0.0005, 0.012, size=(n, k))
    prices = 100.0 * np.cumprod(1.0 + daily_returns, axis=0)
    dates = pd.date_range("2024-01-02", periods=n, freq="B")
    tickers = [f"T{i}" for i in range(k)]
    return pd.DataFrame(prices, index=dates, columns=tickers)


def _returns_and_weights() -> tuple[pd.DataFrame, pd.Series]:
    prices = _make_prices()
    returns = prices.pct_change(fill_method=None).dropna()
    weights = pd.Series(1.0 / returns.shape[1], index=returns.columns)
    return returns, weights


def test_historical_simulation_is_seed_deterministic() -> None:
    returns, weights = _returns_and_weights()
    first = run_historical_simulation(returns, weights, n_simulations=500, seed=5)
    second = run_historical_simulation(returns, weights, n_simulations=500, seed=5)
    other = run_historical_simulation(returns, weights, n_simulations=500, seed=6)
    pd.testing.assert_frame_equal(first, second)
    assert not first.equals(other)


def test_copula_simulation_is_seed_deterministic() -> None:
    returns, weights = _returns_and_weights()
    analyser = CopulaRiskAnalyser(returns, weights)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        analyser.fit_marginal_distributions()
        analyser.fit_copula()
    first = analyser.run_simulation(n_simulations=400, seed=11)
    second = analyser.run_simulation(n_simulations=400, seed=11)
    other = analyser.run_simulation(n_simulations=400, seed=12)
    pd.testing.assert_frame_equal(first, second)
    assert not first.equals(other)


def test_pipeline_is_seed_deterministic_historical() -> None:
    prices = _make_prices()

    def run() -> dict[str, float]:
        repo = FakeMarketDataRepository(prices)
        pipeline = PortfolioPipeline(
            uow=InMemoryUnitOfWork(repo), n_simulations=500, seed=123
        )
        result = pipeline.run(
            tickers=list(prices.columns),
            start_date="2024-01-02",
            use_copula=False,
        )
        return result.risk_metrics

    assert run() == run()


def test_bootstrap_ci_is_seed_deterministic() -> None:
    rng = np.random.default_rng(3)
    returns = pd.Series(rng.normal(0.0005, 0.01, size=800))
    _, _, samples_a = stationary_bootstrap_sharpe_ci(returns, n_resamples=300, seed=9)
    _, _, samples_b = stationary_bootstrap_sharpe_ci(returns, n_resamples=300, seed=9)
    np.testing.assert_array_equal(samples_a, samples_b)

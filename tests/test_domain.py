"""Service-layer smoke test using the fake repository.

Confirms the Protocol-based pipeline assembles correctly without any
network IO. End-to-end pipeline run lives in the notebook smoke job.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from portfolio_optimisation.domain import MarketDataRepository, UnitOfWork
from portfolio_optimisation.infra.repositories import (
    FakeMarketDataRepository,
    InMemoryUnitOfWork,
)
from portfolio_optimisation.services import PortfolioPipeline


def _make_prices(seed: int = 7, n: int = 250, k: int = 6) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    daily_returns = rng.normal(0.0005, 0.012, size=(n, k))
    prices = 100.0 * np.cumprod(1.0 + daily_returns, axis=0)
    dates = pd.date_range("2024-01-02", periods=n, freq="B")
    tickers = [f"T{i}" for i in range(k)]
    return pd.DataFrame(prices, index=dates, columns=tickers)


def test_repository_protocol_runtime_check() -> None:
    repo = FakeMarketDataRepository(_make_prices())
    assert isinstance(repo, MarketDataRepository)


def test_uow_runtime_check() -> None:
    uow = InMemoryUnitOfWork(FakeMarketDataRepository(_make_prices()))
    assert isinstance(uow, UnitOfWork)


def test_pipeline_runs_with_fake_repo_no_copula() -> None:
    prices = _make_prices()
    repo = FakeMarketDataRepository(prices)
    pipeline = PortfolioPipeline(
        uow=InMemoryUnitOfWork(repo),
        n_simulations=200,
    )
    result = pipeline.run(
        tickers=list(prices.columns),
        start_date="2024-01-02",
        use_copula=False,
    )
    assert abs(result.weights.sum() - 1.0) < 1e-9
    assert "Sharpe Ratio" in result.performance_metrics
    assert "VaR" in result.risk_metrics
    assert not result.simulated_returns.empty
    assert result.metadata["use_copula"] is False

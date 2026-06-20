"""Regression tests for request-validated market-data caching.

The previous cache keyed only on a fixed path and would serve stale data when
the ticker universe or start date changed. These tests pin the corrected
behaviour: the cache is reused only when it genuinely covers the request, and
the served frame is always subset and windowed to what was asked for.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from portfolio_optimisation.infra.data import (
    cache_satisfies_request,
    clean_prices,
)
from portfolio_optimisation.infra.repositories import YfinanceParquetRepository


def _make_prices(seed: int = 7, n: int = 300, k: int = 6) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    daily_returns = rng.normal(0.0005, 0.012, size=(n, k))
    prices = 100.0 * np.cumprod(1.0 + daily_returns, axis=0)
    dates = pd.date_range("2024-01-02", periods=n, freq="B")
    tickers = [f"T{i}" for i in range(k)]
    return pd.DataFrame(prices, index=dates, columns=tickers)


def test_cache_satisfies_request_accepts_covered_subset() -> None:
    prices = _make_prices()
    assert cache_satisfies_request(prices, ["T0", "T1"], "2024-03-01") is True


def test_cache_satisfies_request_rejects_missing_ticker() -> None:
    prices = _make_prices()
    assert cache_satisfies_request(prices, ["T0", "ZZZ"], "2024-03-01") is False


def test_cache_satisfies_request_rejects_earlier_start() -> None:
    prices = _make_prices()
    assert cache_satisfies_request(prices, ["T0"], "2023-01-01") is False


def test_clean_prices_subsets_and_windows() -> None:
    prices = _make_prices()
    cleaned, returns = clean_prices(prices, ["T0", "T2"], "2024-02-01")
    assert list(cleaned.columns) == ["T0", "T2"]
    assert (cleaned.index >= pd.Timestamp("2024-02-01")).all()
    assert not returns.empty


def test_repository_serves_covered_request_from_cache(tmp_path: Path) -> None:
    prices = _make_prices()
    cache_file = tmp_path / "snapshot.parquet"
    prices.to_parquet(cache_file, engine="pyarrow")

    repo = YfinanceParquetRepository(cache_path=cache_file)
    out_prices, out_returns = repo.load_prices(["T0", "T1"], "2024-02-01")

    assert list(out_prices.columns) == ["T0", "T1"]
    assert (out_prices.index >= pd.Timestamp("2024-02-01")).all()
    assert not out_returns.empty

"""Golden-master regression over the default pipeline.

Runs the deterministic HRP allocation and historical risk simulation on the
committed market-data snapshot and compares weights and metrics against stored
reference values, catching silent numerical drift from refactors. The tolerance
absorbs cross-platform BLAS noise while remaining tight enough to flag genuine
formula changes.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from portfolio_optimisation.config import Settings
from portfolio_optimisation.services import build_pipeline_from_settings

_GOLDEN = Path(__file__).parent / "golden" / "pipeline_metrics.json"
_TICKERS = ["IYW", "VGT", "IYF", "IYR", "XLU", "IYK"]
_ATOL = 1e-6


def test_pipeline_matches_golden_master() -> None:
    settings = Settings(seed=12345, n_simulations=4000, var_method="empirical")
    pipeline = build_pipeline_from_settings(settings)
    result = pipeline.run(
        tickers=_TICKERS,
        start_date="2017-01-01",
        use_copula=False,
        linkage_method="ward",
    )
    golden = json.loads(_GOLDEN.read_text(encoding="utf-8"))

    for ticker, expected in golden["weights"].items():
        assert np.isclose(result.weights[ticker], expected, atol=_ATOL), ticker
    for key, expected in golden["performance_metrics"].items():
        assert np.isclose(result.performance_metrics[key], expected, atol=_ATOL), key
    for key, expected in golden["risk_metrics"].items():
        assert np.isclose(result.risk_metrics[key], expected, atol=_ATOL), key

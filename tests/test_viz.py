"""Smoke tests for the visualisation layer.

Confirms matplotlib figures build, return a ``(Figure, Axes)`` pair and save to
disk, and that the Plotly bootstrap figures build from a results frame. No
figure is shown; rendering stays the caller's responsibility.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from portfolio_optimisation.optim.hrp import HRPModel
from portfolio_optimisation.viz import (
    PortfolioVisualiser,
    plot_asset_prices,
    plot_performance_distributions,
    plot_risk_return_profiles,
)


def _returns(seed: int = 0, n: int = 300, k: int = 5) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    data = rng.normal(0.0005, 0.012, size=(n, k))
    index = pd.date_range("2022-01-03", periods=n, freq="B")
    return pd.DataFrame(data, index=index, columns=[f"T{i}" for i in range(k)])


def test_portfolio_figures_build_and_save(tmp_path: Path) -> None:
    returns = _returns()
    hrp = HRPModel(returns)
    hrp.optimize()
    weights = hrp.clean_weights()
    viz = PortfolioVisualiser(returns.mean() * 252, returns.cov() * 252, {"HRP": weights})

    fig, _ = viz.plot_comparative_weights(save_path=tmp_path / "weights.png")
    assert (tmp_path / "weights.png").exists()
    assert fig is not None

    viz.plot_correlation_matrix(hrp, ordered=True, save_path=tmp_path / "corr.svg")
    assert (tmp_path / "corr.svg").exists()

    viz.plot_dendrogram(hrp, save_path=tmp_path / "dendro.png")
    assert (tmp_path / "dendro.png").exists()


def test_bootstrap_plotly_figures_build() -> None:
    results = pd.DataFrame(
        {
            "exp_return": [0.10, 0.12, 0.09, 0.11],
            "volatility": [0.15, 0.16, 0.14, 0.15],
            "sharpe_ratio": [0.60, 0.70, 0.50, 0.65],
            "linkage_method": ["ward", "ward", "single", "single"],
        }
    )
    assert plot_performance_distributions(results, "sharpe_ratio").data
    assert plot_risk_return_profiles(results).data

    prices = 100.0 + _returns().cumsum()
    assert plot_asset_prices(prices).data

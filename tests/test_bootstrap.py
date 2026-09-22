"""Tests for the HRP stationary-bootstrap analyser."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from portfolio_optimisation.optim import HRPAnalyser

if TYPE_CHECKING:
    from pathlib import Path

    import pytest


def _returns(seed: int = 3, t: int = 400, n: int = 5) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame(rng.normal(0.0004, 0.01, size=(t, n)), columns=[f"A{i}" for i in range(n)])


def test_bootstrap_is_seeded_paired_and_cached(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.chdir(tmp_path)
    analyser = HRPAnalyser(_returns())
    analyser.run_bootstrap(["ward", "single"], reps=6, verbose=False, seed=1, paired=True)
    first = analyser.bootstrap_results
    assert first is not None
    assert list(first["linkage_method"]) == ["ward"] * 6 + ["single"] * 6
    assert set(first.columns) == {"exp_return", "volatility", "sharpe_ratio", "linkage_method"}

    again = HRPAnalyser(_returns())
    again.run_bootstrap(["ward", "single"], reps=6, verbose=False, seed=1, paired=True)
    pd.testing.assert_frame_equal(first, again.bootstrap_results)
    assert any(tmp_path.joinpath("cache").iterdir())

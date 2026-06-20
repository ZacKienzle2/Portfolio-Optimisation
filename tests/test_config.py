"""Tests for the typed configuration loader and package errors."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from portfolio_optimisation.config import Settings, load_settings
from portfolio_optimisation.infra.errors import MarketDataError, PortfolioError


@pytest.fixture(autouse=True)
def _clear_portfolio_env(  # pyright: ignore[reportUnusedFunction]
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for key in list(os.environ):
        if key.startswith("PORTFOLIO_"):
            monkeypatch.delenv(key, raising=False)


def test_defaults_when_no_sources(tmp_path: Path) -> None:
    settings = load_settings(tmp_path / "absent.toml")
    assert settings.risk_free_rate == 0.02
    assert settings.n_simulations == 10_000
    assert settings.var_method == "empirical"
    assert settings.seed is None


def test_env_overrides_default(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PORTFOLIO_RISK_FREE_RATE", "0.05")
    monkeypatch.setenv("PORTFOLIO_SEED", "7")
    settings = load_settings(tmp_path / "absent.toml")
    assert settings.risk_free_rate == 0.05
    assert settings.seed == 7


def test_toml_is_read(tmp_path: Path) -> None:
    config = tmp_path / "portfolio.toml"
    config.write_text('[tool.portfolio]\nn_simulations = 250\nlinkage_method = "single"\n')
    settings = load_settings(config)
    assert settings.n_simulations == 250
    assert settings.linkage_method == "single"


def test_explicit_override_beats_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PORTFOLIO_N_SIMULATIONS", "999")
    settings = load_settings(tmp_path / "absent.toml", n_simulations=42)
    assert settings.n_simulations == 42


def test_path_fields_are_paths(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PORTFOLIO_CACHE_DIR", "artefacts")
    settings = load_settings(tmp_path / "absent.toml")
    assert isinstance(settings.cache_dir, Path)
    assert settings.cache_dir == Path("artefacts")


def test_to_dict_serialises_paths() -> None:
    serialised = Settings().to_dict()
    assert isinstance(serialised["cache_dir"], str)
    assert isinstance(serialised["data_cache_path"], str)


def test_market_data_error_is_value_error() -> None:
    error = MarketDataError(["B", "A"], "2020-01-01")
    assert isinstance(error, ValueError)
    assert isinstance(error, PortfolioError)
    assert error.tickers == ["B", "A"]
    assert "A" in str(error)
    assert "2020-01-01" in str(error)

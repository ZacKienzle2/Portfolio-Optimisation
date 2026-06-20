"""Tests for the portfolio-opt command-line interface.

The ``run`` test exercises the full pipeline offline by resolving the default
``Initial_Files/market_data.parquet`` snapshot (a tracked fixture), so no
network access is required.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from portfolio_optimisation import __version__
from portfolio_optimisation.cli import main


def test_cli_version(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["version"]) == 0
    assert capsys.readouterr().out.strip() == __version__


def test_cli_config_outputs_json(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["config"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert "risk_free_rate" in payload
    assert isinstance(payload["cache_dir"], str)


def test_cli_run_uses_committed_dataset(tmp_path: Path) -> None:
    output = tmp_path / "result.json"
    code = main(
        [
            "run",
            "--tickers",
            "IYW",
            "VGT",
            "IYF",
            "--start",
            "2018-01-01",
            "--no-copula",
            "--output",
            str(output),
        ]
    )
    assert code == 0
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert abs(sum(payload["weights"].values()) - 1.0) < 1e-6
    assert "VaR" in payload["risk_metrics"]
    assert "Sharpe Ratio" in payload["performance_metrics"]

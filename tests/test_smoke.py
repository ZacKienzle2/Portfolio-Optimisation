"""Import-only smoke tests.

These guard the package layout: every subpackage must import without side
effects so a broken refactor surfaces in CI before any logic test runs.
"""

from __future__ import annotations

import importlib


def test_root_package_importable() -> None:
    module = importlib.import_module("portfolio_optimisation")
    assert hasattr(module, "__version__")


def test_subpackage_imports() -> None:
    for name in (
        "portfolio_optimisation.domain",
        "portfolio_optimisation.econometrics",
        "portfolio_optimisation.infra",
        "portfolio_optimisation.optim",
        "portfolio_optimisation.risk",
        "portfolio_optimisation.sde",
        "portfolio_optimisation.viz",
    ):
        importlib.import_module(name)


def test_risk_exports() -> None:
    risk = importlib.import_module("portfolio_optimisation.risk")
    for symbol in (
        "CopulaRiskAnalyser",
        "calculatePerformanceMetrics",
        "calculateRiskMetrics",
        "plotSimulationResults",
        "run_historical_simulation",
    ):
        assert hasattr(risk, symbol), symbol


def test_optim_exports() -> None:
    optim = importlib.import_module("portfolio_optimisation.optim")
    for symbol in ("HRPAnalyser", "HRPModel"):
        assert hasattr(optim, symbol), symbol


def test_econometrics_exports() -> None:
    econometrics = importlib.import_module("portfolio_optimisation.econometrics")
    assert hasattr(econometrics, "Econometrics")


def test_sde_exports() -> None:
    sde = importlib.import_module("portfolio_optimisation.sde")
    assert hasattr(sde, "SDEFitter")


def test_viz_exports() -> None:
    viz = importlib.import_module("portfolio_optimisation.viz")
    assert hasattr(viz, "PortfolioVisualiser")


def test_infra_exports() -> None:
    infra = importlib.import_module("portfolio_optimisation.infra")
    for symbol in (
        "generateFinalReport",
        "getData",
        "get_discrete_portfolio",
        "inverseVarianceWeights",
    ):
        assert hasattr(infra, symbol), symbol

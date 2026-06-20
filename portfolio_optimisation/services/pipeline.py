"""Portfolio analysis pipeline service.

Orchestrates the standard fetch -> optimise -> analyse -> report flow
without coupling the domain to yfinance or parquet IO directly. Inject
any :class:`MarketDataRepository` to substitute the data source.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import pandas as pd
from rich.console import Console

from portfolio_optimisation.config import Settings
from portfolio_optimisation.domain.repositories import (
    UnitOfWork,
)
from portfolio_optimisation.infra.repositories import (
    InMemoryUnitOfWork,
    YfinanceParquetRepository,
)
from portfolio_optimisation.optim.hrp import HRPModel
from portfolio_optimisation.risk.copula import (
    CopulaRiskAnalyser,
    run_historical_simulation,
)
from portfolio_optimisation.risk.metrics import (
    calculate_performance_metrics,
    calculate_risk_metrics,
)


@dataclass
class PortfolioResult:
    """Bundle of everything the standard pipeline emits."""

    prices: pd.DataFrame
    returns: pd.DataFrame
    weights: pd.Series
    portfolio_returns: pd.Series
    risk_metrics: dict[str, float]
    performance_metrics: dict[str, float]
    simulated_returns: pd.DataFrame
    metadata: dict[str, object] = field(default_factory=dict)


class PortfolioPipeline:
    """Run an HRP optimisation + risk simulation in one call.

    The pipeline is the only place that wires together the optimisation,
    risk and reporting modules; consumers (notebooks, CLI, services) talk
    to this class rather than to the leaves individually.
    """

    def __init__(
        self,
        uow: UnitOfWork,
        risk_free_rate: float = 0.02,
        n_simulations: int = 10_000,
        var_alpha: float = 0.05,
        var_method: Literal["empirical", "parametric"] = "empirical",
        *,
        seed: int | None = None,
    ) -> None:
        self.uow: UnitOfWork = uow
        self.risk_free_rate: float = risk_free_rate
        self.n_simulations: int = n_simulations
        self.var_alpha: float = var_alpha
        self.var_method: Literal["empirical", "parametric"] = var_method
        self.seed: int | None = seed

    def run(
        self,
        tickers: list[str],
        start_date: str,
        *,
        use_copula: bool = True,
        linkage_method: str = "ward",
    ) -> PortfolioResult:
        """Execute the standard pipeline and return aggregated results.

        Args:
            tickers (list[str]): Asset symbols.
            start_date (str): ISO date for history start.
            use_copula (bool): If True simulate via t-copula; if False
                fall back to bootstrap of historical returns.
            linkage_method (str): Hierarchical-clustering linkage.

        Returns:
            PortfolioResult: Aggregated artefacts for downstream reporting.
        """
        with self.uow as uow:
            prices, returns = uow.market_data.load_prices(tickers, start_date)

            hrp = HRPModel(returns)
            hrp.optimize(linkage_method=linkage_method)
            weights = hrp.clean_weights()

            portfolio_returns = returns.dot(weights.reindex(returns.columns).fillna(0.0))

            if use_copula:
                analyser = CopulaRiskAnalyser(returns, weights)
                analyser.fit_marginal_distributions()
                analyser.fit_copula()
                simulated = analyser.run_simulation(
                    n_simulations=self.n_simulations, seed=self.seed
                )
            else:
                simulated = run_historical_simulation(
                    returns, weights, n_simulations=self.n_simulations, seed=self.seed
                )

            risk = calculate_risk_metrics(simulated, alpha=self.var_alpha, method=self.var_method)
            perf = calculate_performance_metrics(
                portfolio_returns, risk_free_rate=self.risk_free_rate
            )

            uow.commit()

        return PortfolioResult(
            prices=prices,
            returns=returns,
            weights=weights,
            portfolio_returns=portfolio_returns,
            risk_metrics=risk,
            performance_metrics=perf,
            simulated_returns=simulated,
            metadata={
                "linkage_method": linkage_method,
                "use_copula": use_copula,
                "n_simulations": self.n_simulations,
                "var_alpha": self.var_alpha,
                "var_method": self.var_method,
                "seed": self.seed,
            },
        )


def build_default_pipeline(
    risk_free_rate: float = 0.02,
    n_simulations: int = 10_000,
    *,
    seed: int | None = None,
) -> PortfolioPipeline:
    """Wire a pipeline backed by the production yfinance+parquet repository."""
    repo = YfinanceParquetRepository()
    uow = InMemoryUnitOfWork(repo)
    return PortfolioPipeline(
        uow=uow,
        risk_free_rate=risk_free_rate,
        n_simulations=n_simulations,
        seed=seed,
    )


def build_pipeline_from_settings(
    settings: Settings, *, console: Console | None = None
) -> PortfolioPipeline:
    """Wire a production pipeline from a :class:`Settings` instance.

    Args:
        settings (Settings): Resolved run configuration.
        console (Console | None): Rich console for data-layer status output.

    Returns:
        PortfolioPipeline: A pipeline backed by the yfinance+parquet repository
        whose cache path and run knobs come from ``settings``.
    """
    repo = YfinanceParquetRepository(cache_path=settings.data_cache_path, console=console)
    uow = InMemoryUnitOfWork(repo)
    return PortfolioPipeline(
        uow=uow,
        risk_free_rate=settings.risk_free_rate,
        n_simulations=settings.n_simulations,
        var_alpha=settings.var_alpha,
        var_method=settings.var_method,
        seed=settings.seed,
    )

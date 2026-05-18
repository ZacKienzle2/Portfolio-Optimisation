from typing import Dict, List, Optional, Tuple

from rich.console import Console
from rich.table import Table


def generateFinalReport(console: Console, metricsData: Dict[str, Dict[str, float]]):
    """Render portfolio performance and risk metrics in a Rich table.

    Args:
        console (Console): Rich console for output.
        metricsData (Dict[str, Dict[str, float]]): Nested dict mapping
            portfolio name -> metric name -> value.
    """
    portfolioNames: List[str] = list(metricsData.keys())
    table = Table(
        show_header=True,
        header_style="bold cyan",
        title="\nPortfolio & Risk Analysis",
        title_style="bold magenta",
    )
    table.add_column("Metric", style="dim", width=25)
    for name in portfolioNames:
        table.add_column(name, justify="right")

    rows: List[Tuple[str, str, Optional[str]]] = [
        ("Annualised Return", ".2%", "green"),
        ("Annualised Volatility", ".2%", None),
        ("Sharpe Ratio", ".2f", "bold"),
        ("Sortino Ratio", ".2f", "bold"),
        ("Max Drawdown", ".2%", "red"),
        ("Daily VaR (95%)", ".2%", "red"),
        ("Daily CVaR (95%)", ".2%", "bold red"),
    ]
    keyMap: Dict[str, str] = {
        "Daily VaR (95%)": "VaR",
        "Daily CVaR (95%)": "CVaR",
    }

    for i, (metric, fmt, style) in enumerate(rows):
        if i in [2, 5]:
            table.add_section()
        key: str = keyMap.get(metric, metric)

        values: List[str] = [f"{metricsData[p][key]:{fmt}}" for p in portfolioNames]

        if style:
            table.add_row(metric, *values, style=style)
        else:
            table.add_row(metric, *values)

    console.print(table)

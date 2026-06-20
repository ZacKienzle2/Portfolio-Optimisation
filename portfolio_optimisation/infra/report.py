"""Rich-console rendering of the final portfolio report."""

from rich.console import Console
from rich.table import Table


def generate_final_report(console: Console, metrics_data: dict[str, dict[str, float]]):
    """Render portfolio performance and risk metrics in a Rich table.

    Args:
        console (Console): Rich console for output.
        metrics_data (Dict[str, Dict[str, float]]): Nested dict mapping
            portfolio name -> metric name -> value.
    """
    portfolio_names: list[str] = list(metrics_data.keys())
    table = Table(
        show_header=True,
        header_style="bold cyan",
        title="\nPortfolio & Risk Analysis",
        title_style="bold magenta",
    )
    table.add_column("Metric", style="dim", width=25)
    for name in portfolio_names:
        table.add_column(name, justify="right")

    rows: list[tuple[str, str, str | None]] = [
        ("Annualised Return", ".2%", "green"),
        ("Annualised Volatility", ".2%", None),
        ("Sharpe Ratio", ".2f", "bold"),
        ("Sortino Ratio", ".2f", "bold"),
        ("Max Drawdown", ".2%", "red"),
        ("Daily VaR (95%)", ".2%", "red"),
        ("Daily CVaR (95%)", ".2%", "bold red"),
    ]
    key_map: dict[str, str] = {
        "Daily VaR (95%)": "VaR",
        "Daily CVaR (95%)": "CVaR",
    }

    for i, (metric, fmt, style) in enumerate(rows):
        if i in [2, 5]:
            table.add_section()
        key: str = key_map.get(metric, metric)

        values: list[str] = [f"{metrics_data[p][key]:{fmt}}" for p in portfolio_names]

        if style:
            table.add_row(metric, *values, style=style)
        else:
            table.add_row(metric, *values)

    console.print(table)

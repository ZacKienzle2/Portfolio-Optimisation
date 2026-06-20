"""Plotly visualisations for HRP stationary-bootstrap diagnostics.

These consume the ``bootstrap_results`` frame produced by
:class:`portfolio_optimisation.optim.bootstrap.HRPAnalyser`, keeping plotting in
the visualisation layer so the optimisation layer carries no Plotly dependency.
Each function returns a :class:`plotly.graph_objects.Figure`; rendering or
export is the caller's choice.
"""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from portfolio_optimisation.viz.style import plotly_template

_LEGEND_TOP = {
    "orientation": "h",
    "yanchor": "bottom",
    "y": 1.02,
    "xanchor": "center",
    "x": 0.5,
}


def plot_asset_prices(prices: pd.DataFrame) -> go.Figure:
    """Grid of historical price series, one panel per asset."""
    tickers = list(prices.columns)
    n_rows = (len(tickers) + 1) // 2
    fig = make_subplots(rows=n_rows, cols=2, subplot_titles=tickers)
    for i, ticker in enumerate(tickers):
        fig.add_trace(
            go.Scatter(x=prices.index, y=prices[ticker], name=ticker),
            row=i // 2 + 1,
            col=i % 2 + 1,
        )
    fig.update_layout(
        height=250 * n_rows,
        title_text="Daily Prices",
        showlegend=False,
        template=plotly_template(),
    )
    return fig


def plot_performance_distributions(
    results: pd.DataFrame, metric: str = "sharpe_ratio"
) -> go.Figure:
    """Violin plot of a bootstrapped performance metric by linkage method."""
    metric_title = metric.replace("_", " ").title()
    medians = results.groupby("linkage_method")[metric].median().reset_index()
    fig = px.violin(
        results,
        x="linkage_method",
        y=metric,
        color="linkage_method",
        title=f"Bootstrapped {metric_title}s of HRP Portfolios",
        labels={"linkage_method": "Linkage Method", metric: metric_title},
    )
    for _, row in medians.iterrows():
        value = row[metric]
        formatted = (
            f"{value * 100:.2f}%"
            if "return" in metric or "volatility" in metric
            else f"{value:.2f}"
        )
        fig.add_annotation(
            x=row["linkage_method"],
            y=value,
            text=f"Median:<br>{formatted}",
            showarrow=True,
            arrowhead=2,
            ax=0,
            ay=-40,
        )
    fig.update_layout(
        xaxis_title="Linkage Method",
        yaxis_title=metric_title,
        legend_title="Linkage Method",
        legend=_LEGEND_TOP,
        template=plotly_template(),
    )
    return fig


def plot_risk_return_profiles(results: pd.DataFrame) -> go.Figure:
    """Risk-return scatter panels coloured by Sharpe ratio, per linkage method."""
    methods = sorted(results["linkage_method"].unique())
    n_cols = 2
    n_rows = (len(methods) + 1) // n_cols
    fig = make_subplots(
        rows=n_rows,
        cols=n_cols,
        subplot_titles=methods,
        x_title="Annualised Volatility (Risk)",
        y_title="Annualised Expected Return",
    )
    sharpe_min = results["sharpe_ratio"].min()
    sharpe_max = results["sharpe_ratio"].max()
    for i, method in enumerate(methods):
        subset = results[results["linkage_method"] == method]
        fig.add_trace(
            go.Scatter(
                x=subset["volatility"],
                y=subset["exp_return"],
                mode="markers",
                name=method,
                hovertext=subset["sharpe_ratio"].round(2),
                marker={
                    "color": subset["sharpe_ratio"],
                    "cmin": sharpe_min,
                    "cmax": sharpe_max,
                    "colorscale": "Cividis",
                    "showscale": (i == 0),
                    "colorbar": {"title": "Sharpe Ratio"} if i == 0 else None,
                },
            ),
            row=i // n_cols + 1,
            col=i % n_cols + 1,
        )
    fig.update_layout(
        title="Risk-Return Profile by Linkage Method",
        showlegend=False,
        height=400 * n_rows,
        template=plotly_template(),
    )
    return fig


def plot_investment_growth(
    results: pd.DataFrame, initial_investment: float, years: int
) -> go.Figure:
    """Projected compounded growth using each method's median expected return."""
    medians = results.groupby("linkage_method")["exp_return"].median().reset_index()
    growth = pd.DataFrame({"Year": range(1, years + 1)})
    for _, row in medians.iterrows():
        rate = row["exp_return"]
        growth[row["linkage_method"]] = initial_investment * ((1 + rate) ** growth["Year"])
    melted = growth.melt(id_vars=["Year"], var_name="Linkage Method", value_name="Investment Value")
    fig = px.line(
        melted,
        x="Year",
        y="Investment Value",
        color="Linkage Method",
        title=(
            f"Projected Growth Over {years} Years (Initial Investment: ${initial_investment:,.0f})"
        ),
    )
    fig.update_layout(
        yaxis_title="Investment Value ($)",
        legend=_LEGEND_TOP,
        template=plotly_template(),
    )
    return fig

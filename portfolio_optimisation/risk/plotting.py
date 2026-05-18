from typing import Dict, Optional

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.axes import Axes


def plotSimulationResults(
    simulatedReturns: pd.DataFrame,
    riskMetrics: Dict[str, float],
):
    """Visualise the distribution of simulated returns with VaR/CVaR.

    Creates a histogram with density curve for simulated returns, overlaying
    VaR and CVaR lines. A rug plot below highlights tail events.

    Args:
        simulatedReturns (pd.DataFrame): DataFrame with 'simulated_returns'.
        riskMetrics (Dict[str, float]): Dictionary containing 'VaR' and 'CVaR'.
    """
    if simulatedReturns.empty or "simulated_returns" not in simulatedReturns.columns:
        print("Warning: Empty or invalid DataFrame provided for plotting.")
        return

    returnsSeries: pd.Series = simulatedReturns["simulated_returns"]
    varValue: Optional[float] = riskMetrics.get("VaR")
    cvarValue: Optional[float] = riskMetrics.get("CVaR")

    if varValue is None or cvarValue is None:
        print("Warning: Valid VaR or CVaR missing in riskMetrics for plotting.")
        return

    tailReturns: pd.Series = returnsSeries[returnsSeries <= varValue]

    fig: plt.Figure
    axHist: Axes
    axRug: Axes
    fig, (axHist, axRug) = plt.subplots(
        2,
        1,
        figsize=(12, 8),
        sharex=True,
        gridspec_kw={"height_ratios": [0.8, 0.2]},
    )

    plotTitle = "Distribution of Simulated Portfolio Returns"
    axHist = sns.histplot(
        x=returnsSeries,
        bins=70,
        stat="density",
        ax=axHist,
        label="Simulated Returns",
        alpha=0.7,
        color="skyblue",
        kde=False,
    )
    axHist.axvline(
        varValue,
        color="red",
        linestyle="--",
        lw=2,
        label=f"VaR (95%): {varValue:.2%}",
    )
    axHist.axvline(
        cvarValue,
        color="purple",
        linestyle="-",
        lw=2,
        label=f"CVaR (95%): {cvarValue:.2%}",
    )
    axHist.set_title(plotTitle)
    axHist.set_ylabel("Density")
    axHist.legend()
    axHist.grid(True, alpha=0.3)

    axRug = sns.rugplot(x=tailReturns, ax=axRug, color="darkviolet", height=0.5)
    axRug.axvline(cvarValue, color="purple", linestyle="-", lw=2)
    axRug.set_title("Individual Tail Events (Losses Beyond VaR)")
    axRug.set_xlabel("Daily Portfolio Return")
    axRug.set_yticks([])
    axRug.grid(True, axis="x", alpha=0.3)

    fig.tight_layout()
    plt.show()

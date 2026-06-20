import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.axes import Axes


def plot_simulation_results(
    simulated_returns: pd.DataFrame,
    risk_metrics: dict[str, float],
):
    """Visualise the distribution of simulated returns with VaR/CVaR.

    Creates a histogram with density curve for simulated returns, overlaying
    VaR and CVaR lines. A rug plot below highlights tail events.

    Args:
        simulated_returns (pd.DataFrame): DataFrame with 'simulated_returns'.
        risk_metrics (Dict[str, float]): Dictionary containing 'VaR' and 'CVaR'.
    """
    if simulated_returns.empty or "simulated_returns" not in simulated_returns.columns:
        print("Warning: Empty or invalid DataFrame provided for plotting.")
        return

    returns_series: pd.Series = simulated_returns["simulated_returns"]
    var_value: float | None = risk_metrics.get("VaR")
    cvar_value: float | None = risk_metrics.get("CVaR")

    if var_value is None or cvar_value is None:
        print("Warning: Valid VaR or CVaR missing in risk_metrics for plotting.")
        return

    tail_returns: pd.Series = returns_series[returns_series <= var_value]

    fig: plt.Figure
    ax_hist: Axes
    ax_rug: Axes
    fig, (ax_hist, ax_rug) = plt.subplots(
        2,
        1,
        figsize=(12, 8),
        sharex=True,
        gridspec_kw={"height_ratios": [0.8, 0.2]},
    )

    plot_title = "Distribution of Simulated Portfolio Returns"
    ax_hist = sns.histplot(
        x=returns_series,
        bins=70,
        stat="density",
        ax=ax_hist,
        label="Simulated Returns",
        alpha=0.7,
        color="skyblue",
        kde=True,
    )
    ax_hist.axvline(
        var_value,
        color="red",
        linestyle="--",
        lw=2,
        label=f"VaR (95%): {var_value:.2%}",
    )
    ax_hist.axvline(
        cvar_value,
        color="purple",
        linestyle="-",
        lw=2,
        label=f"CVaR (95%): {cvar_value:.2%}",
    )
    ax_hist.set_title(plot_title)
    ax_hist.set_ylabel("Density")
    ax_hist.legend()
    ax_hist.grid(True, alpha=0.3)

    ax_rug = sns.rugplot(x=tail_returns, ax=ax_rug, color="darkviolet", height=0.5)
    ax_rug.axvline(cvar_value, color="purple", linestyle="-", lw=2)
    ax_rug.set_title("Individual Tail Events (Losses Beyond VaR)")
    ax_rug.set_xlabel("Daily Portfolio Return")
    ax_rug.set_yticks([])
    ax_rug.grid(True, axis="x", alpha=0.3)

    fig.tight_layout()
    plt.show()

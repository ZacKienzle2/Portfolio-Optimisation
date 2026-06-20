"""Cohesive, publication-grade figure styling.

A single :func:`configure_style` applies matplotlib rcParams and a colour-blind
safe palette so every matplotlib and Plotly figure in the project shares one
visual language suitable for a quantitative report. Helpers standardise figure
export so artefacts are reproducible and archivable.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib as mpl
from matplotlib.figure import Figure

# Okabe-Ito qualitative palette: distinguishable under all common colour-vision
# deficiencies. Used as the categorical colour cycle across the project.
OKABE_ITO: tuple[str, ...] = (
    "#0072B2",  # blue
    "#D55E00",  # vermillion
    "#009E73",  # green
    "#CC79A7",  # reddish purple
    "#E69F00",  # orange
    "#56B4E9",  # sky blue
    "#F0E442",  # yellow
    "#000000",  # black
)

# Perceptually-uniform sequential map and a colour-blind-safe diverging map
# (for correlations centred on zero).
SEQUENTIAL_CMAP = "cividis"
DIVERGING_CMAP = "PuOr"

_RC_PARAMS: dict[str, Any] = {
    "figure.dpi": 120,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "figure.figsize": (9.0, 5.5),
    "figure.constrained_layout.use": True,
    "font.family": "serif",
    "mathtext.fontset": "dejavuserif",
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.titleweight": "bold",
    "axes.labelsize": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "axes.axisbelow": True,
    "grid.alpha": 0.3,
    "grid.linewidth": 0.6,
    "legend.frameon": False,
    "xtick.labelsize": 9.5,
    "ytick.labelsize": 9.5,
    "image.cmap": SEQUENTIAL_CMAP,
}


def configure_style() -> None:
    """Apply the project matplotlib rcParams and colour cycle.

    Idempotent: safe to call before every figure so styling is consistent
    regardless of import order or notebook cell execution order.
    """
    for key, value in _RC_PARAMS.items():
        mpl.rcParams[key] = value
    mpl.rcParams["axes.prop_cycle"] = mpl.cycler(color=list(OKABE_ITO))


def save_figure(fig: Figure, path: str | Path, *, dpi: int = 300) -> Path:
    """Save ``fig`` with a tight bounding box, creating parent directories.

    Args:
        fig (Figure): Figure to write.
        path (str | Path): Destination path; the suffix selects the format.
        dpi (int): Output resolution for raster formats.

    Returns:
        Path: The resolved output path.
    """
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=dpi, bbox_inches="tight")
    return out


def plotly_template() -> str:
    """Register and return the name of the shared Plotly template.

    Mirrors the matplotlib palette and serif typography so interactive Plotly
    figures and static matplotlib figures read as one family.
    """
    import plotly.graph_objects as go
    import plotly.io as pio

    template = go.layout.Template(
        layout={
            "colorway": list(OKABE_ITO),
            "font": {"family": "Georgia, 'Times New Roman', serif", "size": 13},
            "paper_bgcolor": "white",
            "plot_bgcolor": "white",
            "xaxis": {"gridcolor": "#E6E6E6", "zeroline": False},
            "yaxis": {"gridcolor": "#E6E6E6", "zeroline": False},
        }
    )
    pio.templates["portfolio"] = template
    return "portfolio"

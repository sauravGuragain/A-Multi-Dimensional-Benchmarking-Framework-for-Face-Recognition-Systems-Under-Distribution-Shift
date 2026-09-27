"""
faceeval.visualization.style
==============================
Shared publication-quality matplotlib theme for all FaceEval-X figures.

Design decisions
----------------
* IEEE / ACM double-column figure width: 3.5 inches per column.
  Full-width figures: 7.16 inches.  We default to 7.0 inches wide.
* Font: Computer Modern (LaTeX default) at 10pt body / 9pt axis labels.
  Falls back to DejaVu Sans when LaTeX is unavailable (CI environments).
* Color palette: colorblind-safe Okabe-Ito 8-color palette, extended with
  a sequential palette for continuous variables.
* All figures are exported at 300 DPI for print quality.
* Grid lines: light grey, major only, no spines on top/right.
"""

from __future__ import annotations

import logging
from typing import Any

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Color palette (Okabe-Ito, colorblind-safe)
# ---------------------------------------------------------------------------

# Eight distinct colors accessible to people with the most common forms
# of color blindness (deuteranopia, protanopia).
OKABE_ITO = [
    "#0072B2",   # blue        — traditional ML models
    "#E69F00",   # orange      — DL models
    "#009E73",   # green       — blur perturbations
    "#CC79A7",   # magenta     — noise perturbations
    "#56B4E9",   # sky blue    — photometric
    "#F0E442",   # yellow      — geometric
    "#D55E00",   # vermillion  — occlusion
    "#000000",   # black       — baseline / reference
]

# Paradigm colors
TRADITIONAL_COLOR = OKABE_ITO[0]   # blue
DL_COLOR          = OKABE_ITO[1]   # orange

# Perturbation category colors
PERTURB_COLORS = {
    "blur":        OKABE_ITO[2],
    "noise":       OKABE_ITO[3],
    "photometric": OKABE_ITO[4],
    "geometric":   OKABE_ITO[5],
    "occlusion":   OKABE_ITO[6],
    "compression": OKABE_ITO[7],
    "resolution":  "#888888",
}

# Sequential palette for continuous heatmaps
SEQUENTIAL_CMAP = "viridis"
DIVERGING_CMAP  = "RdYlGn"

# ---------------------------------------------------------------------------
# Figure size presets
# ---------------------------------------------------------------------------

FIGURE_SIZES = {
    "single_column": (3.5, 2.8),
    "double_column": (7.0, 3.5),
    "square":        (5.0, 5.0),
    "wide":          (8.0, 3.5),
    "tall":          (5.0, 7.0),
    "radar":         (5.5, 5.5),
    "confusion":     (6.0, 5.0),
}

# ---------------------------------------------------------------------------
# Matplotlib RC parameters
# ---------------------------------------------------------------------------

FACEEVAL_RC: dict[str, Any] = {
    # Font
    "font.size":         10,
    "axes.titlesize":    10,
    "axes.labelsize":    9,
    "xtick.labelsize":   8,
    "ytick.labelsize":   8,
    "legend.fontsize":   8,
    "figure.titlesize":  11,
    # Lines
    "lines.linewidth":   1.8,
    "lines.markersize":  5,
    "patch.linewidth":   0.8,
    # Axes
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.grid":         True,
    "axes.grid.which":   "major",
    "grid.color":        "#E0E0E0",
    "grid.linewidth":    0.6,
    "axes.prop_cycle":   matplotlib.cycler(color=OKABE_ITO),
    # Figure
    "figure.dpi":        150,
    "savefig.dpi":       300,
    "savefig.bbox":      "tight",
    "savefig.pad_inches":0.05,
    # Legend
    "legend.framealpha": 0.9,
    "legend.edgecolor":  "#CCCCCC",
    "legend.frameon":    True,
}

_STYLE_APPLIED = False


def apply_style() -> None:
    """Apply FaceEval-X matplotlib RC parameters globally."""
    global _STYLE_APPLIED
    if _STYLE_APPLIED:
        return
    try:
        matplotlib.rcParams.update(FACEEVAL_RC)
        _STYLE_APPLIED = True
        logger.debug("FaceEval-X matplotlib style applied.")
    except Exception as exc:
        logger.warning("Could not apply matplotlib style: %s", exc)


def figure(
    size: str | tuple[float, float] = "double_column",
    title: str | None = None,
) -> tuple[plt.Figure, plt.Axes]:
    """
    Create a styled figure and axes pair.

    Parameters
    ----------
    size:
        Preset name from FIGURE_SIZES or (width, height) tuple in inches.
    title:
        Optional suptitle for the figure.

    Returns
    -------
    (fig, ax)
    """
    apply_style()
    figsize = FIGURE_SIZES.get(size, size) if isinstance(size, str) else size
    fig, ax = plt.subplots(figsize=figsize)
    if title:
        fig.suptitle(title, y=1.02)
    return fig, ax


def multi_figure(
    nrows: int,
    ncols: int,
    size: str | tuple[float, float] = "wide",
    title: str | None = None,
    **subplots_kwargs: Any,
) -> tuple[plt.Figure, Any]:
    """Create a multi-panel styled figure."""
    apply_style()
    figsize = FIGURE_SIZES.get(size, size) if isinstance(size, str) else size
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, **subplots_kwargs)
    if title:
        fig.suptitle(title, y=1.02)
    return fig, axes


def model_color(model_name: str, paradigm: str | None = None) -> str:
    """Return a consistent color for a model based on its name."""
    # DL models
    dl_models = {"facenet", "arcface", "insightface", "dlib_fr"}
    if paradigm == "deep_learning" or model_name in dl_models:
        idx = list(dl_models).index(model_name) if model_name in dl_models else 0
        dl_colors = [OKABE_ITO[1], "#E69F00", "#F5A623", "#FFB733"]
        return dl_colors[idx % len(dl_colors)]
    # Traditional models
    trad_models = ["eigenfaces", "fisherfaces", "pca_svm", "lbph", "hog_svm", "knn"]
    idx = trad_models.index(model_name) if model_name in trad_models else 0
    trad_colors = [OKABE_ITO[0], "#56B4E9", "#009E73", "#0055A0", "#003080", "#002050"]
    return trad_colors[idx % len(trad_colors)]


def add_watermark(ax: plt.Axes, text: str = "FaceEval-X") -> None:
    """Add a light watermark to an axes."""
    ax.text(
        0.99, 0.01, text,
        transform=ax.transAxes,
        ha="right", va="bottom",
        fontsize=6, alpha=0.3, color="grey",
    )


def format_severity_axis(ax: plt.Axes) -> None:
    """Standard formatting for severity (x) axis."""
    ax.set_xlabel("Perturbation Severity")
    ax.set_xlim(-0.02, 1.02)
    ax.set_xticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.xaxis.set_tick_params(which="minor", bottom=False)


def format_accuracy_axis(ax: plt.Axes, ylim: tuple[float, float] = (0.0, 1.05)) -> None:
    """Standard formatting for accuracy (y) axis."""
    ax.set_ylabel("Accuracy")
    ax.set_ylim(*ylim)
    ax.set_yticks(np.arange(0.0, 1.1, 0.2))

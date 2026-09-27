"""
faceeval.visualization.calibration_plot
=========================================
Calibration reliability diagrams and ECE-vs-severity plots.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from faceeval.core.types import CalibrationMetrics, ModelName
from faceeval.visualization.style import (
    apply_style, figure, model_color, add_watermark,
)


def plot_reliability_diagram(
    calibrations: list[CalibrationMetrics],
    title: str = "Reliability Diagrams",
    figsize: str | tuple = "double_column",
) -> plt.Figure:
    """
    Plot reliability diagrams (one per model) on a shared figure.

    A perfectly calibrated model would lie on the diagonal y=x.
    Bars above diagonal = underconfident; below = overconfident.
    """
    apply_style()
    n = len(calibrations)
    ncols = min(n, 3)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.5 * ncols, 3.5 * nrows))
    axes_flat = np.array(axes).flatten() if n > 1 else [axes]

    for i, cal in enumerate(calibrations):
        ax = axes_flat[i]
        bins = cal.reliability_diagram_bins
        if not bins:
            ax.set_visible(False)
            continue

        confs = [b[0] for b in bins]
        accs  = [b[1] for b in bins]
        counts= [b[2] for b in bins]
        widths = [0.08] * len(confs)

        color = model_color(cal.model_name)
        ax.bar(confs, accs, width=widths, alpha=0.7, color=color,
               label="Accuracy", align="center")
        ax.plot([0, 1], [0, 1], "--", color="#888888", linewidth=1.0,
                label="Perfect calibration")

        ax.set_title(
            f"{cal.model_name}\nECE={cal.ece:.4f}  MCE={cal.mce:.4f}",
            fontsize=8,
        )
        ax.set_xlabel("Confidence", fontsize=7)
        ax.set_ylabel("Accuracy", fontsize=7)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.05)
        add_watermark(ax)

    for idx in range(n, len(axes_flat)):
        axes_flat[idx].set_visible(False)

    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    return fig


def plot_ece_vs_severity(
    calibration_by_model_severity: dict[str, dict[float, CalibrationMetrics]],
    title: str = "ECE vs. Perturbation Severity",
    figsize: str | tuple = "double_column",
) -> plt.Figure:
    """
    Plot how ECE changes with severity for each model.
    Shows calibration degradation under distribution shift.
    """
    apply_style()
    fig, ax = figure(figsize, title)

    for model_name, by_sev in sorted(calibration_by_model_severity.items()):
        sevs = sorted(by_sev.keys())
        eces = [by_sev[s].ece for s in sevs]
        color = model_color(model_name)
        ax.plot(sevs, eces, marker="s", markersize=4, color=color,
                label=model_name, linewidth=1.8)

    ax.set_xlabel("Perturbation Severity")
    ax.set_ylabel("Expected Calibration Error (ECE)")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(bottom=0)
    ax.legend(loc="upper left", fontsize=7)
    add_watermark(ax)
    fig.tight_layout()
    return fig

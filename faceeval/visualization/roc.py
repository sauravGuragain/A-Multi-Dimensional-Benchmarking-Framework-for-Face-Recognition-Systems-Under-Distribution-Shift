"""
faceeval.visualization.roc
============================
ROC curve and ADC (Accuracy-Degradation Curve) figure generators.
"""

from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from faceeval.core.types import (
    ADCCurve,
    EvaluationResult,
    ModelName,
    ROCCurve,
)
from faceeval.visualization.style import (
    apply_style, figure, format_accuracy_axis,
    format_severity_axis, model_color, add_watermark,
    OKABE_ITO,
)

# ---------------------------------------------------------------------------
# ROC curves
# ---------------------------------------------------------------------------

def plot_roc_curves(
    results: list[EvaluationResult],
    title: str = "ROC Curves",
    show_eer: bool = True,
    figsize: str | tuple = "double_column",
) -> plt.Figure:
    """
    Plot ROC curves for multiple models on one axes.

    Parameters
    ----------
    results:
        EvaluationResult objects; must contain non-None ``roc_curve`` fields.
    title:
        Figure title.
    show_eer:
        Overlay EER operating point markers.
    figsize:
        Size preset or (width, height) tuple.

    Returns
    -------
    matplotlib.figure.Figure
    """
    apply_style()
    fig, ax = figure(figsize, title)

    # Chance line
    ax.plot([0, 1], [0, 1], "--", color="#AAAAAA", linewidth=1.0,
            label="Chance (AUC=0.50)", zorder=1)

    plotted: set[str] = set()
    for result in results:
        if result.roc_curve is None:
            continue
        label_key = result.model_name
        if label_key in plotted:
            continue
        plotted.add(label_key)

        roc = result.roc_curve
        color = model_color(result.model_name, result.model_paradigm.value)
        paradigm_tag = "DL" if result.model_paradigm.value == "deep_learning" else "Trad"
        label = f"{result.model_name} [{paradigm_tag}] AUC={roc.auc:.3f}"

        ax.plot(roc.fpr, roc.tpr, color=color, linewidth=1.8,
                label=label, zorder=3)

        if show_eer and result.eer is not None:
            eer = result.eer
            ax.scatter([eer], [1 - eer], marker="o", s=40,
                       color=color, zorder=5, edgecolors="white", linewidths=0.5)

    ax.set_xlabel("False Positive Rate (FAR)")
    ax.set_ylabel("True Positive Rate (1 - FRR)")
    ax.set_xlim(-0.01, 1.01)
    ax.set_ylim(-0.01, 1.05)
    ax.legend(loc="lower right", fontsize=7)
    if show_eer:
        ax.annotate("● = EER point", xy=(0.72, 0.08), xycoords="axes fraction",
                    fontsize=7, color="grey")
    add_watermark(ax)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# ADC curves
# ---------------------------------------------------------------------------

def plot_adc_curves(
    adc_curves: list[ADCCurve],
    perturbation_type: str,
    title: str | None = None,
    baseline_accs: dict[ModelName, float] | None = None,
    figsize: str | tuple = "double_column",
) -> plt.Figure:
    """
    Plot Accuracy-Degradation Curves for multiple models under one perturbation.

    Parameters
    ----------
    adc_curves:
        ADCCurve objects (one per model).
    perturbation_type:
        Human-readable perturbation name for the x-axis label.
    title:
        Figure title (auto-generated if None).
    baseline_accs:
        Optional dict of model_name → clean accuracy, drawn as horizontal
        dashed reference lines.
    figsize:
        Size preset or (width, height) tuple.
    """
    apply_style()
    auto_title = title or f"ADC — {perturbation_type.replace('_', ' ').title()}"
    fig, ax = figure(figsize, auto_title)

    for curve in sorted(adc_curves, key=lambda c: c.model_name):
        sevs = [p.severity for p in curve.points]
        accs = [p.accuracy for p in curve.points]
        color = model_color(curve.model_name)
        label = f"{curve.model_name}  (AUC={curve.area:.3f})"
        ax.plot(sevs, accs, marker="o", markersize=4, color=color,
                label=label, linewidth=1.8, zorder=3)

    if baseline_accs:
        for model_name, base_acc in baseline_accs.items():
            color = model_color(model_name)
            ax.axhline(base_acc, linestyle=":", linewidth=1.0,
                       color=color, alpha=0.5)

    format_severity_axis(ax)
    format_accuracy_axis(ax)
    ax.set_xlabel(f"Severity ({perturbation_type.replace('_', ' ')})")
    ax.legend(loc="upper right", fontsize=7)
    add_watermark(ax)
    fig.tight_layout()
    return fig


def plot_adc_grid(
    curves_by_model: dict[ModelName, list[ADCCurve]],
    perturbation_types: list[str],
    ncols: int = 4,
    figsize: tuple = (14, 10),
) -> plt.Figure:
    """
    Plot a grid of ADC subplots, one per perturbation type.
    All models overlaid on each subplot.
    """
    apply_style()
    n = len(perturbation_types)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    axes_flat = axes.flatten() if hasattr(axes, "flatten") else [axes]

    for idx, pt in enumerate(perturbation_types):
        ax = axes_flat[idx]
        for model_name, curves in sorted(curves_by_model.items()):
            match = next((c for c in curves if c.perturbation_type.value == pt), None)
            if match is None:
                continue
            sevs = [p.severity for p in match.points]
            accs = [p.accuracy for p in match.points]
            ax.plot(sevs, accs, marker=".", markersize=3,
                    color=model_color(model_name),
                    label=model_name, linewidth=1.4)
        ax.set_title(pt.replace("_", " "), fontsize=8)
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(0, 1.05)
        ax.set_xlabel("Severity", fontsize=7)
        ax.set_ylabel("Accuracy", fontsize=7)

    # Hide unused subplots
    for idx in range(n, len(axes_flat)):
        axes_flat[idx].set_visible(False)

    # Shared legend
    handles, labels = axes_flat[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="lower center",
                   ncol=min(6, len(labels)), fontsize=7,
                   bbox_to_anchor=(0.5, -0.02))

    fig.suptitle("Accuracy-Degradation Curves — All Perturbations", fontsize=10)
    fig.tight_layout()
    return fig

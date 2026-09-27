"""
faceeval.visualization.heatmap
================================
Heatmaps: pairwise distance matrices, overlap matrices, subgroup grids.
Confusion matrices in a separate section below.
"""

from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from faceeval.core.types import EvaluationResult
from faceeval.visualization.style import (
    apply_style, figure, add_watermark,
    SEQUENTIAL_CMAP, DIVERGING_CMAP,
)


# ---------------------------------------------------------------------------
# Generic annotated heatmap
# ---------------------------------------------------------------------------

def plot_heatmap(
    matrix: np.ndarray,
    row_labels: list[str],
    col_labels: list[str],
    title: str = "Heatmap",
    cmap: str = SEQUENTIAL_CMAP,
    vmin: float | None = None,
    vmax: float | None = None,
    fmt: str = ".3f",
    annot: bool = True,
    figsize: tuple = (7.0, 5.5),
    xlabel: str = "",
    ylabel: str = "",
    colorbar_label: str = "",
) -> plt.Figure:
    """
    Create an annotated heatmap with customisable color range and labels.
    """
    apply_style()
    fig, ax = plt.subplots(figsize=figsize)

    im = ax.imshow(
        matrix, cmap=cmap,
        vmin=vmin if vmin is not None else matrix.min(),
        vmax=vmax if vmax is not None else matrix.max(),
        aspect="auto",
    )

    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    if colorbar_label:
        cbar.set_label(colorbar_label, fontsize=8)
    cbar.ax.tick_params(labelsize=7)

    ax.set_xticks(range(len(col_labels)))
    ax.set_yticks(range(len(row_labels)))
    ax.set_xticklabels(col_labels, rotation=40, ha="right", fontsize=7)
    ax.set_yticklabels(row_labels, fontsize=7)
    ax.set_xlabel(xlabel, fontsize=8)
    ax.set_ylabel(ylabel, fontsize=8)
    ax.set_title(title, fontsize=10)

    if annot:
        threshold = (matrix.max() + matrix.min()) / 2.0
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                val = matrix[i, j]
                text_color = "white" if val < threshold else "black"
                ax.text(j, i, format(val, fmt),
                        ha="center", va="center",
                        fontsize=6, color=text_color)

    add_watermark(ax)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Fingerprint distance / overlap heatmaps
# ---------------------------------------------------------------------------

def plot_fingerprint_distance_heatmap(
    distance_matrix: np.ndarray,
    model_names: list[str],
    title: str = "Pairwise Behavioral Fingerprint Distances",
    figsize: tuple = (6.5, 5.5),
) -> plt.Figure:
    """
    Symmetric heatmap of pairwise fingerprint distances.
    Diagonal is zero (same model), shown in dark.
    """
    return plot_heatmap(
        matrix=distance_matrix,
        row_labels=model_names,
        col_labels=model_names,
        title=title,
        cmap="Blues",
        vmin=0.0,
        fmt=".3f",
        figsize=figsize,
        colorbar_label="Euclidean Distance",
    )


def plot_failure_overlap_heatmap(
    overlap_matrix: np.ndarray,
    traditional_models: list[str],
    dl_models: list[str],
    title: str = "Cross-Paradigm Failure Overlap",
    figsize: tuple = (6.0, 4.5),
) -> plt.Figure:
    """
    Heatmap showing failure overlap fraction between traditional and DL models.
    Higher overlap = more shared failures = less benefit from ensembling.
    """
    return plot_heatmap(
        matrix=overlap_matrix,
        row_labels=traditional_models,
        col_labels=dl_models,
        title=title,
        cmap=DIVERGING_CMAP,
        vmin=0.0,
        vmax=1.0,
        fmt=".3f",
        figsize=figsize,
        xlabel="Deep Learning Models",
        ylabel="Traditional ML Models",
        colorbar_label="Overlap Fraction",
    )


def plot_subgroup_accuracy_heatmap(
    model_names: list[str],
    subgroup_keys: list[str],
    accuracy_matrix: np.ndarray,
    title: str = "Per-Subgroup Accuracy",
    figsize: tuple = (8.0, 4.5),
) -> plt.Figure:
    """
    Heatmap of per-subgroup accuracy for all models.
    Reveals systematic fairness disparities at a glance.
    """
    return plot_heatmap(
        matrix=accuracy_matrix,
        row_labels=model_names,
        col_labels=subgroup_keys,
        title=title,
        cmap="RdYlGn",
        vmin=0.0,
        vmax=1.0,
        fmt=".3f",
        figsize=figsize,
        xlabel="Demographic Subgroup",
        ylabel="Model",
        colorbar_label="Accuracy",
    )


# ---------------------------------------------------------------------------
# Confusion matrix
# ---------------------------------------------------------------------------

def plot_confusion_matrix(
    confusion_matrix: list[list[int]],
    class_labels: list[str],
    model_name: str = "",
    normalize: bool = True,
    title: str | None = None,
    figsize: tuple = (6.0, 5.0),
    max_labels: int = 20,
) -> plt.Figure:
    """
    Plot a confusion matrix with optional row-normalisation.

    Parameters
    ----------
    confusion_matrix:
        N×N integer matrix (rows = true, columns = predicted).
    class_labels:
        Subject labels corresponding to matrix rows/columns.
    model_name:
        Used in auto-generated title.
    normalize:
        Divide each row by its sum (shows per-class recall).
    title:
        Override auto-generated title.
    max_labels:
        If n_classes > this, skip per-cell annotations to avoid clutter.
    """
    apply_style()
    cm = np.array(confusion_matrix, dtype=np.float64)

    if normalize:
        row_sums = cm.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        cm = cm / row_sums

    auto_title = title or f"Confusion Matrix — {model_name}" + (" (normalised)" if normalize else "")
    fig, ax = plt.subplots(figsize=figsize)

    im = ax.imshow(cm, cmap="Blues", vmin=0.0, vmax=1.0 if normalize else None)
    cbar = fig.colorbar(im, ax=ax, shrink=0.85)
    cbar.set_label("Recall" if normalize else "Count", fontsize=7)
    cbar.ax.tick_params(labelsize=6)

    n = len(class_labels)
    show_annot = n <= max_labels
    tick_labels = class_labels if n <= max_labels else [
        f"S{i}" for i in range(n)
    ]

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(tick_labels, rotation=65, ha="right",
                       fontsize=max(4, min(7, 80 // n)))
    ax.set_yticklabels(tick_labels, fontsize=max(4, min(7, 80 // n)))
    ax.set_xlabel("Predicted Subject", fontsize=8)
    ax.set_ylabel("True Subject", fontsize=8)
    ax.set_title(auto_title, fontsize=9)

    if show_annot:
        thresh = cm.max() / 2.0
        for i in range(n):
            for j in range(n):
                v = cm[i, j]
                txt = f"{v:.2f}" if normalize else f"{int(v)}"
                ax.text(j, i, txt, ha="center", va="center",
                        fontsize=max(4, min(6, 60 // n)),
                        color="white" if v > thresh else "black")

    add_watermark(ax)
    fig.tight_layout()
    return fig

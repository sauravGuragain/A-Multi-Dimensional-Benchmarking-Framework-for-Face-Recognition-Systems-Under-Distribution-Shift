"""
faceeval.visualization.deployment_chart
=========================================
Deployment score visualizations: bar charts, spider charts, scenario
comparison, and the practitioner ranking chart for Chapter 12.
"""

from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from faceeval.core.types import DeploymentRanking, DeploymentScore, DeploymentScenario
from faceeval.visualization.style import (
    apply_style, model_color, add_watermark,
    TRADITIONAL_COLOR, DL_COLOR, OKABE_ITO,
)

_WEIGHT_DIMS = [
    "accuracy", "robustness", "calibration",
    "fairness", "latency", "memory", "computational_cost",
]
_DIM_COLORS = {d: OKABE_ITO[i % len(OKABE_ITO)] for i, d in enumerate(_WEIGHT_DIMS)}


# ---------------------------------------------------------------------------
# Stacked bar chart — deployment score breakdown
# ---------------------------------------------------------------------------

def plot_deployment_score_bars(
    scores: list[DeploymentScore],
    title: str | None = None,
    figsize: tuple = (8.0, 4.5),
    show_total_label: bool = True,
) -> plt.Figure:
    """
    Horizontal stacked bar chart showing weighted component breakdown per model.

    Each bar = total deployment score; segments = weighted contributions.
    Allows practitioners to see *why* a model scored as it did.
    """
    apply_style()
    sorted_scores = sorted(scores, key=lambda s: s.total_score)
    model_names = [s.model_name for s in sorted_scores]

    fig, ax = plt.subplots(figsize=figsize)
    y_pos = np.arange(len(model_names))
    bar_h = 0.6
    left = np.zeros(len(sorted_scores))

    for dim in _WEIGHT_DIMS:
        vals = [s.weighted_components.get(dim, 0.0) for s in sorted_scores]
        ax.barh(y_pos, vals, left=left, height=bar_h,
                label=dim.replace("_", " ").title(),
                color=_DIM_COLORS[dim], alpha=0.85)
        left += np.array(vals)

    if show_total_label:
        for i, score in enumerate(sorted_scores):
            ax.text(score.total_score + 0.005, i, f"{score.total_score:.3f}",
                    va="center", fontsize=7)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(model_names, fontsize=8)
    ax.set_xlabel("Weighted Deployment Score", fontsize=9)
    ax.set_xlim(0, 1.12)
    ax.set_title(
        title or f"Deployment Scores — {sorted_scores[0].scenario.value.replace('_',' ').title()}",
        fontsize=10,
    )
    ax.legend(loc="lower right", fontsize=7, ncol=2)
    add_watermark(ax)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Cross-scenario ranking heatmap
# ---------------------------------------------------------------------------

def plot_cross_scenario_ranking(
    ranks_by_model_scenario: dict[str, dict[str, int]],
    title: str = "Model Rankings Across Deployment Scenarios",
    figsize: tuple = (8.0, 4.5),
) -> plt.Figure:
    """
    Heatmap where cell (model, scenario) = rank in that scenario.
    Lower rank number = better = darker green.
    """
    apply_style()
    models   = sorted(ranks_by_model_scenario.keys())
    scenarios = sorted(
        {s for ranks in ranks_by_model_scenario.values() for s in ranks}
    )

    matrix = np.array([
        [ranks_by_model_scenario[m].get(s, 99) for s in scenarios]
        for m in models
    ], dtype=float)

    fig, ax = plt.subplots(figsize=figsize)
    # Invert colormap: rank 1 = darkest green
    im = ax.imshow(matrix, cmap="RdYlGn_r", vmin=1, vmax=len(models))
    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("Rank (1 = best)", fontsize=7)
    cbar.ax.tick_params(labelsize=6)

    ax.set_xticks(range(len(scenarios)))
    ax.set_yticks(range(len(models)))
    ax.set_xticklabels(
        [s.replace("_", "\n") for s in scenarios], fontsize=7, ha="center"
    )
    ax.set_yticklabels(models, fontsize=7)
    ax.set_title(title, fontsize=10)

    # Annotate with rank numbers
    for i in range(len(models)):
        for j in range(len(scenarios)):
            rank = matrix[i, j]
            ax.text(j, i, f"#{int(rank)}" if rank < 99 else "N/A",
                    ha="center", va="center", fontsize=8,
                    color="white" if rank > len(models) / 2 else "black",
                    fontweight="bold")

    add_watermark(ax)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Component score comparison (grouped bar chart)
# ---------------------------------------------------------------------------

def plot_component_comparison(
    scores: list[DeploymentScore],
    title: str = "Component Score Comparison",
    figsize: tuple = (9.0, 4.5),
) -> plt.Figure:
    """
    Grouped bar chart comparing raw component scores across models.
    Each group = one dimension; bars within group = models.
    """
    apply_style()
    n_models = len(scores)
    n_dims = len(_WEIGHT_DIMS)
    x = np.arange(n_dims)
    width = 0.75 / n_models

    fig, ax = plt.subplots(figsize=figsize)

    for i, score in enumerate(scores):
        vals = [score.component_scores.get(d, 0.0) for d in _WEIGHT_DIMS]
        offset = (i - n_models / 2 + 0.5) * width
        color = model_color(score.model_name, score.paradigm.value)
        bars = ax.bar(x + offset, vals, width, label=score.model_name,
                      color=color, alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(
        [d.replace("_", "\n") for d in _WEIGHT_DIMS], fontsize=7
    )
    ax.set_ylabel("Component Score (0–1)", fontsize=8)
    ax.set_ylim(0, 1.15)
    ax.set_title(title, fontsize=10)
    ax.legend(loc="upper right", fontsize=7, ncol=min(3, n_models))
    add_watermark(ax)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Sensitivity waterfall chart
# ---------------------------------------------------------------------------

def plot_sensitivity_waterfall(
    sensitivity_data: dict[str, Any],
    model_name: str,
    title: str | None = None,
    figsize: tuple = (7.0, 4.0),
) -> plt.Figure:
    """
    Horizontal bar chart showing how much each weight dimension affects
    model_name's total score (∂S/∂w_d = component_score_d).
    """
    apply_style()
    model_sens = sensitivity_data.get(model_name, {})
    if not model_sens:
        raise ValueError(f"No sensitivity data for model '{model_name}'.")

    dims = sorted(model_sens.keys(), key=lambda d: model_sens[d])
    vals = [model_sens[d] for d in dims]

    fig, ax = plt.subplots(figsize=figsize)
    colors = [OKABE_ITO[i % len(OKABE_ITO)] for i in range(len(dims))]
    ax.barh(range(len(dims)), vals, color=colors, alpha=0.85)
    ax.set_yticks(range(len(dims)))
    ax.set_yticklabels([d.replace("_", " ") for d in dims], fontsize=8)
    ax.set_xlabel("Sensitivity (∂Score/∂Weight = Component Score)", fontsize=8)
    ax.set_title(title or f"Score Sensitivity — {model_name}", fontsize=10)
    ax.set_xlim(0, 1.05)
    for i, v in enumerate(vals):
        ax.text(v + 0.01, i, f"{v:.3f}", va="center", fontsize=7)
    add_watermark(ax)
    fig.tight_layout()
    return fig

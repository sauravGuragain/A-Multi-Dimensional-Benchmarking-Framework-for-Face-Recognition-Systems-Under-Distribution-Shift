"""
faceeval.visualization.radar
==============================
Radar (spider) charts for behavioral fingerprint visualization.

Each spoke of the radar represents one fingerprint dimension.
Multiple models are overlaid on the same chart for direct comparison.
The chart reveals at a glance which model excels on which dimensions
and where each paradigm is strong or weak.
"""

from __future__ import annotations

import math

import matplotlib.pyplot as plt
import numpy as np

from faceeval.core.types import BehavioralFingerprint, ModelParadigm
from faceeval.visualization.style import (
    apply_style, model_color, add_watermark,
    TRADITIONAL_COLOR, DL_COLOR, FIGURE_SIZES,
)


def plot_fingerprint_radar(
    fingerprints: list[BehavioralFingerprint],
    dimensions: list[str] | None = None,
    title: str = "Behavioral Fingerprint Radar",
    figsize: tuple[float, float] | None = None,
) -> plt.Figure:
    """
    Plot behavioral fingerprints as overlaid radar charts.

    Parameters
    ----------
    fingerprints:
        List of BehavioralFingerprint objects (all must share the same
        dimension_names, or a subset is selected via ``dimensions``).
    dimensions:
        Subset of dimension names to display.  Defaults to all dimensions
        in the first fingerprint (up to 16 spokes for readability).
    title:
        Figure suptitle.
    figsize:
        Override figure size in inches.
    """
    apply_style()
    if not fingerprints:
        raise ValueError("At least one fingerprint is required.")

    # Select dimensions
    all_dims = fingerprints[0].dimension_names
    if dimensions is None:
        # Prefer a curated subset for readability: robustness dims + meta dims
        rob_dims = [d for d in all_dims if d.startswith("robustness_")][:8]
        meta_dims = [d for d in all_dims if not d.startswith("robustness_")]
        dimensions = rob_dims + meta_dims
    # Keep only dims present in all fingerprints
    valid_dims = [d for d in dimensions if d in all_dims]
    if not valid_dims:
        raise ValueError("No valid shared dimensions found.")

    n_dims = len(valid_dims)
    if n_dims < 3:
        raise ValueError(f"Radar chart requires at least 3 dimensions; got {n_dims}.")

    # Angles: evenly spaced, closed loop
    angles = [2 * math.pi * i / n_dims for i in range(n_dims)]
    angles.append(angles[0])

    size = figsize or FIGURE_SIZES["radar"]
    fig, ax = plt.subplots(figsize=size, subplot_kw={"projection": "polar"})

    # Draw one polygon per fingerprint
    for fp in fingerprints:
        dim_to_idx = {d: i for i, d in enumerate(fp.dimension_names)}
        values = [fp.vector[dim_to_idx[d]] if d in dim_to_idx else 0.0 for d in valid_dims]
        values.append(values[0])   # close the polygon

        color = model_color(fp.model_name, fp.paradigm.value)
        ls = "-" if fp.paradigm == ModelParadigm.TRADITIONAL else "--"
        paradigm_tag = "Trad" if fp.paradigm == ModelParadigm.TRADITIONAL else "DL"
        label = f"{fp.model_name} [{paradigm_tag}]"

        ax.plot(angles, values, ls, color=color, linewidth=1.8, label=label)
        ax.fill(angles, values, color=color, alpha=0.10)

    # Spoke labels
    ax.set_xticks(angles[:-1])
    short_labels = [_shorten_dim(d) for d in valid_dims]
    ax.set_xticklabels(short_labels, size=7)

    # Radial grid
    ax.set_ylim(0, 1)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(["0.2", "0.4", "0.6", "0.8", "1.0"], size=6, color="grey")

    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.15), fontsize=7)
    ax.set_title(title, pad=15, fontsize=10)
    fig.tight_layout()
    return fig


def plot_paradigm_comparison_radar(
    fingerprints: list[BehavioralFingerprint],
    title: str = "Traditional ML vs Deep Learning — Mean Fingerprint",
    figsize: tuple | None = None,
) -> plt.Figure:
    """
    Plot mean fingerprint vectors for each paradigm on one radar chart.
    Highlights systematic differences between paradigm families.
    """
    apply_style()
    trad = [fp for fp in fingerprints if fp.paradigm == ModelParadigm.TRADITIONAL]
    deep = [fp for fp in fingerprints if fp.paradigm == ModelParadigm.DEEP_LEARNING]

    if not trad or not deep:
        raise ValueError("Need at least one fingerprint per paradigm.")

    dims = fingerprints[0].dimension_names
    dim_to_idx = {d: i for i, d in enumerate(dims)}

    # Compute mean vectors
    def mean_vec(fps: list[BehavioralFingerprint]) -> list[float]:
        mat = np.array([[fp.vector[dim_to_idx.get(d, 0)] for d in dims] for fp in fps])
        return mat.mean(axis=0).tolist()

    trad_mean = mean_vec(trad)
    deep_mean = mean_vec(deep)

    # Create synthetic fingerprints for plotting
    from faceeval.core.types import BehavioralFingerprint
    fp_trad_mean = BehavioralFingerprint(
        model_name="Traditional ML (mean)",
        paradigm=ModelParadigm.TRADITIONAL,
        dimension_names=dims,
        vector=trad_mean,
        scores=dict(zip(dims, trad_mean)),
    )
    fp_deep_mean = BehavioralFingerprint(
        model_name="Deep Learning (mean)",
        paradigm=ModelParadigm.DEEP_LEARNING,
        dimension_names=dims,
        vector=deep_mean,
        scores=dict(zip(dims, deep_mean)),
    )
    return plot_fingerprint_radar([fp_trad_mean, fp_deep_mean], title=title, figsize=figsize)


def _shorten_dim(dim: str) -> str:
    """Convert a dimension name to a short radar spoke label."""
    replacements = {
        "robustness_gaussian_blur": "Blur",
        "robustness_motion_blur": "Motion",
        "robustness_gaussian_noise": "Noise",
        "robustness_salt_pepper_noise": "S&P",
        "robustness_speckle_noise": "Speckle",
        "robustness_brightness": "Bright",
        "robustness_contrast": "Contrast",
        "robustness_gamma": "Gamma",
        "robustness_rotation": "Rotate",
        "robustness_scaling": "Scale",
        "robustness_cropping": "Crop",
        "robustness_random_occlusion": "Occl",
        "robustness_face_mask": "Mask",
        "robustness_sunglasses": "Glasses",
        "robustness_jpeg_compression": "JPEG",
        "robustness_resolution_degradation": "Res↓",
        "calibration_quality": "Cal",
        "fairness_gap": "Fair",
        "speed_score": "Speed",
        "memory_efficiency": "Mem",
        "sample_efficiency": "SampleEff",
    }
    return replacements.get(dim, dim.replace("robustness_", "").replace("_", "\n")[:8])

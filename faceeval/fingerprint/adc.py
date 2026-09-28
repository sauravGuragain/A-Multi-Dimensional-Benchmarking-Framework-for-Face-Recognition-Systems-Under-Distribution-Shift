from __future__ import annotations

import logging
from collections import defaultdict
from typing import Any

import numpy as np

from faceeval.core.types import (
    ADCCurve,
    ADCPoint,
    EvaluationResult,
    ModelName,
    PerturbationType,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# ADC computation
# ---------------------------------------------------------------------------

def compute_adc_curve(
    results_for_perturbation: list[EvaluationResult],
    model_name: ModelName,
    perturbation_type: PerturbationType,
    n_classes: int | None = None,
) -> ADCCurve:
    """
    Build one ADC curve from a list of EvaluationResults at different severities.

    Parameters
    ----------
    results_for_perturbation:
        EvaluationResult objects for one (model × perturbation_type) pair,
        one per severity level.  Must contain at least 2 points.
    model_name:
        Name of the model being fingerprinted.
    perturbation_type:
        The perturbation type these results correspond to.
    n_classes:
        Number of subject classes, used to compute normalised AUC.
        If ``None``, raw AUC is reported.

    Returns
    -------
    ADCCurve
    """
    if not results_for_perturbation:
        raise ValueError(
            f"No results provided for model='{model_name}', "
            f"perturbation='{perturbation_type.value}'"
        )

    # Sort by severity
    sorted_results = sorted(
        results_for_perturbation,
        key=lambda r: r.perturbation_spec.severity if r.perturbation_spec else 0.0,
    )

    points: list[ADCPoint] = []
    for result in sorted_results:
        severity = result.perturbation_spec.severity if result.perturbation_spec else 0.0
        points.append(ADCPoint(
            severity=severity,
            accuracy=result.accuracy,
            perturbation_type=perturbation_type,
        ))

    area = _compute_auc(points, n_classes)

    return ADCCurve(
        model_name=model_name,
        perturbation_type=perturbation_type,
        points=points,
        area=area,
    )


def compute_all_adc_curves(
    evaluation_results: list[EvaluationResult],
    n_classes: int | None = None,
) -> dict[ModelName, list[ADCCurve]]:
    """
    Build all ADC curves for every (model × perturbation_type) pair.

    Parameters
    ----------
    evaluation_results:
        All EvaluationResult objects from one experiment run.
    n_classes:
        Number of subjects — passed to ``compute_adc_curve`` for normalisation.

    Returns
    -------
    dict mapping model_name → list of ADCCurve (one per perturbation type)
    """
    # Group by (model_name, perturbation_type)
    grouped: dict[tuple[str, str], list[EvaluationResult]] = defaultdict(list)
    for result in evaluation_results:
        if result.perturbation_spec is None:
            pt_key = "clean"
        else:
            pt_key = result.perturbation_spec.perturbation_type.value
        grouped[(result.model_name, pt_key)].append(result)

    curves: dict[ModelName, list[ADCCurve]] = defaultdict(list)
    for (model_name, pt_key), results in grouped.items():
        if pt_key == "clean" or len(results) < 2:
            continue
        try:
            pt = PerturbationType(pt_key)
            curve = compute_adc_curve(results, model_name, pt, n_classes)
            curves[model_name].append(curve)
            logger.debug(
                "ADC [%s/%s]: %d points, AUC=%.4f",
                model_name, pt_key, len(curve.points), curve.area,
            )
        except (ValueError, KeyError) as exc:
            logger.warning("Could not compute ADC for (%s, %s): %s", model_name, pt_key, exc)

    return dict(curves)


# ---------------------------------------------------------------------------
# Degradation metrics derived from ADC
# ---------------------------------------------------------------------------

def degradation_rate(curve: ADCCurve) -> float:
    """
    Average per-unit-severity accuracy drop.

    degradation_rate = (accuracy[s=0] - accuracy[s=1]) / 1.0

    A higher value means the model collapses faster under increasing severity.
    """
    if not curve.points:
        return 0.0
    acc_clean = curve.points[0].accuracy
    acc_max   = curve.points[-1].accuracy
    return float(acc_clean - acc_max)


def breakdown_severity(curve: ADCCurve, threshold: float = 0.5) -> float | None:
    """
    Find the severity at which accuracy first drops below ``threshold``.

    Returns ``None`` if accuracy never falls below the threshold.
    A higher breakdown severity = more robust model.
    """
    for i in range(len(curve.points) - 1):
        p0, p1 = curve.points[i], curve.points[i + 1]
        if p0.accuracy >= threshold > p1.accuracy:
            # Linear interpolation
            if abs(p1.accuracy - p0.accuracy) < 1e-9:
                return float(p0.severity)
            t = (threshold - p0.accuracy) / (p1.accuracy - p0.accuracy)
            return float(p0.severity + t * (p1.severity - p0.severity))
    if curve.points and curve.points[0].accuracy < threshold:
        return 0.0
    return None


def relative_robustness(
    curve_a: ADCCurve,
    curve_b: ADCCurve,
) -> float:
    """
    Relative robustness of model A versus model B on the same perturbation.

    Returns (AUC_A - AUC_B), so positive means A is more robust.
    """
    return float(curve_a.area - curve_b.area)


def adc_summary_table(
    curves_by_model: dict[ModelName, list[ADCCurve]],
) -> list[dict[str, Any]]:
    """
    Build a flat summary table suitable for CSV export and LaTeX tables.

    One row per (model × perturbation_type) pair.
    """
    rows: list[dict[str, Any]] = []
    for model_name, curves in sorted(curves_by_model.items()):
        for curve in sorted(curves, key=lambda c: c.perturbation_type.value):
            bd = breakdown_severity(curve)
            rows.append({
                "model_name": model_name,
                "perturbation_type": curve.perturbation_type.value,
                "adc_auc": round(curve.area, 4),
                "clean_accuracy": round(curve.points[0].accuracy, 4) if curve.points else None,
                "max_severity_accuracy": round(curve.points[-1].accuracy, 4) if curve.points else None,
                "degradation_rate": round(degradation_rate(curve), 4),
                "breakdown_severity_at_50pct": round(bd, 4) if bd is not None else None,
                "n_severity_levels": len(curve.points),
            })
    return rows


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _compute_auc(points: list[ADCPoint], n_classes: int | None) -> float:
    """Trapezoidal AUC with optional normalisation by chance-level accuracy."""
    if len(points) < 2:
        return points[0].accuracy if points else 0.0

    severities = np.array([p.severity for p in points], dtype=np.float64)
    accuracies = np.array([p.accuracy for p in points], dtype=np.float64)

    # Sort by severity (should already be sorted, but be safe)
    order = np.argsort(severities)
    severities = severities[order]
    accuracies = accuracies[order]

    raw_auc = float(np.trapz(accuracies, severities))

    if n_classes is not None and n_classes > 1:
        chance = 1.0 / n_classes
        normalised = (raw_auc - chance) / max(1.0 - chance, 1e-9)
        return float(np.clip(normalised, 0.0, 1.0))

    return float(np.clip(raw_auc, 0.0, 1.0))

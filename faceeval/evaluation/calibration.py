from __future__ import annotations

import logging
from typing import Any

import numpy as np

from faceeval.core.exceptions import CalibrationError, InsufficientSamplesError
from faceeval.core.types import (
    CalibrationMetrics,
    ModelName,
    PerturbationSpec,
    PredictionResult,
    SubjectID,
)

logger = logging.getLogger(__name__)

_MIN_SAMPLES = 10
_DEFAULT_N_BINS = 10


# ---------------------------------------------------------------------------
# Main calibration function
# ---------------------------------------------------------------------------

def compute_calibration(
    predictions: list[PredictionResult],
    ground_truth: list[SubjectID],
    model_name: ModelName,
    perturbation_spec: PerturbationSpec | None = None,
    n_bins: int = _DEFAULT_N_BINS,
) -> CalibrationMetrics:
    """
    Compute calibration metrics for a set of predictions.

    Parameters
    ----------
    predictions:
        Prediction results including per-prediction confidence scores.
    ground_truth:
        True subject ID for each prediction (parallel list).
    model_name:
        Name of the model being evaluated.
    perturbation_spec:
        The perturbation condition these predictions were generated under.
    n_bins:
        Number of equal-width confidence bins for ECE computation.
        Default: 10 (the standard from Guo et al., 2017).

    Returns
    -------
    CalibrationMetrics
    """
    if len(predictions) < _MIN_SAMPLES:
        raise InsufficientSamplesError("calibration", _MIN_SAMPLES, len(predictions))

    confidences: list[float] = []
    corrects: list[float] = []

    for pred, true in zip(predictions, ground_truth):
        if pred.confidence is None:
            continue
        conf = max(0.0, min(1.0, pred.confidence))
        correct = 1.0 if pred.predicted_subject_id == true else 0.0
        confidences.append(conf)
        corrects.append(correct)

    if len(confidences) < _MIN_SAMPLES:
        raise CalibrationError(
            f"Insufficient predictions with confidence scores "
            f"({len(confidences)} < {_MIN_SAMPLES})"
        )

    confs = np.array(confidences, dtype=np.float64)
    cors  = np.array(corrects,    dtype=np.float64)

    # --- Reliability diagram bins ---
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_data: list[tuple[float, float, int]] = []

    total_weight = 0.0
    ece = 0.0
    mce = 0.0

    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        in_bin = (confs >= lo) & (confs < hi if hi < 1.0 else confs <= hi)
        n_in_bin = in_bin.sum()

        if n_in_bin == 0:
            continue

        mean_conf = float(confs[in_bin].mean())
        mean_acc  = float(cors[in_bin].mean())
        gap = abs(mean_conf - mean_acc)

        bin_data.append((mean_conf, mean_acc, int(n_in_bin)))
        weight = n_in_bin / len(confs)
        ece += weight * gap
        mce = max(mce, gap)
        total_weight += weight

    # --- Brier score ---
    brier = float(np.mean((confs - cors) ** 2))

    # --- Over/under confidence rates ---
    overconf  = sum(1 for mc, ma, _ in bin_data if mc > ma) / max(len(bin_data), 1)
    underconf = sum(1 for mc, ma, _ in bin_data if mc < ma) / max(len(bin_data), 1)

    return CalibrationMetrics(
        model_name=model_name,
        perturbation_spec=perturbation_spec,
        ece=float(ece),
        mce=float(mce),
        brier_score=brier,
        overconfidence_rate=overconf,
        underconfidence_rate=underconf,
        reliability_diagram_bins=bin_data,
        num_samples=len(confidences),
    )


# ---------------------------------------------------------------------------
# Temperature scaling (post-hoc calibration)
# ---------------------------------------------------------------------------

def find_optimal_temperature(
    logits: list[float],
    labels: list[int],
    n_iter: int = 100,
    lr: float = 0.01,
) -> float:
    """
    Find optimal temperature T for temperature scaling via gradient descent.

    Temperature scaling: p = softmax(logits / T)

    Lower T → sharpens distribution (more confident).
    Higher T → flattens distribution (less confident).

    Parameters
    ----------
    logits:
        Raw model logits (pre-softmax) for the positive class.
    labels:
        Binary labels (1 = correct, 0 = incorrect).
    n_iter:
        Number of optimisation steps.
    lr:
        Learning rate.

    Returns
    -------
    float
        Optimal temperature T ∈ (0, ∞).
    """
    T = 1.0
    logits_arr = np.array(logits, dtype=np.float64)
    labels_arr = np.array(labels, dtype=np.float64)

    for _ in range(n_iter):
        scaled = logits_arr / T
        # Sigmoid for binary calibration
        p = 1.0 / (1.0 + np.exp(-scaled))
        p = np.clip(p, 1e-7, 1 - 1e-7)

        # NLL gradient w.r.t. T
        grad = -np.mean(
            (labels_arr - p) * logits_arr / (T ** 2)
        )
        T -= lr * grad
        T = max(0.01, min(T, 100.0))  # clamp to reasonable range

    return float(T)


def apply_temperature_scaling(
    confidences: list[float],
    temperature: float,
) -> list[float]:
    """
    Apply temperature scaling to a list of confidences.

    Converts confidences to logits (via logit transform), scales by T,
    then converts back via sigmoid.  Expects confidences in (0, 1).
    """
    result: list[float] = []
    for c in confidences:
        c_clip = max(1e-7, min(c, 1 - 1e-7))
        logit  = math.log(c_clip / (1.0 - c_clip))
        scaled = logit / temperature
        calibrated = 1.0 / (1.0 + math.exp(-scaled))
        result.append(float(calibrated))
    return result


# ---------------------------------------------------------------------------
# Calibration summary across severity levels
# ---------------------------------------------------------------------------

def calibration_profile(
    calibration_by_severity: dict[float, CalibrationMetrics],
) -> dict[str, Any]:
    """
    Summarise how calibration degrades across severity levels.

    Returns a dict suitable for visualisation: ECE and Brier score
    vs. severity, plus the overall mean and worst-case ECE.
    """
    if not calibration_by_severity:
        return {}

    items = sorted(calibration_by_severity.items())
    severities = [s for s, _ in items]
    eces   = [m.ece for _, m in items]
    briers = [m.brier_score for _, m in items]

    return {
        "severities": severities,
        "eces": eces,
        "brier_scores": briers,
        "mean_ece": float(np.mean(eces)),
        "max_ece": float(np.max(eces)),
        "mean_brier": float(np.mean(briers)),
        "ece_degradation": float(eces[-1] - eces[0]) if len(eces) > 1 else 0.0,
    }


import math  # noqa: E402  (placed after function body to avoid circular ordering)

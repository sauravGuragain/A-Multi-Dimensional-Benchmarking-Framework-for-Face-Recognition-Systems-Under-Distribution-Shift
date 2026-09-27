from __future__ import annotations

import logging
from typing import Any

import numpy as np

from faceeval.core.types import (
    FailureCase,
    FailureMode,
    ModelName,
    PerturbationSpec,
    PredictionResult,
    SubjectID,
)

logger = logging.getLogger(__name__)

# Thresholds for failure mode classification
_HIGH_CONFIDENCE_THRESHOLD = 0.70   # confidence above this = high-confidence prediction
_LOW_CONFIDENCE_THRESHOLD  = 0.40   # confidence below this = low-confidence prediction


# ---------------------------------------------------------------------------
# Core extraction
# ---------------------------------------------------------------------------

def extract_failures(
    predictions: list[PredictionResult],
    ground_truth: list[SubjectID],
    model_name: ModelName,
    perturbation_spec: PerturbationSpec | None = None,
    high_conf_threshold: float = _HIGH_CONFIDENCE_THRESHOLD,
    low_conf_threshold: float = _LOW_CONFIDENCE_THRESHOLD,
    images: list[np.ndarray] | None = None,
) -> list[FailureCase]:
    """
    Extract all failure cases from one (model × perturbation × severity) batch.

    Parameters
    ----------
    predictions:
        Output of ``BaseRecognizer.predict()`` for this batch.
    ground_truth:
        True subject IDs, parallel to ``predictions``.
    model_name:
        Name of the model producing these predictions.
    perturbation_spec:
        Perturbation condition (``None`` for clean images).
    high_conf_threshold:
        Minimum confidence to classify a wrong prediction as HIGH_CONFIDENCE_WRONG.
    low_conf_threshold:
        Maximum confidence to classify a correct prediction as LOW_CONFIDENCE_CORRECT.
    images:
        Optional preprocessed face arrays (parallel to predictions) for
        computing image-level statistics stored in ``FailureCase.metadata``
        (used later by the meta-model).

    Returns
    -------
    list[FailureCase]
        One ``FailureCase`` per incorrect or borderline prediction.
    """
    cases: list[FailureCase] = []

    for i, (pred, true_id) in enumerate(zip(predictions, ground_truth)):
        conf = pred.confidence or 0.0
        predicted_id = pred.predicted_subject_id or ""
        is_correct = predicted_id == true_id

        mode = _classify_failure_mode(
            is_correct, conf, high_conf_threshold, low_conf_threshold
        )
        if mode is None:
            continue  # correct prediction, not a borderline case

        case = FailureCase(
            image_id=pred.image_id,
            subject_id=true_id,
            model_name=model_name,
            perturbation_spec=perturbation_spec,
            failure_mode=mode,
            predicted_subject_id=predicted_id if not is_correct else None,
            confidence=conf,
            similarity_score=pred.similarity_score,
            true_subject_id=true_id,
        )
        cases.append(case)

    logger.debug(
        "[%s] Extracted %d failure cases from %d predictions "
        "(perturbation=%s)",
        model_name, len(cases), len(predictions),
        perturbation_spec.label if perturbation_spec else "clean",
    )
    return cases


def extract_all_failures(
    predictions_by_condition: dict[str, list[PredictionResult]],
    ground_truth_by_condition: dict[str, list[SubjectID]],
    specs_by_condition: dict[str, PerturbationSpec | None],
    model_name: ModelName,
    **kwargs: Any,
) -> list[FailureCase]:
    """
    Extract failures across all (perturbation × severity) conditions.

    Parameters
    ----------
    predictions_by_condition:
        Dict mapping condition_key → list of PredictionResult.
        condition_key is typically ``PerturbationSpec.label``.
    ground_truth_by_condition:
        Parallel dict of true subject IDs per condition.
    specs_by_condition:
        PerturbationSpec for each condition.

    Returns
    -------
    list[FailureCase]  (all conditions concatenated)
    """
    all_cases: list[FailureCase] = []
    for key in predictions_by_condition:
        cases = extract_failures(
            predictions=predictions_by_condition[key],
            ground_truth=ground_truth_by_condition[key],
            model_name=model_name,
            perturbation_spec=specs_by_condition.get(key),
            **kwargs,
        )
        all_cases.extend(cases)
    logger.info(
        "[%s] Total failures across %d conditions: %d",
        model_name, len(predictions_by_condition), len(all_cases),
    )
    return all_cases


# ---------------------------------------------------------------------------
# Failure statistics
# ---------------------------------------------------------------------------

def failure_statistics(
    cases: list[FailureCase],
    model_name: ModelName | None = None,
) -> dict[str, Any]:
    """
    Compute summary statistics over a list of failure cases.

    Returns counts and rates by failure mode, by perturbation type,
    and by severity level.
    """
    if not cases:
        return {
            "total_failures": 0,
            "model_name": model_name,
            "by_failure_mode": {},
            "by_perturbation_type": {},
            "by_severity": {},
            "mean_failure_confidence": None,
        }

    by_mode: dict[str, int] = {}
    by_pt: dict[str, int] = {}
    by_sev: dict[float, int] = {}
    confidences: list[float] = []

    for case in cases:
        mode_key = case.failure_mode.value
        by_mode[mode_key] = by_mode.get(mode_key, 0) + 1

        if case.perturbation_spec:
            pt = case.perturbation_spec.perturbation_type.value
            sev = case.perturbation_spec.severity
            by_pt[pt] = by_pt.get(pt, 0) + 1
            by_sev[sev] = by_sev.get(sev, 0) + 1

        if case.confidence is not None:
            confidences.append(case.confidence)

    return {
        "total_failures": len(cases),
        "model_name": model_name,
        "by_failure_mode": by_mode,
        "by_perturbation_type": dict(sorted(by_pt.items())),
        "by_severity": {str(round(s, 2)): n for s, n in sorted(by_sev.items())},
        "mean_failure_confidence": float(np.mean(confidences)) if confidences else None,
        "high_confidence_wrong_count": by_mode.get(FailureMode.HIGH_CONFIDENCE_WRONG.value, 0),
        "false_accept_count": by_mode.get(FailureMode.FALSE_ACCEPT.value, 0),
    }


def failure_rate_by_severity(
    cases: list[FailureCase],
    total_predictions_by_severity: dict[float, int],
) -> dict[float, float]:
    """
    Compute failure rate (failures / total) at each severity level.
    Complementary to the ADC accuracy curve.
    """
    failures_by_sev: dict[float, int] = {}
    for case in cases:
        if case.perturbation_spec:
            sev = case.perturbation_spec.severity
            failures_by_sev[sev] = failures_by_sev.get(sev, 0) + 1

    rates: dict[float, float] = {}
    for sev, total in total_predictions_by_severity.items():
        n_fail = failures_by_sev.get(sev, 0)
        rates[sev] = n_fail / total if total > 0 else 0.0
    return dict(sorted(rates.items()))


def most_confused_pairs(
    cases: list[FailureCase],
    top_k: int = 10,
) -> list[dict[str, Any]]:
    """
    Find the (true_subject, predicted_subject) pairs that occur most often.

    Returns top_k pairs sorted by confusion frequency — useful for identifying
    subjects that look similar to the model.
    """
    pair_counts: dict[tuple[str, str], int] = {}
    for case in cases:
        if case.predicted_subject_id and case.failure_mode in (
            FailureMode.FALSE_ACCEPT, FailureMode.HIGH_CONFIDENCE_WRONG
        ):
            pair = (case.true_subject_id, case.predicted_subject_id)
            pair_counts[pair] = pair_counts.get(pair, 0) + 1

    sorted_pairs = sorted(pair_counts.items(), key=lambda x: x[1], reverse=True)
    return [
        {
            "true_subject": pair[0],
            "predicted_subject": pair[1],
            "confusion_count": count,
        }
        for pair, count in sorted_pairs[:top_k]
    ]


# ---------------------------------------------------------------------------
# Image-level feature extraction for meta-model
# ---------------------------------------------------------------------------

def extract_image_features(image: np.ndarray) -> dict[str, float]:
    """
    Extract lightweight image statistics used as meta-model features.

    Features:
        mean_brightness     — mean pixel value (normalised to [0, 1])
        local_variance      — mean variance in 8×8 patches (texture measure)
        edge_density        — fraction of pixels with gradient magnitude > threshold
        aspect_ratio        — height / width (1.0 for square images)
        brightness_std      — standard deviation of pixel brightness
    """
    # Ensure float
    if image.dtype == np.uint8:
        img = image.astype(np.float32) / 255.0
    else:
        img = image.astype(np.float32)

    # Grayscale
    if len(img.shape) == 3:
        gray = 0.299 * img[:, :, 0] + 0.587 * img[:, :, 1] + 0.114 * img[:, :, 2]
    else:
        gray = img

    h, w = gray.shape

    # Mean brightness
    mean_brightness = float(gray.mean())

    # Brightness std
    brightness_std = float(gray.std())

    # Local variance (8×8 blocks)
    block = 8
    variances: list[float] = []
    for r in range(0, h - block, block):
        for c in range(0, w - block, block):
            patch = gray[r:r + block, c:c + block]
            variances.append(float(patch.var()))
    local_variance = float(np.mean(variances)) if variances else 0.0

    # Edge density via Sobel-like gradients
    gy = np.diff(gray, axis=0)
    gx = np.diff(gray, axis=1)
    min_h = min(gy.shape[0], gx.shape[0])
    min_w = min(gy.shape[1], gx.shape[1])
    grad_mag = np.sqrt(gy[:min_h, :min_w] ** 2 + gx[:min_h, :min_w] ** 2)
    edge_density = float((grad_mag > 0.05).mean())

    return {
        "mean_brightness": mean_brightness,
        "brightness_std": brightness_std,
        "local_variance": local_variance,
        "edge_density": edge_density,
        "aspect_ratio": h / max(w, 1),
    }


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _classify_failure_mode(
    is_correct: bool,
    confidence: float,
    high_conf_threshold: float,
    low_conf_threshold: float,
) -> FailureMode | None:
    """
    Classify one prediction into a failure mode, or return None if it's a
    normal correct prediction.
    """
    if not is_correct:
        if confidence >= high_conf_threshold:
            return FailureMode.HIGH_CONFIDENCE_WRONG
        else:
            return FailureMode.FALSE_REJECT
    else:
        # Correct prediction, but borderline confidence
        if confidence < low_conf_threshold:
            return FailureMode.LOW_CONFIDENCE_CORRECT
        return None   # Normal correct prediction — not a failure

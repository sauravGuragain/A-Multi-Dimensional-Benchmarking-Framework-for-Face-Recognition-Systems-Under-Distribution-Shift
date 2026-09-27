from __future__ import annotations

import logging
import math
from typing import Any

import numpy as np

from faceeval.core.exceptions import InsufficientSamplesError
from faceeval.core.types import (
    EvaluationResult,
    ModelName,
    ModelParadigm,
    PRCurve,
    PerturbationSpec,
    PredictionResult,
    ROCCurve,
    SubjectID,
)

logger = logging.getLogger(__name__)

# Minimum samples required before computing metrics
_MIN_SAMPLES = 4


# ---------------------------------------------------------------------------
# Identification metrics
# ---------------------------------------------------------------------------

def compute_identification_metrics(
    predictions: list[PredictionResult],
    ground_truth: list[SubjectID],
    class_labels: list[SubjectID] | None = None,
) -> dict[str, Any]:
    """
    Compute closed-set identification metrics.

    Parameters
    ----------
    predictions:
        List of ``PredictionResult`` objects (one per test image).
    ground_truth:
        True subject ID for each prediction, in the same order.
    class_labels:
        All known class labels (used to build the confusion matrix in a
        consistent order across experiments).  Defaults to sorted unique
        labels in ``ground_truth``.

    Returns
    -------
    dict with keys:
        accuracy, precision_macro, recall_macro, f1_macro,
        precision_weighted, recall_weighted, f1_weighted,
        top_k_accuracy  (dict: k → float),
        confusion_matrix (list[list[int]]),
        class_labels (list[str]),
        per_class_metrics (dict: label → {precision, recall, f1, support})
    """
    if len(predictions) < _MIN_SAMPLES:
        raise InsufficientSamplesError("identification_metrics", _MIN_SAMPLES, len(predictions))

    if len(predictions) != len(ground_truth):
        raise ValueError(
            f"predictions length ({len(predictions)}) != "
            f"ground_truth length ({len(ground_truth)})"
        )

    y_true = list(ground_truth)
    y_pred = [p.predicted_subject_id or "" for p in predictions]

    labels = class_labels if class_labels else sorted(set(y_true))
    label_to_idx = {l: i for i, l in enumerate(labels)}
    n = len(labels)

    # --- Confusion matrix ---
    cm = [[0] * n for _ in range(n)]
    for true, pred in zip(y_true, y_pred):
        i = label_to_idx.get(true, -1)
        j = label_to_idx.get(pred, -1)
        if i >= 0 and j >= 0:
            cm[i][j] += 1

    # --- Per-class precision / recall / F1 ---
    per_class: dict[str, dict[str, float]] = {}
    for li, label in enumerate(labels):
        tp = cm[li][li]
        fp = sum(cm[r][li] for r in range(n)) - tp
        fn = sum(cm[li][c] for c in range(n)) - tp
        support = tp + fn

        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        per_class[label] = {
            "precision": prec, "recall": rec,
            "f1": f1, "support": support,
        }

    # --- Macro averages ---
    total_support = sum(v["support"] for v in per_class.values())
    prec_macro = float(np.mean([v["precision"] for v in per_class.values()]))
    rec_macro  = float(np.mean([v["recall"]    for v in per_class.values()]))
    f1_macro   = float(np.mean([v["f1"]        for v in per_class.values()]))

    # --- Weighted averages ---
    def weighted_avg(key: str) -> float:
        if total_support == 0:
            return 0.0
        return float(sum(
            v[key] * v["support"] for v in per_class.values()
        ) / total_support)

    prec_w = weighted_avg("precision")
    rec_w  = weighted_avg("recall")
    f1_w   = weighted_avg("f1")

    # --- Top-k accuracy ---
    top_k_acc: dict[int, float] = {}
    for k in [1, 3, 5]:
        correct_k = 0
        for pred, true in zip(predictions, y_true):
            top_k_ids = [subj for subj, _ in pred.top_k_predictions[:k]]
            if true in top_k_ids:
                correct_k += 1
            elif pred.predicted_subject_id == true:
                correct_k += 1
        top_k_acc[k] = correct_k / len(predictions)

    accuracy = sum(1 for t, p in zip(y_true, y_pred) if t == p) / len(y_true)

    return {
        "accuracy": accuracy,
        "precision_macro": prec_macro,
        "recall_macro": rec_macro,
        "f1_macro": f1_macro,
        "precision_weighted": prec_w,
        "recall_weighted": rec_w,
        "f1_weighted": f1_w,
        "top_k_accuracy": top_k_acc,
        "confusion_matrix": cm,
        "class_labels": labels,
        "per_class_metrics": per_class,
        "num_samples": len(predictions),
    }


# ---------------------------------------------------------------------------
# Verification metrics
# ---------------------------------------------------------------------------

def compute_verification_metrics(
    similarity_scores: list[float],
    is_same_person: list[bool],
    far_operating_points: list[float] | None = None,
    frr_operating_points: list[float] | None = None,
    n_thresholds: int = 500,
) -> dict[str, Any]:
    """
    Compute face verification metrics from similarity scores.

    Parameters
    ----------
    similarity_scores:
        Cosine (or other) similarity score for each pair.  Higher = more similar.
    is_same_person:
        Ground-truth label for each pair.
    far_operating_points:
        FAR levels at which to report the corresponding FRR.
        Default: [0.001, 0.01, 0.1].
    frr_operating_points:
        FRR levels at which to report the corresponding FAR.
        Default: [0.001, 0.01, 0.1].
    n_thresholds:
        Number of threshold points to sweep for curves.

    Returns
    -------
    dict with keys:
        auc, eer, roc_curve (ROCCurve), pr_curve (PRCurve),
        far_at_frr (dict: frr_level → far), frr_at_far (dict: far_level → frr),
        optimal_threshold (float)
    """
    if len(similarity_scores) < _MIN_SAMPLES:
        raise InsufficientSamplesError("verification_metrics", _MIN_SAMPLES, len(similarity_scores))

    far_ops = far_operating_points or [0.001, 0.01, 0.1]
    frr_ops = frr_operating_points or [0.001, 0.01, 0.1]

    scores = np.array(similarity_scores, dtype=np.float64)
    labels = np.array(is_same_person, dtype=bool)

    pos_mask = labels
    neg_mask = ~labels
    n_pos = pos_mask.sum()
    n_neg = neg_mask.sum()

    if n_pos == 0 or n_neg == 0:
        logger.warning("Verification metrics: only one class present in labels.")
        return _empty_verification_metrics()

    # --- Sweep thresholds ---
    thresholds = np.linspace(scores.min(), scores.max(), n_thresholds)

    fprs, tprs, fars, frrs = [], [], [], []
    for thresh in thresholds:
        predicted_same = scores >= thresh
        tp = (predicted_same & pos_mask).sum()
        fp = (predicted_same & neg_mask).sum()
        fn = (~predicted_same & pos_mask).sum()
        tn = (~predicted_same & neg_mask).sum()

        tpr = tp / n_pos if n_pos > 0 else 0.0
        fpr = fp / n_neg if n_neg > 0 else 0.0
        far = fpr           # FAR = FP / N_neg
        frr = fn / n_pos if n_pos > 0 else 0.0

        tprs.append(float(tpr))
        fprs.append(float(fpr))
        fars.append(float(far))
        frrs.append(float(frr))

    # --- AUC (trapezoidal integration, sort by fpr) ---
    sorted_pairs = sorted(zip(fprs, tprs))
    sorted_fprs = [p[0] for p in sorted_pairs]
    sorted_tprs = [p[1] for p in sorted_pairs]
    auc = float(np.trapezoid(sorted_tprs, sorted_fprs))

    # --- EER: linear interpolation where FAR == FRR ---
    eer, eer_thresh = _compute_eer(fars, frrs, thresholds.tolist())

    # --- ROC curve ---
    roc_curve = ROCCurve(
        fpr=sorted_fprs,
        tpr=sorted_tprs,
        thresholds=[t for _, t in sorted(
            zip(fprs, thresholds.tolist()), key=lambda x: x[0]
        )],
        auc=auc,
    )

    # --- Precision-Recall curve ---
    pr_data = _compute_pr_curve(scores, labels)
    pr_curve = PRCurve(**pr_data)

    # --- FAR at fixed FRR levels ---
    far_at_frr: dict[float, float] = {}
    for frr_level in frr_ops:
        far_at_frr[frr_level] = _interpolate_at_level(frrs, fars, frr_level)

    # --- FRR at fixed FAR levels ---
    frr_at_far: dict[float, float] = {}
    for far_level in far_ops:
        frr_at_far[far_level] = _interpolate_at_level(fars, frrs, far_level)

    return {
        "auc": auc,
        "eer": eer,
        "eer_threshold": eer_thresh,
        "optimal_threshold": eer_thresh,
        "roc_curve": roc_curve,
        "pr_curve": pr_curve,
        "far_at_frr": far_at_frr,
        "frr_at_far": frr_at_far,
        "n_pairs": len(similarity_scores),
        "n_positive_pairs": int(n_pos),
        "n_negative_pairs": int(n_neg),
        "fars": fars,
        "frrs": frrs,
        "thresholds": thresholds.tolist(),
    }


# ---------------------------------------------------------------------------
# Full EvaluationResult assembler
# ---------------------------------------------------------------------------

def build_evaluation_result(
    run_id: str,
    model_name: ModelName,
    model_paradigm: ModelParadigm,
    dataset_name: str,
    predictions: list[PredictionResult],
    ground_truth: list[SubjectID],
    perturbation_spec: PerturbationSpec | None,
    baseline_accuracy: float | None = None,
    class_labels: list[SubjectID] | None = None,
    similarity_scores: list[float] | None = None,
    pair_labels: list[bool] | None = None,
) -> EvaluationResult:
    """
    Assemble a complete ``EvaluationResult`` from raw predictions.

    Combines identification metrics (always) and verification metrics
    (when ``similarity_scores`` and ``pair_labels`` are provided).
    """
    id_metrics = compute_identification_metrics(
        predictions, ground_truth, class_labels
    )

    result = EvaluationResult(
        run_id=run_id,
        model_name=model_name,
        model_paradigm=model_paradigm,
        dataset_name=dataset_name,
        perturbation_spec=perturbation_spec,
        accuracy=id_metrics["accuracy"],
        precision=id_metrics["precision_macro"],
        recall=id_metrics["recall_macro"],
        f1_score=id_metrics["f1_macro"],
        top_k_accuracy=id_metrics["top_k_accuracy"],
        confusion_matrix=id_metrics["confusion_matrix"],
        class_labels=id_metrics["class_labels"],
        num_samples=id_metrics["num_samples"],
        baseline_accuracy=baseline_accuracy,
    )

    if similarity_scores is not None and pair_labels is not None:
        ver = compute_verification_metrics(similarity_scores, pair_labels)
        result = EvaluationResult(
            **{
                **{f: getattr(result, f) for f in result.__dataclass_fields__},
                "auc": ver["auc"],
                "eer": ver["eer"],
                "far_at_thresholds": ver["frr_at_far"],
                "frr_at_thresholds": ver["far_at_frr"],
                "roc_curve": ver["roc_curve"],
                "pr_curve": ver["pr_curve"],
            }
        )

    return result


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _compute_eer(
    fars: list[float],
    frrs: list[float],
    thresholds: list[float],
) -> tuple[float, float]:
    """Linear interpolation for EER."""
    diffs = [abs(f - r) for f, r in zip(fars, frrs)]
    idx = int(np.argmin(diffs))

    if idx == 0 or idx == len(fars) - 1:
        eer = (fars[idx] + frrs[idx]) / 2.0
        return eer, thresholds[idx]

    # Interpolate between idx-1 and idx
    for i in range(len(fars) - 1):
        f1, r1 = fars[i], frrs[i]
        f2, r2 = fars[i + 1], frrs[i + 1]
        if (f1 - r1) * (f2 - r2) <= 0:
            denom = (f2 - r2) - (f1 - r1)
            if abs(denom) < 1e-12:
                eer = (f1 + r1) / 2.0
                thresh = thresholds[i]
            else:
                t = (r1 - f1) / denom
                eer = f1 + t * (f2 - f1)
                thresh = thresholds[i] + t * (thresholds[i + 1] - thresholds[i])
            return float(eer), float(thresh)

    eer = (fars[idx] + frrs[idx]) / 2.0
    return float(eer), float(thresholds[idx])


def _compute_pr_curve(
    scores: np.ndarray,
    labels: np.ndarray,
    n_points: int = 200,
) -> dict[str, Any]:
    thresholds = np.linspace(scores.min(), scores.max(), n_points)
    precisions, recalls = [], []
    for thresh in thresholds:
        pred = scores >= thresh
        tp = (pred & labels).sum()
        fp = (pred & ~labels).sum()
        fn = (~pred & labels).sum()
        p = tp / (tp + fp) if (tp + fp) > 0 else 1.0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        precisions.append(float(p))
        recalls.append(float(r))

    ap = float(np.trapezoid(precisions, recalls)) if len(recalls) > 1 else 0.0
    return {
        "precision": precisions,
        "recall": recalls,
        "thresholds": thresholds.tolist(),
        "average_precision": abs(ap),
    }


def _interpolate_at_level(
    x_values: list[float],
    y_values: list[float],
    target_x: float,
) -> float:
    """Return y-value at target_x via linear interpolation in sorted (x, y) pairs."""
    paired = sorted(zip(x_values, y_values))
    xs = [p[0] for p in paired]
    ys = [p[1] for p in paired]

    if target_x <= xs[0]:
        return ys[0]
    if target_x >= xs[-1]:
        return ys[-1]

    for i in range(len(xs) - 1):
        if xs[i] <= target_x <= xs[i + 1]:
            dx = xs[i + 1] - xs[i]
            if dx < 1e-12:
                return (ys[i] + ys[i + 1]) / 2.0
            t = (target_x - xs[i]) / dx
            return ys[i] + t * (ys[i + 1] - ys[i])
    return ys[-1]


def _empty_verification_metrics() -> dict[str, Any]:
    empty_roc = ROCCurve(fpr=[0.0, 1.0], tpr=[0.0, 1.0], thresholds=[1.0, 0.0], auc=0.5)
    empty_pr  = PRCurve(precision=[1.0, 0.0], recall=[0.0, 1.0],
                        thresholds=[1.0, 0.0], average_precision=0.5)
    return {
        "auc": 0.5, "eer": 0.5, "eer_threshold": 0.0, "optimal_threshold": 0.0,
        "roc_curve": empty_roc, "pr_curve": empty_pr,
        "far_at_frr": {}, "frr_at_far": {},
        "n_pairs": 0, "n_positive_pairs": 0, "n_negative_pairs": 0,
        "fars": [], "frrs": [], "thresholds": [],
    }


def compute_degradation_auc(
    accuracy_by_severity: dict[float, float],
) -> float:
    """
    Compute the area under the accuracy-vs-severity curve.

    Higher AUC = model degrades more gracefully (greater robustness).
    AUC is computed over the normalised severity axis [0, 1].
    """
    if len(accuracy_by_severity) < 2:
        return list(accuracy_by_severity.values())[0] if accuracy_by_severity else 0.0
    items = sorted(accuracy_by_severity.items())
    severities = [s for s, _ in items]
    accs = [a for _, a in items]
    return float(np.trapezoid(accs, severities))

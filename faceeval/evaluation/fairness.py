from __future__ import annotations

import logging
from collections import defaultdict
from typing import Any

import numpy as np

from faceeval.core.exceptions import FairnessError
from faceeval.core.types import (
    FairnessReport,
    ImageRecord,
    ModelName,
    PerturbationSpec,
    PredictionResult,
    SubjectID,
    SubgroupMetrics,
)
from faceeval.evaluation.metrics import compute_verification_metrics

logger = logging.getLogger(__name__)

_MIN_SUBGROUP_SAMPLES = 5


# ---------------------------------------------------------------------------
# Main fairness computation
# ---------------------------------------------------------------------------

def compute_fairness(
    predictions: list[PredictionResult],
    ground_truth: list[SubjectID],
    records: list[ImageRecord],
    model_name: ModelName,
    perturbation_spec: PerturbationSpec | None = None,
    attribute_keys: list[str] | None = None,
    similarity_scores: list[float] | None = None,
    pair_labels: list[bool] | None = None,
) -> FairnessReport:
    """
    Compute subgroup fairness metrics.

    Parameters
    ----------
    predictions:
        Model predictions for each image (parallel to ``ground_truth``).
    ground_truth:
        True subject IDs (parallel to ``predictions``).
    records:
        ``ImageRecord`` objects carrying per-image attribute metadata.
        Must be parallel to ``predictions``.
    model_name:
        Model being evaluated.
    perturbation_spec:
        Perturbation condition.
    attribute_keys:
        Which metadata attributes to group by.  ``None`` auto-detects
        available attributes from ``records[0].metadata``.
    similarity_scores:
        If provided, per-subgroup EER and FAR/FRR are also computed.
    pair_labels:
        Parallel boolean labels for ``similarity_scores``.

    Returns
    -------
    FairnessReport
    """
    if len(predictions) != len(records):
        raise FairnessError(
            f"predictions ({len(predictions)}) and records ({len(records)}) "
            "must be the same length"
        )

    # --- Auto-detect attribute keys ---
    if attribute_keys is None:
        attribute_keys = _detect_attribute_keys(records)

    if not attribute_keys:
        logger.warning(
            "[%s] No attribute keys found in record metadata. "
            "Fairness report will have empty subgroup metrics.", model_name
        )
        return FairnessReport(
            model_name=model_name,
            perturbation_spec=perturbation_spec,
            subgroup_metrics=[],
            max_accuracy_gap=0.0,
            max_eer_gap=0.0,
            equal_opportunity_gap=0.0,
            demographic_parity_gap=0.0,
        )

    # --- Group images by attribute value ---
    subgroup_metrics: list[SubgroupMetrics] = []

    for attr_key in attribute_keys:
        groups = _group_by_attribute(records, attr_key)

        for attr_value, indices in groups.items():
            if len(indices) < _MIN_SUBGROUP_SAMPLES:
                logger.debug(
                    "Skipping subgroup '%s:%s' — only %d samples (min %d)",
                    attr_key, attr_value, len(indices), _MIN_SUBGROUP_SAMPLES,
                )
                continue

            group_preds = [predictions[i] for i in indices]
            group_truth = [ground_truth[i] for i in indices]

            # Identification accuracy
            correct = sum(
                1 for p, t in zip(group_preds, group_truth)
                if p.predicted_subject_id == t
            )
            acc = correct / len(indices)

            # Verification metrics (if scores available)
            eer, far, frr, auc = 0.0, 0.0, 0.0, 0.5
            if similarity_scores is not None and pair_labels is not None:
                group_scores = [similarity_scores[i] for i in indices
                                if i < len(similarity_scores)]
                group_pair_labels = [pair_labels[i] for i in indices
                                     if i < len(pair_labels)]
                if len(group_scores) >= _MIN_SUBGROUP_SAMPLES:
                    try:
                        ver = compute_verification_metrics(group_scores, group_pair_labels)
                        eer = ver["eer"]
                        auc = ver["auc"]
                        far = ver["frr_at_far"].get(0.01, 0.0)
                        frr = ver["far_at_frr"].get(0.01, 0.0)
                    except Exception as exc:
                        logger.debug("Subgroup verification metrics failed: %s", exc)

            subgroup_metrics.append(SubgroupMetrics(
                model_name=model_name,
                attribute_key=attr_key,
                attribute_value=attr_value,
                num_samples=len(indices),
                accuracy=acc,
                eer=eer,
                far=far,
                frr=frr,
                auc=auc,
            ))

    # --- Compute gap metrics ---
    report = _compute_gap_metrics(
        subgroup_metrics, model_name, perturbation_spec
    )
    logger.info(
        "[%s] Fairness: %d subgroups, accuracy_gap=%.4f, eer_gap=%.4f",
        model_name, len(subgroup_metrics),
        report.max_accuracy_gap, report.max_eer_gap,
    )
    return report


# ---------------------------------------------------------------------------
# Gap metrics
# ---------------------------------------------------------------------------

def _compute_gap_metrics(
    subgroup_metrics: list[SubgroupMetrics],
    model_name: ModelName,
    perturbation_spec: PerturbationSpec | None,
) -> FairnessReport:
    if not subgroup_metrics:
        return FairnessReport(
            model_name=model_name,
            perturbation_spec=perturbation_spec,
            subgroup_metrics=[],
            max_accuracy_gap=0.0,
            max_eer_gap=0.0,
            equal_opportunity_gap=0.0,
            demographic_parity_gap=0.0,
        )

    accuracies = [m.accuracy for m in subgroup_metrics]
    eers       = [m.eer      for m in subgroup_metrics]
    recalls    = [m.accuracy for m in subgroup_metrics]  # recall ≈ accuracy here

    # Max accuracy gap
    max_acc_gap = float(max(accuracies) - min(accuracies)) if len(accuracies) > 1 else 0.0

    # Max EER gap
    max_eer_gap = float(max(eers) - min(eers)) if len(eers) > 1 else 0.0

    # Equal opportunity (TPR / recall gap)
    eq_opp_gap = float(max(recalls) - min(recalls)) if len(recalls) > 1 else 0.0

    # Demographic parity (positive prediction rate gap)
    # Positive prediction rate = proportion of images predicted as any class
    dem_par_gap = max_acc_gap  # simplified; full version requires pair data

    return FairnessReport(
        model_name=model_name,
        perturbation_spec=perturbation_spec,
        subgroup_metrics=subgroup_metrics,
        max_accuracy_gap=max_acc_gap,
        max_eer_gap=max_eer_gap,
        equal_opportunity_gap=eq_opp_gap,
        demographic_parity_gap=dem_par_gap,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _detect_attribute_keys(records: list[ImageRecord]) -> list[str]:
    """
    Auto-detect available attribute keys from the first 20 records.

    Scans ``record.metadata`` for dict-valued ``"attributes"`` key
    (as loaded by CelebA / VGGFace2 loaders).
    """
    candidate_keys: set[str] = set()
    for rec in records[:20]:
        attrs = rec.metadata.get("attributes", {})
        if isinstance(attrs, dict):
            candidate_keys.update(attrs.keys())

    # Also look for top-level demographic keys
    top_level_keys = {"gender", "age_group", "ethnicity", "race", "sex"}
    for rec in records[:20]:
        for key in top_level_keys:
            if key in rec.metadata:
                candidate_keys.add(key)

    return sorted(candidate_keys)


def _group_by_attribute(
    records: list[ImageRecord],
    attribute_key: str,
) -> dict[str, list[int]]:
    """
    Group record indices by the value of ``attribute_key`` in their metadata.
    """
    groups: dict[str, list[int]] = defaultdict(list)
    for i, rec in enumerate(records):
        # First check nested attributes dict
        attrs = rec.metadata.get("attributes", {})
        if isinstance(attrs, dict) and attribute_key in attrs:
            val = attrs[attribute_key]
            # CelebA attributes are +1 / -1
            if isinstance(val, (int, float)):
                val = "positive" if val > 0 else "negative"
            groups[str(val)].append(i)
        # Then check top-level metadata
        elif attribute_key in rec.metadata:
            val = rec.metadata[attribute_key]
            groups[str(val)].append(i)

    return dict(groups)


def fairness_summary(
    reports: list[FairnessReport],
) -> dict[str, Any]:
    """
    Aggregate fairness reports across multiple (model × perturbation) conditions.

    Returns per-model worst-case and mean fairness gaps.
    """
    by_model: dict[str, list[FairnessReport]] = defaultdict(list)
    for r in reports:
        by_model[r.model_name].append(r)

    summary: dict[str, Any] = {}
    for model_name, model_reports in by_model.items():
        acc_gaps = [r.max_accuracy_gap for r in model_reports]
        eer_gaps = [r.max_eer_gap for r in model_reports]
        summary[model_name] = {
            "worst_accuracy_gap": float(max(acc_gaps)),
            "mean_accuracy_gap":  float(np.mean(acc_gaps)),
            "worst_eer_gap":      float(max(eer_gaps)),
            "mean_eer_gap":       float(np.mean(eer_gaps)),
            "n_conditions":       len(model_reports),
        }
    return summary

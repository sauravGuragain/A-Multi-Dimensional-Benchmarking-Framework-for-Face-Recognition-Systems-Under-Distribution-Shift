from __future__ import annotations

import logging
import math
from datetime import datetime
from typing import Any

import numpy as np

from faceeval.core.types import (
    BehavioralFingerprint,
    CalibrationMetrics,
    EvaluationResult,
    FairnessReport,
    ModelName,
    ModelParadigm,
    PerturbationType,
    ResourceUsage,
    SampleEfficiencyCurve,
)
from faceeval.fingerprint.adc import compute_all_adc_curves

logger = logging.getLogger(__name__)

# Reference values for normalising speed and memory dimensions
_REFERENCE_FPS_LOG     = math.log(100.0)    # 100 fps reference
_REFERENCE_MEMORY_MB   = 1000.0             # 1 GB reference


# ---------------------------------------------------------------------------
# Fingerprint builder
# ---------------------------------------------------------------------------

class FingerprintBuilder:
    """
    Assembles a ``BehavioralFingerprint`` for one model from its evaluation
    results, calibration metrics, fairness report, and resource usage.

    Usage::

        builder = FingerprintBuilder(model_name="facenet", paradigm=ModelParadigm.DEEP_LEARNING)
        builder.add_evaluation_results(results)
        builder.add_calibration(calibration_metrics)     # optional
        builder.add_fairness(fairness_report)            # optional
        builder.add_resource_usage(resource_usages)      # optional
        builder.add_sample_efficiency(curve)             # optional
        fingerprint = builder.build(n_classes=5749)
    """

    def __init__(
        self,
        model_name: ModelName,
        paradigm: ModelParadigm,
    ) -> None:
        self._model_name = model_name
        self._paradigm   = paradigm
        self._eval_results:   list[EvaluationResult]   = []
        self._calibrations:   list[CalibrationMetrics] = []
        self._fairness:       FairnessReport | None    = None
        self._resource_usages: list[ResourceUsage]     = []
        self._sample_eff_curve: SampleEfficiencyCurve | None = None

    # ------------------------------------------------------------------
    # Data ingestion
    # ------------------------------------------------------------------

    def add_evaluation_results(self, results: list[EvaluationResult]) -> None:
        model_results = [r for r in results if r.model_name == self._model_name]
        self._eval_results.extend(model_results)
        logger.debug(
            "[%s] Added %d evaluation results.", self._model_name, len(model_results)
        )

    def add_calibration(self, metrics: CalibrationMetrics | list[CalibrationMetrics]) -> None:
        if isinstance(metrics, list):
            self._calibrations.extend(
                [m for m in metrics if m.model_name == self._model_name]
            )
        elif metrics.model_name == self._model_name:
            self._calibrations.append(metrics)

    def add_fairness(self, report: FairnessReport) -> None:
        if report.model_name == self._model_name:
            self._fairness = report

    def add_resource_usage(self, usages: ResourceUsage | list[ResourceUsage]) -> None:
        if isinstance(usages, list):
            self._resource_usages.extend(
                [u for u in usages if u.model_name == self._model_name]
            )
        else:
            if usages.model_name == self._model_name:
                self._resource_usages.append(usages)

    def add_sample_efficiency(self, curve: SampleEfficiencyCurve) -> None:
        if curve.model_name == self._model_name:
            self._sample_eff_curve = curve

    # ------------------------------------------------------------------
    # Build
    # ------------------------------------------------------------------

    def build(self, n_classes: int | None = None) -> BehavioralFingerprint:
        """
        Assemble and return the ``BehavioralFingerprint``.

        Parameters
        ----------
        n_classes:
            Number of subject classes in the dataset.  Used to normalise
            ADC-AUC scores relative to random-chance baseline.
        """
        scores: dict[str, float] = {}

        # --- 1. Robustness dimensions from ADC curves ---
        adc_curves_all = compute_all_adc_curves(self._eval_results, n_classes)
        model_adc_curves = adc_curves_all.get(self._model_name, [])

        for curve in model_adc_curves:
            key = f"robustness_{curve.perturbation_type.value}"
            scores[key] = float(np.clip(curve.area, 0.0, 1.0))

        # Fill in any missing perturbation types with 0.0
        for pt in PerturbationType:
            key = f"robustness_{pt.value}"
            if key not in scores:
                scores[key] = 0.0

        # --- 2. Calibration quality ---
        scores["calibration_quality"] = self._compute_calibration_score()

        # --- 3. Fairness dimension ---
        scores["fairness_gap"] = self._compute_fairness_score()

        # --- 4. Speed score ---
        scores["speed_score"] = self._compute_speed_score()

        # --- 5. Memory efficiency ---
        scores["memory_efficiency"] = self._compute_memory_score()

        # --- 6. Sample efficiency ---
        scores["sample_efficiency"] = self._compute_sample_efficiency_score()

        # --- Build ordered dimension list ---
        dimension_names = _build_dimension_order(scores)
        vector = [scores[dim] for dim in dimension_names]

        logger.info(
            "[%s] Fingerprint built — %d dimensions, "
            "robustness dims: %d, calibration=%.4f, fairness=%.4f",
            self._model_name, len(dimension_names),
            len(model_adc_curves),
            scores["calibration_quality"],
            scores["fairness_gap"],
        )

        return BehavioralFingerprint(
            model_name=self._model_name,
            paradigm=self._paradigm,
            dimension_names=dimension_names,
            vector=vector,
            scores=scores,
            adc_curves=model_adc_curves,
            sample_efficiency=self._sample_eff_curve,
            computed_at=datetime.utcnow(),
        )

    # ------------------------------------------------------------------
    # Dimension-specific computations
    # ------------------------------------------------------------------

    def _compute_calibration_score(self) -> float:
        """Mean (1 - ECE) across all calibration records for this model."""
        if not self._calibrations:
            return 0.5   # neutral value when unavailable
        eces = [m.ece for m in self._calibrations]
        mean_ece = float(np.mean(eces))
        return float(np.clip(1.0 - mean_ece, 0.0, 1.0))

    def _compute_fairness_score(self) -> float:
        """1 - max_accuracy_gap (higher = fairer)."""
        if self._fairness is None:
            return 0.5
        return float(np.clip(1.0 - self._fairness.max_accuracy_gap, 0.0, 1.0))

    def _compute_speed_score(self) -> float:
        """
        Sigmoid-normalised throughput score in [0, 1].

        log(fps) is mapped through a sigmoid centred at the reference FPS
        so that 100 fps → 0.5, very fast models → approaching 1.0,
        very slow models → approaching 0.0.
        """
        if not self._resource_usages:
            return 0.5
        fps_values = [
            u.throughput_fps for u in self._resource_usages
            if u.throughput_fps is not None and u.throughput_fps > 0
        ]
        if not fps_values:
            return 0.5
        mean_fps = float(np.mean(fps_values))
        log_fps = math.log(max(mean_fps, 0.01))
        score = 1.0 / (1.0 + math.exp(-( log_fps - _REFERENCE_FPS_LOG)))
        return float(np.clip(score, 0.0, 1.0))

    def _compute_memory_score(self) -> float:
        """1 - (peak_memory_MB / reference_MB), clamped to [0, 1]."""
        if not self._resource_usages:
            return 0.5
        mem_values = [
            u.peak_memory_mb for u in self._resource_usages
            if u.peak_memory_mb is not None and u.peak_memory_mb >= 0
        ]
        if not mem_values:
            return 0.5
        mean_mem = float(np.mean(mem_values))
        score = 1.0 - (mean_mem / _REFERENCE_MEMORY_MB)
        return float(np.clip(score, 0.0, 1.0))

    def _compute_sample_efficiency_score(self) -> float:
        """Area under the normalised sample efficiency curve, or 0.5 if unavailable."""
        if self._sample_eff_curve is None:
            return 0.5
        from faceeval.fingerprint.sample_efficiency import sample_efficiency_auc
        return float(np.clip(sample_efficiency_auc(self._sample_eff_curve), 0.0, 1.0))


# ---------------------------------------------------------------------------
# Batch builder
# ---------------------------------------------------------------------------

def build_all_fingerprints(
    evaluation_results: list[EvaluationResult],
    model_paradigms: dict[ModelName, ModelParadigm],
    calibration_metrics: list[CalibrationMetrics] | None = None,
    fairness_reports: list[FairnessReport] | None = None,
    resource_usages: list[ResourceUsage] | None = None,
    sample_efficiency_curves: list[SampleEfficiencyCurve] | None = None,
    n_classes: int | None = None,
) -> list[BehavioralFingerprint]:
    """
    Build fingerprints for all models in one call.

    Parameters
    ----------
    evaluation_results:
        All EvaluationResult objects from one experiment run.
    model_paradigms:
        Mapping from model_name → ModelParadigm (to set the paradigm field).
    calibration_metrics, fairness_reports, resource_usages, sample_efficiency_curves:
        Optional supplementary data; each is distributed to the correct builder
        by model_name.
    n_classes:
        Number of subject classes for ADC normalisation.

    Returns
    -------
    list[BehavioralFingerprint]
        One fingerprint per model in ``model_paradigms``.
    """
    fingerprints: list[BehavioralFingerprint] = []

    for model_name, paradigm in model_paradigms.items():
        builder = FingerprintBuilder(model_name, paradigm)
        builder.add_evaluation_results(evaluation_results)

        if calibration_metrics:
            builder.add_calibration(calibration_metrics)
        if fairness_reports:
            for fr in fairness_reports:
                builder.add_fairness(fr)
        if resource_usages:
            builder.add_resource_usage(resource_usages)
        if sample_efficiency_curves:
            for curve in sample_efficiency_curves:
                builder.add_sample_efficiency(curve)

        fp = builder.build(n_classes=n_classes)
        fingerprints.append(fp)

    return fingerprints


# ---------------------------------------------------------------------------
# Dimension ordering helper
# ---------------------------------------------------------------------------

def _build_dimension_order(scores: dict[str, float]) -> list[str]:
    """
    Return a deterministic, sorted list of dimension names.

    Order:
      1. robustness_* dimensions (sorted alphabetically by perturbation name)
      2. calibration_quality
      3. fairness_gap
      4. speed_score
      5. memory_efficiency
      6. sample_efficiency
    """
    fixed_tail = [
        "calibration_quality",
        "fairness_gap",
        "speed_score",
        "memory_efficiency",
        "sample_efficiency",
    ]
    robustness_dims = sorted(
        k for k in scores if k.startswith("robustness_")
    )
    return robustness_dims + fixed_tail

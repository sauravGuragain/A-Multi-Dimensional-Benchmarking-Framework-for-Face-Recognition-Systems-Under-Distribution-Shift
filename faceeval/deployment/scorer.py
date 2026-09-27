from __future__ import annotations

import logging
import math
from datetime import datetime
from typing import Any

import numpy as np

from faceeval.core.types import (
    BehavioralFingerprint,
    CalibrationMetrics,
    DeploymentRanking,
    DeploymentScenario,
    DeploymentScore,
    DeploymentWeights,
    EvaluationResult,
    FairnessReport,
    ModelName,
    ModelParadigm,
    ResourceUsage,
)
from faceeval.deployment.scenarios import get_weights

logger = logging.getLogger(__name__)

# Normalisation reference values
_REFERENCE_FPS        = 100.0     # 100 fps → latency score = 0.5
_REFERENCE_MEMORY_MB  = 1000.0    # 1 GB → memory score = 0.0
_REFERENCE_MODEL_MB   = 500.0     # 500 MB → cost score = 0.0


# ---------------------------------------------------------------------------
# Component score extractors
# ---------------------------------------------------------------------------

def _score_accuracy(results: list[EvaluationResult]) -> float:
    """Mean accuracy across all evaluation conditions for this model."""
    accs = [r.accuracy for r in results if r.accuracy >= 0]
    return float(np.mean(accs)) if accs else 0.0


def _score_robustness(fingerprint: BehavioralFingerprint | None) -> float:
    """
    Mean ADC-AUC across all perturbation types from the fingerprint.
    Falls back to 0.5 (neutral) when fingerprint is unavailable.
    """
    if fingerprint is None:
        return 0.5
    robustness_dims = [
        v for k, v in fingerprint.scores.items()
        if k.startswith("robustness_")
    ]
    if not robustness_dims:
        return 0.5
    return float(np.mean(robustness_dims))


def _score_calibration(calibrations: list[CalibrationMetrics]) -> float:
    """1 - mean ECE (lower ECE = better calibration = higher score)."""
    if not calibrations:
        return 0.5
    eces = [m.ece for m in calibrations]
    return float(np.clip(1.0 - np.mean(eces), 0.0, 1.0))


def _score_fairness(fairness: FairnessReport | None) -> float:
    """1 - max_accuracy_gap (lower gap = fairer = higher score)."""
    if fairness is None:
        return 0.5
    return float(np.clip(1.0 - fairness.max_accuracy_gap, 0.0, 1.0))


def _score_latency(usages: list[ResourceUsage]) -> float:
    """
    Sigmoid-normalised throughput score.
    fps=100 → 0.5; fps=1000 → ~0.91; fps=10 → ~0.09.
    """
    fps_vals = [
        u.throughput_fps for u in usages
        if u.throughput_fps is not None and u.throughput_fps > 0
    ]
    if not fps_vals:
        return 0.5
    mean_fps = float(np.mean(fps_vals))
    return float(1.0 / (1.0 + math.exp(-(math.log(max(mean_fps, 0.01)) - math.log(_REFERENCE_FPS)))))


def _score_memory(usages: list[ResourceUsage]) -> float:
    """1 - (mean_peak_memory / reference_memory), clamped to [0, 1]."""
    mem_vals = [
        u.peak_memory_mb for u in usages
        if u.peak_memory_mb is not None and u.peak_memory_mb >= 0
    ]
    if not mem_vals:
        return 0.5
    mean_mem = float(np.mean(mem_vals))
    return float(np.clip(1.0 - mean_mem / _REFERENCE_MEMORY_MB, 0.0, 1.0))


def _score_computational_cost(usages: list[ResourceUsage]) -> float:
    """
    Inverse-normalised model size score.
    Small model (few MB) → score near 1.0; large model (500+ MB) → near 0.
    """
    size_vals = [
        u.model_size_mb for u in usages
        if u.model_size_mb is not None and u.model_size_mb >= 0
    ]
    if not size_vals:
        return 0.5
    mean_size = float(np.mean(size_vals))
    return float(np.clip(1.0 - mean_size / _REFERENCE_MODEL_MB, 0.0, 1.0))


# ---------------------------------------------------------------------------
# Main scorer
# ---------------------------------------------------------------------------

class DeploymentScorer:
    """
    Computes ``DeploymentScore`` objects for one or more models.

    Parameters
    ----------
    scenario:
        Deployment context. Determines default weights.
    custom_weights:
        Override the scenario's default weights.
    """

    def __init__(
        self,
        scenario: DeploymentScenario = DeploymentScenario.RESEARCH_BENCHMARK,
        custom_weights: DeploymentWeights | None = None,
    ) -> None:
        self._scenario = scenario
        self._weights  = custom_weights or get_weights(scenario)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def score_model(
        self,
        model_name: ModelName,
        paradigm: ModelParadigm,
        evaluation_results: list[EvaluationResult],
        fingerprint: BehavioralFingerprint | None = None,
        calibration_metrics: list[CalibrationMetrics] | None = None,
        fairness_report: FairnessReport | None = None,
        resource_usages: list[ResourceUsage] | None = None,
    ) -> DeploymentScore:
        """
        Compute the deployment score for one model.

        Parameters
        ----------
        model_name:
            Registered model name.
        paradigm:
            TRADITIONAL or DEEP_LEARNING.
        evaluation_results:
            EvaluationResult objects for this model (all conditions).
        fingerprint:
            Pre-built BehavioralFingerprint (for robustness score).
        calibration_metrics:
            CalibrationMetrics objects for this model.
        fairness_report:
            FairnessReport for this model.
        resource_usages:
            ResourceUsage records for this model.

        Returns
        -------
        DeploymentScore
        """
        cal = calibration_metrics or []
        res = resource_usages or []

        # --- Compute raw component scores ---
        comp = {
            "accuracy":           _score_accuracy(evaluation_results),
            "robustness":         _score_robustness(fingerprint),
            "calibration":        _score_calibration(cal),
            "fairness":           _score_fairness(fairness_report),
            "latency":            _score_latency(res),
            "memory":             _score_memory(res),
            "computational_cost": _score_computational_cost(res),
        }

        # --- Apply weights ---
        w = self._weights
        weight_map = {
            "accuracy":           w.accuracy,
            "robustness":         w.robustness,
            "calibration":        w.calibration,
            "fairness":           w.fairness,
            "latency":            w.latency,
            "memory":             w.memory,
            "computational_cost": w.computational_cost,
        }
        weighted = {k: comp[k] * weight_map[k] for k in comp}
        total = float(sum(weighted.values()))

        recommendation = self._build_recommendation(
            model_name, comp, total, self._scenario
        )

        logger.info(
            "[%s | %s] Deployment score=%.4f "
            "(acc=%.3f rob=%.3f cal=%.3f fair=%.3f lat=%.3f mem=%.3f cost=%.3f)",
            model_name, self._scenario.value, total,
            comp["accuracy"], comp["robustness"], comp["calibration"],
            comp["fairness"], comp["latency"], comp["memory"],
            comp["computational_cost"],
        )

        return DeploymentScore(
            model_name=model_name,
            paradigm=paradigm,
            scenario=self._scenario,
            weights=self._weights,
            component_scores=comp,
            weighted_components=weighted,
            total_score=total,
            recommendation=recommendation,
            computed_at=datetime.utcnow(),
        )

    def score_all_models(
        self,
        model_data: dict[ModelName, dict[str, Any]],
    ) -> list[DeploymentScore]:
        """
        Score all models and return results sorted by descending total score.

        Parameters
        ----------
        model_data:
            Dict mapping model_name → dict with keys:
                'paradigm'             : ModelParadigm
                'evaluation_results'   : list[EvaluationResult]
                'fingerprint'          : BehavioralFingerprint | None
                'calibration_metrics'  : list[CalibrationMetrics] | None
                'fairness_report'      : FairnessReport | None
                'resource_usages'      : list[ResourceUsage] | None

        Returns
        -------
        list[DeploymentScore]  sorted by total_score descending (rank 1 first)
        """
        scores: list[DeploymentScore] = []
        for model_name, data in model_data.items():
            score = self.score_model(
                model_name=model_name,
                paradigm=data.get("paradigm", ModelParadigm.TRADITIONAL),
                evaluation_results=data.get("evaluation_results", []),
                fingerprint=data.get("fingerprint"),
                calibration_metrics=data.get("calibration_metrics"),
                fairness_report=data.get("fairness_report"),
                resource_usages=data.get("resource_usages"),
            )
            scores.append(score)

        # Sort descending by total score
        scores.sort(key=lambda s: s.total_score, reverse=True)

        # Assign ranks. The rank-1 model's recommendation text is rebuilt
        # with rank context so it can correctly say "best available" instead
        # of a flat "Not recommended" when its absolute score sits below the
        # unreserved-recommendation threshold but it is still the strongest
        # of the models actually evaluated.
        ranked: list[DeploymentScore] = []
        for rank, score in enumerate(scores, start=1):
            recommendation = score.recommendation
            if rank == 1 and score.total_score < 0.65:
                recommendation = self._build_recommendation(
                    score.model_name, score.component_scores, score.total_score,
                    self._scenario, rank=rank, is_best_available=True,
                )
            ranked.append(DeploymentScore(
                **{f: getattr(score, f) for f in score.__dataclass_fields__
                   if f not in ("rank", "recommendation")},
                rank=rank,
                recommendation=recommendation,
            ))
        return ranked

    def build_ranking(
        self,
        scores: list[DeploymentScore],
    ) -> DeploymentRanking:
        """Package a list of scored models into a ``DeploymentRanking``."""
        sorted_scores = sorted(scores, key=lambda s: s.total_score, reverse=True)
        return DeploymentRanking(
            scenario=self._scenario,
            weights=self._weights,
            ranked_scores=sorted_scores,
        )

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    @staticmethod
    def _build_recommendation(
        model_name: ModelName,
        comp: dict[str, float],
        total: float,
        scenario: DeploymentScenario,
        rank: int | None = None,
        is_best_available: bool = False,
    ) -> str:
        """
        Generate a human-readable deployment recommendation.

        Highlights the model's strongest and weakest components in context.

        Parameters
        ----------
        rank, is_best_available:
            When the caller knows this model's rank among the full candidate
            set, the recommendation acknowledges it explicitly rather than
            producing a tier label in isolation. This avoids the contradictory
            "Top-ranked model: ... Not recommended" phrasing that results
            from applying an absolute score threshold without rank context —
            a model can simultaneously be the best of the models evaluated
            and still fall below the absolute bar for unreserved
            recommendation; the text should say that plainly instead of
            implying a flat rejection.
        """
        sorted_comp = sorted(comp.items(), key=lambda x: x[1])
        weakest_dim, weakest_val = sorted_comp[0]
        strongest_dim, strongest_val = sorted_comp[-1]
        scenario_label = scenario.value.replace("_", " ")

        if total >= 0.65:
            tier_text = f"Recommended for {scenario_label}."
        elif total >= 0.45:
            tier_text = f"Conditional fit for {scenario_label} — review weakest dimension before deploying."
        elif is_best_available:
            tier_text = (
                f"Best available option for {scenario_label} among the models evaluated, "
                f"but its absolute score is low — none of the candidates may be production-ready "
                f"for this scenario without further tuning or additional models."
            )
        else:
            tier_text = f"Not recommended for {scenario_label}."

        return (
            f"{tier_text} "
            f"Score={total:.3f}. "
            f"Strongest: {strongest_dim} ({strongest_val:.3f}). "
            f"Weakest: {weakest_dim} ({weakest_val:.3f})."
        )
from __future__ import annotations

import logging
from typing import Any

import numpy as np

from faceeval.core.types import (
    DeploymentRanking,
    DeploymentScenario,
    DeploymentScore,
    ModelName,
    ModelParadigm,
)
from faceeval.deployment.scenarios import get_metadata

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Ranking utilities
# ---------------------------------------------------------------------------

def rank_models(scores: list[DeploymentScore]) -> list[DeploymentScore]:
    """
    Sort scores descending by total_score and assign integer ranks (1 = best).
    Returns new DeploymentScore objects with ``rank`` populated.
    """
    sorted_scores = sorted(scores, key=lambda s: s.total_score, reverse=True)
    ranked: list[DeploymentScore] = []
    for rank, score in enumerate(sorted_scores, start=1):
        ranked.append(DeploymentScore(
            **{f: getattr(score, f) for f in score.__dataclass_fields__
               if f != "rank"},
            rank=rank,
        ))
    return ranked


def cross_scenario_ranking(
    scores_by_scenario: dict[DeploymentScenario, list[DeploymentScore]],
) -> dict[str, Any]:
    """
    Compare model rankings across multiple deployment scenarios.

    Parameters
    ----------
    scores_by_scenario:
        Dict mapping scenario → list of DeploymentScore (already ranked).

    Returns
    -------
    dict with:
        rank_matrix     — model × scenario rank matrix (models as rows)
        mean_rank       — mean rank across scenarios per model
        rank_variance   — rank variance (low = consistent across contexts)
        best_universal  — model with lowest mean rank (best overall)
        scenario_specialists — models that rank 1 in only one scenario
    """
    all_models: list[ModelName] = []
    for scores in scores_by_scenario.values():
        for s in scores:
            if s.model_name not in all_models:
                all_models.append(s.model_name)

    scenarios = list(scores_by_scenario.keys())
    rank_matrix: dict[ModelName, dict[str, int]] = {m: {} for m in all_models}

    for scenario, scores in scores_by_scenario.items():
        ranked = rank_models(scores)
        for score in ranked:
            rank_matrix[score.model_name][scenario.value] = score.rank or 999

    # Mean and variance of ranks per model
    stats: dict[ModelName, dict[str, float]] = {}
    for model, ranks in rank_matrix.items():
        rank_vals = [ranks.get(s.value, 999) for s in scenarios]
        stats[model] = {
            "mean_rank": float(np.mean(rank_vals)),
            "rank_variance": float(np.var(rank_vals)),
            "min_rank": int(min(rank_vals)),
            "max_rank": int(max(rank_vals)),
        }

    best_universal = min(stats, key=lambda m: stats[m]["mean_rank"])

    # Specialists: rank 1 in exactly one scenario
    top_per_scenario: dict[str, ModelName] = {}
    for scenario, scores in scores_by_scenario.items():
        ranked = rank_models(scores)
        if ranked:
            top_per_scenario[scenario.value] = ranked[0].model_name

    specialists = [
        m for m in all_models
        if list(top_per_scenario.values()).count(m) == 1
    ]

    return {
        "models": all_models,
        "scenarios": [s.value for s in scenarios],
        "rank_matrix": rank_matrix,
        "model_stats": stats,
        "best_universal_model": best_universal,
        "scenario_specialists": specialists,
        "top_per_scenario": top_per_scenario,
    }


# ---------------------------------------------------------------------------
# Trade-off analysis
# ---------------------------------------------------------------------------

def trade_off_table(
    score_a: DeploymentScore,
    score_b: DeploymentScore,
) -> dict[str, Any]:
    """
    Compare two models dimension-by-dimension.

    Returns a structured dict suitable for a LaTeX table in Chapter 12.
    """
    dims = list(score_a.component_scores.keys())
    a_wins: list[str] = []
    b_wins: list[str] = []
    ties:   list[str] = []

    for dim in dims:
        ca = score_a.component_scores.get(dim, 0.0)
        cb = score_b.component_scores.get(dim, 0.0)
        if abs(ca - cb) < 0.02:
            ties.append(dim)
        elif ca > cb:
            a_wins.append(dim)
        else:
            b_wins.append(dim)

    return {
        "model_a": score_a.model_name,
        "model_b": score_b.model_name,
        "scenario": score_a.scenario.value,
        "score_a_total": round(score_a.total_score, 4),
        "score_b_total": round(score_b.total_score, 4),
        "winner": score_a.model_name if score_a.total_score > score_b.total_score else score_b.model_name,
        "a_wins_on": a_wins,
        "b_wins_on": b_wins,
        "ties_on": ties,
        "dimension_details": {
            dim: {
                "score_a": round(score_a.component_scores.get(dim, 0.0), 4),
                "score_b": round(score_b.component_scores.get(dim, 0.0), 4),
                "advantage": "a" if score_a.component_scores.get(dim, 0.0) > score_b.component_scores.get(dim, 0.0) else "b",
            }
            for dim in dims
        },
    }


# ---------------------------------------------------------------------------
# Practitioner decision guide
# ---------------------------------------------------------------------------

def generate_decision_guide(
    rankings_by_scenario: dict[DeploymentScenario, DeploymentRanking],
    available_models: list[ModelName] | None = None,
) -> dict[str, Any]:
    """
    Generate a practitioner decision guide for all scenarios.

    The guide maps each scenario to:
      - Recommended model (rank 1)
      - Runner-up (rank 2)
      - Key strengths of the recommended model
      - Known limitations
      - Critical threshold warnings (e.g. FAR > acceptable limit)

    This output is structured to populate Chapter 12 directly.
    """
    guide: dict[str, Any] = {
        "title": "FaceEval-X Practitioner Deployment Decision Guide",
        "generated_from_scenarios": [s.value for s in rankings_by_scenario],
        "scenarios": {},
    }

    for scenario, ranking in rankings_by_scenario.items():
        meta = get_metadata(scenario)
        if not ranking.ranked_scores:
            continue

        top   = ranking.ranked_scores[0]
        runner = ranking.ranked_scores[1] if len(ranking.ranked_scores) > 1 else None

        # Identify strengths (top-3 component scores)
        sorted_comps = sorted(
            top.component_scores.items(),
            key=lambda x: x[1], reverse=True,
        )
        strengths   = [f"{k} ({v:.3f})" for k, v in sorted_comps[:3]]
        weaknesses  = [f"{k} ({v:.3f})" for k, v in sorted_comps[-2:]]

        # Threshold warnings
        warnings: list[str] = []
        acc_threshold = meta.get("minimum_accuracy")
        if acc_threshold and top.component_scores.get("accuracy", 0) < acc_threshold:
            warnings.append(
                f"WARNING: Model accuracy ({top.component_scores['accuracy']:.3f}) "
                f"is below the minimum required for this scenario ({acc_threshold:.3f})."
            )

        guide["scenarios"][scenario.value] = {
            "display_name": meta["display_name"],
            "description": meta["description"],
            "recommended_model": {
                "name": top.model_name,
                "paradigm": top.paradigm.value,
                "total_score": round(top.total_score, 4),
                "rank": top.rank,
                "strengths": strengths,
                "weaknesses": weaknesses,
                "recommendation_text": top.recommendation,
            },
            "runner_up": {
                "name": runner.model_name,
                "total_score": round(runner.total_score, 4),
            } if runner else None,
            "critical_metrics": meta["critical_metrics"],
            "warnings": warnings,
            "score_gap": round(
                top.total_score - (runner.total_score if runner else 0.0), 4
            ),
        }

    return guide


# ---------------------------------------------------------------------------
# Ranking summary table (for CSV / LaTeX export)
# ---------------------------------------------------------------------------

def ranking_summary_table(
    rankings_by_scenario: dict[DeploymentScenario, DeploymentRanking],
) -> list[dict[str, Any]]:
    """
    Build a flat summary table: one row per (model × scenario).
    Suitable for CSV export and LaTeX tabular environment.
    """
    rows: list[dict[str, Any]] = []
    for scenario, ranking in rankings_by_scenario.items():
        for score in ranking.ranked_scores:
            rows.append({
                "scenario": scenario.value,
                "rank": score.rank,
                "model_name": score.model_name,
                "paradigm": score.paradigm.value,
                "total_score": round(score.total_score, 4),
                **{f"score_{k}": round(v, 4) for k, v in score.component_scores.items()},
                **{f"weighted_{k}": round(v, 4) for k, v in score.weighted_components.items()},
                "recommendation": score.recommendation,
            })
    return rows

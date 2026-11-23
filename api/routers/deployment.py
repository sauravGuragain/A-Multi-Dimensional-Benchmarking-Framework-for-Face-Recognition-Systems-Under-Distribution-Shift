from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from api.common import as_dict, as_list, load_run, num
from api.dependencies import get_result_store
from api.readonly_store import ReadOnlyResultStore
from api.schemas import (
    DecisionGuideResponse, DeploymentRankingResponse, DeploymentScoreResponse,
)

router = APIRouter(prefix="/api/runs/{run_id}/deployment", tags=["deployment"])


def _floats(d) -> dict[str, float]:
    return {str(k): v for k, v in as_dict(d).items() if num(v) is not None}


def _parse_ranking(r) -> DeploymentRankingResponse | None:
    r = as_dict(r)
    if not isinstance(r.get("scenario"), str) or not isinstance(r.get("ranking"), list):
        return None
    scores = []
    for s in r["ranking"]:
        s = as_dict(s)
        if not isinstance(s.get("model_name"), str):
            continue
        rank = s.get("rank")
        scores.append(DeploymentScoreResponse(
            model_name=s["model_name"],
            paradigm=str(s.get("paradigm") or ""),
            rank=rank if isinstance(rank, int) and not isinstance(rank, bool) else None,
            total_score=num(s.get("total_score")),
            component_scores=_floats(s.get("component_scores")),
            weighted_components=_floats(s.get("weighted_components")),
            recommendation=str(s.get("recommendation") or ""),
        ))
    return DeploymentRankingResponse(
        scenario=r["scenario"], weights=_floats(r.get("weights")), ranked_scores=scores,
    )


def _rankings(store, run_id) -> list[DeploymentRankingResponse]:
    raw = as_list(load_run(store, run_id).get("deployment_rankings"))
    return [p for p in (_parse_ranking(r) for r in raw) if p is not None]


@router.get("/rankings", response_model=list[DeploymentRankingResponse])
def list_rankings(
    run_id: str, store: ReadOnlyResultStore = Depends(get_result_store),
) -> list[DeploymentRankingResponse]:
    """List deployment rankings for all scenarios in this run."""
    return _rankings(store, run_id)


@router.get("/rankings/{scenario}", response_model=DeploymentRankingResponse)
def get_ranking_for_scenario(
    run_id: str, scenario: str, store: ReadOnlyResultStore = Depends(get_result_store),
) -> DeploymentRankingResponse:
    """Get the deployment ranking for one specific scenario."""
    match = next((r for r in _rankings(store, run_id) if r.scenario == scenario), None)
    if match is None:
        raise HTTPException(404, f"No ranking for scenario '{scenario}' in run '{run_id}'.")
    return match


@router.get("/decision-guide", response_model=DecisionGuideResponse)
def get_decision_guide(
    run_id: str, store: ReadOnlyResultStore = Depends(get_result_store),
) -> DecisionGuideResponse:
    """Practitioner decision guide generated from this run's stored rankings."""
    from faceeval.core.types import (
        DeploymentRanking, DeploymentScenario, DeploymentScore, DeploymentWeights, ModelParadigm,
    )
    from faceeval.deployment.ranker import generate_decision_guide

    raw = as_list(load_run(store, run_id).get("deployment_rankings"))
    if not raw:
        raise HTTPException(404, f"No deployment rankings found for run '{run_id}'.")
    try:
        by_scenario = {}
        for r in raw:
            scenario = DeploymentScenario(r["scenario"])
            weights = DeploymentWeights(**r["weights"])
            scores = [
                DeploymentScore(
                    model_name=s["model_name"], paradigm=ModelParadigm(s["paradigm"]),
                    scenario=scenario, weights=weights,
                    component_scores=s["component_scores"],
                    weighted_components=s["weighted_components"],
                    total_score=s["total_score"], rank=s.get("rank"),
                    recommendation=s.get("recommendation", ""),
                )
                for s in r["ranking"]
            ]
            by_scenario[scenario] = DeploymentRanking(scenario, weights, scores)
        guide = generate_decision_guide(by_scenario)
        return DecisionGuideResponse(title=guide["title"], scenarios=guide["scenarios"])
    except (KeyError, TypeError, ValueError) as exc:
        raise HTTPException(422, f"Stored rankings for run '{run_id}' cannot be used: {exc}")

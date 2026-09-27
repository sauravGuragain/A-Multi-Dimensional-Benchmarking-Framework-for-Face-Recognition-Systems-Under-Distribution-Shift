from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from api.dependencies import get_result_store
from api.schemas import (
    DecisionGuideResponse, DeploymentRankingResponse, DeploymentScoreResponse,
)
from faceeval.core.exceptions import RunNotFoundError
from faceeval.storage.result_store import ResultStore

router = APIRouter(prefix="/api/runs/{run_id}/deployment", tags=["deployment"])


@router.get("/rankings", response_model=list[DeploymentRankingResponse])
def list_rankings(
    run_id: str,
    store: ResultStore = Depends(get_result_store),
) -> list[DeploymentRankingResponse]:
    """List deployment rankings for all scenarios in this run."""
    try:
        data = store.get(run_id)
    except RunNotFoundError:
        raise HTTPException(404, f"Run '{run_id}' not found.")

    rankings = data.get("deployment_rankings", [])
    return [
        DeploymentRankingResponse(
            scenario=r["scenario"],
            weights=r["weights"],
            ranked_scores=[
                DeploymentScoreResponse(
                    model_name=s["model_name"],
                    paradigm=s["paradigm"],
                    rank=s.get("rank"),
                    total_score=s["total_score"],
                    component_scores=s["component_scores"],
                    weighted_components=s["weighted_components"],
                    recommendation=s.get("recommendation", ""),
                )
                for s in r["ranking"]
            ],
        )
        for r in rankings
    ]


@router.get("/rankings/{scenario}", response_model=DeploymentRankingResponse)
def get_ranking_for_scenario(
    run_id: str,
    scenario: str,
    store: ResultStore = Depends(get_result_store),
) -> DeploymentRankingResponse:
    """Get the deployment ranking for one specific scenario."""
    try:
        data = store.get(run_id)
    except RunNotFoundError:
        raise HTTPException(404, f"Run '{run_id}' not found.")

    rankings = data.get("deployment_rankings", [])
    match = next((r for r in rankings if r["scenario"] == scenario), None)
    if match is None:
        raise HTTPException(404, f"No ranking for scenario '{scenario}' in run '{run_id}'.")

    return DeploymentRankingResponse(
        scenario=match["scenario"],
        weights=match["weights"],
        ranked_scores=[
            DeploymentScoreResponse(
                model_name=s["model_name"],
                paradigm=s["paradigm"],
                rank=s.get("rank"),
                total_score=s["total_score"],
                component_scores=s["component_scores"],
                weighted_components=s["weighted_components"],
                recommendation=s.get("recommendation", ""),
            )
            for s in match["ranking"]
        ],
    )


@router.get("/decision-guide", response_model=DecisionGuideResponse)
def get_decision_guide(
    run_id: str,
    store: ResultStore = Depends(get_result_store),
) -> DecisionGuideResponse:
    """
    Generate the practitioner decision guide from this run's deployment rankings.
    """
    from faceeval.deployment.ranker import generate_decision_guide
    from faceeval.core.types import (
        DeploymentRanking, DeploymentScore, DeploymentWeights,
        DeploymentScenario, ModelParadigm,
    )

    try:
        data = store.get(run_id)
    except RunNotFoundError:
        raise HTTPException(404, f"Run '{run_id}' not found.")

    rankings_data = data.get("deployment_rankings", [])
    if not rankings_data:
        raise HTTPException(404, f"No deployment rankings found for run '{run_id}'.")

    rankings_by_scenario = {}
    for r in rankings_data:
        scenario = DeploymentScenario(r["scenario"])
        weights = DeploymentWeights(**r["weights"])
        scores = [
            DeploymentScore(
                model_name=s["model_name"],
                paradigm=ModelParadigm(s["paradigm"]),
                scenario=scenario,
                weights=weights,
                component_scores=s["component_scores"],
                weighted_components=s["weighted_components"],
                total_score=s["total_score"],
                rank=s.get("rank"),
                recommendation=s.get("recommendation", ""),
            )
            for s in r["ranking"]
        ]
        rankings_by_scenario[scenario] = DeploymentRanking(scenario, weights, scores)

    guide = generate_decision_guide(rankings_by_scenario)
    return DecisionGuideResponse(title=guide["title"], scenarios=guide["scenarios"])

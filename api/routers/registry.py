from __future__ import annotations

from fastapi import APIRouter

from api.schemas import (
    ModelInfoResponse, ModelListResponse,
    PerturbationInfoResponse, PerturbationListResponse,
    ScenarioInfoResponse, ScenarioListResponse,
)

router = APIRouter(prefix="/api/registry", tags=["registry"])


@router.get("/models", response_model=ModelListResponse)
def list_registered_models() -> ModelListResponse:
    """List all registered recognition models, grouped by paradigm."""
    import faceeval.models  # noqa: F401  ensure registrations are loaded
    from faceeval.core.registry import model_registry
    from faceeval.core.types import ModelParadigm

    traditional = []
    deep = []
    for key in model_registry.list_registered():
        meta = model_registry.get_metadata(key)
        info = ModelInfoResponse(
            name=key,
            paradigm=meta["paradigm"].value,
            model_type=meta["model_type"],
            description=meta.get("description", ""),
        )
        if meta["paradigm"] == ModelParadigm.TRADITIONAL:
            traditional.append(info)
        else:
            deep.append(info)

    return ModelListResponse(traditional=traditional, deep_learning=deep)


@router.get("/perturbations", response_model=PerturbationListResponse)
def list_registered_perturbations() -> PerturbationListResponse:
    """List all registered perturbation types."""
    import faceeval.perturbation  # noqa: F401
    from faceeval.core.registry import perturbation_registry

    perturbations = []
    for key in perturbation_registry.list_registered():
        meta = perturbation_registry.get_metadata(key)
        perturbations.append(PerturbationInfoResponse(
            name=key,
            category=meta.get("category", ""),
            description=meta.get("description", ""),
        ))
    return PerturbationListResponse(perturbations=perturbations)


@router.get("/scenarios", response_model=ScenarioListResponse)
def list_deployment_scenarios() -> ScenarioListResponse:
    """List all built-in deployment scenarios with their default weights."""
    from faceeval.deployment.scenarios import list_scenarios

    scenarios = list_scenarios()
    return ScenarioListResponse(
        scenarios=[
            ScenarioInfoResponse(
                scenario=s["scenario"],
                display_name=s["display_name"],
                description=s["description"],
                weights=s["weights"],
            )
            for s in scenarios
        ]
    )

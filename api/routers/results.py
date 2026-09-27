from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query

from api.dependencies import get_result_store
from api.schemas import EvaluationResultListResponse, EvaluationResultResponse
from faceeval.core.exceptions import RunNotFoundError
from faceeval.storage.result_store import ResultStore

router = APIRouter(prefix="/api/runs/{run_id}/results", tags=["results"])


@router.get("", response_model=EvaluationResultListResponse)
def list_evaluation_results(
    run_id: str,
    model_name: str | None = Query(None),
    perturbation_type: str | None = Query(None),
    min_severity: float | None = Query(None, ge=0.0, le=1.0),
    max_severity: float | None = Query(None, ge=0.0, le=1.0),
    store: ResultStore = Depends(get_result_store),
) -> EvaluationResultListResponse:
    """
    List evaluation results for one run, with optional filtering.

    Filter combinations support the dashboard's "compare models under
    perturbation X" and "show severity sweep for model Y" views.
    """
    try:
        data = store.get(run_id)
    except RunNotFoundError:
        raise HTTPException(404, f"Run '{run_id}' not found.")

    raw_results = data.get("evaluation_results", [])
    filtered = []

    for r in raw_results:
        if model_name and r.get("model_name") != model_name:
            continue

        pert_spec = r.get("perturbation_spec")
        pt = pert_spec.get("perturbation_type") if pert_spec else None
        sev = pert_spec.get("severity") if pert_spec else None

        if perturbation_type and pt != perturbation_type:
            continue
        if min_severity is not None and (sev is None or sev < min_severity):
            continue
        if max_severity is not None and (sev is None or sev > max_severity):
            continue

        filtered.append(EvaluationResultResponse(
            model_name=r.get("model_name", ""),
            model_paradigm=r.get("model_paradigm", ""),
            dataset_name=r.get("dataset_name", ""),
            perturbation_type=pt,
            severity=sev,
            accuracy=r.get("accuracy", 0.0),
            precision=r.get("precision", 0.0),
            recall=r.get("recall", 0.0),
            f1_score=r.get("f1_score", 0.0),
            auc=r.get("auc", 0.0),
            eer=r.get("eer", 0.0),
            num_samples=r.get("num_samples", 0),
        ))

    return EvaluationResultListResponse(results=filtered, total=len(filtered))


@router.get("/adc")
def get_adc_curve(
    run_id: str,
    model_name: str = Query(...),
    perturbation_type: str = Query(...),
    store: ResultStore = Depends(get_result_store),
) -> dict:
    """
    Return ADC curve data (severity → accuracy points) for one model and
    perturbation type, ready for plotting on the dashboard.
    """
    try:
        data = store.get(run_id)
    except RunNotFoundError:
        raise HTTPException(404, f"Run '{run_id}' not found.")

    points = []
    for r in data.get("evaluation_results", []):
        if r.get("model_name") != model_name:
            continue
        spec = r.get("perturbation_spec")
        if not spec or spec.get("perturbation_type") != perturbation_type:
            continue
        points.append({"severity": spec.get("severity"), "accuracy": r.get("accuracy")})

    points.sort(key=lambda p: p["severity"])
    return {
        "model_name": model_name,
        "perturbation_type": perturbation_type,
        "points": points,
    }

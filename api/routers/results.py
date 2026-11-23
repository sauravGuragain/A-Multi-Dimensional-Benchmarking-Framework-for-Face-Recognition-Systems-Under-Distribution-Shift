from __future__ import annotations

from fastapi import APIRouter, Depends, Query

from api.common import as_dict, as_list, load_run, num
from api.dependencies import get_result_store
from api.readonly_store import ReadOnlyResultStore
from api.schemas import EvaluationResultListResponse, EvaluationResultResponse

router = APIRouter(prefix="/api/runs/{run_id}/results", tags=["results"])


def _int_or_none(v):
    return v if isinstance(v, int) and not isinstance(v, bool) else None


@router.get("", response_model=EvaluationResultListResponse)
def list_evaluation_results(
    run_id: str,
    model_name: str | None = Query(None),
    perturbation_type: str | None = Query(None),
    min_severity: float | None = Query(None, ge=0.0, le=1.0),
    max_severity: float | None = Query(None, ge=0.0, le=1.0),
    store: ReadOnlyResultStore = Depends(get_result_store),
) -> EvaluationResultListResponse:
    """List evaluation results for one run, with optional filtering."""
    data = load_run(store, run_id)
    filtered = []

    for r in as_list(data.get("evaluation_results")):
        if not isinstance(r, dict):
            continue
        if model_name and r.get("model_name") != model_name:
            continue

        spec = as_dict(r.get("perturbation_spec"))
        pt = spec.get("perturbation_type")
        sev = num(spec.get("severity"))

        if perturbation_type and pt != perturbation_type:
            continue
        if min_severity is not None and (sev is None or sev < min_severity):
            continue
        if max_severity is not None and (sev is None or sev > max_severity):
            continue

        filtered.append(EvaluationResultResponse(
            model_name=str(r.get("model_name") or ""),
            model_paradigm=str(r.get("model_paradigm") or ""),
            dataset_name=str(r.get("dataset_name") or ""),
            perturbation_type=pt if isinstance(pt, str) else None,
            severity=sev,
            accuracy=num(r.get("accuracy")),
            precision=num(r.get("precision")),
            recall=num(r.get("recall")),
            f1_score=num(r.get("f1_score")),
            auc=num(r.get("auc")),
            eer=num(r.get("eer")),
            num_samples=_int_or_none(r.get("num_samples")),
        ))

    return EvaluationResultListResponse(results=filtered, total=len(filtered))


@router.get("/adc")
def get_adc_curve(
    run_id: str,
    model_name: str = Query(...),
    perturbation_type: str = Query(...),
    store: ReadOnlyResultStore = Depends(get_result_store),
) -> dict:
    """ADC data (severity -> accuracy) for one model and perturbation type.

    Points with a missing severity or accuracy are dropped, not zero-filled.
    """
    data = load_run(store, run_id)
    points = []
    for r in as_list(data.get("evaluation_results")):
        if not isinstance(r, dict):
            continue
        if r.get("model_name") != model_name:
            continue
        spec = as_dict(r.get("perturbation_spec"))
        if spec.get("perturbation_type") != perturbation_type:
            continue
        sev, acc = num(spec.get("severity")), num(r.get("accuracy"))
        if sev is None or acc is None:
            continue
        points.append({"severity": sev, "accuracy": acc})

    points.sort(key=lambda p: p["severity"])
    return {"model_name": model_name, "perturbation_type": perturbation_type, "points": points}

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query

from api.common import as_dict, as_list, load_run, num
from api.dependencies import get_result_store
from api.readonly_store import ReadOnlyResultStore
from api.schemas import RunDetailResponse, RunListResponse, RunSummary
from faceeval.core.types import ExperimentStatus

router = APIRouter(prefix="/api/runs", tags=["runs"])

# The dashboard is read-only: there is intentionally no DELETE/PUT/POST here.


@router.get("", response_model=RunListResponse)
def list_runs(
    experiment_name: str | None = Query(None, description="Filter by experiment name"),
    status: str | None = Query(None, description="Filter by status (pending/running/completed/failed)"),
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    store: ReadOnlyResultStore = Depends(get_result_store),
) -> RunListResponse:
    """List experiment runs, most recent first."""
    if status:
        try:
            ExperimentStatus(status)
        except ValueError:
            raise HTTPException(400, f"Invalid status '{status}'. Valid: {[s.value for s in ExperimentStatus]}")

    runs = store.list_runs(experiment_name=experiment_name, status=status, limit=limit, offset=offset)
    total = store.count_runs(experiment_name=experiment_name, status=status)
    return RunListResponse(
        runs=[RunSummary(**r) for r in runs], total=total, limit=limit, offset=offset,
    )


@router.get("/{run_id}", response_model=RunDetailResponse)
def get_run(
    run_id: str,
    store: ReadOnlyResultStore = Depends(get_result_store),
) -> RunDetailResponse:
    """Get full details for one experiment run."""
    data = load_run(store, run_id)
    return RunDetailResponse(
        run_id=str(data.get("run_id", run_id)),
        status=str(data.get("status", "unknown")),
        config=as_dict(data.get("config")),
        started_at=data.get("started_at"),
        completed_at=data.get("completed_at"),
        duration_seconds=num(data.get("duration_seconds")),
        error_message=str(data.get("error_message") or ""),
        num_evaluation_results=len(as_list(data.get("evaluation_results"))),
        num_fingerprints=len(as_list(data.get("fingerprints"))),
        num_deployment_rankings=len(as_list(data.get("deployment_rankings"))),
    )

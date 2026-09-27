from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query

from api.dependencies import get_result_store
from api.schemas import RunDetailResponse, RunListResponse, RunSummary
from faceeval.core.exceptions import RunNotFoundError
from faceeval.core.types import ExperimentStatus
from faceeval.storage.result_store import ResultStore

router = APIRouter(prefix="/api/runs", tags=["runs"])


@router.get("", response_model=RunListResponse)
def list_runs(
    experiment_name: str | None = Query(None, description="Filter by experiment name"),
    status: str | None = Query(None, description="Filter by status (pending/running/completed/failed)"),
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    store: ResultStore = Depends(get_result_store),
) -> RunListResponse:
    """List experiment runs, most recent first."""
    status_enum = None
    if status:
        try:
            status_enum = ExperimentStatus(status)
        except ValueError:
            raise HTTPException(400, f"Invalid status '{status}'. Valid: {[s.value for s in ExperimentStatus]}")

    runs = store.list_runs(
        experiment_name=experiment_name, status=status_enum,
        limit=limit, offset=offset,
    )
    total = store.count_runs(experiment_name=experiment_name, status=status_enum)

    return RunListResponse(
        runs=[RunSummary(**r) for r in runs],
        total=total, limit=limit, offset=offset,
    )


@router.get("/{run_id}", response_model=RunDetailResponse)
def get_run(
    run_id: str,
    store: ResultStore = Depends(get_result_store),
) -> RunDetailResponse:
    """Get full details for one experiment run."""
    try:
        data = store.get(run_id)
    except RunNotFoundError:
        raise HTTPException(404, f"Run '{run_id}' not found.")

    return RunDetailResponse(
        run_id=data["run_id"],
        status=data["status"],
        config=data["config"],
        started_at=data.get("started_at"),
        completed_at=data.get("completed_at"),
        duration_seconds=data.get("duration_seconds"),
        error_message=data.get("error_message", ""),
        num_evaluation_results=len(data.get("evaluation_results", [])),
        num_fingerprints=len(data.get("fingerprints", [])),
        num_deployment_rankings=len(data.get("deployment_rankings", [])),
    )


@router.delete("/{run_id}")
def delete_run(
    run_id: str,
    store: ResultStore = Depends(get_result_store),
) -> dict[str, bool]:
    """Delete an experiment run."""
    deleted = store.delete(run_id)
    if not deleted:
        raise HTTPException(404, f"Run '{run_id}' not found.")
    return {"deleted": True}

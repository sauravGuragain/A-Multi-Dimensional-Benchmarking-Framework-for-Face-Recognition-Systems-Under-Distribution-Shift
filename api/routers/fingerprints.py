from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query

from api.dependencies import get_result_store
from api.schemas import (
    FingerprintComparisonResponse, FingerprintListResponse, FingerprintResponse,
)
from faceeval.core.exceptions import RunNotFoundError
from faceeval.storage.result_store import ResultStore

router = APIRouter(prefix="/api/runs/{run_id}/fingerprints", tags=["fingerprints"])


@router.get("", response_model=FingerprintListResponse)
def list_fingerprints(
    run_id: str,
    store: ResultStore = Depends(get_result_store),
) -> FingerprintListResponse:
    """List behavioral fingerprints for all models in this run."""
    try:
        data = store.get(run_id)
    except RunNotFoundError:
        raise HTTPException(404, f"Run '{run_id}' not found.")

    fps = data.get("fingerprints", [])
    return FingerprintListResponse(
        fingerprints=[
            FingerprintResponse(
                model_name=fp["model_name"],
                paradigm=fp["paradigm"],
                dimension_names=fp["dimension_names"],
                vector=fp["vector"],
                scores=fp["scores"],
            )
            for fp in fps
        ]
    )


@router.get("/{model_name}", response_model=FingerprintResponse)
def get_fingerprint(
    run_id: str,
    model_name: str,
    store: ResultStore = Depends(get_result_store),
) -> FingerprintResponse:
    """Get the fingerprint for one specific model."""
    try:
        data = store.get(run_id)
    except RunNotFoundError:
        raise HTTPException(404, f"Run '{run_id}' not found.")

    fps = data.get("fingerprints", [])
    match = next((fp for fp in fps if fp["model_name"] == model_name), None)
    if match is None:
        raise HTTPException(404, f"No fingerprint for model '{model_name}' in run '{run_id}'.")

    return FingerprintResponse(
        model_name=match["model_name"],
        paradigm=match["paradigm"],
        dimension_names=match["dimension_names"],
        vector=match["vector"],
        scores=match["scores"],
    )


@router.get("/compare/{model_a}/{model_b}", response_model=FingerprintComparisonResponse)
def compare_fingerprints(
    run_id: str,
    model_a: str,
    model_b: str,
    store: ResultStore = Depends(get_result_store),
) -> FingerprintComparisonResponse:
    """Compare two models' fingerprints: distances and per-dimension gaps."""
    try:
        data = store.get(run_id)
    except RunNotFoundError:
        raise HTTPException(404, f"Run '{run_id}' not found.")

    fps = {fp["model_name"]: fp for fp in data.get("fingerprints", [])}
    if model_a not in fps:
        raise HTTPException(404, f"No fingerprint for '{model_a}'.")
    if model_b not in fps:
        raise HTTPException(404, f"No fingerprint for '{model_b}'.")

    fa, fb = fps[model_a], fps[model_b]
    if fa["dimension_names"] != fb["dimension_names"]:
        raise HTTPException(400, "Fingerprints have mismatched dimensions.")

    import math
    va, vb = fa["vector"], fb["vector"]
    euclidean = math.sqrt(sum((a - b) ** 2 for a, b in zip(va, vb)))
    norm_a = math.sqrt(sum(a * a for a in va))
    norm_b = math.sqrt(sum(b * b for b in vb))
    cosine = 1.0 - (sum(a * b for a, b in zip(va, vb)) / (norm_a * norm_b)) if norm_a > 0 and norm_b > 0 else 1.0

    gaps = {dim: vb[i] - va[i] for i, dim in enumerate(fa["dimension_names"])}

    return FingerprintComparisonResponse(
        model_a=model_a, model_b=model_b,
        euclidean_distance=euclidean,
        cosine_distance=cosine,
        dimension_gaps=gaps,
    )

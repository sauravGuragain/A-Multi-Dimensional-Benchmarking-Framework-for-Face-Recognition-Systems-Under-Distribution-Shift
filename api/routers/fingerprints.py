from __future__ import annotations

import math

from fastapi import APIRouter, Depends, HTTPException

from api.common import as_dict, as_list, load_run, num
from api.dependencies import get_result_store
from api.readonly_store import ReadOnlyResultStore
from api.schemas import (
    FingerprintComparisonResponse, FingerprintListResponse, FingerprintResponse,
)

router = APIRouter(prefix="/api/runs/{run_id}/fingerprints", tags=["fingerprints"])


def _parse(fp) -> FingerprintResponse | None:
    """Parse one stored fingerprint, or None if it is malformed (never guessed)."""
    fp = as_dict(fp)
    names, vector = fp.get("dimension_names"), fp.get("vector")
    if not isinstance(fp.get("model_name"), str) or not isinstance(names, list) \
            or not isinstance(vector, list) or len(names) != len(vector):
        return None
    vec = [num(v) for v in vector]
    if any(v is None for v in vec):
        return None
    scores = {k: v for k, v in as_dict(fp.get("scores")).items() if num(v) is not None}
    return FingerprintResponse(
        model_name=fp["model_name"], paradigm=str(fp.get("paradigm") or ""),
        dimension_names=[str(n) for n in names], vector=vec, scores=scores,
    )


def _all(store, run_id) -> tuple[list[FingerprintResponse], int]:
    raw = as_list(load_run(store, run_id).get("fingerprints"))
    parsed = [_parse(fp) for fp in raw]
    return [p for p in parsed if p is not None], sum(p is None for p in parsed)


@router.get("", response_model=FingerprintListResponse)
def list_fingerprints(
    run_id: str, store: ReadOnlyResultStore = Depends(get_result_store),
) -> FingerprintListResponse:
    """List behavioral fingerprints for all models in this run."""
    fps, skipped = _all(store, run_id)
    return FingerprintListResponse(fingerprints=fps, skipped=skipped)


# NOTE: declared before "/{model_name}" so "compare" is not captured as a model name.
@router.get("/compare/{model_a}/{model_b}", response_model=FingerprintComparisonResponse)
def compare_fingerprints(
    run_id: str, model_a: str, model_b: str,
    store: ReadOnlyResultStore = Depends(get_result_store),
) -> FingerprintComparisonResponse:
    """Compare two models' fingerprints: distances and per-dimension gaps."""
    fps = {fp.model_name: fp for fp in _all(store, run_id)[0]}
    for m in (model_a, model_b):
        if m not in fps:
            raise HTTPException(404, f"No fingerprint for '{m}'.")
    fa, fb = fps[model_a], fps[model_b]
    if fa.dimension_names != fb.dimension_names:
        raise HTTPException(400, "Fingerprints have mismatched dimensions.")

    va, vb = fa.vector, fb.vector
    euclidean = math.sqrt(sum((a - b) ** 2 for a, b in zip(va, vb)))
    na, nb = math.sqrt(sum(a * a for a in va)), math.sqrt(sum(b * b for b in vb))
    cosine = 1.0 - sum(a * b for a, b in zip(va, vb)) / (na * nb) if na > 0 and nb > 0 else 1.0
    gaps = {dim: vb[i] - va[i] for i, dim in enumerate(fa.dimension_names)}
    return FingerprintComparisonResponse(
        model_a=model_a, model_b=model_b,
        euclidean_distance=euclidean, cosine_distance=cosine, dimension_gaps=gaps,
    )


@router.get("/{model_name}", response_model=FingerprintResponse)
def get_fingerprint(
    run_id: str, model_name: str, store: ReadOnlyResultStore = Depends(get_result_store),
) -> FingerprintResponse:
    """Get the fingerprint for one specific model."""
    match = next((fp for fp in _all(store, run_id)[0] if fp.model_name == model_name), None)
    if match is None:
        raise HTTPException(404, f"No fingerprint for model '{model_name}' in run '{run_id}'.")
    return match

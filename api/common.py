"""Small helpers shared by the routers."""

from __future__ import annotations

import math
from typing import Any

from fastapi import HTTPException

from api.readonly_store import ReadOnlyResultStore
from faceeval.core.exceptions import RunNotFoundError


def load_run(store: ReadOnlyResultStore, run_id: str) -> dict[str, Any]:
    """Load one stored run or raise a 404."""
    try:
        return store.get(run_id)
    except RunNotFoundError:
        raise HTTPException(404, f"Run '{run_id}' not found.")


def num(value: Any) -> float | None:
    """Return a finite float, or None.  Missing values are never turned into 0."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    f = float(value)
    return f if math.isfinite(f) else None


def as_list(value: Any) -> list:
    return value if isinstance(value, list) else []


def as_dict(value: Any) -> dict:
    return value if isinstance(value, dict) else {}

from __future__ import annotations

import re
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse

from api.dependencies import Settings, get_settings

router = APIRouter(prefix="/api/runs/{run_id}/reports", tags=["reports"])

# HTML is intentionally not served: a report file is data, not a page to execute.
_MEDIA_TYPES = {
    ".csv": "text/csv",
    ".json": "application/json",
    ".pdf": "application/pdf",
}
_RUN_ID = re.compile(r"^[A-Za-z0-9_-]{1,64}$")
_FILENAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,200}$")   # no separators or glob characters


def _check_run_id(run_id: str) -> None:
    if not _RUN_ID.match(run_id):
        raise HTTPException(400, "Invalid run id.")


def _run_files(base: Path, run_id: str) -> list[Path]:
    """Report files whose name contains the run id, confined to ``base``."""
    root = base.resolve()
    out = []
    for p in root.rglob("*"):
        if run_id not in p.name or p.suffix.lower() not in _MEDIA_TYPES or not p.is_file():
            continue
        try:
            p.resolve().relative_to(root)
        except ValueError:
            continue   # symlink pointing outside the reports directory
        out.append(p)
    return sorted(out)


@router.get("/list")
def list_report_files(run_id: str, settings: Settings = Depends(get_settings)) -> dict:
    """List report files (csv/json/pdf) whose names contain this run id."""
    _check_run_id(run_id)
    base = Path(settings.reports_dir)
    if not base.is_dir():
        return {"files": []}
    return {
        "files": [
            {
                "filename": p.name,
                "format": p.suffix.lstrip(".").lower(),
                "download_url": f"/api/runs/{run_id}/reports/file/{p.name}",
                "size_bytes": p.stat().st_size,
            }
            for p in _run_files(base, run_id)
        ]
    }


@router.get("/file/{filename}")
def download_report_file(
    run_id: str, filename: str, settings: Settings = Depends(get_settings),
) -> FileResponse:
    """Download one report file; it must belong to this run (name contains the run id)."""
    _check_run_id(run_id)
    if not _FILENAME.match(filename) or run_id not in filename:
        raise HTTPException(400, "Invalid report file name.")
    media_type = _MEDIA_TYPES.get(Path(filename).suffix.lower())
    if media_type is None:
        raise HTTPException(415, "Only .csv, .json and .pdf reports are served.")

    base = Path(settings.reports_dir)
    match = next((p for p in _run_files(base, run_id) if p.name == filename), None) \
        if base.is_dir() else None
    if match is None:
        raise HTTPException(404, f"Report file '{filename}' not found.")
    return FileResponse(str(match), media_type=media_type, filename=match.name)

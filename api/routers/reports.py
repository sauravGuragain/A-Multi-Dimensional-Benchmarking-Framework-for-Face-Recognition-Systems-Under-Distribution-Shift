from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse

from api.dependencies import get_settings, Settings

router = APIRouter(prefix="/api/runs/{run_id}/reports", tags=["reports"])

_MEDIA_TYPES = {
    ".csv": "text/csv",
    ".json": "application/json",
    ".pdf": "application/pdf",
    ".html": "text/html",
}


@router.get("/list")
def list_report_files(
    run_id: str,
    settings: Settings = Depends(get_settings),
) -> dict:
    """List all available report files for a run."""
    base = Path(settings.reports_dir)
    if not base.exists():
        return {"files": []}

    matches = [
        p for p in base.rglob(f"*{run_id}*")
        if p.is_file() and p.suffix in _MEDIA_TYPES
    ]
    return {
        "files": [
            {
                "filename": p.name,
                "format": p.suffix.lstrip("."),
                "download_url": f"/api/runs/{run_id}/reports/file/{p.name}",
                "size_bytes": p.stat().st_size,
            }
            for p in sorted(matches)
        ]
    }


@router.get("/file/{filename}")
def download_report_file(
    run_id: str,
    filename: str,
    settings: Settings = Depends(get_settings),
) -> FileResponse:
    """Download a specific report file by filename."""
    base = Path(settings.reports_dir)
    # Search recursively for the file (reports may be nested by run_id)
    matches = list(base.rglob(filename))
    if not matches:
        raise HTTPException(404, f"Report file '{filename}' not found.")

    path = matches[0]
    # Path traversal guard
    try:
        path.resolve().relative_to(base.resolve())
    except ValueError:
        raise HTTPException(400, "Invalid file path.")

    media_type = _MEDIA_TYPES.get(path.suffix, "application/octet-stream")
    return FileResponse(str(path), media_type=media_type, filename=path.name)

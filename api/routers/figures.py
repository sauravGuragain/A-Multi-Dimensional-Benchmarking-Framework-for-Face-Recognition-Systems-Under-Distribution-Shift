from __future__ import annotations

from pathlib import Path
from urllib.parse import quote

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse

from api.dependencies import get_settings, Settings
from api.schemas import FigureInfo, FigureListResponse

router = APIRouter(prefix="/api/figures", tags=["figures"])


_FIGURE_TYPES = {".png": "image/png", ".svg": "image/svg+xml"}


def _safe_join(base: Path, rel: str) -> Path | None:
    """Resolve ``rel`` under ``base``; None if it would escape (``..``, absolute, symlink)."""
    try:
        full = (base / rel).resolve()
        full.relative_to(base.resolve())
    except (ValueError, OSError):
        return None
    return full


@router.get("", response_model=FigureListResponse)
def list_figures(
    subfolder: str = "",
    settings: Settings = Depends(get_settings),
) -> FigureListResponse:
    """
    List available PNG figures, optionally scoped to a subfolder
    (e.g. a specific run_id).  A missing folder gives an empty list.
    """
    root = Path(settings.figures_dir)
    base = _safe_join(root, subfolder) if subfolder else root.resolve()
    if base is None:
        raise HTTPException(400, "Invalid subfolder.")
    if not base.is_dir():
        return FigureListResponse(figures=[])

    root_resolved = root.resolve()
    figures: list[FigureInfo] = []

    for png_path in sorted(base.rglob("*.png")):
        if not png_path.is_file() or _safe_join(root_resolved, str(png_path.relative_to(root_resolved))) is None:
            continue
        figure_id = png_path.stem   # not unique across folders; the URL is the unique key

        svg_path = png_path.with_suffix(".svg")
        rel = png_path.relative_to(root_resolved)
        rel_svg = svg_path.relative_to(root_resolved) if svg_path.is_file() else None

        figures.append(FigureInfo(
            figure_id=figure_id,
            figure_type=_infer_type(figure_id),
            title=figure_id.replace("_", " ").title(),
            caption="",
            png_url=f"/api/figures/file/{quote(rel.as_posix())}",
            svg_url=f"/api/figures/file/{quote(rel_svg.as_posix())}" if rel_svg else None,
        ))

    return FigureListResponse(figures=figures)


@router.get("/file/{file_path:path}")
def get_figure_file(
    file_path: str,
    settings: Settings = Depends(get_settings),
) -> FileResponse:
    """Serve one PNG or SVG figure by path relative to the figures directory."""
    full_path = _safe_join(Path(settings.figures_dir), file_path)
    if full_path is None:
        raise HTTPException(400, "Invalid file path.")
    media_type = _FIGURE_TYPES.get(full_path.suffix.lower())
    if media_type is None:
        raise HTTPException(415, "Only .png and .svg figures are served.")
    if not full_path.is_file():
        raise HTTPException(404, f"Figure file not found: '{file_path}'")
    return FileResponse(str(full_path), media_type=media_type)


def _infer_type(figure_id: str) -> str:
    """Infer figure type from its filename for UI grouping."""
    mapping = {
        "roc": "roc_curve", "adc": "adc_curve", "reliability": "calibration",
        "ece": "calibration", "radar": "radar_chart", "heatmap": "heatmap",
        "confusion": "confusion_matrix", "deploy": "deployment_chart",
        "sensitivity": "sensitivity_chart", "cross_scenario": "ranking_chart",
        "distance": "distance_heatmap", "overlap": "overlap_heatmap",
        "subgroup": "fairness_heatmap", "component": "comparison_chart",
    }
    for key, val in mapping.items():
        if key in figure_id:
            return val
    return "other"

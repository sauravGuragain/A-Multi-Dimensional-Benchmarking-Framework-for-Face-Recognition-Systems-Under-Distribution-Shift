from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse

from api.dependencies import get_settings, Settings
from api.schemas import FigureInfo, FigureListResponse

router = APIRouter(prefix="/api/figures", tags=["figures"])


@router.get("", response_model=FigureListResponse)
def list_figures(
    subfolder: str = "",
    settings: Settings = Depends(get_settings),
) -> FigureListResponse:
    """
    List all available figures, optionally scoped to a subfolder
    (e.g. a specific run_id).
    """
    base = Path(settings.figures_dir) / subfolder if subfolder else Path(settings.figures_dir)
    if not base.exists():
        return FigureListResponse(figures=[])

    figures: list[FigureInfo] = []
    seen_ids: set[str] = set()

    for png_path in sorted(base.rglob("*.png")):
        figure_id = png_path.stem
        if figure_id in seen_ids:
            continue
        seen_ids.add(figure_id)

        svg_path = png_path.with_suffix(".svg")
        rel = png_path.relative_to(Path(settings.figures_dir))
        rel_svg = svg_path.relative_to(Path(settings.figures_dir)) if svg_path.exists() else None

        figures.append(FigureInfo(
            figure_id=figure_id,
            figure_type=_infer_type(figure_id),
            title=figure_id.replace("_", " ").title(),
            caption="",
            png_url=f"/api/figures/file/{rel.as_posix()}",
            svg_url=f"/api/figures/file/{rel_svg.as_posix()}" if rel_svg else None,
        ))

    return FigureListResponse(figures=figures)


@router.get("/file/{file_path:path}")
def get_figure_file(
    file_path: str,
    settings: Settings = Depends(get_settings),
) -> FileResponse:
    """Serve a specific figure file by relative path."""
    full_path = Path(settings.figures_dir) / file_path
    # Prevent path traversal outside the figures directory
    try:
        full_path = full_path.resolve()
        base = Path(settings.figures_dir).resolve()
        full_path.relative_to(base)
    except ValueError:
        raise HTTPException(400, "Invalid file path.")

    if not full_path.exists() or not full_path.is_file():
        raise HTTPException(404, f"Figure file not found: '{file_path}'")

    media_type = "image/svg+xml" if full_path.suffix == ".svg" else "image/png"
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

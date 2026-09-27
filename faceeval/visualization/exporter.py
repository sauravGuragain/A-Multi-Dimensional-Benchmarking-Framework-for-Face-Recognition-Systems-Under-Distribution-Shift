"""
faceeval.visualization.exporter
=================================
Saves matplotlib figures as SVG and PNG with run metadata embedded.

Design decisions
----------------
* Every export call returns paths to both the SVG and PNG files so the
  caller never needs to call ``fig.savefig`` directly.
* Metadata (model names, perturbation types, run_id) is embedded in the
  SVG ``<metadata>`` block for traceability.
* Figure IDs are deterministic from figure_type + model + perturbation
  so re-exporting overwrites rather than duplicating.
* All output files go under ``<report_dir>/figures/``.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)


class FigureExporter:
    """
    Handles saving matplotlib figures to disk in SVG and PNG formats.

    Parameters
    ----------
    output_dir:
        Root directory for exported figures.
    export_svg:
        Whether to save SVG files.
    export_png:
        Whether to save PNG files.
    dpi:
        PNG resolution.
    """

    def __init__(
        self,
        output_dir: str = "reports/figures",
        export_svg: bool = True,
        export_png: bool = True,
        dpi: int = 300,
    ) -> None:
        self._root = Path(output_dir)
        self._root.mkdir(parents=True, exist_ok=True)
        self._export_svg = export_svg
        self._export_png = export_png
        self._dpi = dpi

    def save(
        self,
        fig: plt.Figure,
        figure_id: str,
        subfolder: str = "",
        close_after: bool = True,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, str]:
        """
        Export a figure to SVG and/or PNG.

        Parameters
        ----------
        fig:
            The matplotlib Figure to save.
        figure_id:
            Filename stem (e.g. ``"roc_eigenfaces_gaussian_blur"``).
        subfolder:
            Optional subdirectory under ``output_dir``.
        close_after:
            Whether to call ``plt.close(fig)`` after saving.
        metadata:
            Optional dict embedded in SVG metadata comments.

        Returns
        -------
        dict with keys ``"svg"`` and ``"png"`` containing saved file paths
        (empty string if that format was not exported).
        """
        dest = self._root / subfolder if subfolder else self._root
        dest.mkdir(parents=True, exist_ok=True)

        paths: dict[str, str] = {"svg": "", "png": ""}

        if self._export_svg:
            svg_path = dest / f"{figure_id}.svg"
            try:
                fig.savefig(
                    str(svg_path),
                    format="svg",
                    bbox_inches="tight",
                    pad_inches=0.05,
                    metadata={"Creator": "FaceEval-X", **(metadata or {})},
                )
                paths["svg"] = str(svg_path)
                logger.debug("Saved SVG: %s", svg_path)
            except Exception as exc:
                logger.warning("SVG export failed for '%s': %s", figure_id, exc)

        if self._export_png:
            png_path = dest / f"{figure_id}.png"
            try:
                fig.savefig(
                    str(png_path),
                    format="png",
                    dpi=self._dpi,
                    bbox_inches="tight",
                    pad_inches=0.05,
                )
                paths["png"] = str(png_path)
                logger.debug("Saved PNG: %s", png_path)
            except Exception as exc:
                logger.warning("PNG export failed for '%s': %s", figure_id, exc)

        if close_after:
            plt.close(fig)

        return paths

    def figure_path(self, figure_id: str, fmt: str = "png", subfolder: str = "") -> str:
        """Return the expected path for a figure without saving."""
        dest = self._root / subfolder if subfolder else self._root
        return str(dest / f"{figure_id}.{fmt}")

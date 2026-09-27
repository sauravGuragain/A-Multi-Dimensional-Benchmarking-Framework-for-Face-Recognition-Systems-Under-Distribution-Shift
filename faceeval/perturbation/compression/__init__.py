"""
faceeval.perturbation.compression
====================================
Compression and resolution perturbations: JPEG artefacts and downscale-upscale.

Severity → parameter mappings
-------------------------------
JPEG compression    : severity → quality ∈ [95, 5]
    quality=95 → nearly lossless; quality=5 → heavy block artefacts.
    Inverted: higher severity = lower quality.

Resolution degradation : severity → downscale_factor ∈ [1.0, 8.0]
    severity=0 → factor=1.0 (no change)
    severity=1 → factor=8.0 (image shrunk 8× then upscaled back)
    Simulates low-resolution surveillance camera imagery.
"""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from faceeval.core.registry import register_perturbation
from faceeval.core.types import PerturbationCategory, PerturbationType
from faceeval.perturbation.base import BasePerturbation


# ---------------------------------------------------------------------------
# JPEG compression
# ---------------------------------------------------------------------------

class JPEGCompression(BasePerturbation):
    """
    JPEG encode-decode cycle simulating lossy compression artefacts.

    Higher severity → lower JPEG quality → more block artefacts.
    Uses OpenCV's JPEG codec via imencode/imdecode round-trip.
    """

    PERTURBATION_TYPE = PerturbationType.JPEG_COMPRESSION
    CATEGORY          = PerturbationCategory.COMPRESSION

    _MIN_QUALITY = 5
    _MAX_QUALITY = 95

    def _severity_to_params(self, severity: float) -> dict[str, Any]:
        # Inverted: severity=0 → quality=95, severity=1 → quality=5
        quality = int(self._MAX_QUALITY - severity * (self._MAX_QUALITY - self._MIN_QUALITY))
        return {"jpeg_quality": quality}

    def _apply(self, image: np.ndarray, severity: float) -> np.ndarray:
        quality = self._severity_to_params(severity)["jpeg_quality"]

        # JPEG codec works only on uint8
        working = self._to_uint8(image)
        encode_params = [cv2.IMWRITE_JPEG_QUALITY, quality]

        success, buf = cv2.imencode(".jpg", working, encode_params)
        if not success:
            return image.copy()

        decoded = cv2.imdecode(buf, cv2.IMREAD_UNCHANGED)
        if decoded is None:
            return image.copy()

        # Restore shape (grayscale images may lose channel dim after decode)
        if len(image.shape) == 2 and len(decoded.shape) == 3:
            decoded = cv2.cvtColor(decoded, cv2.COLOR_BGR2GRAY)

        return self._restore_dtype(decoded, image)


# ---------------------------------------------------------------------------
# Resolution degradation
# ---------------------------------------------------------------------------

class ResolutionDegradation(BasePerturbation):
    """
    Downscale then upscale — simulates low-resolution capture.

    The image is shrunk by ``downscale_factor`` using area interpolation
    (best for downscaling) then enlarged back to original dimensions using
    bilinear interpolation (introduces visible pixelation / blurring).
    """

    PERTURBATION_TYPE = PerturbationType.RESOLUTION_DEGRADATION
    CATEGORY          = PerturbationCategory.RESOLUTION

    _MIN_FACTOR = 1.0
    _MAX_FACTOR = 8.0

    def _severity_to_params(self, severity: float) -> dict[str, Any]:
        factor = self._MIN_FACTOR + severity * (self._MAX_FACTOR - self._MIN_FACTOR)
        return {"downscale_factor": round(factor, 3)}

    def _apply(self, image: np.ndarray, severity: float) -> np.ndarray:
        factor = self._severity_to_params(severity)["downscale_factor"]
        h, w = image.shape[:2]

        small_h = max(1, int(h / factor))
        small_w = max(1, int(w / factor))

        small = cv2.resize(image, (small_w, small_h), interpolation=cv2.INTER_AREA)
        upscaled = cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)
        return upscaled


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

@register_perturbation(
    key=PerturbationType.JPEG_COMPRESSION.value,
    category=PerturbationCategory.COMPRESSION.value,
    description="JPEG encode-decode (quality 95→5 as severity increases)",
    severity_param="jpeg_quality",
)
def _jpeg_factory(**_: Any) -> JPEGCompression:
    return JPEGCompression()


@register_perturbation(
    key=PerturbationType.RESOLUTION_DEGRADATION.value,
    category=PerturbationCategory.RESOLUTION.value,
    description="Downscale-upscale resolution degradation (factor 1×–8×)",
    severity_param="downscale_factor",
)
def _res_factory(**_: Any) -> ResolutionDegradation:
    return ResolutionDegradation()

"""
faceeval.perturbation.geometric
=================================
Geometric perturbations: in-plane rotation, scaling, random crop-and-resize.

These model pose variability — the second major degradation axis for face
recognition after illumination.

Severity → parameter mappings
-------------------------------
Rotation : severity → angle ∈ [0°, 45°]
    Images are rotated about the image centre.  Background fill is
    reflect-border to avoid introducing black-corner artefacts.

Scaling  : severity → scale ∈ [0.5, 1.0]
    Zoom-out: image is scaled down and padded to original size.
    severity=0 → scale=1.0 (identity); severity=1 → scale=0.5 (50% zoom out).

Cropping : severity → crop_fraction ∈ [0.0, 0.40]
    A random-seed-fixed central crop is resized back to original dimensions.
    Simulates camera zoom-in or off-centre face alignment.
"""

from __future__ import annotations

import math
from typing import Any

import cv2
import numpy as np

from faceeval.core.registry import register_perturbation
from faceeval.core.types import PerturbationCategory, PerturbationType
from faceeval.perturbation.base import BasePerturbation

_SEED = 7   # fixed seed for all geometric RNG operations


# ---------------------------------------------------------------------------
# Rotation
# ---------------------------------------------------------------------------

class RotationPerturbation(BasePerturbation):
    """In-plane rotation about image centre."""

    PERTURBATION_TYPE = PerturbationType.ROTATION
    CATEGORY          = PerturbationCategory.GEOMETRIC

    _MAX_ANGLE = 45.0

    def _severity_to_params(self, severity: float) -> dict[str, Any]:
        angle = severity * self._MAX_ANGLE
        return {"angle_deg": round(angle, 3)}

    def _apply(self, image: np.ndarray, severity: float) -> np.ndarray:
        angle = self._severity_to_params(severity)["angle_deg"]
        h, w = image.shape[:2]
        centre = (w / 2.0, h / 2.0)
        M = cv2.getRotationMatrix2D(centre, angle, scale=1.0)
        return cv2.warpAffine(
            image, M, (w, h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT_101,
        )


# ---------------------------------------------------------------------------
# Scaling (zoom-out)
# ---------------------------------------------------------------------------

class ScalingPerturbation(BasePerturbation):
    """
    Zoom-out scaling: shrinks the face and pads with reflected border.

    Models situations where the face occupies a smaller fraction of the frame
    (e.g. subject further from camera, surveillance scenario).
    """

    PERTURBATION_TYPE = PerturbationType.SCALING
    CATEGORY          = PerturbationCategory.GEOMETRIC

    _MIN_SCALE = 0.50
    _MAX_SCALE = 1.00

    def _severity_to_params(self, severity: float) -> dict[str, Any]:
        scale = self._MAX_SCALE - severity * (self._MAX_SCALE - self._MIN_SCALE)
        return {"scale_factor": round(scale, 4)}

    def _apply(self, image: np.ndarray, severity: float) -> np.ndarray:
        scale = self._severity_to_params(severity)["scale_factor"]
        h, w = image.shape[:2]

        new_h, new_w = max(1, int(h * scale)), max(1, int(w * scale))
        shrunk = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        # Place centred in a canvas of original size
        canvas = image.copy()
        pad_y = (h - new_h) // 2
        pad_x = (w - new_w) // 2

        if image.dtype == np.float32:
            canvas[:] = image.mean()
        else:
            canvas[:] = 128

        canvas[pad_y: pad_y + new_h, pad_x: pad_x + new_w] = shrunk
        return canvas


# ---------------------------------------------------------------------------
# Cropping
# ---------------------------------------------------------------------------

class CroppingPerturbation(BasePerturbation):
    """
    Random-crop followed by resize back to original dimensions.

    Simulates off-centre alignment or zoom-in.  The crop origin is fixed by
    a deterministic seed so the same severity always produces the same crop
    offset (reproducibility requirement).
    """

    PERTURBATION_TYPE = PerturbationType.CROPPING
    CATEGORY          = PerturbationCategory.GEOMETRIC

    _MAX_CROP = 0.40    # maximum fraction cropped from each edge

    def _severity_to_params(self, severity: float) -> dict[str, Any]:
        crop_frac = severity * self._MAX_CROP
        return {"crop_fraction": round(crop_frac, 4)}

    def _apply(self, image: np.ndarray, severity: float) -> np.ndarray:
        crop_frac = self._severity_to_params(severity)["crop_fraction"]
        h, w = image.shape[:2]

        rng = np.random.default_rng(_SEED)
        max_offset_y = int(h * crop_frac)
        max_offset_x = int(w * crop_frac)

        y0 = rng.integers(0, max(1, max_offset_y))
        x0 = rng.integers(0, max(1, max_offset_x))
        y1 = h - rng.integers(0, max(1, max_offset_y))
        x1 = w - rng.integers(0, max(1, max_offset_x))

        y1, x1 = max(y1, y0 + 1), max(x1, x0 + 1)
        crop = image[y0:y1, x0:x1]
        return cv2.resize(crop, (w, h), interpolation=cv2.INTER_LINEAR)


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

@register_perturbation(
    key=PerturbationType.ROTATION.value,
    category=PerturbationCategory.GEOMETRIC.value,
    description="In-plane rotation 0°–45°",
    severity_param="angle_deg",
)
def _rot_factory(**_: Any) -> RotationPerturbation:
    return RotationPerturbation()


@register_perturbation(
    key=PerturbationType.SCALING.value,
    category=PerturbationCategory.GEOMETRIC.value,
    description="Zoom-out scaling (scale 1.0→0.5)",
    severity_param="scale_factor",
)
def _scale_factory(**_: Any) -> ScalingPerturbation:
    return ScalingPerturbation()


@register_perturbation(
    key=PerturbationType.CROPPING.value,
    category=PerturbationCategory.GEOMETRIC.value,
    description="Central crop-and-resize (crop fraction 0–0.40)",
    severity_param="crop_fraction",
)
def _crop_factory(**_: Any) -> CroppingPerturbation:
    return CroppingPerturbation()

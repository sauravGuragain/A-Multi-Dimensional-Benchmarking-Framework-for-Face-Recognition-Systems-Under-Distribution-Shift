"""
faceeval.perturbation.occlusion
=================================
Occlusion perturbations: random block occlusion, face mask, sunglasses.

These model real-world partial face coverage — a dominant failure source
for recognition systems in the wild.

Severity → parameter mappings
-------------------------------
Random occlusion : severity → occluded_area_fraction ∈ [0.0, 0.40]
    A grey filled rectangle covers this fraction of the total image area.
    Position is fixed (upper-left of the face region) for reproducibility.

Face mask        : severity → mask_opacity ∈ [0.0, 1.0]
    A synthetic lower-face mask (covering nose+mouth region) is blended
    with configurable opacity.  Opacity=1 gives a fully opaque mask.

Sunglasses       : severity → glasses_opacity ∈ [0.0, 1.0]
    A synthetic dark rectangle over the eye region simulates sunglasses.
"""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from faceeval.core.registry import register_perturbation
from faceeval.core.types import PerturbationCategory, PerturbationType
from faceeval.perturbation.base import BasePerturbation

_SEED = 13


# ---------------------------------------------------------------------------
# Random occlusion (generic rectangle)
# ---------------------------------------------------------------------------

class RandomOcclusion(BasePerturbation):
    """
    Grey filled rectangle occlusion at a fixed position.

    The rectangle is placed at the upper-left quadrant of the face to
    consistently cover the forehead and one eye — the most diagnostically
    informative region in identity-based recognition.
    """

    PERTURBATION_TYPE = PerturbationType.RANDOM_OCCLUSION
    CATEGORY          = PerturbationCategory.OCCLUSION

    _MAX_AREA = 0.40

    def _severity_to_params(self, severity: float) -> dict[str, Any]:
        return {"occluded_area_fraction": round(severity * self._MAX_AREA, 4)}

    def _apply(self, image: np.ndarray, severity: float) -> np.ndarray:
        frac = self._severity_to_params(severity)["occluded_area_fraction"]
        h, w = image.shape[:2]
        area = h * w * frac
        rect_h = int((area / (w / 2)) ** 0.5 * (h / w) ** 0.5 * (w / 2) ** 0.5)
        rect_w = int(area / max(rect_h, 1))
        rect_h = min(rect_h, h)
        rect_w = min(rect_w, w)

        out = image.copy()
        fill = 128 if image.dtype == np.uint8 else 0.5
        out[:rect_h, :rect_w] = fill
        return out


# ---------------------------------------------------------------------------
# Face mask
# ---------------------------------------------------------------------------

class FaceMask(BasePerturbation):
    """
    Synthetic lower-face mask (covers nose and mouth, ~35%–75% of face height).

    Severity controls mask opacity: severity=1.0 → fully opaque light-blue
    surgical-mask colour; severity=0.5 → semi-transparent.
    """

    PERTURBATION_TYPE = PerturbationType.FACE_MASK
    CATEGORY          = PerturbationCategory.OCCLUSION

    # Surgical mask colour in BGR (approximate light blue)
    _MASK_COLOUR_BGR = np.array([200, 180, 140], dtype=np.uint8)
    # Face mask covers rows from 50% to 95% of image height
    _Y_START_FRAC = 0.50
    _Y_END_FRAC   = 0.95

    def _severity_to_params(self, severity: float) -> dict[str, Any]:
        return {"mask_opacity": round(severity, 4)}

    def _apply(self, image: np.ndarray, severity: float) -> np.ndarray:
        opacity = self._severity_to_params(severity)["mask_opacity"]
        h, w = image.shape[:2]

        y0 = int(h * self._Y_START_FRAC)
        y1 = min(h, int(h * self._Y_END_FRAC))

        out = image.copy()
        if image.dtype == np.uint8:
            colour = self._MASK_COLOUR_BGR
            if len(image.shape) == 2:
                colour_val = int(colour.mean())
                mask_region = np.full((y1 - y0, w), colour_val, dtype=np.uint8)
            else:
                mask_region = np.tile(colour, (y1 - y0, w, 1)).reshape(y1 - y0, w, 3)

            out[y0:y1] = cv2.addWeighted(
                image[y0:y1], 1.0 - opacity,
                mask_region.astype(np.uint8), opacity, 0,
            )
        else:
            colour_f = self._MASK_COLOUR_BGR.astype(np.float32) / 255.0
            if len(image.shape) == 2:
                mask_region_f = np.full((y1 - y0, w), colour_f.mean(), dtype=np.float32)
            else:
                mask_region_f = np.tile(colour_f, (y1 - y0, w, 1)).reshape(y1 - y0, w, 3)
            out[y0:y1] = (1.0 - opacity) * image[y0:y1] + opacity * mask_region_f
        return out


# ---------------------------------------------------------------------------
# Sunglasses
# ---------------------------------------------------------------------------

class SunglassesPerturbation(BasePerturbation):
    """
    Synthetic dark rectangular glasses bar over the eye region.

    Covers the horizontal band from ~30% to 50% of face height and the
    full width minus 10% margins (to simulate typical frame extent).
    """

    PERTURBATION_TYPE = PerturbationType.SUNGLASSES
    CATEGORY          = PerturbationCategory.OCCLUSION

    _Y_START_FRAC = 0.28
    _Y_END_FRAC   = 0.52
    _X_MARGIN     = 0.08

    def _severity_to_params(self, severity: float) -> dict[str, Any]:
        return {"glasses_opacity": round(severity, 4)}

    def _apply(self, image: np.ndarray, severity: float) -> np.ndarray:
        opacity = self._severity_to_params(severity)["glasses_opacity"]
        h, w = image.shape[:2]

        y0 = int(h * self._Y_START_FRAC)
        y1 = int(h * self._Y_END_FRAC)
        x0 = int(w * self._X_MARGIN)
        x1 = int(w * (1.0 - self._X_MARGIN))

        out = image.copy()
        # Dark brownish frame colour
        if image.dtype == np.uint8:
            lens_colour = np.array([20, 15, 10], dtype=np.uint8)
            if len(image.shape) == 2:
                region = np.full((y1 - y0, x1 - x0), int(lens_colour.mean()), dtype=np.uint8)
            else:
                region = np.tile(lens_colour, (y1 - y0, x1 - x0, 1)).reshape(y1 - y0, x1 - x0, 3)
            out[y0:y1, x0:x1] = cv2.addWeighted(
                image[y0:y1, x0:x1], 1.0 - opacity,
                region, opacity, 0,
            )
        else:
            lens_f = np.array([0.08, 0.06, 0.04], dtype=np.float32)
            if len(image.shape) == 2:
                region_f = np.full((y1 - y0, x1 - x0), float(lens_f.mean()), dtype=np.float32)
            else:
                region_f = np.tile(lens_f, (y1 - y0, x1 - x0, 1)).reshape(y1 - y0, x1 - x0, 3)
            out[y0:y1, x0:x1] = (
                (1.0 - opacity) * image[y0:y1, x0:x1] + opacity * region_f
            )
        return out


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

@register_perturbation(
    key=PerturbationType.RANDOM_OCCLUSION.value,
    category=PerturbationCategory.OCCLUSION.value,
    description="Grey rectangle occlusion (area fraction 0–0.40)",
    severity_param="occluded_area_fraction",
)
def _ro_factory(**_: Any) -> RandomOcclusion:
    return RandomOcclusion()


@register_perturbation(
    key=PerturbationType.FACE_MASK.value,
    category=PerturbationCategory.OCCLUSION.value,
    description="Lower-face surgical mask (opacity 0–1.0)",
    severity_param="mask_opacity",
)
def _fm_factory(**_: Any) -> FaceMask:
    return FaceMask()


@register_perturbation(
    key=PerturbationType.SUNGLASSES.value,
    category=PerturbationCategory.OCCLUSION.value,
    description="Dark sunglasses bar over eye region (opacity 0–1.0)",
    severity_param="glasses_opacity",
)
def _sg_factory(**_: Any) -> SunglassesPerturbation:
    return SunglassesPerturbation()

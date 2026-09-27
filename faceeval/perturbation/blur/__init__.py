"""
faceeval.perturbation.blur
===========================
Blur perturbations: Gaussian blur and motion blur.

Severity → parameter mappings
-------------------------------
Gaussian blur: severity [0,1] → sigma [0, 8.0]
    sigma=0   → identity; sigma=8 → very strong blur
    Kernel size is always odd: 2*ceil(3*sigma)+1, capped at image min-dim.

Motion blur: severity [0,1] → kernel_length [0, 40] pixels
    length=0  → identity
    length=40 → strongly motion-blurred (unrecognisable at high severity)
    Direction angle is fixed at 45° to produce consistent cross-image comparisons
    (random angles would add variance that doesn't belong in the severity axis).
"""

from __future__ import annotations

import math
from typing import Any

import cv2
import numpy as np

from faceeval.core.registry import register_perturbation
from faceeval.core.types import PerturbationCategory, PerturbationType
from faceeval.perturbation.base import BasePerturbation

# ---------------------------------------------------------------------------
# Gaussian blur
# ---------------------------------------------------------------------------

_MAX_SIGMA = 8.0    # at severity=1.0


class GaussianBlur(BasePerturbation):
    """Isotropic Gaussian blur — the canonical spatial-frequency perturbation."""

    PERTURBATION_TYPE = PerturbationType.GAUSSIAN_BLUR
    CATEGORY          = PerturbationCategory.BLUR

    def _severity_to_params(self, severity: float) -> dict[str, Any]:
        sigma = severity * _MAX_SIGMA
        ksize = max(3, 2 * math.ceil(3 * sigma) + 1) if sigma > 0 else 1
        if ksize % 2 == 0:
            ksize += 1
        return {"sigma": round(sigma, 4), "kernel_size": ksize}

    def _apply(self, image: np.ndarray, severity: float) -> np.ndarray:
        params = self._severity_to_params(severity)
        sigma  = params["sigma"]
        ksize  = params["kernel_size"]
        # Cap kernel to image size
        h, w = image.shape[:2]
        ksize = min(ksize, min(h, w) if min(h, w) % 2 == 1 else min(h, w) - 1)
        return cv2.GaussianBlur(image, (ksize, ksize), sigmaX=sigma, sigmaY=sigma)


# ---------------------------------------------------------------------------
# Motion blur
# ---------------------------------------------------------------------------

_MAX_LENGTH = 40    # pixels at severity=1.0
_ANGLE_DEG  = 45.0  # fixed diagonal direction


class MotionBlur(BasePerturbation):
    """Linear motion blur simulating camera shake or subject motion."""

    PERTURBATION_TYPE = PerturbationType.MOTION_BLUR
    CATEGORY          = PerturbationCategory.BLUR

    def _severity_to_params(self, severity: float) -> dict[str, Any]:
        length = max(1, round(severity * _MAX_LENGTH))
        return {"kernel_length": length, "angle_deg": _ANGLE_DEG}

    def _apply(self, image: np.ndarray, severity: float) -> np.ndarray:
        params = self._severity_to_params(severity)
        length = params["kernel_length"]
        angle  = math.radians(params["angle_deg"])

        # Build a line kernel
        kernel = np.zeros((length, length), dtype=np.float32)
        cx, cy = length // 2, length // 2
        cos_a, sin_a = math.cos(angle), math.sin(angle)
        for i in range(length):
            t = i - length // 2
            x = int(round(cx + t * cos_a))
            y = int(round(cy + t * sin_a))
            if 0 <= x < length and 0 <= y < length:
                kernel[y, x] = 1.0
        k_sum = kernel.sum()
        if k_sum > 0:
            kernel /= k_sum

        if len(image.shape) == 2:
            blurred = cv2.filter2D(image, -1, kernel)
        else:
            blurred = cv2.filter2D(image, -1, kernel)
        return self._restore_dtype(blurred, image)


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

@register_perturbation(
    key=PerturbationType.GAUSSIAN_BLUR.value,
    category=PerturbationCategory.BLUR.value,
    description="Isotropic Gaussian blur (sigma 0–8)",
    severity_param="sigma",
)
def _gaussian_blur_factory(**_: Any) -> GaussianBlur:
    return GaussianBlur()


@register_perturbation(
    key=PerturbationType.MOTION_BLUR.value,
    category=PerturbationCategory.BLUR.value,
    description="Linear motion blur (length 0–40 px, 45° angle)",
    severity_param="kernel_length",
)
def _motion_blur_factory(**_: Any) -> MotionBlur:
    return MotionBlur()

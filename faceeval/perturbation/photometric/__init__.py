"""
faceeval.perturbation.photometric
===================================
Photometric perturbations: brightness shift, contrast scaling, gamma correction.

These model real-world illumination variability — the dominant source of
performance degradation in classical face recognition systems and a key
axis for comparing traditional vs. DL robustness.

Severity → parameter mappings
-------------------------------
Brightness : severity → additive_shift ∈ [-0.40, +0.40]
    Negative shifts (darkening) are applied for severity < 0.5;
    positive shifts (brightening) for severity > 0.5.
    severity=0.5 → shift=0 (identity).
    Full range used: split around 0.5 so the ADC curve captures both ends.

Contrast   : severity → scale_factor ∈ [0.3, 3.0]
    severity=0   → scale=0.3  (washed out)
    severity=0.5 → scale=1.0  (identity; implicit in ADC mid-point)
    severity=1.0 → scale=3.0  (high contrast)

Gamma      : severity → gamma ∈ [0.25, 4.0]
    severity=0.5 → gamma=1.0 (identity)
    severity<0.5 → gamma<1.0 (brightening)
    severity>0.5 → gamma>1.0 (darkening)
"""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from faceeval.core.registry import register_perturbation
from faceeval.core.types import PerturbationCategory, PerturbationType
from faceeval.perturbation.base import BasePerturbation


# ---------------------------------------------------------------------------
# Brightness
# ---------------------------------------------------------------------------

class BrightnessPerturbation(BasePerturbation):
    """Additive brightness shift modelling over/under-exposure."""

    PERTURBATION_TYPE = PerturbationType.BRIGHTNESS
    CATEGORY          = PerturbationCategory.PHOTOMETRIC

    _MAX_SHIFT = 0.40

    def _severity_to_params(self, severity: float) -> dict[str, Any]:
        # Centre at 0.5 so severity=0.5 means no shift
        shift = (severity - 0.5) * 2.0 * self._MAX_SHIFT
        return {"additive_shift": round(shift, 5)}

    def _apply(self, image: np.ndarray, severity: float) -> np.ndarray:
        shift = self._severity_to_params(severity)["additive_shift"]
        f = self._to_float32(image)
        return self._restore_dtype(f + shift, image)


# ---------------------------------------------------------------------------
# Contrast
# ---------------------------------------------------------------------------

class ContrastPerturbation(BasePerturbation):
    """
    Contrast scaling around the per-image mean.

    Images are scaled as: output = mean + scale * (input - mean),
    so the mean luminance is preserved and only the dynamic range changes.
    """

    PERTURBATION_TYPE = PerturbationType.CONTRAST
    CATEGORY          = PerturbationCategory.PHOTOMETRIC

    _MIN_SCALE = 0.3
    _MAX_SCALE = 3.0

    def _severity_to_params(self, severity: float) -> dict[str, Any]:
        # Log-interpolate so mid-severity ≈ natural contrast range
        log_min = np.log(self._MIN_SCALE)
        log_max = np.log(self._MAX_SCALE)
        scale = float(np.exp(log_min + severity * (log_max - log_min)))
        return {"scale_factor": round(scale, 5)}

    def _apply(self, image: np.ndarray, severity: float) -> dict[str, Any]:  # type: ignore[override]
        scale = self._severity_to_params(severity)["scale_factor"]
        f = self._to_float32(image)
        mean = f.mean()
        return self._restore_dtype(mean + scale * (f - mean), image)


# ---------------------------------------------------------------------------
# Gamma correction
# ---------------------------------------------------------------------------

class GammaPerturbation(BasePerturbation):
    """
    Power-law (gamma) transformation.

    output = input ^ gamma

    gamma < 1 brightens the image (compensates under-exposure).
    gamma > 1 darkens it (over-exposure simulation).
    """

    PERTURBATION_TYPE = PerturbationType.GAMMA
    CATEGORY          = PerturbationCategory.PHOTOMETRIC

    _MIN_GAMMA = 0.25
    _MAX_GAMMA = 4.00

    def _severity_to_params(self, severity: float) -> dict[str, Any]:
        # Log-interpolate; severity=0.5 → gamma=1.0
        log_min = np.log(self._MIN_GAMMA)
        log_max = np.log(self._MAX_GAMMA)
        gamma = float(np.exp(log_min + severity * (log_max - log_min)))
        return {"gamma": round(gamma, 5)}

    def _apply(self, image: np.ndarray, severity: float) -> np.ndarray:
        gamma = self._severity_to_params(severity)["gamma"]
        f = self._to_float32(image).clip(0.0, 1.0)
        corrected = np.power(f, gamma)
        return self._restore_dtype(corrected, image)


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

@register_perturbation(
    key=PerturbationType.BRIGHTNESS.value,
    category=PerturbationCategory.PHOTOMETRIC.value,
    description="Additive brightness shift (±0.40 around identity at severity=0.5)",
    severity_param="additive_shift",
)
def _brightness_factory(**_: Any) -> BrightnessPerturbation:
    return BrightnessPerturbation()


@register_perturbation(
    key=PerturbationType.CONTRAST.value,
    category=PerturbationCategory.PHOTOMETRIC.value,
    description="Contrast scaling 0.3× – 3.0× around per-image mean",
    severity_param="scale_factor",
)
def _contrast_factory(**_: Any) -> ContrastPerturbation:
    return ContrastPerturbation()


@register_perturbation(
    key=PerturbationType.GAMMA.value,
    category=PerturbationCategory.PHOTOMETRIC.value,
    description="Power-law gamma correction (0.25 – 4.0)",
    severity_param="gamma",
)
def _gamma_factory(**_: Any) -> GammaPerturbation:
    return GammaPerturbation()

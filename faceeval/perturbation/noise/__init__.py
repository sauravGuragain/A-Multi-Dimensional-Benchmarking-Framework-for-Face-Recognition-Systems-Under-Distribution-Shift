"""
faceeval.perturbation.noise
============================
Noise perturbations: Gaussian, salt-and-pepper, speckle.

Severity → parameter mappings
-------------------------------
Gaussian noise  : severity → std_dev  [0, 0.20] (fraction of pixel range)
Salt-and-pepper : severity → density  [0, 0.20] (fraction of corrupted pixels)
Speckle noise   : severity → variance [0, 0.15] (multiplicative noise variance)
"""

from __future__ import annotations

from typing import Any

import numpy as np

from faceeval.core.registry import register_perturbation
from faceeval.core.types import PerturbationCategory, PerturbationType
from faceeval.perturbation.base import BasePerturbation

_SEED = 0   # fixed seed so the same (image, severity) always yields the same noise


# ---------------------------------------------------------------------------
# Gaussian noise
# ---------------------------------------------------------------------------

class GaussianNoise(BasePerturbation):
    """Additive zero-mean Gaussian noise."""

    PERTURBATION_TYPE = PerturbationType.GAUSSIAN_NOISE
    CATEGORY          = PerturbationCategory.NOISE

    _MAX_STD = 0.20     # fraction of full [0,1] pixel range at severity=1.0

    def _severity_to_params(self, severity: float) -> dict[str, Any]:
        return {"std_dev": round(severity * self._MAX_STD, 5)}

    def _apply(self, image: np.ndarray, severity: float) -> np.ndarray:
        std = self._severity_to_params(severity)["std_dev"]
        rng = np.random.default_rng(_SEED)
        f = self._to_float32(image)
        noise = rng.normal(0.0, std, f.shape).astype(np.float32)
        return self._restore_dtype(f + noise, image)


# ---------------------------------------------------------------------------
# Salt-and-pepper noise
# ---------------------------------------------------------------------------

class SaltPepperNoise(BasePerturbation):
    """Impulse noise — random pixels set to min or max value."""

    PERTURBATION_TYPE = PerturbationType.SALT_PEPPER_NOISE
    CATEGORY          = PerturbationCategory.NOISE

    _MAX_DENSITY = 0.20

    def _severity_to_params(self, severity: float) -> dict[str, Any]:
        return {"density": round(severity * self._MAX_DENSITY, 5)}

    def _apply(self, image: np.ndarray, severity: float) -> np.ndarray:
        density = self._severity_to_params(severity)["density"]
        rng = np.random.default_rng(_SEED)
        out = image.copy()
        mask = rng.random(image.shape[:2])
        salt_mask   = mask < density / 2
        pepper_mask = (mask >= density / 2) & (mask < density)

        if image.dtype == np.uint8:
            out[salt_mask]   = 255
            out[pepper_mask] = 0
        else:
            out[salt_mask]   = 1.0
            out[pepper_mask] = 0.0
        return out


# ---------------------------------------------------------------------------
# Speckle noise
# ---------------------------------------------------------------------------

class SpeckleNoise(BasePerturbation):
    """Multiplicative speckle noise (common in radar / ultrasound imagery)."""

    PERTURBATION_TYPE = PerturbationType.SPECKLE_NOISE
    CATEGORY          = PerturbationCategory.NOISE

    _MAX_VAR = 0.15

    def _severity_to_params(self, severity: float) -> dict[str, Any]:
        return {"variance": round(severity * self._MAX_VAR, 5)}

    def _apply(self, image: np.ndarray, severity: float) -> np.ndarray:
        variance = self._severity_to_params(severity)["variance"]
        rng = np.random.default_rng(_SEED)
        f = self._to_float32(image)
        speckle = rng.normal(0.0, variance ** 0.5, f.shape).astype(np.float32)
        return self._restore_dtype(f + f * speckle, image)


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

@register_perturbation(
    key=PerturbationType.GAUSSIAN_NOISE.value,
    category=PerturbationCategory.NOISE.value,
    description="Additive Gaussian noise (std 0–0.20)",
    severity_param="std_dev",
)
def _gn_factory(**_: Any) -> GaussianNoise:
    return GaussianNoise()


@register_perturbation(
    key=PerturbationType.SALT_PEPPER_NOISE.value,
    category=PerturbationCategory.NOISE.value,
    description="Salt-and-pepper impulse noise (density 0–0.20)",
    severity_param="density",
)
def _sp_factory(**_: Any) -> SaltPepperNoise:
    return SaltPepperNoise()


@register_perturbation(
    key=PerturbationType.SPECKLE_NOISE.value,
    category=PerturbationCategory.NOISE.value,
    description="Multiplicative speckle noise (variance 0–0.15)",
    severity_param="variance",
)
def _sk_factory(**_: Any) -> SpeckleNoise:
    return SpeckleNoise()

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

import hashlib
from typing import Any

import numpy as np

from faceeval.core.registry import register_perturbation
from faceeval.core.types import PerturbationCategory, PerturbationType
from faceeval.perturbation.base import BasePerturbation, current_image_id

# Per-image noise seeding. seed = SHA-256(master seed | image id | perturbation type); severity is NOT part of
# the seed, so an image's noise pattern is the same at every severity (scaled by severity). Changing
# NOISE_MASTER_SEED changes every noise realisation and must be treated as a protocol change.
NOISE_MASTER_SEED = 42


def noise_seed(image_id: str, perturbation_type: str) -> int:
    """Deterministic 64-bit seed for one (image id, perturbation type); the same on every platform and run."""
    key = f"{NOISE_MASTER_SEED}|{image_id.replace(chr(92), '/')}|{perturbation_type}"
    return int.from_bytes(hashlib.sha256(key.encode("utf-8")).digest()[:8], "big")


def _noise_rng(perturbation_type: PerturbationType, image: np.ndarray) -> np.random.Generator:
    image_id = current_image_id()
    if image_id is None:
        # shortcut: ad-hoc callers without an id are seeded from the pixel content (still independent per
        # image and reproducible, but not stable across models/preprocessing); upgrade by passing image_id.
        image_id = f"content:{image.dtype}:{image.shape}:{hashlib.sha256(np.ascontiguousarray(image).tobytes()).hexdigest()}"
    return np.random.default_rng(noise_seed(image_id, perturbation_type.value))


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
        rng = _noise_rng(self.PERTURBATION_TYPE, image)
        f = self._to_float32(image)
        noise = rng.normal(0.0, std, f.shape).astype(np.float32)
        return self._from_unit(f + noise, image)


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
        rng = _noise_rng(self.PERTURBATION_TYPE, image)
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
        rng = _noise_rng(self.PERTURBATION_TYPE, image)
        f = self._to_float32(image)
        speckle = rng.normal(0.0, variance ** 0.5, f.shape).astype(np.float32)
        return self._from_unit(f + f * speckle, image)


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

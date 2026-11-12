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
from faceeval.perturbation.base import BasePerturbation, current_image_id, normalize_image_id

# Per-image noise seeding. seed = SHA-256(master seed | image id | perturbation type); severity is NOT part of
# the seed, so an image's noise pattern is the same at every severity (scaled by severity). Changing
# NOISE_MASTER_SEED or anything below must be treated as a protocol change and bump NOISE_PROTOCOL_VERSION.
NOISE_MASTER_SEED = 42
# per_image_v1: per-image seeding (patch 1).  per_image_v2: + NFC-normalised ids and salt-and-pepper polarity
# drawn from its own uniform array, so a pixel keeps its polarity at every severity.
NOISE_PROTOCOL_VERSION = "per_image_v2"


# Fingerprint of the NumPy random stream the noise is drawn from. Recorded in every run's metadata so a change of
# NumPy version (or generator) that alters the stream is visible, never silent.
#
# Approval policy (one source of truth, used by the metadata AND by the golden tests):
#   * NOISE_STREAM_REFERENCE is the base approved stream (produced with the NumPy version named in it). Any NumPy
#     whose stream is identical matches it, whatever its version string.
#   * NOISE_STREAM_APPROVED_BY_VERSION holds explicit, reviewed approvals {numpy version: stream sha256} for a NumPy
#     whose stream differs. It is empty on purpose; add an entry ONLY after a reviewed decision that the new stream is
#     acceptable (and NOISE_PROTOCOL_VERSION has been bumped). An approval applies to that exact version only and
#     REPLACES the base reference for it.
#   * Anything else is an unapproved stream: ``matches_approved_reference`` is False and the golden test fails.
NOISE_STREAM_PROBE_SEED = 42
NOISE_STREAM_REFERENCE = {
    "numpy_version": "2.5.3",
    "bit_generator": "PCG64",
    "sha256": "b09dc4e7603081f3ab0bb681b5377d8827c01bcadb5ec9c987dc01c49792adae",
}
NOISE_STREAM_APPROVED_BY_VERSION: dict[str, str] = {}


def approved_stream_reference(numpy_version: str | None = None) -> dict[str, str]:
    """The approved stream fingerprint for ``numpy_version`` (default: the running NumPy): its explicit
    version-specific approval if one exists, else the base reference. ``source`` says which.

    A version-specific approval records only ``{numpy version: stream sha256}``, so it carries NO ``bit_generator``
    claim (the base reference states PCG64 because it was produced with it; an approval for another NumPy must not
    inherit that claim). The generator actually used by a run is always reported separately, as the top-level
    ``bit_generator`` of ``noise_stream_fingerprint()``.
    """
    version = np.__version__ if numpy_version is None else numpy_version
    if version in NOISE_STREAM_APPROVED_BY_VERSION:
        return {"numpy_version": version, "sha256": NOISE_STREAM_APPROVED_BY_VERSION[version],
                "source": "version_specific_approval"}
    return {**NOISE_STREAM_REFERENCE, "source": "base_reference"}


def noise_stream_fingerprint() -> dict[str, Any]:
    """Deterministic fingerprint of the random stream: ``default_rng(42)`` -> ``random(8)`` then ``normal(size=8)``
    (the two draw methods the noise uses), float64 little-endian bytes, SHA-256. Carries the generator and NumPy
    version needed to interpret it, the approved reference it is judged against (``approved_stream_reference``:
    the version-specific approval if any, else the base reference) and whether it matches."""
    rng = np.random.default_rng(NOISE_STREAM_PROBE_SEED)
    uniform = rng.random(8)
    normal = rng.normal(size=8)
    digest = hashlib.sha256(uniform.astype("<f8").tobytes() + normal.astype("<f8").tobytes()).hexdigest()
    approved = approved_stream_reference()
    return {
        "generator": "numpy.random.Generator via default_rng",
        "bit_generator": type(rng.bit_generator).__name__,
        "numpy_version": np.__version__,
        "probe": f"default_rng({NOISE_STREAM_PROBE_SEED}): random(8) then normal(size=8); float64 little-endian bytes",
        "sha256": digest,
        "approved_reference": approved,
        "matches_approved_reference": digest == approved["sha256"],
    }


def noise_seed(image_id: str, perturbation_type: str) -> int:
    """Deterministic 64-bit seed for one (image id, perturbation type); the same on every platform and run."""
    key = f"{NOISE_MASTER_SEED}|{normalize_image_id(image_id)}|{perturbation_type}"
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

    USES_IMAGE_ID = True

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

    USES_IMAGE_ID = True

    PERTURBATION_TYPE = PerturbationType.SALT_PEPPER_NOISE
    CATEGORY          = PerturbationCategory.NOISE

    _MAX_DENSITY = 0.20

    def _severity_to_params(self, severity: float) -> dict[str, Any]:
        return {"density": round(severity * self._MAX_DENSITY, 5)}

    def _apply(self, image: np.ndarray, severity: float) -> np.ndarray:
        density = self._severity_to_params(severity)["density"]
        rng = _noise_rng(self.PERTURBATION_TYPE, image)
        out = image.copy()
        # Two fixed-size arrays from the same per-image stream, always drawn in this order and at full size, so
        # neither depends on severity: ``where`` picks WHICH pixels are corrupted (nested across severities
        # because the set is ``where < density``), ``polarity`` picks salt vs pepper (a pixel keeps its polarity
        # at every severity). Marginally: fraction ``density`` corrupted, half salt / half pepper.
        where = rng.random(image.shape[:2])
        polarity = rng.random(image.shape[:2])
        corrupted = where < density
        salt_mask   = corrupted & (polarity < 0.5)
        pepper_mask = corrupted & (polarity >= 0.5)

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

    USES_IMAGE_ID = True

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

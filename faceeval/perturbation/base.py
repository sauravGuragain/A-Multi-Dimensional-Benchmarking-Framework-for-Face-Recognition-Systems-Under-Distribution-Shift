"""
faceeval.perturbation.base
===========================
Abstract base class for all image perturbations and the severity scheduler.

Design contract
---------------
Every perturbation in FaceEval-X is a stateless callable that maps

    (image: np.ndarray, severity: float) → np.ndarray

where ``severity ∈ [0.0, 1.0]`` is a normalised intensity value.

Key invariants
--------------
1.  ``severity = 0.0`` MUST return the image unchanged (or as close as
    possible given integer rounding).  This is the clean-image baseline.
2.  The output dtype MUST match the input dtype.
3.  The output shape MUST match the input shape.
4.  Perturbations are STATELESS — ``apply`` may be called concurrently
    from multiple threads on different images without synchronisation.

Severity ↔ raw parameter mapping
---------------------------------
Each concrete class defines ``_severity_to_params(severity)`` which maps
the normalised ``[0, 1]`` severity to the actual parameter values used
(e.g. Gaussian sigma, JPEG quality, rotation degrees).  This mapping is
stored in ``PerturbationSpec.raw_params`` so every result is fully
reproducible from the spec alone.
"""

from __future__ import annotations

import contextvars
import logging
import unicodedata
from abc import ABC, abstractmethod
from typing import Any

import numpy as np

from faceeval.core.exceptions import InvalidSeverityError, PerturbationError

# Stable id of the image being perturbed; read by perturbations that draw random numbers (noise).
_IMAGE_ID: contextvars.ContextVar[str | None] = contextvars.ContextVar("faceeval_image_id", default=None)


def current_image_id() -> str | None:
    """Image id passed to the running ``BasePerturbation.apply`` call (None if the caller gave none)."""
    return _IMAGE_ID.get()


def normalize_image_id(image_id: str) -> str:
    """Canonical form of an image id used for noise seeding: Unicode NFC, then backslashes -> forward slashes.
    Nothing else is changed (case and surrounding spaces are significant)."""
    return unicodedata.normalize("NFC", image_id).replace(chr(92), "/")


# Characters that make an id invisible or ambiguous: control characters (Cc: NUL, newline, tab, DEL, C1),
# format characters (Cf: zero-width space/joiners, BOM, bidi marks) and lone surrogates (Cs). Letters, accents,
# CJK, emoji etc. are untouched. (Cf also contains the zero-width non-joiner used in some Persian/Indic scripts;
# LFW file names are Latin, and an id containing one is rejected on purpose rather than risk invisible differences.)
_FORBIDDEN_ID_CATEGORIES = {"Cc": "control", "Cf": "format / zero-width", "Cs": "surrogate"}


def check_image_id(image_id: object, position: int | None = None) -> None:
    """Raise if ``image_id`` is not a usable image id: a non-empty, non-whitespace ``str`` without control,
    format/zero-width or surrogate characters."""
    where = "" if position is None else f" at position {position}"
    if not isinstance(image_id, str):
        raise TypeError(f"image id{where} must be a str, got {type(image_id).__name__}: {image_id!r}")
    if not image_id.strip():
        raise ValueError(f"image id{where} is empty or whitespace-only: {image_id!r}")
    for ch in image_id:
        kind = _FORBIDDEN_ID_CATEGORIES.get(unicodedata.category(ch))
        if kind:
            raise ValueError(f"image id{where} contains a {kind} character U+{ord(ch):04X}: {image_id!r}")


def validate_image_ids(image_ids: object, n_images: int, what: str = "images") -> None:
    """Strict validation of the ids of a benchmark batch.

    Rejects ``None``, a bare ``str``/``bytes`` passed instead of a collection of ids, a length different from
    ``n_images``, any id rejected by ``check_image_id``, and duplicate ids (two images with one id would silently
    share one noise realisation). Duplicates are judged on the canonical form (``normalize_image_id``), because
    that is what seeds the noise: ``a\\b`` and ``a/b``, or an NFC and an NFD spelling of one name, are the same id.
    """
    if image_ids is None:
        raise ValueError(f"image_ids is required for a benchmark batch of {n_images} {what}; got None")
    if isinstance(image_ids, (str, bytes, bytearray)):
        raise TypeError(f"image_ids must be a collection with one id per image, not a single {type(image_ids).__name__}: "
                        f"{image_ids!r}")
    if len(image_ids) != n_images:  # type: ignore[arg-type]
        raise ValueError(f"image_ids has {len(image_ids)} entries for {n_images} {what}")  # type: ignore[arg-type]
    seen: dict[str, int] = {}
    for pos, image_id in enumerate(image_ids):  # type: ignore[attr-defined]
        check_image_id(image_id, pos)
        key = normalize_image_id(image_id)
        if key in seen:
            raise ValueError(f"duplicate image id {image_id!r} (canonical {key!r}) at positions {seen[key]} and {pos}")
        seen[key] = pos


from faceeval.core.types import (
    PerturbationCategory,
    PerturbationSpec,
    PerturbationType,
    Severity,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------

class BasePerturbation(ABC):
    """
    Abstract base for a single image perturbation.

    Subclasses must implement:
        ``PERTURBATION_TYPE``   : PerturbationType  — enum value for this class.
        ``CATEGORY``            : PerturbationCategory — broad family.
        ``_severity_to_params`` — maps [0,1] severity to concrete parameters.
        ``_apply``              — applies the perturbation to one image.
    """

    PERTURBATION_TYPE: PerturbationType    = NotImplemented  # type: ignore[assignment]
    CATEGORY:          PerturbationCategory = NotImplemented  # type: ignore[assignment]
    # True for perturbations whose randomness is seeded per image id (noise). Benchmark batches must then
    # supply image ids; ad-hoc ``apply(image, severity)`` calls still work (content-hash fallback).
    USES_IMAGE_ID: bool = False

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def apply(self, image: np.ndarray, severity: Severity, image_id: str | None = None) -> np.ndarray:
        """
        Apply this perturbation at ``severity`` to ``image``.

        Parameters
        ----------
        image:
            Input image as a numpy array.  May be uint8 or float32.
            Shape: (H, W) for grayscale or (H, W, 3) for colour.
        severity:
            Normalised intensity in [0.0, 1.0].  ``0.0`` returns a copy
            of the original image.
        image_id:
            Stable id of the source image. Only perturbations with randomness (noise) use it, to seed
            their per-image noise; every other perturbation ignores it. A supplied id is validated
            (``check_image_id``) at every severity, including 0.0.

        Returns
        -------
        np.ndarray
            Perturbed image with same dtype and shape as input.
        """
        if not 0.0 <= severity <= 1.0:
            raise InvalidSeverityError(severity)

        if image_id is not None:
            check_image_id(image_id)        # a supplied id must be valid at EVERY severity (also 0.0)

        if severity == 0.0:
            return image.copy()

        original_dtype = image.dtype
        token = _IMAGE_ID.set(image_id)
        try:
            result = self._apply(image, severity)
        finally:
            _IMAGE_ID.reset(token)

        # Enforce dtype contract
        if result.dtype != original_dtype:
            if original_dtype == np.uint8:
                result = result.clip(0, 255).astype(np.uint8)
            else:
                result = result.astype(original_dtype)

        # Enforce shape contract
        if result.shape != image.shape:
            raise RuntimeError(
                f"{self.__class__.__name__}: output shape {result.shape} "
                f"!= input shape {image.shape}"
            )
        return result

    def build_spec(self, severity: Severity) -> PerturbationSpec:
        """Build a ``PerturbationSpec`` for this perturbation at ``severity``."""
        return PerturbationSpec(
            perturbation_type=self.PERTURBATION_TYPE,
            category=self.CATEGORY,
            severity=severity,
            raw_params=self._severity_to_params(severity),
            description=self.__class__.__doc__.split("\n")[0].strip()
            if self.__class__.__doc__ else "",
        )

    # ------------------------------------------------------------------
    # Abstract hooks
    # ------------------------------------------------------------------

    @abstractmethod
    def _apply(self, image: np.ndarray, severity: Severity) -> np.ndarray:
        """Apply the perturbation.  Called only when severity > 0."""
        ...

    @abstractmethod
    def _severity_to_params(self, severity: Severity) -> dict[str, Any]:
        """Map normalised severity to concrete parameter values."""
        ...

    # ------------------------------------------------------------------
    # Shared helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _to_uint8(image: np.ndarray) -> np.ndarray:
        if image.dtype == np.uint8:
            return image
        return (image * 255.0).clip(0, 255).astype(np.uint8)

    @staticmethod
    def _to_float32(image: np.ndarray) -> np.ndarray:
        if image.dtype == np.float32:
            return image
        if image.dtype == np.uint8:
            return image.astype(np.float32) / 255.0
        return image.astype(np.float32)

    @staticmethod
    def _from_unit(result: np.ndarray, original: np.ndarray) -> np.ndarray:
        """Inverse of ``_to_float32``: map a [0, 1]-scale result back to ``original``'s dtype and range."""
        result = np.clip(result, 0.0, 1.0)
        if original.dtype == np.uint8:
            return np.rint(result * 255.0).astype(np.uint8)
        return result.astype(original.dtype)

    @staticmethod
    def _restore_dtype(result: np.ndarray, original: np.ndarray) -> np.ndarray:
        """Cast ``result`` to ``original``'s dtype. ``result`` must already be in ``original``'s value range
        (use ``_from_unit`` for a [0, 1]-scale result from ``_to_float32``)."""
        if original.dtype == np.uint8:
            return result.clip(0, 255).astype(np.uint8)
        if original.dtype == np.float32:
            return result.astype(np.float32)
        return result.astype(original.dtype)


# ---------------------------------------------------------------------------
# Severity scheduler
# ---------------------------------------------------------------------------

class SeverityScheduler:
    """
    Maps ``PerturbationSchedule`` severity levels to ``PerturbationSpec`` objects.

    The scheduler is the single place where the normalised severity grid
    (e.g. ``[0.0, 0.2, 0.4, 0.6, 0.8, 1.0]``) is combined with a
    ``BasePerturbation`` instance to produce the ``PerturbationSpec`` list
    used by the evaluation loop.

    Usage::

        scheduler = SeverityScheduler(severity_levels=[0.0, 0.25, 0.5, 0.75, 1.0])
        specs = scheduler.specs_for(gaussian_blur_instance)
        # → 5 PerturbationSpec objects, one per level
    """

    def __init__(self, severity_levels: list[Severity] | None = None) -> None:
        self._levels = sorted(set(severity_levels or [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]))

    def specs_for(self, perturbation: BasePerturbation) -> list[PerturbationSpec]:
        """Return one ``PerturbationSpec`` per severity level."""
        return [perturbation.build_spec(s) for s in self._levels]

    def all_specs(
        self, perturbations: list[BasePerturbation]
    ) -> list[PerturbationSpec]:
        """Return specs for every (perturbation × severity) combination."""
        specs: list[PerturbationSpec] = []
        for p in perturbations:
            specs.extend(self.specs_for(p))
        return specs

    @property
    def levels(self) -> list[Severity]:
        return self._levels

    @property
    def num_levels(self) -> int:
        return len(self._levels)

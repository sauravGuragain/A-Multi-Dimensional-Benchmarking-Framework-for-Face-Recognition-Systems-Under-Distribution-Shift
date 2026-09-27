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

import logging
from abc import ABC, abstractmethod
from typing import Any

import numpy as np

from faceeval.core.exceptions import InvalidSeverityError
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

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def apply(self, image: np.ndarray, severity: Severity) -> np.ndarray:
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

        Returns
        -------
        np.ndarray
            Perturbed image with same dtype and shape as input.
        """
        if not 0.0 <= severity <= 1.0:
            raise InvalidSeverityError(severity)

        if severity == 0.0:
            return image.copy()

        original_dtype = image.dtype
        result = self._apply(image, severity)

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
    def _restore_dtype(result: np.ndarray, original: np.ndarray) -> np.ndarray:
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

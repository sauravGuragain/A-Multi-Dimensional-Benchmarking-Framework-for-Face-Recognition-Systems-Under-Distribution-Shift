"""
faceeval.perturbation.composer
================================
``PerturbationComposer`` — applies perturbation specs to image batches and
produces ``PerturbedBatch`` objects consumed by the evaluation loop.

Responsibilities
----------------
* Look up ``BasePerturbation`` instances from ``perturbation_registry``.
* Apply the perturbation to every image in a batch at the specified severity.
* Optionally compose (chain) multiple perturbations in sequence.
* Return ``PerturbedBatch`` typed objects that carry both the perturbed
  pixels and the full ``PerturbationSpec`` provenance.

Thread safety
-------------
``PerturbationComposer`` is stateless after construction.  The underlying
perturbation objects are also stateless.  Safe to use from multiple threads.
"""

from __future__ import annotations

import logging
import time
from typing import Any

import numpy as np

from faceeval.core.exceptions import UnknownPluginError
from faceeval.core.registry import perturbation_registry
from faceeval.core.types import (
    ImageRecord,
    PerturbationSpec,
    PerturbationType,
    PerturbedBatch,
    SubjectID,
)
from faceeval.perturbation.base import BasePerturbation, SeverityScheduler

logger = logging.getLogger(__name__)


class PerturbationComposer:
    """
    Applies one or more ``PerturbationSpec`` objects to image batches.

    Parameters
    ----------
    severity_levels:
        Default severity grid used when building specs from ``PerturbationType``
        names.  Individual specs override this.
    compose:
        When ``True`` and multiple perturbation types are specified, apply
        them in sequence on each image (compounding degradation).
        When ``False`` (default), each perturbation type is applied
        independently, producing one batch per type.
    """

    def __init__(
        self,
        severity_levels: list[float] | None = None,
        compose: bool = False,
    ) -> None:
        self._scheduler = SeverityScheduler(severity_levels)
        self._compose = compose
        self._cache: dict[str, BasePerturbation] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def apply_spec(
        self,
        images: list[np.ndarray],
        subject_ids: list[SubjectID],
        image_ids: list[str],
        spec: PerturbationSpec,
        dataset_name: str = "",
    ) -> PerturbedBatch:
        """
        Apply one ``PerturbationSpec`` to every image in ``images``.

        Parameters
        ----------
        images:
            List of preprocessed face arrays (any dtype).
        subject_ids:
            Subject ID for each image (parallel list).
        image_ids:
            Image ID for each image (parallel list).
        spec:
            Fully described perturbation with type, category, and severity.
        dataset_name:
            Dataset this batch originates from.

        Returns
        -------
        PerturbedBatch
        """
        perturbation = self._get_perturbation(spec.perturbation_type)
        t0 = time.perf_counter()

        perturbed: list[Any] = []
        for img in images:
            try:
                out = perturbation.apply(img, spec.severity)
            except Exception as exc:
                logger.warning(
                    "Perturbation '%s' at severity=%.2f failed on one image: %s; "
                    "returning original.",
                    spec.perturbation_type.value, spec.severity, exc,
                )
                out = img.copy()
            perturbed.append(out)

        elapsed = (time.perf_counter() - t0) * 1000
        logger.debug(
            "Applied '%s' s=%.2f to %d images in %.1f ms",
            spec.perturbation_type.value, spec.severity, len(images), elapsed,
        )
        return PerturbedBatch(
            original_ids=image_ids,
            subject_ids=subject_ids,
            image_tensors=perturbed,
            perturbation_spec=spec,
            dataset_name=dataset_name,
        )

    def apply_all_specs(
        self,
        images: list[np.ndarray],
        subject_ids: list[SubjectID],
        image_ids: list[str],
        specs: list[PerturbationSpec],
        dataset_name: str = "",
    ) -> list[PerturbedBatch]:
        """
        Apply every spec in ``specs`` to the same image batch.

        Returns one ``PerturbedBatch`` per spec (len(specs) batches total).
        Use this for the outer evaluation loop where every
        (perturbation × severity) condition is applied independently.
        """
        return [
            self.apply_spec(images, subject_ids, image_ids, spec, dataset_name)
            for spec in specs
        ]

    def apply_from_records(
        self,
        records: list[ImageRecord],
        raw_images: list[np.ndarray],
        specs: list[PerturbationSpec],
        dataset_name: str = "",
    ) -> list[PerturbedBatch]:
        """
        Convenience wrapper that extracts subject_ids and image_ids from records.
        """
        subject_ids = [r.subject_id for r in records]
        image_ids   = [r.image_id   for r in records]
        return self.apply_all_specs(
            raw_images, subject_ids, image_ids, specs, dataset_name
        )

    def compose_specs(
        self,
        images: list[np.ndarray],
        subject_ids: list[SubjectID],
        image_ids: list[str],
        specs: list[PerturbationSpec],
        dataset_name: str = "",
    ) -> PerturbedBatch:
        """
        Apply multiple specs in sequence (chaining / compounding).

        All specs are applied one after another on each image, accumulating
        the degradation.  The ``PerturbationSpec`` of the returned batch is
        the LAST spec applied (for traceability).  This mode is used when
        ``compose=True`` is set on the composer.
        """
        if not specs:
            raise ValueError("compose_specs requires at least one spec")

        current_images = [img.copy() for img in images]
        for spec in specs:
            perturbation = self._get_perturbation(spec.perturbation_type)
            current_images = [perturbation.apply(img, spec.severity) for img in current_images]

        return PerturbedBatch(
            original_ids=image_ids,
            subject_ids=subject_ids,
            image_tensors=current_images,
            perturbation_spec=specs[-1],
            dataset_name=dataset_name,
        )

    def specs_for_type(self, ptype: PerturbationType) -> list[PerturbationSpec]:
        """
        Build a list of ``PerturbationSpec`` objects for all severity levels
        for one perturbation type.
        """
        perturbation = self._get_perturbation(ptype)
        return self._scheduler.specs_for(perturbation)

    def specs_for_types(
        self, ptypes: list[PerturbationType]
    ) -> list[PerturbationSpec]:
        """Build specs for all (perturbation_type × severity_level) combinations."""
        specs: list[PerturbationSpec] = []
        for pt in ptypes:
            specs.extend(self.specs_for_type(pt))
        return specs

    def clean_batch(
        self,
        images: list[np.ndarray],
        subject_ids: list[SubjectID],
        image_ids: list[str],
        dataset_name: str = "",
    ) -> PerturbedBatch:
        """Return an identity batch (severity=0.0 Gaussian blur) for the clean baseline."""
        spec = PerturbationSpec(
            perturbation_type=PerturbationType.GAUSSIAN_BLUR,
            category=self._get_perturbation(PerturbationType.GAUSSIAN_BLUR).CATEGORY,
            severity=0.0,
            raw_params={"sigma": 0.0, "kernel_size": 1},
            description="clean (no perturbation)",
        )
        return PerturbedBatch(
            original_ids=image_ids,
            subject_ids=subject_ids,
            image_tensors=[img.copy() for img in images],
            perturbation_spec=spec,
            dataset_name=dataset_name,
        )

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _get_perturbation(self, ptype: PerturbationType) -> BasePerturbation:
        key = ptype.value
        if key not in self._cache:
            try:
                self._cache[key] = perturbation_registry.get(key)
            except UnknownPluginError:
                raise UnknownPluginError(
                    f"Perturbation '{key}' is not registered. "
                    f"Registered: {perturbation_registry.list_registered()}"
                )
        return self._cache[key]

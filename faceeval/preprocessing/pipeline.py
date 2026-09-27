"""
faceeval.preprocessing.pipeline
=================================
End-to-end preprocessing pipeline: load → detect → align → normalize.

``PreprocessingPipeline`` is the single entry point for all pixel work.
It composes ``FaceDetector``, ``FaceAligner``, and ``Normalizer`` into one
callable that accepts an ``ImageRecord`` (or a raw path) and returns a
normalised numpy array ready for feature extraction.

Design decisions
----------------
* The pipeline is **stateless across images** — one instance processes
  any number of images without accumulating per-image state.
* Model-specific pipelines are produced by ``pipeline_for_model()``, which
  wires the correct normalisation mode and output size.
* The pipeline can process both lists and single images.
* Failure handling: ``ProcessedImage.success == False`` with an error message
  rather than raising, so batch processing can continue past bad images and
  collect failures for the failure-analysis module.
* Image loading is the pipeline's responsibility — downstream modules never
  call ``cv2.imread`` directly.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from faceeval.core.exceptions import PreprocessingError
from faceeval.core.types import ImageRecord
from faceeval.preprocessing.aligner import AlignedFace, FaceAligner
from faceeval.preprocessing.detector import DetectionResult, FaceDetector
from faceeval.preprocessing.normalizer import Normalizer, normalizer_for_model

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------

@dataclass
class ProcessedImage:
    """
    Output of the preprocessing pipeline for one image.

    ``success == True``  → ``array`` contains a valid normalised face chip.
    ``success == False`` → ``error`` describes what went wrong.
    """
    image_id: str
    image_path: str
    success: bool
    array: np.ndarray | None             # float32 or uint8 depending on normalizer
    detection: DetectionResult | None    # full detection metadata
    aligned: AlignedFace | None          # aligned chip before normalisation
    total_time_ms: float = 0.0
    error: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "image_id": self.image_id,
            "image_path": self.image_path,
            "success": self.success,
            "shape": list(self.array.shape) if self.array is not None else None,
            "total_time_ms": round(self.total_time_ms, 2),
            "alignment_method": (
                self.aligned.alignment_method if self.aligned else None
            ),
            "detection_backend": (
                self.detection.backend_used if self.detection else None
            ),
            "error": self.error,
        }


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

class PreprocessingPipeline:
    """
    Full preprocessing pipeline: load → detect → align → normalize.

    Parameters
    ----------
    detector:
        Configured ``FaceDetector`` instance.
    aligner:
        Configured ``FaceAligner`` instance.
    normalizer:
        Configured ``Normalizer`` instance.
    skip_detection_for_pre_cropped:
        When ``True``, skips detection and treats the entire image as the
        face crop.  Recommended for datasets like LFW and VGGFace2 where
        images are already tightly cropped.
    """

    def __init__(
        self,
        detector: FaceDetector,
        aligner: FaceAligner,
        normalizer: Normalizer,
        skip_detection_for_pre_cropped: bool = True,
    ) -> None:
        self._detector = detector
        self._aligner = aligner
        self._normalizer = normalizer
        self._skip_detection = skip_detection_for_pre_cropped

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def process(self, record: ImageRecord) -> ProcessedImage:
        """
        Process one ``ImageRecord`` through the full pipeline.

        Returns a ``ProcessedImage`` with ``success=False`` on error rather
        than raising, so batch callers can accumulate failures.
        """
        t_start = time.perf_counter()
        try:
            result = self._process_inner(record)
        except Exception as exc:
            logger.warning(
                "Preprocessing failed for '%s': %s", record.image_path, exc
            )
            result = ProcessedImage(
                image_id=record.image_id,
                image_path=record.image_path,
                success=False,
                array=None,
                detection=None,
                aligned=None,
                error=str(exc),
            )
        result.total_time_ms = (time.perf_counter() - t_start) * 1000
        return result

    def process_batch(
        self,
        records: list[ImageRecord],
        progress_callback: Any = None,
    ) -> list[ProcessedImage]:
        """
        Process a batch of records.

        Parameters
        ----------
        records:
            List of ``ImageRecord`` objects.
        progress_callback:
            Optional callable ``(completed: int, total: int) -> None``
            called after each image.
        """
        results: list[ProcessedImage] = []
        total = len(records)
        for i, record in enumerate(records):
            results.append(self.process(record))
            if progress_callback is not None:
                progress_callback(i + 1, total)
        return results

    def process_path(self, image_path: str, image_id: str = "") -> ProcessedImage:
        """Process a raw image path without an ``ImageRecord``."""
        record = ImageRecord(
            image_id=image_id or image_path,
            image_path=image_path,
            subject_id="unknown",
            dataset_name="unknown",
            split="unknown",
        )
        return self.process(record)

    @property
    def detector(self) -> FaceDetector:
        return self._detector

    @property
    def aligner(self) -> FaceAligner:
        return self._aligner

    @property
    def normalizer(self) -> Normalizer:
        return self._normalizer

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _process_inner(self, record: ImageRecord) -> ProcessedImage:
        # 1. Load image
        img_bgr = self._load(record.image_path)

        # 2. Detect
        if self._skip_detection:
            detection = self._whole_image_detection(img_bgr, record.image_path)
        else:
            detection = self._detector.detect(img_bgr, record.image_path)

        if detection.num_faces == 0:
            return ProcessedImage(
                image_id=record.image_id,
                image_path=record.image_path,
                success=False,
                array=None,
                detection=detection,
                aligned=None,
                error="No face detected",
            )

        # 3. Align
        aligned = self._aligner.align(img_bgr, detection)
        if aligned is None:
            return ProcessedImage(
                image_id=record.image_id,
                image_path=record.image_path,
                success=False,
                array=None,
                detection=detection,
                aligned=None,
                error="Alignment produced no output",
            )

        # 4. Normalize
        normalised = self._normalizer.normalize(aligned.image)

        return ProcessedImage(
            image_id=record.image_id,
            image_path=record.image_path,
            success=True,
            array=normalised,
            detection=detection,
            aligned=aligned,
        )

    @staticmethod
    def _load(path: str) -> np.ndarray:
        img = cv2.imread(path)
        if img is None:
            raise PreprocessingError(f"cv2.imread returned None for '{path}'")
        return img

    @staticmethod
    def _whole_image_detection(
        img_bgr: np.ndarray, image_path: str
    ) -> DetectionResult:
        """Synthetic detection result treating the entire image as a face."""
        from faceeval.preprocessing.detector import (
            DetectionResult, FaceBoundingBox, FaceLandmarks,
        )
        h, w = img_bgr.shape[:2]
        box = FaceBoundingBox(0, 0, w, h, 1.0)
        lm = FaceLandmarks(
            left_eye=(w * 0.35, h * 0.37),
            right_eye=(w * 0.65, h * 0.37),
            nose=(w * 0.50, h * 0.55),
            mouth_left=(w * 0.35, h * 0.72),
            mouth_right=(w * 0.65, h * 0.72),
        )
        return DetectionResult(
            image_path=image_path, image_h=h, image_w=w,
            faces=[box], landmarks=[lm],
            backend_used="whole_image",
        )


# ---------------------------------------------------------------------------
# Factory: build a model-specific pipeline
# ---------------------------------------------------------------------------

def pipeline_for_model(
    model_name: str,
    detector_backends: list[str] | None = None,
    skip_detection_for_pre_cropped: bool = True,
) -> PreprocessingPipeline:
    """
    Build a ``PreprocessingPipeline`` wired for a specific recognition model.

    Output size and normalisation mode are determined by ``model_name``
    via ``normalizer_for_model()``.

    Parameters
    ----------
    model_name:
        Registered model name (e.g. ``"facenet"``, ``"arcface"``,
        ``"eigenfaces"``).
    detector_backends:
        Override the detector backend order.  Defaults to
        ``["opencv_dnn", "haar", "whole_image"]``.
    skip_detection_for_pre_cropped:
        Pass ``True`` for LFW / VGGFace2; ``False`` for in-the-wild images.
    """
    from faceeval.preprocessing.normalizer import _MODEL_TO_SIZE
    output_size = _MODEL_TO_SIZE.get(model_name, 112)

    detector = FaceDetector(backends=detector_backends or ["opencv_dnn", "haar", "whole_image"])
    aligner = FaceAligner(output_size=output_size)
    normalizer = normalizer_for_model(model_name)

    return PreprocessingPipeline(
        detector=detector,
        aligner=aligner,
        normalizer=normalizer,
        skip_detection_for_pre_cropped=skip_detection_for_pre_cropped,
    )


def default_pipeline(skip_detection_for_pre_cropped: bool = True) -> PreprocessingPipeline:
    """
    Build a sensible default pipeline (ArcFace normalisation, 112×112).

    Suitable for general use when the model is not yet selected.
    """
    return pipeline_for_model(
        "arcface",
        skip_detection_for_pre_cropped=skip_detection_for_pre_cropped,
    )

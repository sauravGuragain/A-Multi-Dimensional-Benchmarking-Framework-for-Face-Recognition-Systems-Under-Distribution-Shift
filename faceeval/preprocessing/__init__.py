"""
faceeval.preprocessing
========================
Face detection, alignment, and normalisation pipeline.

Public API::

    from faceeval.preprocessing import (
        FaceDetector, FaceAligner, Normalizer,
        PreprocessingPipeline, pipeline_for_model, default_pipeline,
    )
"""

from faceeval.preprocessing.detector import (
    FaceDetector, DetectionResult, FaceBoundingBox, FaceLandmarks,
)
from faceeval.preprocessing.aligner import FaceAligner, AlignedFace
from faceeval.preprocessing.normalizer import (
    Normalizer, NormalizationMode, normalizer_for_model,
)
from faceeval.preprocessing.pipeline import (
    PreprocessingPipeline, ProcessedImage,
    pipeline_for_model, default_pipeline,
)

__all__ = [
    "FaceDetector", "DetectionResult", "FaceBoundingBox", "FaceLandmarks",
    "FaceAligner", "AlignedFace",
    "Normalizer", "NormalizationMode", "normalizer_for_model",
    "PreprocessingPipeline", "ProcessedImage",
    "pipeline_for_model", "default_pipeline",
]

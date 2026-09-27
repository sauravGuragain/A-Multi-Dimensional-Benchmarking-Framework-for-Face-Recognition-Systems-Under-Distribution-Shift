"""
faceeval.models
================
Recognition model package.
Importing triggers all @register_model decorators for both pipelines.
"""

import faceeval.models.traditional   # noqa: F401
import faceeval.models.deep          # noqa: F401

from faceeval.models.base import BaseRecognizer
from faceeval.models.traditional import (
    TraditionalRecognizerMixin, EigenfacesRecognizer,
    FisherfacesRecognizer, PCASVMRecognizer,
    LBPHRecognizer, HOGSVMRecognizer, KNNRecognizer,
)
from faceeval.models.deep import (
    DeepRecognizerBase, FaceNetRecognizer,
    ArcFaceRecognizer, InsightFaceRecognizer, DlibFaceRecognizer,
)

__all__ = [
    "BaseRecognizer",
    "TraditionalRecognizerMixin", "EigenfacesRecognizer", "FisherfacesRecognizer",
    "PCASVMRecognizer", "LBPHRecognizer", "HOGSVMRecognizer", "KNNRecognizer",
    "DeepRecognizerBase", "FaceNetRecognizer", "ArcFaceRecognizer",
    "InsightFaceRecognizer", "DlibFaceRecognizer",
]

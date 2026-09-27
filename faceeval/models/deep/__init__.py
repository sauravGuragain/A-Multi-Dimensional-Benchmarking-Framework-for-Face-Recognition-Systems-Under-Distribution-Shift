"""
faceeval.models.deep
=====================
Deep learning face recognizers.
Importing this package registers all DL models via @register_model.
"""

from faceeval.models.deep.base import DeepRecognizerBase
from faceeval.models.deep.facenet import FaceNetRecognizer
from faceeval.models.deep.arcface import ArcFaceRecognizer
from faceeval.models.deep.insightface import InsightFaceRecognizer, DlibFaceRecognizer

__all__ = [
    "DeepRecognizerBase",
    "FaceNetRecognizer",
    "ArcFaceRecognizer",
    "InsightFaceRecognizer",
    "DlibFaceRecognizer",
]

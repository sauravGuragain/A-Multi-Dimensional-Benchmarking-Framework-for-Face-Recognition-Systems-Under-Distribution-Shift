"""
faceeval.models.traditional
============================
Traditional ML face recognizers.
Importing this package registers all traditional models via @register_model.
"""

from faceeval.models.traditional.base import TraditionalRecognizerMixin
from faceeval.models.traditional.eigenfaces import (
    EigenfacesRecognizer, FisherfacesRecognizer, PCASVMRecognizer,
)
from faceeval.models.traditional.lbph import LBPHRecognizer
from faceeval.models.traditional.hog_svm import HOGSVMRecognizer, KNNRecognizer

__all__ = [
    "TraditionalRecognizerMixin",
    "EigenfacesRecognizer", "FisherfacesRecognizer", "PCASVMRecognizer",
    "LBPHRecognizer", "HOGSVMRecognizer", "KNNRecognizer",
]

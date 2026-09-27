"""
faceeval.preprocessing.normalizer
===================================
Pixel-level normalisation of aligned face chips.

Different model families expect different input statistics:

* **ArcFace / InsightFace** — subtract 127.5, divide by 128.0 → range [-1, 1]
* **FaceNet** — per-image standardisation (zero-mean, unit-variance)
* **Dlib FR** — [0, 1] float normalisation, RGB channel order
* **Classical ML** (Eigenfaces, LBP, HOG) — grayscale, optionally
  histogram-equalised

All methods accept a uint8 BGR numpy array (output of ``FaceAligner``)
and return a float32 array ready for the model.

``Normalizer`` is stateless — every call is independent.
"""

from __future__ import annotations

import logging
from enum import Enum

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class NormalizationMode(str, Enum):
    """Normalisation strategy identifier."""
    ARCFACE       = "arcface"         # [-1, 1], BGR
    FACENET       = "facenet"         # zero-mean unit-var, RGB
    DLIB          = "dlib"            # [0, 1], RGB
    UNIT_RANGE    = "unit_range"      # [0, 1], BGR (generic DL)
    IMAGENET      = "imagenet"        # ImageNet mean/std, RGB
    GRAYSCALE     = "grayscale"       # single-channel float, [0, 1]
    GRAYSCALE_EQ  = "grayscale_eq"    # grayscale + histogram equalisation
    RAW_UINT8     = "raw_uint8"       # no normalisation, returns uint8 BGR


# ImageNet channel statistics (mean, std) — RGB order
_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)


class Normalizer:
    """
    Pixel normalisation for aligned face chips.

    Parameters
    ----------
    mode:
        Normalisation strategy.  See ``NormalizationMode`` for options.
    target_size:
        If not ``None``, resize the chip to ``(target_size, target_size)``
        before normalising.  Useful when the aligner output size differs
        from the model's expected input resolution.
    """

    def __init__(
        self,
        mode: NormalizationMode | str = NormalizationMode.ARCFACE,
        target_size: int | None = None,
    ) -> None:
        self._mode = NormalizationMode(mode)
        self._target_size = target_size

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def normalize(self, face_bgr: np.ndarray) -> np.ndarray:
        """
        Normalise a single aligned face chip.

        Parameters
        ----------
        face_bgr:
            uint8 BGR array, shape ``(H, W, 3)`` or ``(H, W)`` for grayscale.

        Returns
        -------
        np.ndarray
            float32 (or uint8 for RAW_UINT8) array ready for model input.
        """
        img = self._maybe_resize(face_bgr)
        return self._apply(img)

    def normalize_batch(self, faces: list[np.ndarray]) -> list[np.ndarray]:
        """Normalise a list of face chips."""
        return [self.normalize(f) for f in faces]

    @property
    def mode(self) -> NormalizationMode:
        return self._mode

    # ------------------------------------------------------------------
    # Strategy dispatch
    # ------------------------------------------------------------------

    def _apply(self, img: np.ndarray) -> np.ndarray:
        m = self._mode

        if m == NormalizationMode.ARCFACE:
            return self._arcface(img)
        if m == NormalizationMode.FACENET:
            return self._facenet(img)
        if m == NormalizationMode.DLIB:
            return self._dlib(img)
        if m == NormalizationMode.UNIT_RANGE:
            return self._unit_range(img)
        if m == NormalizationMode.IMAGENET:
            return self._imagenet(img)
        if m == NormalizationMode.GRAYSCALE:
            return self._grayscale(img)
        if m == NormalizationMode.GRAYSCALE_EQ:
            return self._grayscale_eq(img)
        if m == NormalizationMode.RAW_UINT8:
            return img  # no normalisation
        raise ValueError(f"Unknown NormalizationMode: {m}")

    # ------------------------------------------------------------------
    # Individual normalisations
    # ------------------------------------------------------------------

    @staticmethod
    def _arcface(img: np.ndarray) -> np.ndarray:
        """
        ArcFace standard: (pixel - 127.5) / 128.0 → float32 in [-1, 1].
        Keeps BGR channel order.
        """
        return (img.astype(np.float32) - 127.5) / 128.0

    @staticmethod
    def _facenet(img: np.ndarray) -> np.ndarray:
        """
        FaceNet per-image standardisation: (x - mean(x)) / std(x).
        Converts to RGB.
        """
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)
        mean = rgb.mean()
        std = max(rgb.std(), 1.0 / np.sqrt(rgb.size))
        return (rgb - mean) / std

    @staticmethod
    def _dlib(img: np.ndarray) -> np.ndarray:
        """Dlib: [0, 1] float, RGB order."""
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return rgb.astype(np.float32) / 255.0

    @staticmethod
    def _unit_range(img: np.ndarray) -> np.ndarray:
        """Generic deep learning: [0, 1] float, BGR order."""
        return img.astype(np.float32) / 255.0

    @staticmethod
    def _imagenet(img: np.ndarray) -> np.ndarray:
        """ImageNet statistics normalisation, RGB order."""
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        return (rgb - _IMAGENET_MEAN) / _IMAGENET_STD

    @staticmethod
    def _grayscale(img: np.ndarray) -> np.ndarray:
        """
        Grayscale [0, 1] float.  For traditional ML models (Eigenfaces, HOG, LBP).
        Input may already be grayscale; handles both.
        """
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img
        return gray.astype(np.float32) / 255.0

    @staticmethod
    def _grayscale_eq(img: np.ndarray) -> np.ndarray:
        """Grayscale with CLAHE histogram equalisation then [0, 1] float."""
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        eq = clahe.apply(gray)
        return eq.astype(np.float32) / 255.0

    # ------------------------------------------------------------------
    # Resize helper
    # ------------------------------------------------------------------

    def _maybe_resize(self, img: np.ndarray) -> np.ndarray:
        if self._target_size is None:
            return img
        h, w = img.shape[:2]
        if h == self._target_size and w == self._target_size:
            return img
        return cv2.resize(img, (self._target_size, self._target_size),
                          interpolation=cv2.INTER_LINEAR)


# ---------------------------------------------------------------------------
# Convenience factory: build the correct normalizer for a given model type
# ---------------------------------------------------------------------------

_MODEL_TO_MODE: dict[str, NormalizationMode] = {
    "facenet":      NormalizationMode.FACENET,
    "arcface":      NormalizationMode.ARCFACE,
    "insightface":  NormalizationMode.ARCFACE,
    "dlib_fr":      NormalizationMode.DLIB,
    "eigenfaces":   NormalizationMode.GRAYSCALE_EQ,
    "fisherfaces":  NormalizationMode.GRAYSCALE_EQ,
    "lbph":         NormalizationMode.GRAYSCALE,
    "pca_svm":      NormalizationMode.GRAYSCALE_EQ,
    "hog_svm":      NormalizationMode.GRAYSCALE,
    "knn":          NormalizationMode.GRAYSCALE,
}

_MODEL_TO_SIZE: dict[str, int] = {
    "facenet":      160,
    "arcface":      112,
    "insightface":  112,
    "dlib_fr":      150,
    "eigenfaces":   100,
    "fisherfaces":  100,
    "lbph":         100,
    "pca_svm":      100,
    "hog_svm":      128,
    "knn":          100,
}


def normalizer_for_model(model_name: str) -> Normalizer:
    """
    Return a pre-configured ``Normalizer`` for a given model name.

    Falls back to ``UNIT_RANGE`` + ``112`` for unknown models.
    """
    mode = _MODEL_TO_MODE.get(model_name, NormalizationMode.UNIT_RANGE)
    size = _MODEL_TO_SIZE.get(model_name, 112)
    return Normalizer(mode=mode, target_size=size)

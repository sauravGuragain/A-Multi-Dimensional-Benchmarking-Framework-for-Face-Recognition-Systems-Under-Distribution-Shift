"""
faceeval.models.traditional.lbph
==================================
Local Binary Pattern Histogram (LBPH) face recognizer.

LBPH (Ahonen, Hadid & Pietikäinen 2006)
-----------------------------------------
Divide the face image into a grid of cells.  In each cell, compare each
pixel to its circular neighbours — encoding the comparison outcomes as a
binary string (the Local Binary Pattern).  Concatenate cell histograms into
a single feature vector.

LBPH is robust to monotonic illumination changes and is the most commonly
used classical texture-based face descriptor.  It does not require a training
set for the feature extraction step (histograms are computed from raw pixels)
but does require one for the classifier.

Implementation
--------------
We use scikit-image's ``local_binary_pattern`` rather than OpenCV's
``face.LBPHFaceRecognizer`` so that the feature vector is explicit and
can be stored in the ``FeatureCache``, compared cross-model, and used for
behavioral fingerprinting.

The classifier is a ``LinearSVC`` with probability calibration by default,
mirroring the other traditional models.
"""

from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import Any

import numpy as np

from faceeval.core.registry import register_model
from faceeval.core.types import (
    DistanceMetric,
    EmbeddingVector,
    ModelParadigm,
    PerturbationSpec,
    PredictionResult,
    SubjectID,
    TraditionalModelType,
)
from faceeval.models.traditional.base import TraditionalRecognizerMixin

logger = logging.getLogger(__name__)


class LBPHRecognizer(TraditionalRecognizerMixin):
    """
    LBPH face recognizer backed by a linear SVM classifier.

    Parameters
    ----------
    radius:
        Radius of the circular LBP neighbourhood in pixels.
    n_points:
        Number of neighbours on the circular pattern (typically 8 × radius).
    grid_x, grid_y:
        Number of cell columns / rows the face image is divided into.
        Total histogram length = grid_x × grid_y × (n_points + 2).
    method:
        LBP variant: ``"uniform"`` (default, 59 bins), ``"default"``
        (2^n_points bins), or ``"nri_uniform"`` (non-rotation-invariant).
    svm_C:
        SVM regularisation strength.
    """

    MODEL_NAME      = TraditionalModelType.LBPH.value
    MODEL_TYPE      = TraditionalModelType.LBPH.value
    PARADIGM        = ModelParadigm.TRADITIONAL
    VERSION         = "1.0.0"
    EMBEDDING_DIM   = None          # set after fit (depends on grid + method)
    DISTANCE_METRIC = DistanceMetric.EUCLIDEAN

    def __init__(
        self,
        radius: int = 1,
        n_points: int = 8,
        grid_x: int = 8,
        grid_y: int = 8,
        method: str = "uniform",
        svm_C: float = 1.0,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            radius=radius,
            n_points=n_points,
            grid_x=grid_x,
            grid_y=grid_y,
            method=method,
            svm_C=svm_C,
            **kwargs,
        )
        self._radius   = radius
        self._n_points = n_points
        self._grid_x   = grid_x
        self._grid_y   = grid_y
        self._method   = method
        self._svm_C    = svm_C

    # ------------------------------------------------------------------
    # Feature extraction
    # ------------------------------------------------------------------

    def _compute_lbp_histogram(self, face_gray: np.ndarray) -> np.ndarray:
        """
        Compute the LBPH feature vector for one grayscale face array.

        Divides the face into a (grid_y × grid_x) grid of cells and
        concatenates per-cell normalised histograms into a single 1-D vector.
        """
        from skimage.feature import local_binary_pattern

        # Ensure uint8 grayscale
        if face_gray.dtype != np.uint8:
            face_u8 = (face_gray * 255).clip(0, 255).astype(np.uint8)
        else:
            face_u8 = face_gray

        if len(face_u8.shape) == 3:
            import cv2
            face_u8 = cv2.cvtColor(face_u8, cv2.COLOR_BGR2GRAY)

        h, w = face_u8.shape
        lbp = local_binary_pattern(
            face_u8, self._n_points, self._radius, method=self._method
        )

        # Histogram bins depend on LBP method
        if self._method == "uniform":
            n_bins = self._n_points + 2
        elif self._method == "nri_uniform":
            n_bins = self._n_points * (self._n_points - 1) + 3
        else:
            n_bins = 2 ** self._n_points

        cell_h = h // self._grid_y
        cell_w = w // self._grid_x
        histograms: list[np.ndarray] = []

        for gy in range(self._grid_y):
            for gx in range(self._grid_x):
                y0, y1 = gy * cell_h, (gy + 1) * cell_h
                x0, x1 = gx * cell_w, (gx + 1) * cell_w
                cell = lbp[y0:y1, x0:x1]
                hist, _ = np.histogram(cell.ravel(), bins=n_bins,
                                       range=(0, n_bins), density=True)
                histograms.append(hist)

        return np.concatenate(histograms).astype(np.float32)

    def extract_features(self, X: list[np.ndarray]) -> list[EmbeddingVector]:
        """Compute LBPH histograms without needing a trained classifier."""
        features = [self._compute_lbp_histogram(x) for x in X]
        if features:
            self.EMBEDDING_DIM = len(features[0])
        return [f.tolist() for f in features]

    # ------------------------------------------------------------------
    # Override fit / predict to use custom feature extraction
    # ------------------------------------------------------------------

    def fit(self, X_train: list[np.ndarray], y_train: list[SubjectID]) -> None:
        from sklearn.svm import LinearSVC
        from sklearn.calibration import CalibratedClassifierCV

        self._class_labels = sorted(set(y_train))
        logger.debug("[%s] Extracting LBPH features from %d samples", self.MODEL_NAME, len(X_train))

        X_feat = np.array([self._compute_lbp_histogram(x) for x in X_train])
        self.EMBEDDING_DIM = X_feat.shape[1]

        base = LinearSVC(C=self._svm_C, max_iter=3000)
        self._clf = CalibratedClassifierCV(base, cv=min(3, len(self._class_labels)))
        self._clf.fit(X_feat, y_train)
        self._is_trained = True
        logger.info("[%s] Trained — histogram_dim=%d, %d subjects",
                    self.MODEL_NAME, self.EMBEDDING_DIM, len(self._class_labels))

    def predict(
        self,
        X: list[np.ndarray],
        perturbation_spec: PerturbationSpec | None = None,
    ) -> list[PredictionResult]:
        self._require_trained()
        results: list[PredictionResult] = []
        for i, face in enumerate(X):
            t0 = self._time_ms()
            hist = self._compute_lbp_histogram(face).reshape(1, -1)
            predicted = self._clf.predict(hist)[0]
            proba = self._clf.predict_proba(hist)[0]
            confidence = float(proba.max())
            order = np.argsort(proba)[::-1]
            top_k = [(self._clf.classes_[j], float(proba[j])) for j in order[:5]]
            elapsed = self._time_ms() - t0
            results.append(self._make_prediction(
                image_id=str(i),
                predicted_subject_id=str(predicted),
                confidence=confidence,
                top_k=top_k,
                embedding=hist.flatten().tolist(),
                inference_time_ms=elapsed,
                perturbation_spec=perturbation_spec,
            ))
        return results

    # Override mixin methods not used (we bypass the extractor/flatten path)
    def _build_extractor(self) -> Any:
        return None

    def _build_classifier(self) -> Any:
        return None

    def save(self, path: str | Path) -> None:
        self._require_trained()
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        state = {
            "clf": self._clf,
            "class_labels": self._class_labels,
            "hyperparameters": self._hyperparameters,
            "embedding_dim": self.EMBEDDING_DIM,
        }
        with open(path, "wb") as f:
            pickle.dump(state, f, protocol=pickle.HIGHEST_PROTOCOL)

    def load(self, path: str | Path) -> None:
        with open(path, "rb") as f:
            state = pickle.load(f)
        self._clf = state["clf"]
        self._class_labels = state["class_labels"]
        self._hyperparameters = state["hyperparameters"]
        self.EMBEDDING_DIM = state.get("embedding_dim")
        self._is_trained = True


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

@register_model(
    key=TraditionalModelType.LBPH.value,
    paradigm=ModelParadigm.TRADITIONAL,
    model_type=TraditionalModelType.LBPH.value,
    description="Local Binary Pattern Histogram recognizer with LinearSVC",
)
def _lbph_factory(**kwargs: Any) -> LBPHRecognizer:
    return LBPHRecognizer(**kwargs)

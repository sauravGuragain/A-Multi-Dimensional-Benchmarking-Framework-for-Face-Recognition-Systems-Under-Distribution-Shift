"""
faceeval.models.traditional.hog_svm
=====================================
Histogram of Oriented Gradients (HOG) + SVM and KNN recognizers.

HOG + SVM (Dalal & Triggs 2005 descriptor, face recognition application)
--------------------------------------------------------------------------
HOG captures edge and gradient structure in localised image patches,
making it more robust to uniform illumination changes than raw pixel PCA.
The HOG descriptor is fed to an RBF-kernel SVM classifier.

HOG descriptor parameters
--------------------------
Optimal for 128×128 face images (our default for HOG):
  pixels_per_cell = (8, 8)
  cells_per_block = (2, 2)
  orientations    = 9
  → feature vector length ≈ 2916 for 128×128 (resized from pipeline output)

KNN (k-Nearest Neighbour)
--------------------------
A simple non-parametric baseline that classifies a new face by majority
vote among its k nearest neighbours in the raw pixel or HOG feature space.
Provides a lower-bound reference for classical methods.
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


# ---------------------------------------------------------------------------
# HOG descriptor helper
# ---------------------------------------------------------------------------

def compute_hog_features(
    face: np.ndarray,
    target_size: int = 128,
    orientations: int = 9,
    pixels_per_cell: tuple[int, int] = (8, 8),
    cells_per_block: tuple[int, int] = (2, 2),
) -> np.ndarray:
    """
    Compute a normalised HOG descriptor for one face image.

    Parameters
    ----------
    face:
        float32 or uint8 array, grayscale (H, W) or colour (H, W, 3).
    target_size:
        Resize face to ``(target_size, target_size)`` before extraction.

    Returns
    -------
    np.ndarray  shape (D,)  float32 HOG feature vector.
    """
    import cv2
    from skimage.feature import hog

    # Ensure grayscale uint8
    if len(face.shape) == 3:
        gray = cv2.cvtColor(
            (face * 255).clip(0, 255).astype(np.uint8)
            if face.dtype != np.uint8 else face,
            cv2.COLOR_BGR2GRAY,
        )
    else:
        gray = (face * 255).clip(0, 255).astype(np.uint8) if face.dtype != np.uint8 else face

    # Resize to canonical size
    if gray.shape[0] != target_size or gray.shape[1] != target_size:
        gray = cv2.resize(gray, (target_size, target_size), interpolation=cv2.INTER_LINEAR)

    descriptor = hog(
        gray,
        orientations=orientations,
        pixels_per_cell=pixels_per_cell,
        cells_per_block=cells_per_block,
        block_norm="L2-Hys",
        feature_vector=True,
    )
    return descriptor.astype(np.float32)


# ---------------------------------------------------------------------------
# HOG + SVM
# ---------------------------------------------------------------------------

class HOGSVMRecognizer(TraditionalRecognizerMixin):
    """
    HOG feature extraction + RBF-SVM classifier.

    Parameters
    ----------
    target_size:
        Resize face to this square size before HOG extraction.
    orientations:
        Number of gradient orientation bins.
    pixels_per_cell:
        HOG cell size (height, width) in pixels.
    cells_per_block:
        HOG block size in cells.
    svm_C:
        SVM regularisation.
    svm_gamma:
        RBF kernel width (``"scale"`` recommended).
    """

    MODEL_NAME      = TraditionalModelType.HOG_SVM.value
    MODEL_TYPE      = TraditionalModelType.HOG_SVM.value
    PARADIGM        = ModelParadigm.TRADITIONAL
    VERSION         = "1.0.0"
    EMBEDDING_DIM   = None
    DISTANCE_METRIC = DistanceMetric.EUCLIDEAN

    def __init__(
        self,
        target_size: int = 128,
        orientations: int = 9,
        pixels_per_cell: tuple[int, int] = (8, 8),
        cells_per_block: tuple[int, int] = (2, 2),
        svm_C: float = 10.0,
        svm_gamma: str | float = "scale",
        **kwargs: Any,
    ) -> None:
        super().__init__(
            target_size=target_size,
            orientations=orientations,
            pixels_per_cell=list(pixels_per_cell),
            cells_per_block=list(cells_per_block),
            svm_C=svm_C,
            svm_gamma=svm_gamma,
            **kwargs,
        )
        self._target_size    = target_size
        self._orientations   = orientations
        self._pixels_per_cell = pixels_per_cell
        self._cells_per_block = cells_per_block
        self._svm_C          = svm_C
        self._svm_gamma      = svm_gamma

    def _hog(self, face: np.ndarray) -> np.ndarray:
        return compute_hog_features(
            face,
            target_size=self._target_size,
            orientations=self._orientations,
            pixels_per_cell=self._pixels_per_cell,
            cells_per_block=self._cells_per_block,
        )

    def fit(self, X_train: list[np.ndarray], y_train: list[SubjectID]) -> None:
        from sklearn.svm import SVC

        self._class_labels = sorted(set(y_train))
        logger.debug("[%s] Extracting HOG features from %d samples", self.MODEL_NAME, len(X_train))
        X_feat = np.array([self._hog(x) for x in X_train])
        self.EMBEDDING_DIM = X_feat.shape[1]

        self._clf = SVC(
            C=self._svm_C, kernel="rbf", gamma=self._svm_gamma,
            probability=True, max_iter=5000,
        )
        self._clf.fit(X_feat, y_train)
        self._is_trained = True
        logger.info("[%s] Trained — hog_dim=%d, %d subjects",
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
            feat = self._hog(face).reshape(1, -1)
            predicted = self._clf.predict(feat)[0]
            proba = self._clf.predict_proba(feat)[0]
            confidence = float(proba.max())
            order = np.argsort(proba)[::-1]
            top_k = [(self._clf.classes_[j], float(proba[j])) for j in order[:5]]
            elapsed = self._time_ms() - t0
            results.append(self._make_prediction(
                image_id=str(i),
                predicted_subject_id=str(predicted),
                confidence=confidence,
                top_k=top_k,
                embedding=feat.flatten().tolist(),
                inference_time_ms=elapsed,
                perturbation_spec=perturbation_spec,
            ))
        return results

    def extract_features(self, X: list[np.ndarray]) -> list[EmbeddingVector]:
        return [self._hog(x).tolist() for x in X]

    def _build_extractor(self) -> Any: return None
    def _build_classifier(self) -> Any: return None

    def save(self, path: str | Path) -> None:
        self._require_trained()
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({
                "clf": self._clf, "class_labels": self._class_labels,
                "hyperparameters": self._hyperparameters,
                "embedding_dim": self.EMBEDDING_DIM,
            }, f, protocol=pickle.HIGHEST_PROTOCOL)

    def load(self, path: str | Path) -> None:
        with open(path, "rb") as f:
            state = pickle.load(f)
        self._clf = state["clf"]
        self._class_labels = state["class_labels"]
        self._hyperparameters = state["hyperparameters"]
        self.EMBEDDING_DIM = state.get("embedding_dim")
        self._is_trained = True


# ---------------------------------------------------------------------------
# KNN
# ---------------------------------------------------------------------------

class KNNRecognizer(TraditionalRecognizerMixin):
    """
    k-Nearest Neighbour recognizer on HOG feature space.

    Used as the non-parametric lower-bound baseline.

    Parameters
    ----------
    n_neighbors:
        k for voting.
    metric:
        Distance metric: ``"euclidean"`` or ``"cosine"``.
    weights:
        ``"uniform"`` or ``"distance"`` (inverse-distance weighting).
    use_hog:
        If ``True`` (default), extract HOG features before KNN.
        If ``False``, use raw flattened pixels.
    target_size:
        Resize to this square before feature extraction.
    """

    MODEL_NAME      = TraditionalModelType.KNN.value
    MODEL_TYPE      = TraditionalModelType.KNN.value
    PARADIGM        = ModelParadigm.TRADITIONAL
    VERSION         = "1.0.0"
    EMBEDDING_DIM   = None
    DISTANCE_METRIC = DistanceMetric.EUCLIDEAN

    def __init__(
        self,
        n_neighbors: int = 5,
        metric: str = "euclidean",
        weights: str = "distance",
        use_hog: bool = True,
        target_size: int = 64,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            n_neighbors=n_neighbors,
            metric=metric,
            weights=weights,
            use_hog=use_hog,
            target_size=target_size,
            **kwargs,
        )
        self._k           = n_neighbors
        self._metric      = metric
        self._weights     = weights
        self._use_hog     = use_hog
        self._target_size = target_size

    def _featurize(self, face: np.ndarray) -> np.ndarray:
        if self._use_hog:
            return compute_hog_features(face, target_size=self._target_size)
        import cv2
        if len(face.shape) == 3:
            gray = cv2.cvtColor(
                (face * 255).astype(np.uint8) if face.dtype != np.uint8 else face,
                cv2.COLOR_BGR2GRAY,
            )
        else:
            gray = face
        resized = cv2.resize(gray, (self._target_size, self._target_size))
        return resized.flatten().astype(np.float32)

    def _build_extractor(self) -> Any: return None

    def _build_classifier(self) -> Any:
        from sklearn.neighbors import KNeighborsClassifier
        return KNeighborsClassifier(
            n_neighbors=self._k,
            metric=self._metric,
            weights=self._weights,
        )

    def fit(self, X_train: list[np.ndarray], y_train: list[SubjectID]) -> None:
        self._class_labels = sorted(set(y_train))
        X_feat = np.array([self._featurize(x) for x in X_train])
        self.EMBEDDING_DIM = X_feat.shape[1]
        self._clf = self._build_classifier()
        self._clf.fit(X_feat, y_train)
        self._is_trained = True
        logger.info("[%s] Trained — feat_dim=%d, k=%d, %d subjects",
                    self.MODEL_NAME, self.EMBEDDING_DIM, self._k, len(self._class_labels))

    def predict(
        self,
        X: list[np.ndarray],
        perturbation_spec: PerturbationSpec | None = None,
    ) -> list[PredictionResult]:
        self._require_trained()
        results: list[PredictionResult] = []
        for i, face in enumerate(X):
            t0 = self._time_ms()
            feat = self._featurize(face).reshape(1, -1)
            predicted = self._clf.predict(feat)[0]
            proba = self._clf.predict_proba(feat)[0]
            confidence = float(proba.max())
            order = np.argsort(proba)[::-1]
            top_k = [(self._clf.classes_[j], float(proba[j])) for j in order[:5]]
            elapsed = self._time_ms() - t0
            results.append(self._make_prediction(
                image_id=str(i),
                predicted_subject_id=str(predicted),
                confidence=confidence,
                top_k=top_k,
                embedding=feat.flatten().tolist(),
                inference_time_ms=elapsed,
                perturbation_spec=perturbation_spec,
            ))
        return results

    def extract_features(self, X: list[np.ndarray]) -> list[EmbeddingVector]:
        return [self._featurize(x).tolist() for x in X]

    def save(self, path: str | Path) -> None:
        self._require_trained()
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({
                "clf": self._clf, "class_labels": self._class_labels,
                "hyperparameters": self._hyperparameters,
                "embedding_dim": self.EMBEDDING_DIM,
            }, f, protocol=pickle.HIGHEST_PROTOCOL)

    def load(self, path: str | Path) -> None:
        with open(path, "rb") as f:
            state = pickle.load(f)
        self._clf = state["clf"]
        self._class_labels = state["class_labels"]
        self._hyperparameters = state["hyperparameters"]
        self.EMBEDDING_DIM = state.get("embedding_dim")
        self._is_trained = True


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

@register_model(
    key=TraditionalModelType.HOG_SVM.value,
    paradigm=ModelParadigm.TRADITIONAL,
    model_type=TraditionalModelType.HOG_SVM.value,
    description="HOG descriptor + RBF-SVM classifier",
)
def _hog_svm_factory(**kwargs: Any) -> HOGSVMRecognizer:
    return HOGSVMRecognizer(**kwargs)


@register_model(
    key=TraditionalModelType.KNN.value,
    paradigm=ModelParadigm.TRADITIONAL,
    model_type=TraditionalModelType.KNN.value,
    description="k-Nearest Neighbour on HOG feature space (non-parametric baseline)",
)
def _knn_factory(**kwargs: Any) -> KNNRecognizer:
    return KNNRecognizer(**kwargs)

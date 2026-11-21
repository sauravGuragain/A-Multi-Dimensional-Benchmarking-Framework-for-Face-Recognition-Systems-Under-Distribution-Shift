"""
faceeval.models.traditional.base
==================================
Shared utilities for scikit-learn-backed traditional ML recognizers.

All classical models follow the same pattern:

  1. ``_flatten``  — reshape (H, W[, C]) arrays to 1-D feature vectors.
  2. ``_extract``  — sub-class defined feature transformation (PCA, HOG, …).
  3. ``_classify`` — sklearn classifier (SVM, KNN, or the model itself).

``TraditionalRecognizerMixin`` provides the ``fit`` / ``predict`` /
``save`` / ``load`` scaffolding so concrete subclasses only need to
implement ``_build_extractor``, ``_build_classifier``, and
``extract_features``.
"""

from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import Any

import numpy as np

from faceeval.core.exceptions import ModelNotTrainedError
from faceeval.core.types import (
    DistanceMetric,
    EmbeddingVector,
    ModelParadigm,
    PerturbationSpec,
    PredictionResult,
    SubjectID,
)
from faceeval.models.base import BaseRecognizer

logger = logging.getLogger(__name__)


class TraditionalRecognizerMixin(BaseRecognizer):
    """
    Mixin that wires a scikit-learn extractor + classifier into
    ``BaseRecognizer``'s ``fit`` / ``predict`` / ``save`` / ``load``.

    Concrete subclasses set:
        ``_extractor``  — sklearn transform with ``.fit_transform`` / ``.transform``
        ``_clf``        — sklearn classifier with ``.fit`` / ``.predict_proba``

    Both are set to ``None`` until ``fit`` is called.
    """

    PARADIGM: ModelParadigm = ModelParadigm.TRADITIONAL

    def __init__(self, **hyperparameters: Any) -> None:
        super().__init__(**hyperparameters)
        self._extractor: Any = None
        self._clf: Any = None
        self._random_state: int | None = None      # None = scikit-learn's historical behaviour (the global NumPy stream)

    # ------------------------------------------------------------------
    # Abstract hooks for subclasses
    # ------------------------------------------------------------------

    def _build_extractor(self) -> Any:
        """Return an unfitted sklearn transformer, or None if not needed."""
        return None

    def _build_classifier(self) -> Any:
        """Return an unfitted sklearn classifier."""
        raise NotImplementedError

    # ------------------------------------------------------------------
    # BaseRecognizer interface
    # ------------------------------------------------------------------

    def fit(self, X_train: list[np.ndarray], y_train: list[SubjectID]) -> None:
        self._class_labels = sorted(set(y_train))
        X_flat = self._flatten_batch(X_train)

        self._extractor = self._build_extractor()
        self._clf = self._build_classifier()

        if self._extractor is not None:
            logger.debug("[%s] Fitting extractor on %d samples", self.MODEL_NAME, len(X_flat))
            X_feat = self._extractor.fit_transform(X_flat)
        else:
            X_feat = X_flat

        logger.debug("[%s] Fitting classifier on feature shape %s", self.MODEL_NAME, X_feat.shape)
        self._clf.fit(X_feat, y_train)
        self._is_trained = True
        logger.info("[%s] Training complete — %d subjects, %d samples",
                    self.MODEL_NAME, len(self._class_labels), len(X_train))

    def predict(
        self,
        X: list[np.ndarray],
        perturbation_spec: PerturbationSpec | None = None,
    ) -> list[PredictionResult]:
        self._require_trained()
        X_flat = self._flatten_batch(X)
        X_feat = self._transform(X_flat)

        results: list[PredictionResult] = []
        for i, feat_row in enumerate(X_feat):
            t0 = self._time_ms()
            feat_2d = feat_row.reshape(1, -1)

            predicted = self._clf.predict(feat_2d)[0]
            confidence: float | None = None
            top_k: list[tuple[SubjectID, float]] = []

            if hasattr(self._clf, "predict_proba"):
                proba = self._clf.predict_proba(feat_2d)[0]
                confidence = float(proba.max())
                order = np.argsort(proba)[::-1]
                top_k = [
                    (self._clf.classes_[j], float(proba[j]))
                    for j in order[:5]
                ]
            elif hasattr(self._clf, "decision_function"):
                scores = self._clf.decision_function(feat_2d)[0]
                if scores.ndim == 0:
                    scores = np.array([scores])
                softmax = self._softmax(scores)
                confidence = float(softmax.max())

            elapsed = self._time_ms() - t0
            results.append(self._make_prediction(
                image_id=str(i),
                predicted_subject_id=str(predicted),
                confidence=confidence,
                top_k=top_k,
                embedding=feat_row.tolist(),
                inference_time_ms=elapsed,
                perturbation_spec=perturbation_spec,
            ))
        return results

    def fit_embedding(self, X_train: list[np.ndarray], y_train: list[SubjectID]) -> None:
        """Fit only the feature extractor (no classifier head); stateless models just mark trained."""
        self._class_labels = sorted(set(y_train))
        extractor = self._build_extractor()
        if extractor is not None:
            extractor.fit(self._flatten_batch(X_train))
            self._extractor = extractor
        self._is_trained = True

    def extract_features(self, X: list[np.ndarray]) -> list[EmbeddingVector]:
        self._require_trained()
        X_flat = self._flatten_batch(X)
        X_feat = self._transform(X_flat)
        return [row.tolist() for row in X_feat]

    def save(self, path: str | Path) -> None:
        self._require_trained()
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        state = {
            "extractor": self._extractor,
            "clf": self._clf,
            "class_labels": self._class_labels,
            "hyperparameters": self._hyperparameters,
            "model_name": self.MODEL_NAME,
        }
        with open(path, "wb") as f:
            pickle.dump(state, f, protocol=pickle.HIGHEST_PROTOCOL)
        logger.info("[%s] Model saved to '%s'", self.MODEL_NAME, path)

    def load(self, path: str | Path) -> None:
        with open(path, "rb") as f:
            state = pickle.load(f)
        self._extractor = state["extractor"]
        self._clf = state["clf"]
        self._class_labels = state["class_labels"]
        self._hyperparameters = state["hyperparameters"]
        self._is_trained = True
        logger.info("[%s] Model loaded from '%s'", self.MODEL_NAME, path)

    # ------------------------------------------------------------------
    # Shared helpers
    # ------------------------------------------------------------------

    def _transform(self, X_flat: np.ndarray) -> np.ndarray:
        """Apply extractor transform (if any) to already-flattened data."""
        if self._extractor is not None:
            return self._extractor.transform(X_flat)
        return X_flat

    @staticmethod
    def _flatten_batch(X: list[np.ndarray]) -> np.ndarray:
        """Flatten each (H, W[, C]) array to a 1-D row; return (N, D) matrix."""
        rows = [x.flatten().astype(np.float64) for x in X]
        return np.vstack(rows)

    @staticmethod
    def _softmax(x: np.ndarray) -> np.ndarray:
        e = np.exp(x - x.max())
        return e / e.sum()


CLASSICAL_RNG_GLOBAL = "global_seed"       # historical protocol: scikit-learn draws from the process-global NumPy stream
CLASSICAL_RNG_ISOLATED = "isolated_seed"   # separately versioned protocol: every stochastic estimator gets random_state=seed


def apply_classical_rng(model: Any, config: Any) -> bool:
    """Apply ``config.classical_rng`` to a model. Historical default (``global_seed``): do nothing, return False.

    ``isolated_seed``: set ``random_state = config.random_seed`` on the model's stochastic scikit-learn estimators (PCA's
    randomized solver, LinearSVC, SVC), so a fit no longer depends on how much of the global stream earlier code consumed.
    This CAN change the numbers relative to the historical protocol, which is why the option is part of the config hash."""
    if getattr(config, "classical_rng", CLASSICAL_RNG_GLOBAL) != CLASSICAL_RNG_ISOLATED:
        return False
    if getattr(model, "PARADIGM", None) != ModelParadigm.TRADITIONAL:      # deep models have no scikit-learn estimators
        return False
    model._random_state = int(config.random_seed)
    return True

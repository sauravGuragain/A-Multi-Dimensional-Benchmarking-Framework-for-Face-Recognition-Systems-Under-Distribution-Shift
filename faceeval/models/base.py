from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import numpy as np

from faceeval.core.exceptions import ModelNotTrainedError
from faceeval.core.types import (
    DistanceMetric,
    EmbeddingVector,
    ModelInfo,
    ModelName,
    ModelParadigm,
    PerturbationSpec,
    PredictionResult,
    SubjectID,
)

logger = logging.getLogger(__name__)


class BaseRecognizer(ABC):
    """
    Abstract face recognizer.

    Class variables (override in subclasses)
    -----------------------------------------
    MODEL_NAME      : str             Registered model key.
    PARADIGM        : ModelParadigm   TRADITIONAL or DEEP_LEARNING.
    MODEL_TYPE      : str             TraditionalModelType / DeepModelType value.
    VERSION         : str             Semantic version string.
    EMBEDDING_DIM   : int | None      None for non-embedding models (e.g. LBPH).
    DISTANCE_METRIC : DistanceMetric  Metric used for similarity comparison.
    REQUIRES_GPU    : bool            Whether the model benefits from / requires GPU.
    """

    MODEL_NAME:      str              = "base"
    PARADIGM:        ModelParadigm    = ModelParadigm.TRADITIONAL
    MODEL_TYPE:      str              = "base"
    VERSION:         str              = "0.1.0"
    EMBEDDING_DIM:   int | None       = None
    DISTANCE_METRIC: DistanceMetric   = DistanceMetric.EUCLIDEAN
    REQUIRES_GPU:    bool             = False

    def __init__(self, **hyperparameters: Any) -> None:
        self._hyperparameters: dict[str, Any] = hyperparameters
        self._is_trained: bool = False
        self._class_labels: list[SubjectID] = []
        self._model_size_mb: float | None = None

    # ------------------------------------------------------------------
    # Abstract interface — every subclass must implement these
    # ------------------------------------------------------------------

    @abstractmethod
    def fit(self, X_train: list[np.ndarray], y_train: list[SubjectID]) -> None:
        """
        Train the model on pre-processed face arrays.

        Parameters
        ----------
        X_train:
            List of float32 numpy arrays, each the output of the preprocessing
            pipeline for one image.  Shape per array: (H, W) for grayscale
            traditional models or (H, W, 3) for colour/DL models.
        y_train:
            Subject ID label for each corresponding array in X_train.
        """
        ...

    @abstractmethod
    def predict(
        self,
        X: list[np.ndarray],
        perturbation_spec: PerturbationSpec | None = None,
    ) -> list[PredictionResult]:
        """
        Run inference on pre-processed face arrays.

        Parameters
        ----------
        X:
            List of float32 numpy arrays (same format as X_train).
        perturbation_spec:
            Attached to each ``PredictionResult`` for traceability.
            ``None`` for clean (unperturbed) inference.

        Returns
        -------
        list[PredictionResult]
            One result per input array, in the same order.
        """
        ...

    @abstractmethod
    def extract_features(self, X: list[np.ndarray]) -> list[EmbeddingVector]:
        """
        Extract raw feature vectors without performing classification.

        For embedding models (FaceNet, ArcFace): returns L2-normalised embeddings.
        For feature-vector models (HOG, Eigenfaces): returns the feature vector.
        For direct-comparison models (LBPH): returns the histogram.

        Returns
        -------
        list[EmbeddingVector]
            One vector per input array, length = EMBEDDING_DIM (if defined).
        """
        ...

    @abstractmethod
    def save(self, path: str | Path) -> None:
        """Serialise the trained model to ``path``."""
        ...

    @abstractmethod
    def load(self, path: str | Path) -> None:
        """Deserialise a trained model from ``path``."""
        ...

    # ------------------------------------------------------------------
    # Concrete helpers available to all subclasses
    # ------------------------------------------------------------------

    def get_info(self) -> ModelInfo:
        """Return a ``ModelInfo`` descriptor for this model instance."""
        return ModelInfo(
            name=self.MODEL_NAME,
            paradigm=self.PARADIGM,
            model_type=self.MODEL_TYPE,
            version=self.VERSION,
            embedding_dim=self.EMBEDDING_DIM,
            distance_metric=self.DISTANCE_METRIC,
            requires_gpu=self.REQUIRES_GPU,
            model_size_mb=self._model_size_mb,
            hyperparameters=self._hyperparameters,
        )

    def _require_trained(self) -> None:
        """Raise ``ModelNotTrainedError`` if ``fit`` has not been called."""
        if not self._is_trained:
            raise ModelNotTrainedError(self.MODEL_NAME)

    def _make_prediction(
        self,
        image_id: str,
        predicted_subject_id: SubjectID | None,
        confidence: float | None,
        top_k: list[tuple[SubjectID, float]],
        embedding: EmbeddingVector | None,
        inference_time_ms: float,
        perturbation_spec: PerturbationSpec | None,
    ) -> PredictionResult:
        """Factory helper so subclasses don't repeat boilerplate."""
        return PredictionResult(
            image_id=image_id,
            model_name=self.MODEL_NAME,
            perturbation_spec=perturbation_spec,
            predicted_subject_id=predicted_subject_id,
            confidence=confidence,
            top_k_predictions=top_k,
            embedding=embedding,
            inference_time_ms=inference_time_ms,
        )

    @staticmethod
    def _time_ms() -> float:
        return time.perf_counter() * 1000.0

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def is_trained(self) -> bool:
        return self._is_trained

    @property
    def class_labels(self) -> list[SubjectID]:
        return self._class_labels

    @property
    def hyperparameters(self) -> dict[str, Any]:
        return self._hyperparameters

    def __repr__(self) -> str:
        status = "trained" if self._is_trained else "untrained"
        return (
            f"<{self.__class__.__name__} "
            f"name={self.MODEL_NAME!r} status={status} "
            f"hp={self._hyperparameters}>"
        )

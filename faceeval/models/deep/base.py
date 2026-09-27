"""
faceeval.models.deep.base
==========================
Abstract base class for deep learning face recognizers.

Architecture shared by all DL models
--------------------------------------
All four DL models (FaceNet, ArcFace, InsightFace, Dlib FR) work by:

  1. Passing a preprocessed face through a backbone CNN.
  2. Projecting the final feature map to a fixed-size embedding vector.
  3. L2-normalising the embedding so cosine similarity == dot product.
  4. At inference time, comparing the probe embedding against a gallery of
     per-subject centroid embeddings (computed during ``fit``).

This is fundamentally different from the traditional ML models which learn
a discriminative classifier during training.  DL models learn generalised
embeddings from large external datasets; ``fit`` here only builds the
gallery from the provided training faces — the backbone weights are frozen.

Classification protocol
------------------------
For each probe embedding, compute cosine similarity to every gallery
centroid, take argmax as the predicted class, and use the maximum
similarity value as the confidence score.  ``top_k`` returns the k
highest-similarity classes.

Verification protocol
----------------------
Compare two embeddings directly.  Compute cosine similarity; threshold
at ``decision_threshold`` to decide same/different person.

Calibration note
-----------------
Raw cosine similarity is not a calibrated probability.  The evaluation
engine applies Platt scaling / temperature scaling to convert similarities
to calibrated probabilities, consistent with how traditional models report
``predict_proba`` outputs.
"""

from __future__ import annotations

import logging
import pickle
from abc import abstractmethod
from pathlib import Path
from typing import Any

import numpy as np

from faceeval.core.exceptions import EmbeddingError, ModelNotTrainedError
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


class DeepRecognizerBase(BaseRecognizer):
    """
    Abstract base for CNN embedding-based face recognizers.

    Subclasses must implement:
        ``_load_backbone()``   → load / initialise the pretrained model
        ``_embed(X)``         → forward pass returning (N, D) float32 embeddings

    Parameters
    ----------
    device:
        ``"auto"``, ``"cpu"``, ``"cuda"``, or ``"cuda:N"``.
    decision_threshold:
        Cosine similarity threshold for same/different person decisions.
        Typical values: FaceNet 0.6, ArcFace 0.5, Dlib 0.6.
    """

    PARADIGM:        ModelParadigm  = ModelParadigm.DEEP_LEARNING
    DISTANCE_METRIC: DistanceMetric = DistanceMetric.COSINE
    REQUIRES_GPU:    bool           = False  # runs on CPU; GPU optional

    def __init__(
        self,
        device: str = "auto",
        decision_threshold: float = 0.5,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            device=device,
            decision_threshold=decision_threshold,
            **kwargs,
        )
        self._device_str = device
        self._decision_threshold = decision_threshold
        self._backbone: Any = None
        self._gallery: dict[SubjectID, np.ndarray] = {}  # subject → centroid embedding
        self._gallery_embeddings: dict[SubjectID, list[np.ndarray]] = {}  # for variance

    # ------------------------------------------------------------------
    # Abstract backbone interface
    # ------------------------------------------------------------------

    @abstractmethod
    def _load_backbone(self) -> None:
        """Initialise self._backbone with pre-trained weights."""
        ...

    @abstractmethod
    def _embed(self, X: list[np.ndarray]) -> np.ndarray:
        """
        Forward pass through the backbone.

        Parameters
        ----------
        X:
            List of pre-processed face arrays in the format expected by
            this model (RGB float32 for FaceNet/Dlib, BGR float32 for
            ArcFace/InsightFace).

        Returns
        -------
        np.ndarray  shape (N, embedding_dim)  float32, L2-normalised.
        """
        ...

    # ------------------------------------------------------------------
    # BaseRecognizer interface
    # ------------------------------------------------------------------

    def fit(self, X_train: list[np.ndarray], y_train: list[SubjectID]) -> None:
        """
        Build per-subject gallery centroids from training embeddings.

        The backbone is frozen — only gallery statistics are computed here.
        """
        if self._backbone is None:
            self._load_backbone()

        self._class_labels = sorted(set(y_train))
        logger.debug("[%s] Computing training embeddings for %d images …",
                     self.MODEL_NAME, len(X_train))

        embeddings = self._embed(X_train)   # (N, D)
        if embeddings is None or len(embeddings) == 0:
            raise EmbeddingError(f"[{self.MODEL_NAME}] _embed returned empty result")

        # Accumulate per-subject embeddings
        subject_embeds: dict[SubjectID, list[np.ndarray]] = {}
        for emb, subj in zip(embeddings, y_train):
            subject_embeds.setdefault(subj, []).append(emb)

        # Compute L2-normalised centroids
        self._gallery = {}
        self._gallery_embeddings = subject_embeds
        for subj, embs in subject_embeds.items():
            centroid = np.mean(embs, axis=0).astype(np.float32)
            norm = np.linalg.norm(centroid)
            self._gallery[subj] = centroid / norm if norm > 0 else centroid

        self._is_trained = True
        logger.info("[%s] Gallery built — %d subjects, embedding_dim=%d",
                    self.MODEL_NAME, len(self._gallery), embeddings.shape[1])

    def predict(
        self,
        X: list[np.ndarray],
        perturbation_spec: PerturbationSpec | None = None,
    ) -> list[PredictionResult]:
        self._require_trained()
        if self._backbone is None:
            self._load_backbone()

        embeddings = self._embed(X)   # (N, D)
        gallery_subjects = list(self._gallery.keys())
        gallery_matrix = np.stack([self._gallery[s] for s in gallery_subjects])  # (C, D)

        results: list[PredictionResult] = []
        for i, emb in enumerate(embeddings):
            t0 = self._time_ms()
            # Cosine similarity to all gallery centroids
            sims = gallery_matrix @ emb   # (C,) — already L2-normalised
            order = np.argsort(sims)[::-1]
            predicted = gallery_subjects[order[0]]
            confidence = float((sims[order[0]] + 1.0) / 2.0)  # map [-1,1] → [0,1]
            top_k = [
                (gallery_subjects[j], float((sims[j] + 1.0) / 2.0))
                for j in order[:5]
            ]
            elapsed = self._time_ms() - t0
            results.append(self._make_prediction(
                image_id=str(i),
                predicted_subject_id=predicted,
                confidence=confidence,
                top_k=top_k,
                embedding=emb.tolist(),
                inference_time_ms=elapsed,
                perturbation_spec=perturbation_spec,
            ))
        return results

    def verify(
        self,
        emb_a: np.ndarray,
        emb_b: np.ndarray,
    ) -> tuple[bool, float]:
        """
        Face verification: decide if two embeddings belong to the same person.

        Returns
        -------
        (is_same_person, cosine_similarity)
        """
        if self._backbone is None:
            self._load_backbone()
        sim = float(np.dot(emb_a / (np.linalg.norm(emb_a) + 1e-8),
                           emb_b / (np.linalg.norm(emb_b) + 1e-8)))
        return sim >= self._decision_threshold, sim

    def extract_features(self, X: list[np.ndarray]) -> list[EmbeddingVector]:
        if self._backbone is None:
            self._load_backbone()
        embeddings = self._embed(X)
        return [row.tolist() for row in embeddings]

    def save(self, path: str | Path) -> None:
        self._require_trained()
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        state = {
            "gallery": self._gallery,
            "gallery_embeddings": self._gallery_embeddings,
            "class_labels": self._class_labels,
            "hyperparameters": self._hyperparameters,
            "model_name": self.MODEL_NAME,
            "embedding_dim": self.EMBEDDING_DIM,
        }
        with open(path, "wb") as f:
            pickle.dump(state, f, protocol=pickle.HIGHEST_PROTOCOL)
        logger.info("[%s] Gallery saved to '%s'", self.MODEL_NAME, path)

    def load(self, path: str | Path) -> None:
        with open(path, "rb") as f:
            state = pickle.load(f)
        self._gallery = state["gallery"]
        self._gallery_embeddings = state.get("gallery_embeddings", {})
        self._class_labels = state["class_labels"]
        self._hyperparameters = state["hyperparameters"]
        self.EMBEDDING_DIM = state.get("embedding_dim")
        self._is_trained = True
        logger.info("[%s] Gallery loaded from '%s' (%d subjects)",
                    self.MODEL_NAME, path, len(self._gallery))

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _l2_normalize(x: np.ndarray) -> np.ndarray:
        """L2-normalise a (D,) or (N, D) array along the last axis."""
        norm = np.linalg.norm(x, axis=-1, keepdims=True)
        return x / np.where(norm > 0, norm, 1.0)

    def _resolve_device(self) -> str:
        """Resolve 'auto' to 'cuda' or 'cpu' depending on availability."""
        if self._device_str != "auto":
            return self._device_str
        try:
            import torch
            return "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            return "cpu"

    @property
    def gallery(self) -> dict[SubjectID, np.ndarray]:
        return self._gallery

    @property
    def decision_threshold(self) -> float:
        return self._decision_threshold

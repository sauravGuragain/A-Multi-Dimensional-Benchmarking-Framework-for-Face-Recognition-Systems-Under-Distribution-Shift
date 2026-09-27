"""
faceeval.models.deep.arcface
==============================
ArcFace face recognizer using the InsightFace ONNX runtime backend.

ArcFace (Deng et al., 2019)
-----------------------------
ArcFace introduces an Additive Angular Margin Loss that enforces a geometric
margin on the hypersphere, producing more discriminative embeddings than
softmax cross-entropy or triplet loss.

State-of-the-art results on LFW (99.83%), CFP-FP (98.27%), AgeDB-30 (98.28%).

Backend
-------
We use the ``insightface`` library's ONNX-based inference backend
(``buffalo_l`` or ``buffalo_sc`` model packs).  This avoids having to
re-implement the ResNet-100 backbone while staying framework-agnostic
(ONNX Runtime runs on CPU, CUDA, and Apple Silicon without code changes).

Model pack ``buffalo_l``  — large, ResNet100, 512-dim, highest accuracy.
Model pack ``buffalo_sc`` — small, MobileNet, 512-dim, faster on CPU.

Input contract
--------------
ArcFace expects:
    * BGR channel order.
    * (pixel − 127.5) / 128.0  → float32 in [-1, 1].
    * Shape: (N, 3, 112, 112) NCHW.

The preprocessing pipeline's ``NormalizationMode.ARCFACE`` handles this.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from faceeval.core.registry import register_model
from faceeval.core.types import DeepModelType, ModelParadigm
from faceeval.models.deep.base import DeepRecognizerBase

logger = logging.getLogger(__name__)

_INSIGHTFACE_AVAILABLE: bool | None = None


def _check_insightface() -> bool:
    global _INSIGHTFACE_AVAILABLE
    if _INSIGHTFACE_AVAILABLE is None:
        try:
            import insightface  # noqa: F401
            _INSIGHTFACE_AVAILABLE = True
        except ImportError:
            _INSIGHTFACE_AVAILABLE = False
    return _INSIGHTFACE_AVAILABLE


class ArcFaceRecognizer(DeepRecognizerBase):
    """
    ArcFace face recognizer backed by InsightFace ONNX inference.

    Parameters
    ----------
    model_pack:
        InsightFace model pack name: ``"buffalo_l"`` (default, most accurate)
        or ``"buffalo_sc"`` (smaller, faster CPU inference).
    providers:
        ONNX Runtime execution providers in priority order.
        Default: ``["CUDAExecutionProvider", "CPUExecutionProvider"]``.
    batch_size:
        Batch size for ONNX inference.
    decision_threshold:
        Cosine similarity threshold for verification.
    """

    MODEL_NAME      = DeepModelType.ARCFACE.value
    MODEL_TYPE      = DeepModelType.ARCFACE.value
    PARADIGM        = ModelParadigm.DEEP_LEARNING
    VERSION         = "1.0.0"
    EMBEDDING_DIM   = 512
    REQUIRES_GPU    = False

    def __init__(
        self,
        model_pack: str = "buffalo_l",
        providers: list[str] | None = None,
        batch_size: int = 32,
        device: str = "auto",
        decision_threshold: float = 0.5,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            device=device,
            decision_threshold=decision_threshold,
            model_pack=model_pack,
            batch_size=batch_size,
            **kwargs,
        )
        self._model_pack = model_pack
        self._providers = providers or ["CUDAExecutionProvider", "CPUExecutionProvider"]
        self._batch_size = batch_size
        self._recognizer: Any = None   # insightface FaceAnalysis or direct model

    def _load_backbone(self) -> None:
        if not _check_insightface():
            logger.warning(
                "insightface is not installed. "
                "Install with: pip install insightface onnxruntime\n"
                "Running in stub mode."
            )
            self._backbone = _ArcFaceStub(self.EMBEDDING_DIM)
            return

        import insightface
        from insightface.model_zoo import get_model

        logger.info("[%s] Loading ArcFace model pack '%s' …",
                    self.MODEL_NAME, self._model_pack)
        try:
            # Use the recognition model directly for embedding extraction
            app = insightface.app.FaceAnalysis(
                name=self._model_pack,
                providers=self._providers,
            )
            app.prepare(ctx_id=0 if "CUDA" in self._providers[0] else -1)
            self._backbone = _InsightFaceWrapper(app)
        except Exception as exc:
            logger.warning(
                "[%s] Could not load '%s': %s. Falling back to stub.",
                self.MODEL_NAME, self._model_pack, exc,
            )
            self._backbone = _ArcFaceStub(self.EMBEDDING_DIM)

        logger.info("[%s] ArcFace backbone ready.", self.MODEL_NAME)

    def _embed(self, X: list[np.ndarray]) -> np.ndarray:
        if self._backbone is None:
            self._load_backbone()

        if isinstance(self._backbone, (_ArcFaceStub, _InsightFaceWrapper)):
            return self._backbone.embed(X)

        return _ArcFaceStub(self.EMBEDDING_DIM).embed(X)


class _InsightFaceWrapper:
    """Wraps insightface FaceAnalysis to produce embedding vectors."""

    def __init__(self, app: Any) -> None:
        self._app = app

    def embed(self, X: list[np.ndarray]) -> np.ndarray:
        import cv2
        embeddings: list[np.ndarray] = []
        for face_arr in X:
            # Convert float32 normalised → uint8 BGR for insightface detection
            if face_arr.dtype != np.uint8:
                bgr = (face_arr * 128.0 + 127.5).clip(0, 255).astype(np.uint8)
            else:
                bgr = face_arr

            faces = self._app.get(bgr)
            if faces:
                emb = faces[0].embedding.astype(np.float32)
            else:
                # Fallback: zero embedding (will lower similarity scores)
                emb = np.zeros(512, dtype=np.float32)

            norm = np.linalg.norm(emb)
            embeddings.append(emb / norm if norm > 0 else emb)

        return np.array(embeddings, dtype=np.float32)


class _ArcFaceStub:
    """Deterministic stub when insightface is not installed."""

    def __init__(self, dim: int = 512) -> None:
        self._dim = dim

    def embed(self, X: list[np.ndarray]) -> np.ndarray:
        results = []
        for face in X:
            seed = int(np.abs(face).sum() * 137) % (2 ** 31)
            rng = np.random.default_rng(seed)
            v = rng.standard_normal(self._dim).astype(np.float32)
            norm = np.linalg.norm(v)
            results.append(v / norm if norm > 0 else v)
        return np.array(results, dtype=np.float32)


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

@register_model(
    key=DeepModelType.ARCFACE.value,
    paradigm=ModelParadigm.DEEP_LEARNING,
    model_type=DeepModelType.ARCFACE.value,
    description=(
        "ArcFace ResNet100 (512-dim embedding, buffalo_l pack). "
        "Requires: pip install insightface onnxruntime"
    ),
)
def _arcface_factory(**kwargs: Any) -> ArcFaceRecognizer:
    return ArcFaceRecognizer(**kwargs)

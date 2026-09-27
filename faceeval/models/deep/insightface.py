"""
faceeval.models.deep.insightface
==================================
InsightFace recognizer (AdaFace / alternative buffalo packs) and Dlib FR.

InsightFace (separate from ArcFace)
-------------------------------------
While ArcFaceRecognizer uses the buffalo_l pack (ResNet-100 + ArcFace loss),
InsightFaceRecognizer exposes the same library but with:
    * ``antelopev2`` pack — ResNet-50 backbone, faster inference.
    * ``buffalo_s``  pack — lightweight MobileNet backbone.

This lets the thesis benchmark compare backbone architectures and model
sizes independently of the loss function.

Dlib Face Recognition
-----------------------
Dlib's face recognition module uses a ResNet-29 model trained with metric
learning (Davis King, 2017) to produce 128-dimensional face embeddings.
It has been trained on a proprietary ~3M-image dataset and achieves
99.38% accuracy on LFW.

Dlib is:
    * Fully CPU-based (no GPU required).
    * Self-contained (no ONNX Runtime, no PyTorch).
    * The lightest DL model in the benchmark (~22 MB).

Input contract (Dlib)
---------------------
    * RGB float64 array in [0, 1], shape (150, 150, 3).
    * Aligned face chip produced by dlib.get_face_chip().
    * The preprocessing pipeline's NormalizationMode.DLIB produces this.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from faceeval.core.registry import register_model
from faceeval.core.types import DeepModelType, ModelParadigm
from faceeval.models.deep.base import DeepRecognizerBase

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# InsightFace (alternative packs)
# ---------------------------------------------------------------------------

class InsightFaceRecognizer(DeepRecognizerBase):
    """
    InsightFace recognizer using antelopev2 or buffalo_s model packs.

    Shares the same ONNX infrastructure as ArcFaceRecognizer but exposes
    different backbone options for benchmarking.

    Parameters
    ----------
    model_pack:
        ``"antelopev2"`` (default, ResNet-50) or ``"buffalo_s"``
        (MobileNet, fastest CPU inference).
    providers:
        ONNX Runtime execution providers.
    batch_size:
        Forward-pass batch size.
    decision_threshold:
        Cosine similarity threshold for verification.
    """

    MODEL_NAME    = DeepModelType.INSIGHTFACE.value
    MODEL_TYPE    = DeepModelType.INSIGHTFACE.value
    PARADIGM      = ModelParadigm.DEEP_LEARNING
    VERSION       = "1.0.0"
    EMBEDDING_DIM = 512
    REQUIRES_GPU  = False

    def __init__(
        self,
        model_pack: str = "antelopev2",
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

    def _load_backbone(self) -> None:
        try:
            import insightface
            logger.info("[%s] Loading InsightFace pack '%s' …",
                        self.MODEL_NAME, self._model_pack)
            app = insightface.app.FaceAnalysis(
                name=self._model_pack,
                providers=self._providers,
            )
            app.prepare(ctx_id=0 if "CUDA" in self._providers[0] else -1)
            self._backbone = _InsightFaceEmbedder(app, self.EMBEDDING_DIM)
        except Exception as exc:
            logger.warning(
                "[%s] InsightFace load failed (%s). Using stub.", self.MODEL_NAME, exc
            )
            self._backbone = _EmbeddingStub(self.EMBEDDING_DIM, salt=42)

    def _embed(self, X: list[np.ndarray]) -> np.ndarray:
        if self._backbone is None:
            self._load_backbone()
        return self._backbone.embed(X)


class _InsightFaceEmbedder:
    def __init__(self, app: Any, dim: int) -> None:
        self._app = app
        self._dim = dim

    def embed(self, X: list[np.ndarray]) -> np.ndarray:
        embeddings: list[np.ndarray] = []
        for face_arr in X:
            if face_arr.dtype != np.uint8:
                bgr = (face_arr * 128.0 + 127.5).clip(0, 255).astype(np.uint8)
            else:
                bgr = face_arr
            faces = self._app.get(bgr)
            emb = faces[0].embedding.astype(np.float32) if faces else np.zeros(self._dim, dtype=np.float32)
            norm = np.linalg.norm(emb)
            embeddings.append(emb / norm if norm > 0 else emb)
        return np.array(embeddings, dtype=np.float32)


# ---------------------------------------------------------------------------
# Dlib Face Recognition
# ---------------------------------------------------------------------------

class DlibFaceRecognizer(DeepRecognizerBase):
    """
    Dlib ResNet-29 face recognizer producing 128-dimensional embeddings.

    Requires:
        pip install dlib cmake

    Model weights are downloaded automatically on first use to
    ``~/.cache/dlib/``.  The file ``dlib_face_recognition_resnet_model_v1.dat``
    is ~22 MB.

    Parameters
    ----------
    model_path:
        Override the default weights path.  Leave as ``None`` to use
        the standard dlib model downloader.
    decision_threshold:
        Euclidean distance threshold.  Dlib recommends 0.6; cosine
        similarity 0.5 is the default here since we L2-normalise.
    """

    MODEL_NAME    = DeepModelType.DLIB_FR.value
    MODEL_TYPE    = DeepModelType.DLIB_FR.value
    PARADIGM      = ModelParadigm.DEEP_LEARNING
    VERSION       = "1.0.0"
    EMBEDDING_DIM = 128
    REQUIRES_GPU  = False

    def __init__(
        self,
        model_path: str | None = None,
        device: str = "cpu",
        decision_threshold: float = 0.5,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            device=device,
            decision_threshold=decision_threshold,
            model_path=model_path,
            **kwargs,
        )
        self._model_path = model_path
        self._face_rec_model: Any = None
        self._shape_predictor: Any = None

    def _load_backbone(self) -> None:
        try:
            import dlib
            self._load_dlib_models(dlib)
            self._backbone = _DlibEmbedder(self._face_rec_model)
            logger.info("[%s] Dlib face recognition model loaded.", self.MODEL_NAME)
        except ImportError:
            logger.warning(
                "[%s] dlib not installed (pip install dlib). Using stub.",
                self.MODEL_NAME,
            )
            self._backbone = _EmbeddingStub(self.EMBEDDING_DIM, salt=99)
        except Exception as exc:
            logger.warning(
                "[%s] Dlib model load failed: %s. Using stub.", self.MODEL_NAME, exc
            )
            self._backbone = _EmbeddingStub(self.EMBEDDING_DIM, salt=99)

    def _load_dlib_models(self, dlib: Any) -> None:
        """
        Load dlib face recognition model weights.

        Looks in these locations in order:
        1. ``self._model_path`` if provided.
        2. ``~/.cache/dlib/dlib_face_recognition_resnet_model_v1.dat``.
        3. Download via dlib.load_model (requires internet access).
        """
        import os
        cache_dir = os.path.expanduser("~/.cache/dlib")
        default_path = os.path.join(
            cache_dir, "dlib_face_recognition_resnet_model_v1.dat"
        )
        model_path = self._model_path or default_path

        if os.path.exists(model_path):
            self._face_rec_model = dlib.face_recognition_model_v1(model_path)
        else:
            # Attempt to download using dlib's built-in downloader
            os.makedirs(cache_dir, exist_ok=True)
            try:
                import urllib.request
                url = (
                    "http://dlib.net/files/"
                    "dlib_face_recognition_resnet_model_v1.dat.bz2"
                )
                bz2_path = default_path + ".bz2"
                logger.info("[%s] Downloading dlib model from %s …", self.MODEL_NAME, url)
                urllib.request.urlretrieve(url, bz2_path)
                import bz2
                with bz2.open(bz2_path) as src, open(default_path, "wb") as dst:
                    dst.write(src.read())
                os.remove(bz2_path)
                self._face_rec_model = dlib.face_recognition_model_v1(default_path)
            except Exception as exc:
                raise RuntimeError(
                    f"Could not load or download dlib face recognition model: {exc}"
                ) from exc

    def _embed(self, X: list[np.ndarray]) -> np.ndarray:
        if self._backbone is None:
            self._load_backbone()
        return self._backbone.embed(X)


class _DlibEmbedder:
    """Wraps dlib face_recognition_model_v1 to extract embeddings."""

    def __init__(self, model: Any) -> None:
        self._model = model

    def embed(self, X: list[np.ndarray]) -> np.ndarray:
        import dlib
        embeddings: list[np.ndarray] = []
        for face_arr in X:
            # Dlib expects RGB uint8 matrix
            if face_arr.dtype != np.uint8:
                # float32 [0,1] RGB
                rgb_u8 = (face_arr.clip(0, 1) * 255).astype(np.uint8)
            else:
                rgb_u8 = face_arr

            if len(rgb_u8.shape) == 2:
                # Grayscale → RGB
                import cv2
                rgb_u8 = cv2.cvtColor(rgb_u8, cv2.COLOR_GRAY2RGB)

            # dlib expects the image as a dlib.matrix or numpy array (H,W,3) uint8 RGB
            descriptor = self._model.compute_face_descriptor(
                rgb_u8, num_jitters=0
            )
            emb = np.array(descriptor, dtype=np.float32)
            norm = np.linalg.norm(emb)
            embeddings.append(emb / norm if norm > 0 else emb)

        return np.array(embeddings, dtype=np.float32)


# ---------------------------------------------------------------------------
# Shared stub
# ---------------------------------------------------------------------------

class _EmbeddingStub:
    """
    Deterministic stub used when optional dependencies are missing.
    Different ``salt`` values ensure stubs for different models produce
    different (but still consistent) embeddings.
    """

    def __init__(self, dim: int, salt: int = 0) -> None:
        self._dim = dim
        self._salt = salt

    def embed(self, X: list[np.ndarray]) -> np.ndarray:
        results: list[np.ndarray] = []
        for face in X:
            seed = (int(np.abs(face).sum() * 31 + self._salt)) % (2 ** 31)
            rng = np.random.default_rng(seed)
            v = rng.standard_normal(self._dim).astype(np.float32)
            norm = np.linalg.norm(v)
            results.append(v / norm if norm > 0 else v)
        return np.array(results, dtype=np.float32)


# ---------------------------------------------------------------------------
# Registrations
# ---------------------------------------------------------------------------

@register_model(
    key=DeepModelType.INSIGHTFACE.value,
    paradigm=ModelParadigm.DEEP_LEARNING,
    model_type=DeepModelType.INSIGHTFACE.value,
    description=(
        "InsightFace antelopev2 (ResNet-50, 512-dim). "
        "Requires: pip install insightface onnxruntime"
    ),
)
def _insightface_factory(**kwargs: Any) -> InsightFaceRecognizer:
    return InsightFaceRecognizer(**kwargs)


@register_model(
    key=DeepModelType.DLIB_FR.value,
    paradigm=ModelParadigm.DEEP_LEARNING,
    model_type=DeepModelType.DLIB_FR.value,
    description=(
        "Dlib ResNet-29 face recognizer (128-dim, CPU-only). "
        "Requires: pip install dlib cmake"
    ),
)
def _dlib_factory(**kwargs: Any) -> DlibFaceRecognizer:
    return DlibFaceRecognizer(**kwargs)

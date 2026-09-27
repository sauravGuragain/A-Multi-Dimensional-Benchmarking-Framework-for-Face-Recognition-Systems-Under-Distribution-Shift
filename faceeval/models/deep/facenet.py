"""
faceeval.models.deep.facenet
==============================
FaceNet face recognizer using the ``facenet-pytorch`` library.

FaceNet (Schroff, Kalenichenko & Philbin, 2015)
------------------------------------------------
FaceNet trains a deep CNN with a triplet loss to embed faces into a
128-dimensional (or 512-dim InceptionResNetV1) Euclidean space where
distances directly correspond to face similarity.

The ``facenet-pytorch`` library (Tim Esler) provides two pre-trained
InceptionResNetV1 checkpoints:
    * ``"vggface2"``  — trained on VGGFace2, stronger for identity tasks.
    * ``"casia-webface"`` — trained on CASIA-WebFace.

Both are downloaded automatically on first use and cached in
``~/.cache/torch/checkpoints/``.

Input contract
--------------
FaceNet expects:
    * RGB channel order (not BGR).
    * Per-image standardisation: (x − mean(x)) / std(x).
    * Shape: (N, 3, H, W) PyTorch tensor, H = W = 160.

The preprocessing pipeline's ``NormalizationMode.FACENET`` mode handles
this.  The pipeline passes float32 arrays of shape (160, 160, 3) in RGB
order (already channel-swapped from BGR by the normalizer).
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from faceeval.core.registry import register_model
from faceeval.core.types import DeepModelType, ModelParadigm
from faceeval.models.deep.base import DeepRecognizerBase

logger = logging.getLogger(__name__)

_FACENET_AVAILABLE: bool | None = None


def _check_facenet() -> bool:
    global _FACENET_AVAILABLE
    if _FACENET_AVAILABLE is None:
        try:
            import facenet_pytorch  # noqa: F401
            _FACENET_AVAILABLE = True
        except ImportError:
            _FACENET_AVAILABLE = False
    return _FACENET_AVAILABLE


class FaceNetRecognizer(DeepRecognizerBase):
    """
    FaceNet face recognizer backed by InceptionResNetV1.

    Parameters
    ----------
    pretrained:
        Pre-trained weights to use: ``"vggface2"`` (default) or
        ``"casia-webface"``.
    embedding_dim:
        Output embedding size.  512 for InceptionResNetV1 (default).
    device:
        ``"auto"``, ``"cpu"``, or ``"cuda[:N]"``.
    decision_threshold:
        Cosine similarity threshold for verification decisions.
    batch_size:
        Number of images to forward through the backbone at once.
    """

    MODEL_NAME      = DeepModelType.FACENET.value
    MODEL_TYPE      = DeepModelType.FACENET.value
    PARADIGM        = ModelParadigm.DEEP_LEARNING
    VERSION         = "1.0.0"
    EMBEDDING_DIM   = 512
    REQUIRES_GPU    = False

    def __init__(
        self,
        pretrained: str = "vggface2",
        embedding_dim: int = 512,
        device: str = "auto",
        decision_threshold: float = 0.6,
        batch_size: int = 32,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            device=device,
            decision_threshold=decision_threshold,
            pretrained=pretrained,
            embedding_dim=embedding_dim,
            batch_size=batch_size,
            **kwargs,
        )
        self.EMBEDDING_DIM   = embedding_dim
        self._pretrained     = pretrained
        self._batch_size     = batch_size

    # ------------------------------------------------------------------
    # DeepRecognizerBase interface
    # ------------------------------------------------------------------

    def _load_backbone(self) -> None:
        if not _check_facenet():
            logger.warning(
                "facenet-pytorch is not installed. "
                "Install with: pip install facenet-pytorch\n"
                "Running in stub mode — embeddings will be random unit vectors."
            )
            self._backbone = _FaceNetStub(self.EMBEDDING_DIM)
            return

        import torch
        from facenet_pytorch import InceptionResnetV1

        device = self._resolve_device()
        logger.info("[%s] Loading backbone (pretrained=%s, device=%s) …",
                    self.MODEL_NAME, self._pretrained, device)
        self._backbone = InceptionResnetV1(
            pretrained=self._pretrained,
            classify=False,
        ).eval().to(device)
        self._device_str = device
        logger.info("[%s] Backbone loaded.", self.MODEL_NAME)

    def _embed(self, X: list[np.ndarray]) -> np.ndarray:
        """
        Embed a list of face arrays through InceptionResNetV1.

        Input arrays are expected to be float32 RGB (160, 160, 3) from
        ``NormalizationMode.FACENET`` — already per-image standardised and
        in RGB order.
        """
        if self._backbone is None:
            self._load_backbone()

        # Stub path — no PyTorch needed
        if isinstance(self._backbone, _FaceNetStub):
            return self._backbone.embed(X)

        import torch

        device = self._resolve_device()
        all_embeddings: list[np.ndarray] = []

        for i in range(0, len(X), self._batch_size):
            batch = X[i: i + self._batch_size]
            # (H, W, 3) → (N, 3, H, W)
            tensors = []
            for face in batch:
                if face.shape[-1] == 3:
                    t = np.transpose(face, (2, 0, 1))   # HWC → CHW
                else:
                    t = face
                tensors.append(t)

            batch_tensor = torch.from_numpy(
                np.stack(tensors).astype(np.float32)
            ).to(device)

            with torch.no_grad():
                embs = self._backbone(batch_tensor).cpu().numpy()

            # L2-normalise
            all_embeddings.append(self._l2_normalize(embs))

        return np.vstack(all_embeddings).astype(np.float32)


class _FaceNetStub:
    """
    Deterministic stub used when facenet-pytorch is not installed.

    Produces normalised random embeddings seeded from the pixel sum of each
    input image so the same image always yields the same stub embedding.
    This allows all framework tests to run without the optional dependency.
    """

    def __init__(self, dim: int = 512) -> None:
        self._dim = dim

    def embed(self, X: list[np.ndarray]) -> np.ndarray:
        results = []
        for face in X:
            seed = int(np.abs(face).sum()) % (2 ** 31)
            rng = np.random.default_rng(seed)
            v = rng.standard_normal(self._dim).astype(np.float32)
            norm = np.linalg.norm(v)
            results.append(v / norm if norm > 0 else v)
        return np.array(results, dtype=np.float32)


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

@register_model(
    key=DeepModelType.FACENET.value,
    paradigm=ModelParadigm.DEEP_LEARNING,
    model_type=DeepModelType.FACENET.value,
    description=(
        "FaceNet InceptionResNetV1 (512-dim embedding, VGGFace2 pretrained). "
        "Requires: pip install facenet-pytorch"
    ),
)
def _facenet_factory(**kwargs: Any) -> FaceNetRecognizer:
    return FaceNetRecognizer(**kwargs)

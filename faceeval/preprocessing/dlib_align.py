"""
faceeval.preprocessing.dlib_align
=================================
The alignment the pretrained dlib ResNet (``dlib_face_recognition_resnet_model_v1``) was built for.

The network consumes a 150x150 face CHIP produced by dlib's own geometry: dlib's HOG frontal-face detector, then the
5-point landmark model, then ``dlib.get_face_chip(image, landmarks, size=150, padding=0.25)``. The framework's generic
:class:`~faceeval.preprocessing.aligner.FaceAligner` instead maps (pseudo-)landmarks onto the ArcFace template scaled to
150 px, which frames the face differently, so its output is not what the network was trained on.

``DlibChipAligner`` reproduces the dlib procedure inside the framework's aligner interface:

* deterministic: dlib's HOG detector, the landmark model and ``get_face_chip`` contain no randomness;
* face selection: the largest bounding box (ties: higher detector score, then smaller left edge);
* no face found -> ``align`` returns ``None`` -> the image fails like any other detector failure (zero embedding,
  scored, counted), exactly as for ArcFace. There is no centre-crop fallback;
* the framework detector is bypassed: the image is searched in full by dlib.

If dlib itself cannot be imported (the model is then a placeholder STUB anyway) the aligner falls back to the generic
aligner and says so in :meth:`describe` and in every chip's ``alignment_method``; it never does so silently when dlib is
installed (a missing or corrupt landmark file raises).
"""
from __future__ import annotations

import logging
from typing import Any

import cv2
import numpy as np

from faceeval.models.deep import dlib_weights as W
from faceeval.preprocessing.aligner import AlignedFace, FaceAligner
from faceeval.preprocessing.detector import FaceBoundingBox

logger = logging.getLogger(__name__)

CHIP_SIZE = 150
PADDING = 0.25
UPSAMPLE = 1
METHOD = "dlib_5pt_face_chip"
LEGACY_FALLBACK_METHOD = "affine_fixed_landmarks(dlib_unavailable_fallback)"


def dlib_available() -> bool:
    try:
        import dlib  # noqa: F401
        return True
    except ImportError:
        return False


def select_face(rects: list, scores: list) -> int:
    """Index of the face to align: largest area; ties by higher score, then smaller left edge. Deterministic."""
    if not rects:
        raise ValueError("no rectangles")
    keys = [(-(r.width() * r.height()), -float(scores[i]), r.left(), r.top(), i) for i, r in enumerate(rects)]
    return min(keys)[-1]


class DlibChipAligner(FaceAligner):
    def __init__(self, output_size: int = CHIP_SIZE, padding: float = PADDING, upsample: int = UPSAMPLE,
                 predictor_path: str | None = None, allow_download: bool = True) -> None:
        if output_size != CHIP_SIZE:
            raise ValueError(f"the dlib recognition network takes {CHIP_SIZE}px chips, not {output_size}")
        super().__init__(output_size=output_size)
        self._padding, self._upsample = float(padding), int(upsample)
        self._predictor_path, self._allow_download = predictor_path, allow_download
        self._detector: Any = None
        self._predictor: Any = None
        self._dlib: Any = None

    # ----------------------------------------------------------------- loading
    def _load(self) -> bool:
        if self._detector is not None:
            return True
        try:
            import dlib
        except ImportError:
            return False
        path = W.resolve(W.SHAPE_PREDICTOR_5, self._predictor_path, self._allow_download)   # verified, or raises
        self._dlib = dlib
        self._detector = dlib.get_frontal_face_detector()
        self._predictor = dlib.shape_predictor(str(path))
        return True

    @property
    def method(self) -> str:
        return METHOD if dlib_available() else LEGACY_FALLBACK_METHOD

    # --------------------------------------------------------------------- API
    def align(self, image_bgr: np.ndarray, result: Any = None) -> AlignedFace | None:
        if not self._load():                                              # dlib not importable: placeholder environment
            logger.warning("dlib is not importable: using the generic aligner (placeholder environment only).")
            face = super().align(image_bgr, result) if result is not None else None
            if face is not None:
                face.alignment_method = LEGACY_FALLBACK_METHOD
            return face
        rgb = np.ascontiguousarray(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))
        rects, scores, _ = self._detector.run(rgb, self._upsample, 0.0)
        if len(rects) == 0:
            return None
        i = select_face(list(rects), list(scores))
        rect = rects[i]
        shape = self._predictor(rgb, rect)
        chip_rgb = self._dlib.get_face_chip(rgb, shape, size=self._size, padding=self._padding)
        chip_bgr = np.ascontiguousarray(cv2.cvtColor(np.asarray(chip_rgb), cv2.COLOR_RGB2BGR))
        box = FaceBoundingBox(int(rect.left()), int(rect.top()), int(rect.width()), int(rect.height()), float(scores[i]))
        return AlignedFace(image=chip_bgr, source_path=getattr(result, "image_path", ""), bounding_box=box,
                           landmarks_used=None, transform_matrix=None, alignment_method=METHOD)

    def align_all(self, image_bgr: np.ndarray, result: Any = None) -> list:
        face = self.align(image_bgr, result)
        return [] if face is None else [face]

    # --------------------------------------------------------------- provenance
    def describe(self) -> dict[str, Any]:
        info: dict[str, Any] = {
            "alignment": self.method, "chip_size": self._size, "padding": self._padding,
            "detector": f"dlib.get_frontal_face_detector (HOG), upsample={self._upsample}",
            "face_selection": "largest bounding box; ties: higher score, then smaller left edge",
            "no_face_policy": "alignment returns None -> image fails (zero embedding); no crop fallback",
            "landmark_model": W.SHAPE_PREDICTOR_5.name, "landmark_model_sha256_pinned": W.SHAPE_PREDICTOR_5.sha256,
            "weights_source": f"{W.SOURCE_REPO}@{W.SOURCE_COMMIT}", "dlib_version": "NOT_RECORDED"}
        try:
            import dlib
            info["dlib_version"] = str(dlib.__version__)
        except ImportError:
            pass
        return info

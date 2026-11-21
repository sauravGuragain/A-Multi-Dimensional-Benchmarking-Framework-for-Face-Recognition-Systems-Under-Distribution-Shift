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
* NO FACE FOUND: the experiment runner treats an aligner that returns ``None`` as a fatal error for the whole model, so
  returning ``None`` would abort a run at the first miss. Instead the landmark model is run on a fixed centred square
  (``fallback_box_fraction`` of the shorter side) and the chip is labelled ``dlib_5pt_face_chip(centre_box_fallback)``.
  Every such image is COUNTED (:meth:`stats`) and the share is recorded in the run provenance. These images are NOT
  failed embeddings: end-to-end failure accounting for dlib_fr is therefore not comparable with ArcFace's (a detector
  miss there is a zero embedding scored as a non-match). If more than ``MAX_NO_FACE_FRACTION`` of the images need the
  fallback the run is refused (see ``faceeval.orchestration.verification``);
* the framework detector is bypassed: the image is searched in full by dlib;
* MISSING / CORRUPT MODEL FILES are a ``FatalPreprocessingError`` (never a per-image failure);
* dlib not importable (a placeholder STUB environment): the generic aligner is used and every chip and
  :meth:`describe` say so (``LEGACY_FALLBACK_METHOD``), warned ONCE. With ``strict=True`` (``require_real_models``)
  :meth:`ensure_ready` refuses instead.
"""
from __future__ import annotations

import logging
from typing import Any

import cv2
import numpy as np

from faceeval.models.deep import dlib_weights as W
from faceeval.preprocessing.aligner import AlignedFace, FaceAligner, FatalPreprocessingError
from faceeval.preprocessing.detector import FaceBoundingBox

logger = logging.getLogger(__name__)

CHIP_SIZE = 150
PADDING = 0.25
UPSAMPLE = 1
METHOD = "dlib_5pt_face_chip"
LEGACY_FALLBACK_METHOD = "affine_fixed_landmarks(dlib_unavailable_fallback)"
CENTRE_BOX_METHOD = "dlib_5pt_face_chip(centre_box_fallback)"
FALLBACK_BOX_FRACTION = 0.5
MAX_NO_FACE_FRACTION = 0.5          # above this share the run is refused; recorded and warned about from WARN_NO_FACE_FRACTION
WARN_NO_FACE_FRACTION = 0.10


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
                 predictor_path: str | None = None, allow_download: bool = True,
                 fallback_box_fraction: float = FALLBACK_BOX_FRACTION) -> None:
        if output_size != CHIP_SIZE:
            raise ValueError(f"the dlib recognition network takes {CHIP_SIZE}px chips, not {output_size}")
        super().__init__(output_size=output_size)
        self._padding, self._upsample = float(padding), int(upsample)
        self._predictor_path, self._allow_download = predictor_path, allow_download
        self._detector: Any = None
        self._predictor: Any = None
        self._dlib: Any = None
        self._box_fraction = float(fallback_box_fraction)
        self._n_images = self._n_detected = self._n_centre_box = self._n_legacy = 0
        self._warned_missing_dlib = False

    # ----------------------------------------------------------------- loading
    def _load(self) -> bool:
        if self._detector is not None:
            return True
        try:
            import dlib
        except ImportError:
            return False
        try:
            path = W.resolve(W.SHAPE_PREDICTOR_5, self._predictor_path, self._allow_download)   # verified, or raises
            detector = dlib.get_frontal_face_detector()
            predictor = dlib.shape_predictor(str(path))
        except Exception as exc:
            raise FatalPreprocessingError(
                f"dlib_chip alignment cannot start: {type(exc).__name__}: {exc}. No image was processed; this is an "
                f"installation/weights problem, not an image failure.") from exc
        self._dlib, self._detector, self._predictor = dlib, detector, predictor
        return True

    def ensure_ready(self, strict: bool = False) -> bool:
        """Load and verify the landmark model NOW (fail fast, before any image is processed). Returns True when real dlib
        alignment is available. If dlib is not importable: ``strict`` raises, otherwise returns False (the generic
        aligner will be used and labelled as such)."""
        if self._load():
            return True
        if strict:
            raise FatalPreprocessingError("alignment 'dlib_chip' was requested with require_real_models=true but dlib is "
                                          "not importable; refusing to substitute the generic aligner.")
        return False

    def stats(self) -> dict[str, Any]:
        """Counters since construction: images aligned, found by dlib, aligned via the centre-box fallback, aligned by the
        generic aligner (dlib missing)."""
        n = max(self._n_images, 1)
        return {"n_images": self._n_images, "n_detected_by_dlib": self._n_detected,
                "n_centre_box_fallback": self._n_centre_box, "n_generic_aligner_dlib_missing": self._n_legacy,
                "centre_box_fallback_fraction": self._n_centre_box / n if self._n_images else 0.0,
                "max_allowed_fraction": MAX_NO_FACE_FRACTION}

    @property
    def method(self) -> str:
        return METHOD if dlib_available() else LEGACY_FALLBACK_METHOD

    # --------------------------------------------------------------------- API
    def align(self, image_bgr: np.ndarray, result: Any = None) -> AlignedFace | None:
        if not self._load():                                              # dlib not importable: placeholder environment
            if not self._warned_missing_dlib:
                logger.warning("dlib is not importable: using the generic aligner (placeholder environment only).")
                self._warned_missing_dlib = True
            if result is None:
                return None
            face = super().align(image_bgr, result)
            if face is not None:
                face.alignment_method = LEGACY_FALLBACK_METHOD
                self._n_images += 1
                self._n_legacy += 1
            return face
        rgb = np.ascontiguousarray(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))
        rects, scores, _ = self._detector.run(rgb, self._upsample, 0.0)
        self._n_images += 1
        if len(rects) == 0:
            h, w = rgb.shape[:2]
            side = max(8, int(round(self._box_fraction * min(h, w))))
            x0, y0 = (w - side) // 2, (h - side) // 2
            rect, score, method = self._dlib.rectangle(x0, y0, x0 + side, y0 + side), 0.0, CENTRE_BOX_METHOD
            self._n_centre_box += 1
        else:
            i = select_face(list(rects), list(scores))
            rect, score, method = rects[i], float(scores[i]), METHOD
            self._n_detected += 1
        shape = self._predictor(rgb, rect)
        chip_rgb = self._dlib.get_face_chip(rgb, shape, size=self._size, padding=self._padding)
        chip_bgr = np.ascontiguousarray(cv2.cvtColor(np.asarray(chip_rgb), cv2.COLOR_RGB2BGR))
        box = FaceBoundingBox(int(rect.left()), int(rect.top()), int(rect.width()), int(rect.height()), score)
        return AlignedFace(image=chip_bgr, source_path=getattr(result, "image_path", ""), bounding_box=box,
                           landmarks_used=None, transform_matrix=None, alignment_method=method)

    def align_all(self, image_bgr: np.ndarray, result: Any = None) -> list:
        face = self.align(image_bgr, result)
        return [] if face is None else [face]

    # --------------------------------------------------------------- provenance
    def describe(self) -> dict[str, Any]:
        info: dict[str, Any] = {
            "alignment": self.method, "chip_size": self._size, "padding": self._padding,
            "detector": f"dlib.get_frontal_face_detector (HOG), upsample={self._upsample}",
            "face_selection": "largest bounding box; ties: higher score, then smaller left edge",
            "requested_alignment": "dlib_chip",
            "no_face_policy": (f"centre square of {self._box_fraction:g} x shorter side goes to the landmark model; chip labelled "
                               f"'{CENTRE_BOX_METHOD}'; counted in stats(); run refused above {MAX_NO_FACE_FRACTION:g}"),
            "landmark_model": W.SHAPE_PREDICTOR_5.name, "landmark_model_sha256_pinned": W.SHAPE_PREDICTOR_5.sha256,
            "weights_source": f"{W.SOURCE_REPO}@{W.SOURCE_COMMIT}", "dlib_version": "NOT_RECORDED"}
        try:
            import dlib
            info["dlib_version"] = str(dlib.__version__)
        except ImportError:
            pass
        return info

"""
faceeval.preprocessing.aligner
================================
Affine face alignment to a canonical reference pose.

Alignment approach
------------------
Five-point landmark alignment is the standard for deep-learning face
recognition pipelines (FaceNet, ArcFace).  Given detected landmarks
(left eye, right eye, nose, mouth left, mouth right), an affine transform
is computed that maps them to a fixed reference template sized to the target
output resolution.  The result is a tightly cropped, pose-normalised face
chip ready for feature extraction.

Reference templates
-------------------
``TEMPLATE_112`` is the 112×112 canonical template used by ArcFace/InsightFace
(coordinates from the ArcFace paper).
``TEMPLATE_160`` is the 160×160 template used by FaceNet.

When landmarks are unavailable (detection backend returned ``None``), a
geometry-only crop (bounding-box expansion + resize) is used as fallback.

Design notes
------------
* All arithmetic uses float32 to avoid integer truncation in landmark math.
* The output is always a uint8 BGR numpy array of exactly ``output_size``.
* Alignment is stateless — ``FaceAligner`` holds only parameters, no state.
"""

from __future__ import annotations

import logging
from typing import Any

import cv2
import numpy as np

from faceeval.core.exceptions import AlignmentError
from faceeval.preprocessing.detector import DetectionResult, FaceLandmarks

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Canonical landmark reference templates
# ---------------------------------------------------------------------------

# ArcFace / InsightFace canonical template — 112×112
TEMPLATE_112 = np.array([
    [38.2946, 51.6963],   # left eye
    [73.5318, 51.5014],   # right eye
    [56.0252, 71.7366],   # nose tip
    [41.5493, 92.3655],   # mouth left
    [70.7299, 92.2041],   # mouth right
], dtype=np.float32)

# FaceNet canonical template — 160×160 (scaled from 112)
_SCALE_FACTOR = 160.0 / 112.0
TEMPLATE_160 = (TEMPLATE_112 * _SCALE_FACTOR).astype(np.float32)

# Standard 250×250 for classical models (LFW native)
TEMPLATE_250 = (TEMPLATE_112 * (250.0 / 112.0)).astype(np.float32)

_TEMPLATES: dict[int, np.ndarray] = {
    112: TEMPLATE_112,
    160: TEMPLATE_160,
    250: TEMPLATE_250,
}

# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------

class AlignedFace:
    """
    One aligned face chip with provenance metadata.

    Attributes
    ----------
    image:
        uint8 BGR numpy array of shape ``(output_size, output_size, 3)``.
    source_path:
        Original image path.
    bounding_box:
        Detection bounding box (in original image coordinates).
    landmarks_used:
        ``FaceLandmarks`` used for alignment, or ``None`` if geometry fallback.
    transform_matrix:
        2×3 affine transform applied to produce this chip.
    alignment_method:
        ``"affine_5pt"`` | ``"geometry_crop"``
    """

    def __init__(
        self,
        image: np.ndarray,
        source_path: str,
        bounding_box: Any,
        landmarks_used: FaceLandmarks | None,
        transform_matrix: np.ndarray | None,
        alignment_method: str,
    ) -> None:
        self.image = image
        self.source_path = source_path
        self.bounding_box = bounding_box
        self.landmarks_used = landmarks_used
        self.transform_matrix = transform_matrix
        self.alignment_method = alignment_method

    @property
    def shape(self) -> tuple[int, int, int]:
        return self.image.shape  # type: ignore[return-value]

    def to_rgb(self) -> np.ndarray:
        """Return a copy of the face chip in RGB channel order."""
        return cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)

    def to_float32(self) -> np.ndarray:
        """Return a float32 copy normalised to [0, 1]."""
        return self.image.astype(np.float32) / 255.0


# ---------------------------------------------------------------------------
# Aligner
# ---------------------------------------------------------------------------

class FaceAligner:
    """
    Affine face aligner producing canonical-pose face chips.

    Parameters
    ----------
    output_size:
        Side length of the square output chip in pixels.
        Common values: 112 (ArcFace), 160 (FaceNet), 250 (LFW native).
        For sizes without a pre-built template, the 112-template is scaled.
    margin_fraction:
        Extra context to include around the tight bounding box when using
        the geometry-fallback path (landmark-free crops).  0.2 = 20% margin
        on each side.
    """

    def __init__(
        self,
        output_size: int = 112,
        margin_fraction: float = 0.2,
    ) -> None:
        self._size = output_size
        self._margin = margin_fraction
        self._template = self._get_template(output_size)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def align(self, image_bgr: np.ndarray, result: DetectionResult) -> AlignedFace | None:
        """
        Align the primary (highest-confidence) face in ``result``.

        Returns
        -------
        AlignedFace or None
            ``None`` if ``result`` contains no detected faces.
        """
        if result.num_faces == 0:
            return None
        box = result.primary_box
        landmarks = result.primary_landmarks
        assert box is not None  # guaranteed by num_faces > 0

        if landmarks is not None:
            return self._affine_align(image_bgr, box, landmarks, result.image_path)
        return self._geometry_crop(image_bgr, box, result.image_path)

    def align_all(
        self, image_bgr: np.ndarray, result: DetectionResult
    ) -> list[AlignedFace]:
        """Align all detected faces, returning one ``AlignedFace`` per face."""
        aligned: list[AlignedFace] = []
        for box, lm in zip(result.faces, result.landmarks):
            if lm is not None:
                chip = self._affine_align(image_bgr, box, lm, result.image_path)
            else:
                chip = self._geometry_crop(image_bgr, box, result.image_path)
            aligned.append(chip)
        return aligned

    # ------------------------------------------------------------------
    # Alignment implementations
    # ------------------------------------------------------------------

    def _affine_align(
        self,
        image_bgr: np.ndarray,
        box: Any,
        landmarks: FaceLandmarks,
        source_path: str,
    ) -> AlignedFace:
        """
        Compute and apply an affine transform from detected landmarks
        to the canonical template.

        Uses ``cv2.estimateAffinePartial2D`` (rotation + uniform scale +
        translation, 4 DOF) which is more stable than a full 6-DOF affine
        when landmarks are noisy.
        """
        src_pts = landmarks.as_array()  # (5, 2) float32
        dst_pts = self._template        # (5, 2) float32

        M, _ = cv2.estimateAffinePartial2D(src_pts, dst_pts, method=cv2.LMEDS)
        if M is None:
            logger.debug(
                "estimateAffinePartial2D failed for '%s'; falling back to geometry crop.",
                source_path,
            )
            return self._geometry_crop(image_bgr, box, source_path)

        chip = cv2.warpAffine(
            image_bgr, M, (self._size, self._size),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT_101,
        )
        return AlignedFace(
            image=chip,
            source_path=source_path,
            bounding_box=box,
            landmarks_used=landmarks,
            transform_matrix=M,
            alignment_method="affine_5pt",
        )

    def _geometry_crop(
        self,
        image_bgr: np.ndarray,
        box: Any,
        source_path: str,
    ) -> AlignedFace:
        """
        Fallback alignment: expand bounding box by ``margin_fraction``,
        clamp to image bounds, and resize to ``output_size``.
        """
        h_img, w_img = image_bgr.shape[:2]
        margin_x = int(box.w * self._margin)
        margin_y = int(box.h * self._margin)

        x1 = max(0, box.x - margin_x)
        y1 = max(0, box.y - margin_y)
        x2 = min(w_img, box.x2 + margin_x)
        y2 = min(h_img, box.y2 + margin_y)

        if x2 <= x1 or y2 <= y1:
            raise AlignmentError(
                f"Degenerate bounding box after margin expansion for '{source_path}': "
                f"({x1},{y1})-({x2},{y2})"
            )

        crop = image_bgr[y1:y2, x1:x2]
        chip = cv2.resize(crop, (self._size, self._size), interpolation=cv2.INTER_LINEAR)

        return AlignedFace(
            image=chip,
            source_path=source_path,
            bounding_box=box,
            landmarks_used=None,
            transform_matrix=None,
            alignment_method="geometry_crop",
        )

    # ------------------------------------------------------------------
    # Template management
    # ------------------------------------------------------------------

    @staticmethod
    def _get_template(size: int) -> np.ndarray:
        if size in _TEMPLATES:
            return _TEMPLATES[size]
        # Scale from the 112 template for arbitrary sizes
        scale = size / 112.0
        return (TEMPLATE_112 * scale).astype(np.float32)

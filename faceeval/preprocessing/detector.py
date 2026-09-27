"""
faceeval.preprocessing.detector
================================
Face detection with a pluggable multi-backend architecture.

Backends (in priority order)
-----------------------------
1. ``opencv_dnn``  — OpenCV Caffe DNN detector (fast, good accuracy, CPU-friendly).
2. ``haar``        — OpenCV Haar cascade (lightest, lowest accuracy, always available).
3. ``dlib_hog``    — Dlib HOG + SVM detector (accurate, pure CPU, no OpenCV DNN needed).

Design notes
------------
* ``FaceDetector`` tries backends in the order specified at construction.
  If a backend's model weights are unavailable, it is skipped gracefully.
* All backends return the same ``DetectionResult`` dataclass so downstream
  code never branches on which backend was used.
* A ``DetectionResult`` with ``num_faces == 0`` is a valid (non-exception)
  outcome — the caller decides whether to raise ``FaceDetectionError``.
* Detection is intentionally separate from alignment: this module only finds
  bounding boxes and landmark points.  Pixel transformation happens in
  ``aligner.py``.
* For the thesis benchmark, most images are already tightly cropped faces
  (LFW, VGGFace2).  Detection still runs to locate the canonical face region
  and extract landmarks for alignment.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from faceeval.core.exceptions import FaceDetectionError, PreprocessingError

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------

@dataclass
class FaceBoundingBox:
    """Axis-aligned bounding box for one detected face."""
    x: int          # left
    y: int          # top
    w: int          # width
    h: int          # height
    confidence: float = 1.0

    @property
    def x2(self) -> int:
        return self.x + self.w

    @property
    def y2(self) -> int:
        return self.y + self.h

    @property
    def area(self) -> int:
        return self.w * self.h

    def to_dict(self) -> dict[str, Any]:
        return {"x": self.x, "y": self.y, "w": self.w, "h": self.h,
                "confidence": self.confidence}


@dataclass
class FaceLandmarks:
    """
    Five-point facial landmarks: left eye, right eye, nose tip,
    left mouth corner, right mouth corner.

    Coordinates are (x, y) pixel positions in the original image space.
    """
    left_eye:          tuple[float, float]
    right_eye:         tuple[float, float]
    nose:              tuple[float, float]
    mouth_left:        tuple[float, float]
    mouth_right:       tuple[float, float]

    def as_array(self) -> np.ndarray:
        """Return (5, 2) float32 array of landmark coordinates."""
        return np.array([
            self.left_eye, self.right_eye, self.nose,
            self.mouth_left, self.mouth_right,
        ], dtype=np.float32)

    def to_dict(self) -> dict[str, Any]:
        return {
            "left_eye": self.left_eye, "right_eye": self.right_eye,
            "nose": self.nose, "mouth_left": self.mouth_left,
            "mouth_right": self.mouth_right,
        }


@dataclass
class DetectionResult:
    """
    Complete detection output for one image.

    ``faces`` is ordered by descending confidence.  The primary face
    (index 0) is used by the preprocessing pipeline.
    """
    image_path: str
    image_h: int
    image_w: int
    faces: list[FaceBoundingBox] = field(default_factory=list)
    landmarks: list[FaceLandmarks | None] = field(default_factory=list)
    backend_used: str = ""
    detection_time_ms: float = 0.0

    @property
    def num_faces(self) -> int:
        return len(self.faces)

    @property
    def primary_box(self) -> FaceBoundingBox | None:
        return self.faces[0] if self.faces else None

    @property
    def primary_landmarks(self) -> FaceLandmarks | None:
        return self.landmarks[0] if self.landmarks else None

    def to_dict(self) -> dict[str, Any]:
        return {
            "image_path": self.image_path,
            "num_faces": self.num_faces,
            "backend_used": self.backend_used,
            "detection_time_ms": self.detection_time_ms,
            "faces": [f.to_dict() for f in self.faces],
        }


# ---------------------------------------------------------------------------
# Backend base class
# ---------------------------------------------------------------------------

class _DetectorBackend:
    name: str = "base"

    def is_available(self) -> bool:
        return True

    def detect(self, image_bgr: np.ndarray, image_path: str) -> DetectionResult:
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Backend: OpenCV DNN (Caffe ResNet-SSD)
# ---------------------------------------------------------------------------

# Standard paths where the model files may live (in order of preference).
_DNN_PROTO_CANDIDATES = [
    "models/deploy.prototxt",
    "/usr/share/opencv4/haarcascades/deploy.prototxt",
]
_DNN_WEIGHTS_CANDIDATES = [
    "models/res10_300x300_ssd_iter_140000.caffemodel",
    "/usr/share/opencv4/res10_300x300_ssd_iter_140000.caffemodel",
]


class _OpenCVDNNBackend(_DetectorBackend):
    name = "opencv_dnn"

    def __init__(self) -> None:
        self._net: Any = None
        self._loaded = False

    def is_available(self) -> bool:
        return self._try_load()

    def detect(self, image_bgr: np.ndarray, image_path: str) -> DetectionResult:
        import time
        if not self._try_load() or self._net is None:
            return DetectionResult(image_path, image_bgr.shape[0], image_bgr.shape[1])

        t0 = time.perf_counter()
        h, w = image_bgr.shape[:2]
        blob = cv2.dnn.blobFromImage(
            cv2.resize(image_bgr, (300, 300)), 1.0,
            (300, 300), (104.0, 177.0, 123.0),
        )
        self._net.setInput(blob)
        detections = self._net.forward()

        faces: list[FaceBoundingBox] = []
        for i in range(detections.shape[2]):
            conf = float(detections[0, 0, i, 2])
            if conf < 0.5:
                continue
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            x1, y1, x2, y2 = box.astype(int)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            if x2 > x1 and y2 > y1:
                faces.append(FaceBoundingBox(x1, y1, x2 - x1, y2 - y1, conf))

        faces.sort(key=lambda f: f.confidence, reverse=True)
        elapsed = (time.perf_counter() - t0) * 1000
        return DetectionResult(
            image_path=image_path, image_h=h, image_w=w,
            faces=faces, landmarks=[None] * len(faces),
            backend_used=self.name, detection_time_ms=elapsed,
        )

    def _try_load(self) -> bool:
        if self._loaded:
            return self._net is not None
        self._loaded = True
        for proto in _DNN_PROTO_CANDIDATES:
            for weights in _DNN_WEIGHTS_CANDIDATES:
                if Path(proto).exists() and Path(weights).exists():
                    try:
                        self._net = cv2.dnn.readNetFromCaffe(proto, weights)
                        logger.debug("OpenCV DNN model loaded from '%s'", weights)
                        return True
                    except Exception as e:
                        logger.debug("Failed to load DNN model: %s", e)
        return False


# ---------------------------------------------------------------------------
# Backend: OpenCV Haar Cascade
# ---------------------------------------------------------------------------

class _HaarBackend(_DetectorBackend):
    name = "haar"

    def __init__(self) -> None:
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        self._cascade = cv2.CascadeClassifier(cascade_path)

    def is_available(self) -> bool:
        return not self._cascade.empty()

    def detect(self, image_bgr: np.ndarray, image_path: str) -> DetectionResult:
        import time
        t0 = time.perf_counter()
        h, w = image_bgr.shape[:2]
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        rects = self._cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
        )
        faces: list[FaceBoundingBox] = []
        if len(rects) > 0:
            for (x, y, fw, fh) in rects:
                faces.append(FaceBoundingBox(int(x), int(y), int(fw), int(fh), 1.0))
            # Sort largest-first as a proxy for primary face
            faces.sort(key=lambda f: f.area, reverse=True)

        elapsed = (time.perf_counter() - t0) * 1000
        return DetectionResult(
            image_path=image_path, image_h=h, image_w=w,
            faces=faces, landmarks=[None] * len(faces),
            backend_used=self.name, detection_time_ms=elapsed,
        )


# ---------------------------------------------------------------------------
# Backend: Dlib HOG
# ---------------------------------------------------------------------------

class _DlibHOGBackend(_DetectorBackend):
    name = "dlib_hog"

    def __init__(self) -> None:
        self._detector: Any = None
        self._predictor: Any = None

    def is_available(self) -> bool:
        try:
            import dlib  # noqa: F401
            return True
        except ImportError:
            return False

    def detect(self, image_bgr: np.ndarray, image_path: str) -> DetectionResult:
        import time
        try:
            import dlib
        except ImportError:
            return DetectionResult(image_path, image_bgr.shape[0], image_bgr.shape[1])

        if self._detector is None:
            self._detector = dlib.get_frontal_face_detector()

        t0 = time.perf_counter()
        h, w = image_bgr.shape[:2]
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        dets, scores, _ = self._detector.run(rgb, 1)

        faces: list[FaceBoundingBox] = []
        for det, score in zip(dets, scores):
            x1, y1 = max(0, det.left()), max(0, det.top())
            x2, y2 = min(w, det.right()), min(h, det.bottom())
            if x2 > x1 and y2 > y1:
                faces.append(FaceBoundingBox(x1, y1, x2 - x1, y2 - y1, float(score)))

        faces.sort(key=lambda f: f.confidence, reverse=True)
        elapsed = (time.perf_counter() - t0) * 1000
        return DetectionResult(
            image_path=image_path, image_h=h, image_w=w,
            faces=faces, landmarks=[None] * len(faces),
            backend_used=self.name, detection_time_ms=elapsed,
        )


# ---------------------------------------------------------------------------
# Whole-image fallback (for pre-cropped datasets like LFW)
# ---------------------------------------------------------------------------

class _WholeImageBackend(_DetectorBackend):
    """
    Fallback that treats the entire image as the face region.

    Used when real detection fails on pre-cropped datasets (LFW, VGGFace2)
    where the face fills most of the frame anyway.  Landmark positions are
    estimated from fixed fractions of the image size.
    """
    name = "whole_image"

    def detect(self, image_bgr: np.ndarray, image_path: str) -> DetectionResult:
        h, w = image_bgr.shape[:2]
        box = FaceBoundingBox(0, 0, w, h, 0.5)
        # Approximate 5-point landmarks from image fractions
        landmarks = FaceLandmarks(
            left_eye=(w * 0.35, h * 0.37),
            right_eye=(w * 0.65, h * 0.37),
            nose=(w * 0.50, h * 0.55),
            mouth_left=(w * 0.35, h * 0.72),
            mouth_right=(w * 0.65, h * 0.72),
        )
        return DetectionResult(
            image_path=image_path, image_h=h, image_w=w,
            faces=[box], landmarks=[landmarks],
            backend_used=self.name, detection_time_ms=0.0,
        )


# ---------------------------------------------------------------------------
# Public detector class
# ---------------------------------------------------------------------------

class FaceDetector:
    """
    Multi-backend face detector with automatic fallback.

    Parameters
    ----------
    backends:
        Ordered list of backend names to try.  First available backend wins.
        Valid names: ``"opencv_dnn"``, ``"haar"``, ``"dlib_hog"``,
        ``"whole_image"``.
    raise_on_no_face:
        If ``True``, raises ``FaceDetectionError`` when no face is found.
        If ``False`` (default), returns a ``DetectionResult`` with zero faces.
    min_confidence:
        Minimum detection confidence threshold (0–1).  Detections below
        this threshold are discarded.
    """

    _BACKEND_MAP: dict[str, type[_DetectorBackend]] = {
        "opencv_dnn": _OpenCVDNNBackend,
        "haar": _HaarBackend,
        "dlib_hog": _DlibHOGBackend,
        "whole_image": _WholeImageBackend,
    }

    def __init__(
        self,
        backends: list[str] | None = None,
        raise_on_no_face: bool = False,
        min_confidence: float = 0.5,
    ) -> None:
        self._backends_requested = backends or ["opencv_dnn", "haar", "whole_image"]
        self._raise_on_no_face = raise_on_no_face
        self._min_confidence = min_confidence
        self._active_backends: list[_DetectorBackend] = self._build_backends()

    def detect(self, image_bgr: np.ndarray, image_path: str = "") -> DetectionResult:
        """
        Detect faces in ``image_bgr`` (OpenCV BGR format, uint8).

        Returns the first backend's result that contains at least one face.
        Falls through all backends before returning zero-face result.
        """
        for backend in self._active_backends:
            result = backend.detect(image_bgr, image_path)
            # Filter by confidence
            if result.faces:
                result.faces = [
                    f for f in result.faces if f.confidence >= self._min_confidence
                ]
                result.landmarks = result.landmarks[:len(result.faces)]
            if result.num_faces > 0:
                return result

        # All backends exhausted — no face found
        h, w = image_bgr.shape[:2]
        empty = DetectionResult(
            image_path=image_path, image_h=h, image_w=w,
            faces=[], landmarks=[], backend_used="none",
        )
        if self._raise_on_no_face:
            raise FaceDetectionError(image_path, self._active_backends[-1].name)
        return empty

    def detect_from_path(self, image_path: str) -> DetectionResult:
        """Load an image from disk and detect faces."""
        img = cv2.imread(image_path)
        if img is None:
            raise PreprocessingError(f"Could not read image: '{image_path}'")
        return self.detect(img, image_path)

    @property
    def active_backend_names(self) -> list[str]:
        return [b.name for b in self._active_backends]

    def _build_backends(self) -> list[_DetectorBackend]:
        active: list[_DetectorBackend] = []
        for name in self._backends_requested:
            cls = self._BACKEND_MAP.get(name)
            if cls is None:
                logger.warning("Unknown detector backend '%s'; skipping.", name)
                continue
            backend = cls()
            if backend.is_available():
                active.append(backend)
                logger.debug("Detector backend '%s' is available.", name)
            else:
                logger.debug("Detector backend '%s' not available; skipping.", name)
        if not active:
            logger.warning(
                "No requested detector backends are available. "
                "Falling back to whole_image mode."
            )
            active.append(_WholeImageBackend())
        return active

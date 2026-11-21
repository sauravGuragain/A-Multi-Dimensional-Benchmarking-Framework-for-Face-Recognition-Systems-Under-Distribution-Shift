"""
faceeval.models.deep.dlib_weights
=================================
Pinned, checksum-verified access to the two pretrained dlib model files used by ``dlib_fr``:

* ``dlib_face_recognition_resnet_model_v1.dat``  - the 128-d face recognition network;
* ``shape_predictor_5_face_landmarks.dat``       - the 5-point landmark model dlib's own face chip is built from.

Trust and integrity
-------------------
Both files come from the model repository of dlib's author (``github.com/davisking/dlib-models``, CC0 1.0 licence) at
the pinned commit :data:`SOURCE_COMMIT`, over HTTPS. The old plain-HTTP ``dlib.net`` download is no longer used.
The SHA-256 values below were computed from files fetched from that commit and cross-checked against the Git blob ids
that the Git protocol reports for the same commit (see :data:`GIT_BLOB_IDS`). This is trust-on-first-use of the
author's repository: the pinned values detect corruption, truncation and any later substitution, but a checksum cannot
prove the upstream file was the intended one. A cached or user-supplied file whose SHA-256 differs is REFUSED, never used.
"""
from __future__ import annotations

import bz2
import hashlib
import logging
import os
import tempfile
import urllib.request
from pathlib import Path
from typing import NamedTuple

logger = logging.getLogger(__name__)

SOURCE_REPO = "https://github.com/davisking/dlib-models"
SOURCE_COMMIT = "fd81b6308a6a73d4ce08859eb2f4b628a21e27a2"
_RAW = "https://raw.githubusercontent.com/davisking/dlib-models/" + SOURCE_COMMIT + "/"
NO_DOWNLOAD_ENV = "FACEEVAL_NO_DOWNLOAD"


class WeightSpec(NamedTuple):
    name: str                 # file name of the decompressed .dat
    sha256: str               # of the decompressed .dat
    archive_sha256: str       # of the .bz2 as downloaded
    url: str                  # HTTPS URL of the .bz2 at the pinned commit
    n_bytes: int              # size of the decompressed .dat


RECOGNITION = WeightSpec(
    "dlib_face_recognition_resnet_model_v1.dat",
    "55533b28a95800a551ba546ba62fe69625c7e95a7061c338adffead08719da30",
    "abb1f61041e434465855ce81c2bd546e830d28bcbed8d27ffbe5bb408b11553a",
    _RAW + "dlib_face_recognition_resnet_model_v1.dat.bz2", 22466066)
SHAPE_PREDICTOR_5 = WeightSpec(
    "shape_predictor_5_face_landmarks.dat",
    "c4b1e9804792707d3a405c2c16a80a20269e6675021f64a41d30fffafbc41888",
    "6e787bbebf5c9efdb793f6cd1f023230c4413306605f24f299f12869f95aa472",
    _RAW + "shape_predictor_5_face_landmarks.dat.bz2", 9150489)
GIT_BLOB_IDS = {   # of the .bz2 files at SOURCE_COMMIT, as reported by `git ls-tree` over the Git protocol
    "dlib_face_recognition_resnet_model_v1.dat.bz2": "6c3a71f376b4a6df36d6db00847253b0461cf5ab",
    "shape_predictor_5_face_landmarks.dat.bz2": "3021e2b5f7074adc3cf0ea7590478aaf0b871876",
}


class WeightError(RuntimeError):
    pass


def cache_dir() -> Path:
    return Path(os.path.expanduser("~/.cache/dlib"))


def sha256_file(path: Path | str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def verify(path: Path | str, spec: WeightSpec) -> Path:
    """Return ``path`` if its SHA-256 equals the pinned value, else raise ``WeightError`` (the file is never used)."""
    path = Path(path)
    got = sha256_file(path)
    if got != spec.sha256:
        raise WeightError(f"{path}: SHA-256 {got} != pinned {spec.sha256} for {spec.name}; refusing to use it "
                          f"(expected the file from {SOURCE_REPO} at commit {SOURCE_COMMIT[:10]})")
    return path


def _download_verified(spec: WeightSpec, dest: Path, opener=urllib.request.urlopen) -> None:
    if not spec.url.startswith("https://"):
        raise WeightError("refusing a non-HTTPS download URL")
    dest.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Downloading %s over HTTPS from %s", spec.name, spec.url)
    with opener(spec.url, timeout=120) as resp:
        raw = resp.read()
    if hashlib.sha256(raw).hexdigest() != spec.archive_sha256:                # verify BEFORE decompressing
        raise WeightError(f"downloaded archive for {spec.name} has the wrong SHA-256; nothing was written")
    data = bz2.decompress(raw)
    if hashlib.sha256(data).hexdigest() != spec.sha256 or len(data) != spec.n_bytes:
        raise WeightError(f"decompressed {spec.name} has the wrong SHA-256/size; nothing was written")
    fd, tmp = tempfile.mkstemp(dir=str(dest.parent), prefix=dest.name + ".", suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(data)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, dest)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


def resolve(spec: WeightSpec, explicit_path: str | None = None, allow_download: bool = True, opener=urllib.request.urlopen) -> Path:
    """Verified path of a weight file: the explicit path, else the cache, else (if allowed) a verified HTTPS download."""
    if explicit_path:
        p = Path(explicit_path)
        if not p.is_file():
            raise WeightError(f"{p} does not exist")
        return verify(p, spec)
    dest = cache_dir() / spec.name
    if dest.is_file():
        return verify(dest, spec)
    if not allow_download or os.environ.get(NO_DOWNLOAD_ENV):
        raise WeightError(f"{spec.name} is not in {cache_dir()} and downloading is disabled "
                          f"({NO_DOWNLOAD_ENV} or allow_download=False)")
    _download_verified(spec, dest, opener)
    return verify(dest, spec)

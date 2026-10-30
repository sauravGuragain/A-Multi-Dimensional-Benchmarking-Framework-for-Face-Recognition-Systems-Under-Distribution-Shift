"""Split manifest (Revision 4.1, section 2.2): persisted Fold 0 identity/image lists + canonical hash.

Pure functions over supplied partition tables; no splitter, no models, no images. The manifest, not a
re-run of the shuffle, is the source of truth. Acceptance (never adjusted silently): counts equal V1
(identities 4,601 / 574 / 574; images 10,636 / 1,259 / 1,338), identity sets and image sets pairwise
disjoint, repeated loads give the same hash. A count mismatch stops execution.

Image identity. Section 2.2 requires "identity sets and image sets pairwise disjoint" and does not say
that image ids are globally unique, so an image is the RECORD ``(identity, image_id)``: the same
``image_id`` string under two different identities is two different images and is allowed. Record
disjointness is checked on its own (before identity disjointness) so each rule has an isolated, reachable
error.

Canonical serialization: UTF-8, LF, one sorted ``key=value|...`` metadata header line, then rows
``partition|identity|image_id`` sorted as strings, one trailing LF. SHA-256 covers header and rows.
Metadata values may contain ``=`` (the header parser splits each field on its FIRST ``=``); keys may not.
Neither keys nor values may contain ``|``, control characters, or leading/trailing whitespace.
"""
from __future__ import annotations

import errno
import hashlib
import numbers
import os
import re
import tempfile
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from faceeval.evaluation.pairs import validate_images

MANIFEST_VERSION = "faceeval-split-manifest/r4.1/1"
PARTITIONS = ("train", "validation", "test")
V1_IDENTITIES = MappingProxyType({"train": 4601, "validation": 574, "test": 574})
V1_IMAGES = MappingProxyType({"train": 10636, "validation": 1259, "test": 1338})
# Shuffled-order SHA-256 recorded for the standalone splitter replication (Revision 4.1, V25).
# Comparing a SUPPLIED digest with this constant is not the same as reproducing it: reproduction needs the
# real Fold 0 splitter and source data, and how the digest is computed is not specified in this repository.
V25_SHUFFLED_ORDER_SHA256 = "8cad997cea549ca5fb4abb7f3efb588c69f5b5dd78a03a5d7b8a896b7f53ec05"
NOT_RECORDED = "not_recorded"
REQUIRED_METADATA = ("dataset_name", "fold", "seed", "python_version", "numpy_version", "code_version",
                     "shuffled_order_sha256")
_HEX64 = re.compile(r"[0-9a-f]{64}\Z")
_SIDECAR = re.compile(r"([0-9a-f]{64})  ([^\n\r]+)\n\Z")


class ManifestAcceptanceError(RuntimeError):
    """Counts, disjointness, format or hash checks failed; nothing was adjusted."""


def _unsafe_text(v: str) -> bool:
    return v == "" or v != v.strip() or "|" in v or any(ord(c) < 32 or ord(c) == 127 for c in v)


def _clean_meta(key: Any, value: Any) -> Tuple[str, str]:
    if not isinstance(key, str) or key == "" or "=" in key or _unsafe_text(key):
        raise ValueError(f"metadata key must be a non-empty string without '=', '|' or control characters: {key!r}")
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (str, numbers.Integral)):
        raise ValueError(f"metadata {key!r} must be a string or integer, got {value!r}")
    v = str(int(value)) if isinstance(value, numbers.Integral) else str(value)
    if _unsafe_text(v):
        raise ValueError(f"metadata {key!r} has an empty or unsafe value: {value!r}")
    return str(key), v


def _check_sha(value: str) -> None:
    if value != NOT_RECORDED and not _HEX64.match(value):
        raise ValueError("shuffled_order_sha256 must be 64 lowercase hex characters or 'not_recorded'")


def _int_count(name: str, key: str, v: Any) -> int:
    if isinstance(v, (bool, np.bool_)) or not isinstance(v, numbers.Integral):
        raise ValueError(f"{name}[{key!r}] must be an integer, got {v!r}")
    if int(v) < 1:
        raise ValueError(f"{name}[{key!r}] must be at least 1, got {int(v)}")
    return int(v)


def _expected(name: str, value: Optional[Mapping[str, Any]], default: Mapping[str, int]) -> Dict[str, int]:
    if value is None:
        return dict(default)
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping with keys {PARTITIONS}")
    keys = list(value)
    if not all(isinstance(k, str) for k in keys) or set(keys) != set(PARTITIONS) or len(keys) != len(PARTITIONS):
        raise ValueError(f"{name} must have exactly the keys {PARTITIONS}, got {keys!r}")
    return {p: _int_count(name, p, value[p]) for p in PARTITIONS}


@dataclass(frozen=True, eq=False)
class Manifest:
    """Immutable manifest. ``partitions`` maps partition -> tuple of (identity, image_id) tuples and
    ``metadata`` maps str -> str; both are read-only mappings over private copies (item assignment
    raises ``TypeError``; the rows are tuples). Callers' input containers are copied, never kept."""
    partitions: Mapping[str, Tuple[Tuple[str, str], ...]]
    metadata: Mapping[str, str]

    def __post_init__(self):
        object.__setattr__(self, "partitions", MappingProxyType(
            {p: tuple((str(i), str(im)) for i, im in rows) for p, rows in self.partitions.items()}))
        object.__setattr__(self, "metadata", MappingProxyType({str(k): str(v) for k, v in self.metadata.items()}))

    def __eq__(self, other):
        return isinstance(other, Manifest) and manifest_bytes(self) == manifest_bytes(other)

    def __hash__(self):
        return hash(manifest_bytes(self))

    def __reduce__(self):        # pickle / copy.deepcopy: rebuild from plain containers
        return (Manifest, (dict(self.partitions), dict(self.metadata)))


def build_manifest(partitions: Mapping[str, Sequence[Tuple[str, str]]], metadata: Mapping[str, Any],
                   expected_identities: Optional[Mapping[str, int]] = None,
                   expected_images: Optional[Mapping[str, int]] = None) -> Manifest:
    """Validate and freeze a manifest. ``expected_*`` default to the V1 counts; pass your own only
    for a different fold, never to make a mismatch pass."""
    exp_id = _expected("expected_identities", expected_identities, V1_IDENTITIES)
    exp_im = _expected("expected_images", expected_images, V1_IMAGES)
    if not isinstance(partitions, Mapping) or set(partitions) != set(PARTITIONS):
        raise ValueError(f"partitions must be a mapping with exactly the keys {PARTITIONS}")
    if not isinstance(metadata, Mapping):
        raise ValueError("metadata must be a mapping")
    meta = dict(_clean_meta(k, v) for k, v in metadata.items())
    missing = [k for k in REQUIRED_METADATA if k not in meta]
    extra = sorted(k for k in meta if k not in REQUIRED_METADATA)
    if missing or extra:
        raise ValueError(f"metadata keys: missing {missing}, unexpected {extra}")
    _check_sha(meta["shuffled_order_sha256"])
    meta["manifest_version"] = MANIFEST_VERSION
    rows: Dict[str, Tuple[Tuple[str, str], ...]] = {}
    for p in PARTITIONS:
        recs = sorted(validate_images(partitions[p]))
        for a, b in zip(recs, recs[1:]):
            if a == b:
                raise ValueError(f"duplicate record in partition {p}: {a}")
        rows[p] = tuple(recs)
    ids = {p: {i for i, _ in rows[p]} for p in PARTITIONS}
    for i, a in enumerate(PARTITIONS):                       # image (record) sets first, then identity sets
        for b in PARTITIONS[i + 1:]:
            img = set(rows[a]) & set(rows[b])
            if img:
                raise ManifestAcceptanceError(f"image records of {a} and {b} overlap: {sorted(img)[:5]}")
    for i, a in enumerate(PARTITIONS):
        for b in PARTITIONS[i + 1:]:
            common = ids[a] & ids[b]
            if common:
                raise ManifestAcceptanceError(f"identity sets of {a} and {b} overlap: {sorted(common)[:5]}")
    got_id = {p: len(ids[p]) for p in PARTITIONS}
    got_im = {p: len(rows[p]) for p in PARTITIONS}
    if got_id != exp_id or got_im != exp_im:
        raise ManifestAcceptanceError(
            f"counts differ from the required values: identities {got_id} vs {exp_id}; images {got_im} vs {exp_im}")
    return Manifest(rows, meta)


def manifest_bytes(m: Manifest) -> bytes:
    head = "|".join(f"{k}={m.metadata[k]}" for k in sorted(m.metadata))
    lines = sorted(f"{p}|{i}|{im}" for p in PARTITIONS for i, im in m.partitions[p])
    return (head + "\n" + "".join(l + "\n" for l in lines)).encode("utf-8")


def manifest_sha256(m: Manifest) -> str:
    return hashlib.sha256(manifest_bytes(m)).hexdigest()


def compare_shuffled_order(m: Manifest) -> Dict[str, Any]:
    """Report (never reconcile) how the RECORDED shuffled-order digest relates to the V25 value.

    ``status`` is ``match``, ``MISMATCH`` or ``not_recorded``. This compares a supplied string with a
    constant; ``reproduced_from_source`` is always False because this module has no splitter and no
    source data, so it cannot reproduce the digest."""
    got = m.metadata["shuffled_order_sha256"]
    status = "not_recorded" if got == NOT_RECORDED else ("match" if got == V25_SHUFFLED_ORDER_SHA256 else "MISMATCH")
    return {"expected": V25_SHUFFLED_ORDER_SHA256, "recorded": got, "status": status,
            "reproduced_from_source": False}


def _write_temp(directory: str, base: str, data: bytes) -> str:
    fd, tmp = tempfile.mkstemp(prefix=f".{base}.", suffix=".tmp", dir=directory)
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(data)
            fh.flush()
            os.fsync(fh.fileno())
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise
    return tmp


def write_manifest(m: Manifest, path: str) -> str:
    """Write the manifest and ``<path>.sha256`` ("<sha256>  <basename>\\n"); return the SHA-256.

    Method: both contents are written to private temporary files in the destination directory and
    fsynced; each is then published with ``os.link`` (create-only: it fails if the name already exists,
    so nothing is ever overwritten, including under concurrent creation by another process). If any
    step fails, the destination names created by THIS call are removed again (only when they are
    still the very file this call linked) and the temporary files are always removed.

    Guarantees: no overwrite; after any recoverable exception (including ``FileExistsError`` and I/O
    errors) this call leaves no partial pair, no temporary file and no changed pre-existing file.
    Limitations: two files cannot be made one atomic transaction. A crash (kill -9, power loss) between
    the two publish steps can leave the manifest without its sidecar; ``load_manifest`` then refuses it
    and ``write_manifest`` refuses to write over it, so the leftover is detected, never silently
    accepted. Requires a filesystem that supports hard links (APFS/ext4/NTFS do)."""
    data = manifest_bytes(m)
    sha = hashlib.sha256(data).hexdigest()
    path = os.fspath(path)
    side = path + ".sha256"
    for p in (path, side):
        if os.path.lexists(p):
            raise FileExistsError(errno.EEXIST, "refusing to overwrite", p)
    directory = os.path.dirname(os.path.abspath(path))
    base = os.path.basename(path)
    if "\n" in base or "\r" in base:           # the sidecar line "<sha>  <name>\n" could not be parsed back
        raise ValueError(f"manifest file name must not contain CR or LF: {base!r}")
    sidecar_data = f"{sha}  {base}\n".encode("utf-8")
    tmps: List[str] = []
    published: List[Tuple[str, str]] = []
    try:
        tmp_m = _write_temp(directory, base, data)
        tmps.append(tmp_m)
        tmp_s = _write_temp(directory, base + ".sha256", sidecar_data)
        tmps.append(tmp_s)
        os.link(tmp_m, path)
        published.append((path, tmp_m))
        os.link(tmp_s, side)
        published.append((side, tmp_s))
    except BaseException:
        for dest, tmp in reversed(published):
            try:
                if os.path.samefile(dest, tmp):
                    os.unlink(dest)
            except OSError:
                pass
        raise
    finally:
        for tmp in tmps:
            try:
                os.unlink(tmp)
            except OSError:
                pass
    try:                                        # make the new directory entries durable (best effort)
        dfd = os.open(directory, os.O_RDONLY)
        try:
            os.fsync(dfd)
        finally:
            os.close(dfd)
    except OSError:
        pass
    return sha


def _fail(msg: str):
    raise ManifestAcceptanceError(msg)


def load_manifest(path: str, expected_sha256: Optional[str] = None,
                  expected_identities: Optional[Mapping[str, int]] = None,
                  expected_images: Optional[Mapping[str, int]] = None) -> Manifest:
    """Load, verify the bytes against the sidecar (and ``expected_sha256`` if given), re-validate every
    acceptance check, and confirm the canonical re-serialization is byte-identical to the file.

    Malformed manifest or sidecar content raises ``ManifestAcceptanceError`` with a specific message.
    Ordinary filesystem errors (permissions, I/O) propagate as ``OSError``. A malformed caller argument
    (``expected_sha256`` not 64 lowercase hex, bad ``expected_*``) raises ``ValueError``; a well-formed
    but different ``expected_sha256`` raises ``ManifestAcceptanceError``."""
    if expected_sha256 is not None and (not isinstance(expected_sha256, str) or not _HEX64.match(expected_sha256)):
        raise ValueError("expected_sha256 must be 64 lowercase hex characters")
    exp_id = _expected("expected_identities", expected_identities, V1_IDENTITIES)
    exp_im = _expected("expected_images", expected_images, V1_IMAGES)
    path = os.fspath(path)
    with open(path, "rb") as fh:
        data = fh.read()
    side = path + ".sha256"
    if not os.path.exists(side):
        _fail(f"missing sidecar hash file {side}")
    with open(side, "rb") as fh:
        side_raw = fh.read()
    if not data:
        _fail("manifest is empty")
    if not side_raw.strip():
        _fail("sidecar hash file is empty")
    try:
        side_text = side_raw.decode("utf-8")
    except UnicodeDecodeError:
        _fail("sidecar is not valid UTF-8")
    sm = _SIDECAR.match(side_text)
    if not sm:
        _fail("malformed sidecar: expected '<64 lowercase hex>  <name>' and one trailing LF")
    sha = hashlib.sha256(data).hexdigest()
    if sm.group(1) != sha:
        _fail(f"manifest bytes hash to {sha}, sidecar says {sm.group(1)}")
    if sm.group(2) != os.path.basename(path):
        _fail(f"sidecar is bound to the file name {sm.group(2)!r}, not {os.path.basename(path)!r}")
    if expected_sha256 is not None and expected_sha256 != sha:
        _fail(f"manifest hash {sha} differs from the expected {expected_sha256}")
    if b"\r" in data:
        _fail("manifest contains CR; only LF line endings are canonical")
    if not data.endswith(b"\n"):
        _fail("manifest does not end with LF")
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError:
        _fail("manifest is not valid UTF-8")
    lines = text.split("\n")[:-1]
    if len(lines) < 2:
        _fail("manifest has no header and rows")
    meta: Dict[str, str] = {}
    for field in lines[0].split("|"):
        if "=" not in field or field.startswith("="):
            _fail(f"malformed metadata field in header: {field!r}")
        k, v = field.split("=", 1)
        if k in meta:
            _fail(f"duplicate metadata key {k!r}")
        meta[k] = v
    version = meta.pop("manifest_version", None)
    if version != MANIFEST_VERSION:
        _fail(f"unsupported or missing manifest version {version!r}")
    missing = [k for k in REQUIRED_METADATA if k not in meta]
    extra = sorted(k for k in meta if k not in REQUIRED_METADATA)
    if missing or extra:
        _fail(f"metadata keys: missing {missing}, unexpected {extra}")
    parts: Dict[str, List[Tuple[str, str]]] = {p: [] for p in PARTITIONS}
    for ln in lines[1:]:
        f = ln.split("|")
        if len(f) != 3:
            _fail(f"malformed manifest row: {ln!r}")
        if f[0] not in parts:
            _fail(f"invalid partition name {f[0]!r} in row {ln!r}")
        parts[f[0]].append((f[1], f[2]))
    try:
        m = build_manifest(parts, meta, exp_id, exp_im)
    except ValueError as exc:
        raise ManifestAcceptanceError(f"manifest content is invalid: {exc}") from exc
    if manifest_bytes(m) != data:
        _fail("manifest is not in canonical form (row order, header order or formatting differ)")
    return m

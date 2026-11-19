"""LFW Fold 0 verification stage for the orchestration runner.

Wires the Revision 4.1 split manifest (``faceeval.evaluation.split_manifest``) and the
deterministic pair builder (``faceeval.evaluation.pairs``) into the experiment run:

1. Fold 0 = ``DatasetSplitter(seed).all_k_folds(records, k=10)[0]`` (identity-disjoint).
2. The split is frozen as a manifest (written with a ``.sha256`` sidecar) and validated.
3. Validation / test pair sets are built (genuine, impostor pool, balanced subset) and
   written as pair manifests.
4. For every model the embedding extractor is fitted on the *train* partition only, a
   decision threshold is chosen on *clean validation* pairs only, and the *test* pairs are
   scored under every perturbation spec with that frozen threshold.

Results go to ``<output_dir>/verification/<run_id>/`` and are kept separate from the
closed-set identification results.
"""
from __future__ import annotations

import csv
import hashlib
import importlib.metadata as importlib_metadata
import json
import logging
import os
import platform
import random
import subprocess
from pathlib import Path
from typing import Any, Callable

import numpy as np

from faceeval.core.exceptions import EmbeddingError, OrchestratorError
from faceeval.evaluation import pairs as pairs_mod
from faceeval.evaluation import split_manifest as sm
from faceeval.evaluation.metrics import compute_verification_metrics
from faceeval.evaluation.verification_calibration import CAL_SUFFIX, CalibrationRecorder
from faceeval.orchestration.perturbed_inputs import needs_chips, perturb_batch
from faceeval.core.registry import perturbation_registry
from faceeval.perturbation import base as _base
from faceeval.perturbation import noise as _noise

logger = logging.getLogger(__name__)

FOLD_INDEX = 0                   # the REFERENCE fold; other folds via config.verification_fold
REFERENCE_SEED = 42              # seed of the reference Fold 0 (the committed Fold-0 results)
K_FOLDS = 10
FULL_LFW_IDENTITIES = 5749
FULL_LFW_IMAGES = 13233
# Content fingerprints (see ``dataset_fingerprint``) of verified official LFW copies.
# This fingerprint was independently reproduced from a fresh extraction of the original
# lfw.tgz archive. The archive SHA-256 is the published scikit-learn LFW checksum:
# 055f7d9c632d7370e6fb4afc7468d40f970c34a80d4c6f50ffec63f5a8d536c0
OFFICIAL_LFW_FINGERPRINTS: frozenset[str] = frozenset({
    "df3e734c6e5033224011c2587fd2c7671ce2f86d499cb57404f50f0d12337230",
})
PARTITION_CODE = {"validation": 1, "test": 2}
FAR_POINTS = [0.001, 0.01, 0.1]


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _id_seeded_perturbation_types() -> list[str]:
    """Registered perturbation types whose randomness is seeded per image id (``USES_IMAGE_ID``).

    Derived from the registry, not hard-coded. Import order is safe: importing ``faceeval.perturbation.noise``
    (done at the top of this module) first runs ``faceeval.perturbation.__init__``, which registers every built-in
    perturbation. A plugin registered later would appear automatically.
    """
    return sorted(key for key in perturbation_registry.list_registered()
                  if getattr(perturbation_registry.get(key), "USES_IMAGE_ID", False))


def _module_sha256(module) -> str:
    return hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()


def noise_seeding_metadata() -> dict[str, Any]:
    """Provenance of the noise seeding used by this run (recorded in the verification summary).

    The seed depends on code in TWO files, so both are fingerprinted separately: ``noise/__init__.py`` (seed
    derivation, noise generation) and ``base.py`` (``normalize_image_id``, id validation). Changing either one
    changes a recorded hash.
    """
    return {
        "protocol_version": _noise.NOISE_PROTOCOL_VERSION,
        "master_seed": _noise.NOISE_MASTER_SEED,
        "scheme": "sha256(master|image_id|perturbation_type)",
        "image_id_normalization": "Unicode NFC; then backslash -> forward slash; then UTF-8 encoded; no other change",
        "digest_to_seed": "first 8 bytes of the SHA-256 digest, big-endian unsigned 64-bit integer, "
                          "passed to numpy.random.default_rng",
        "severity_in_seed": False,
        "id_source": "ImageRecord.image_id",
        "perturbation_types": _id_seeded_perturbation_types(),
        "salt_pepper": "corrupted = U1 < density; salt if U2 < 0.5 else pepper; U1, U2 drawn in that order "
                       "from the per-image stream (polarity independent of severity)",
        "numpy_version": np.__version__,
        "stream_fingerprint": _noise.noise_stream_fingerprint(),
        "noise_module_sha256": _module_sha256(_noise),
        "base_module_sha256": _module_sha256(_base),
    }


def shuffled_order_sha256(records, seed: int) -> str:
    """SHA-256 of the seed-shuffled sorted subject IDs (same shuffle the splitter performs)."""
    subjects = sorted({r.subject_id for r in records})
    random.Random(seed).shuffle(subjects)
    return hashlib.sha256("\n".join(subjects).encode("utf-8")).hexdigest()


def _code_version() -> str:
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                             timeout=10, check=True)
        return out.stdout.strip() or sm.NOT_RECORDED
    except Exception:
        return sm.NOT_RECORDED


# Packages whose versions decide what a deep-model run computes. Recorded in the summary (best effort).
PROVENANCE_PACKAGES = ("torch", "torchvision", "facenet-pytorch", "insightface", "onnxruntime", "onnx",
                       "opencv-python", "scikit-image", "scikit-learn", "scipy", "Pillow", "dlib")


def _file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _weight_files(model_name: str) -> list[Path]:
    """Cached pretrained-weight files the named deep model would load by default (empty if unknown or absent).

    FaceNet (facenet-pytorch, ``vggface2``): ``<TORCH_HOME or $XDG_CACHE_HOME/torch or ~/.cache/torch>/checkpoints``.
    ArcFace (InsightFace pack ``buffalo_l``): ``~/.insightface/models/buffalo_l/*.onnx``. InsightFace (pack
    ``antelopev2``): ``~/.insightface/models/antelopev2/**.onnx``. dlib_fr: ``~/.cache/dlib/dlib_face_recognition_resnet_model_v1.dat``. Only the default
    locations are looked at; a model loaded from a custom location is reported as not recorded."""
    home = Path(os.path.expanduser("~"))
    if model_name == "facenet":
        torch_home = os.environ.get("TORCH_HOME") or os.path.join(os.environ.get("XDG_CACHE_HOME") or str(home / ".cache"), "torch")
        return sorted(Path(os.path.expanduser(torch_home), "checkpoints").glob("*vggface2*"))
    if model_name == "arcface":
        return sorted((home / ".insightface" / "models" / "buffalo_l").glob("*.onnx"))
    if model_name == "insightface":                       # antelopev2 pack; may be unzipped one level deeper
        base = home / ".insightface" / "models" / "antelopev2"
        return sorted(set(base.glob("*.onnx")) | set(base.glob("*/*.onnx")))
    if model_name == "dlib_fr":
        d = home / ".cache" / "dlib"                     # recognition network + the 5-point landmark model of the chip aligner
        return sorted(set(d.glob("dlib_face_recognition_resnet_model_v1.dat")) | set(d.glob("shape_predictor_5_face_landmarks.dat")))
    return []


def run_provenance(model_names) -> dict[str, Any]:
    """Best-effort provenance that the three ``environment`` fields do not cover (recorded under ``provenance``).

    Records: operating system and machine, whether tracked files of the code checkout were modified (``code_dirty``;
    untracked files are ignored, and ``GIT_OPTIONAL_LOCKS=0`` keeps the check from touching the git index), the version
    of every package in ``PROVENANCE_PACKAGES``, and the SHA-256 of the default cached weight files of the deep models in
    the run. Anything that cannot be determined is ``NOT_RECORDED``; this function NEVER raises, because a provenance
    problem must not cost a multi-hour run its results. It changes no result and no manifest or content hash."""
    nr = sm.NOT_RECORDED
    out: dict[str, Any] = {"platform": nr, "machine": nr, "code_dirty": nr, "packages": {}, "weights": {}}
    try:
        out["platform"], out["machine"] = platform.platform(), platform.machine()
    except Exception:
        pass
    try:
        env = dict(os.environ, GIT_OPTIONAL_LOCKS="0")
        st = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], capture_output=True, text=True,
                            timeout=30, check=True, env=env)
        out["code_dirty"] = bool(st.stdout.strip())
    except Exception:
        pass
    for pkg in PROVENANCE_PACKAGES:
        try:
            out["packages"][pkg] = importlib_metadata.version(pkg)
        except Exception:
            out["packages"][pkg] = nr
    for name in sorted({m for m in model_names if m in ("facenet", "arcface", "insightface", "dlib_fr")}):
        try:
            files = _weight_files(name)
            out["weights"][name] = ({f.name: _file_sha256(f) for f in files} if files else nr)
        except Exception:
            out["weights"][name] = nr
    return out


def dataset_fingerprint(records) -> str:
    """SHA-256 over the sorted (identity, image_id, file-content SHA-256) rows of the dataset used.

    Raises ``OrchestratorError`` if any image file is missing, so a damaged or partial copy is
    caught before any model runs. Reads each image file once.
    """
    rows, missing = [], []
    for r in records:
        if not os.path.isfile(r.image_path):
            missing.append(r.image_path)
            continue
        h = hashlib.sha256()
        with open(r.image_path, "rb") as fh:
            for block in iter(lambda: fh.read(1 << 20), b""):
                h.update(block)
        rows.append(f"{r.subject_id}|{r.image_id}|{h.hexdigest()}")
    if missing:
        raise OrchestratorError(f"{len(missing)} dataset image file(s) missing, e.g. {missing[:3]}")
    return hashlib.sha256(("\n".join(sorted(rows)) + "\n").encode("utf-8")).hexdigest()


def split_content_sha256(manifest) -> str:
    """Hash of the split itself (partition rows + non-environment metadata).

    Unlike ``manifest_sha256`` it excludes python/numpy/code versions, so it is identical for the
    same dataset and seed on any machine or NumPy version."""
    keep = ("dataset_name", "fold", "seed", "shuffled_order_sha256", "manifest_version")
    lines = [f"{k}={manifest.metadata[k]}" for k in keep]
    for part in sm.PARTITIONS:
        lines += [f"{part}|{ident}|{imid}" for ident, imid in manifest.partitions[part]]
    return hashlib.sha256(("\n".join(lines) + "\n").encode("utf-8")).hexdigest()


def pair_content_sha256(ps) -> str:
    """Hash of the pair set (parameters except ``numpy_version``, plus every canonical row)."""
    head = "|".join(f"{k}={ps.params[k]}" for k in sorted(ps.params) if k != "numpy_version")
    body = "\n".join(pairs_mod.manifest_rows(ps)) + "\n"
    return hashlib.sha256((head + "\n" + body).encode("utf-8")).hexdigest()


def _l2(emb: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(emb, axis=1, keepdims=True)
    return emb / np.where(n > 0, n, 1.0)


def _scores(emb: np.ndarray, idx_by_pair: list[tuple[int, int]]) -> np.ndarray:
    a = np.fromiter((p[0] for p in idx_by_pair), dtype=np.int64, count=len(idx_by_pair))
    b = np.fromiter((p[1] for p in idx_by_pair), dtype=np.int64, count=len(idx_by_pair))
    return np.einsum("ij,ij->i", emb[a], emb[b])


def best_accuracy_threshold(scores, labels) -> float:
    """Threshold ``t`` (accept iff ``score >= t``) that maximises accuracy on labelled scores.

    Every distinct outcome is considered: reject every pair (``t`` just above the maximum
    score), and accept the top-k pairs for each distinct score value, where tied scores are
    always accepted or rejected together (the last candidate, ``t == min(score)``, accepts
    every pair). Ties in accuracy resolve to the strictest (highest) threshold.

    Raises ``ValueError`` for empty input, mismatched lengths, non-1-D input, non-finite
    scores, or labels that are not boolean / 0-1.
    """
    s = np.asarray(scores)
    y = np.asarray(labels)
    if s.ndim != 1 or y.ndim != 1:
        raise ValueError("scores and labels must be 1-D")
    if s.size == 0:
        raise ValueError("scores and labels must not be empty")
    if s.shape != y.shape:
        raise ValueError(f"scores and labels differ in length: {s.shape[0]} vs {y.shape[0]}")
    s = s.astype(np.float64)
    if not np.isfinite(s).all():
        raise ValueError("scores must be finite")
    if y.dtype != np.bool_:
        if not np.isin(y, (0, 1)).all():
            raise ValueError("labels must be boolean or 0/1")
        y = y.astype(bool)

    n = s.size
    n_neg = int((~y).sum())
    order = np.argsort(-s, kind="mergesort")
    s_sorted = s[order]
    y_sorted = y[order].astype(np.int64)
    tp = np.cumsum(y_sorted)                       # genuine accepted among top-(i+1)
    fp = np.cumsum(1 - y_sorted)                   # impostors accepted among top-(i+1)
    acc_topk = (tp + (n_neg - fp)) / n
    last_of_tie = np.r_[s_sorted[1:] != s_sorted[:-1], True]   # cut only between distinct values
    cand_acc = np.r_[n_neg / n, acc_topk[last_of_tie]]          # first candidate: reject all
    cand_thr = np.r_[np.nextafter(s_sorted[0], np.inf), s_sorted[last_of_tie]]
    return float(cand_thr[int(np.argmax(cand_acc))])               # argmax: first max = strictest


def _accuracy_at(scores: np.ndarray, labels: np.ndarray, thr: float) -> dict[str, float]:
    pred = scores >= thr
    pos, neg = labels, ~labels
    tar = float(pred[pos].mean()) if pos.any() else float("nan")
    far = float(pred[neg].mean()) if neg.any() else float("nan")
    return {"accuracy": float((pred == labels).mean()), "tar": tar, "far": far}


def _valid_rows(emb: np.ndarray) -> np.ndarray:
    """True for every row that is a real embedding; False for the all-zero vector a failed acquisition
    (e.g. a detector miss) produces, and for non-finite rows. Both deep wrappers L2-normalise, so a real
    row has norm 1 and a failure has norm exactly 0; the cut-off 1e-6 sits far from both, so no real
    embedding is ever called failed. Model-agnostic: it looks only at the embedding."""
    arr = np.asarray(emb, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError(f"embeddings must be a 2-D (n_images, dim) array, got shape {arr.shape}")
    norm = np.linalg.norm(arr, axis=1)
    return np.isfinite(norm) & (norm > 1e-6)


def _sanitize(emb: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Failed rows (zero, NaN or Inf) become the zero vector, so every failure is scored the same documented
    way (cosine 0 with everything) and cannot put NaN/Inf into scores, thresholds or metrics. Valid rows and
    the array itself are returned unchanged when nothing failed."""
    if valid.all():
        return emb
    out = np.array(emb, copy=True)
    out[~valid] = 0.0
    return out


def _warn_if_not_unit(emb: np.ndarray, valid: np.ndarray, model_name: str) -> None:
    """Cosine == dot product only for unit-norm rows; say so if a backbone ever breaks that."""
    n = np.linalg.norm(np.asarray(emb, dtype=np.float64)[valid], axis=1)
    if n.size and float(np.abs(n - 1.0).max()) > 1e-3:
        logger.warning("[%s] valid embeddings are not unit-norm (norm range %.4f..%.4f): scores are dot "
                       "products, not cosine similarities.", model_name, float(n.min()), float(n.max()))


def _pairs_with_failure(valid: np.ndarray, idx_by_pair: list[tuple[int, int]]) -> int:
    """Number of pairs in which at least one of the two images has no valid embedding."""
    a = np.fromiter((p[0] for p in idx_by_pair), dtype=np.int64, count=len(idx_by_pair))
    b = np.fromiter((p[1] for p in idx_by_pair), dtype=np.int64, count=len(idx_by_pair))
    return int((~valid[a] | ~valid[b]).sum())


_POSIX = os.name == "posix"


def _unlink_quietly(path: Path, what: str) -> None:
    """Best-effort removal used on failure paths. A failure to remove is logged and swallowed: it must never
    replace the exception that made the cleanup necessary."""
    try:
        path.unlink(missing_ok=True)
    except OSError:
        logger.error("Could not remove %s %s.", what, path)


def _fsync_dir(directory: Path) -> bool:
    """Best-effort fsync of a directory, so a just-completed ``os.replace`` survives a power loss.

    ``os.replace`` alone makes the swap atomic for readers (they see the old or the new file, never a mix) but
    does not by itself make the new directory entry durable. This is only attempted on POSIX (Windows cannot
    open a directory this way). A failure is logged, never raised: the replace already happened, and the
    run's safety does not depend on it (a lost rename reverts to the older, INCOMPLETE summary). On macOS
    ``fsync`` does not force the drive's own cache to flush, so this is a best effort, not a hard guarantee."""
    if not _POSIX:
        return False
    try:
        fd = os.open(directory, os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
        return True
    except OSError as exc:
        logger.warning("Could not fsync directory %s (%s); the rename is atomic but may not survive a power loss.",
                       directory, exc)
        return False


def _fsync_file(path: Path) -> None:
    """fsync a completed file by path. Raises on failure: unlike a directory fsync, a file that is not durable
    must not be published (a recorded row would then point at data that a power loss can empty)."""
    fd = os.open(path, os.O_RDWR)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _atomic_savez(path: Path, **arrays: Any) -> None:
    """Save arrays so that ``path`` is either absent/old or complete AND durable, never partial.

    Order: write the temporary file, fsync it, ``os.replace`` it over ``path``, then fsync the directory
    (best effort, like ``_write_json_atomic``). The caller records the row naming this file only after this
    returns, so a recorded row never refers to a score file whose data was still only in the page cache."""
    tmp = path.with_name(path.stem + ".tmp.npz")
    try:
        np.savez_compressed(tmp, **arrays)
        _fsync_file(tmp)
        os.replace(tmp, path)
    except BaseException:
        _unlink_quietly(tmp, "the temporary score file")     # never leave a partial temporary file behind
        raise
    _fsync_dir(path.parent)              # after the replace; best effort, see _fsync_dir


def _discard_score_file(path: Path) -> None:
    """Remove the score file of a condition whose row was not recorded. A failure to remove it is logged and
    swallowed: it must never replace the exception that made the cleanup necessary (the file stays detectable
    as an orphan, because every recorded row names its own score file)."""
    _unlink_quietly(path, "the unrecorded score file")


def _write_json_atomic(path: Path, obj: Any) -> None:
    """Write ``obj`` as JSON through a temporary file and ``os.replace``: readers see the old file or the new
    one, never a partial one, and a failure leaves the old file in place and no temporary file."""
    tmp = path.with_name(path.name + ".tmp")
    try:
        with open(tmp, "w") as fh:
            json.dump(obj, fh, indent=2, default=str)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    except BaseException:
        _unlink_quietly(tmp, "the temporary summary file")
        raise
    _fsync_dir(path.parent)              # after the replace; best effort, see _fsync_dir


class _RowStream:
    """Append-only JSON-lines record of a run, written unbuffered and fsynced after every record.

    It lets a killed or interrupted run (sleep, closed terminal, out-of-memory) keep every condition that
    finished. The final ``verification_results.csv/.json`` are written exactly as before; a stream with no
    final ``end`` record therefore marks an interrupted run.

    Commit semantics. A record is one line written with unbuffered ``write`` calls on an ``O_APPEND`` file, so
    there is no user-space buffer that could flush a record later than the caller believes. The FILE is the only
    source of truth for "is this record recorded": ``has_row`` reads it back. If a write is interrupted or
    fails, the line is rolled back when torn (or, after an ordinary error, when not durably flushed) and kept
    only when it is complete and the interruption was a KeyboardInterrupt/SystemExit, so a record can never
    be half there, and a caller never has to guess from its own state whether a row exists."""

    def __init__(self, path: Path) -> None:
        # O_EXCL: exclusive create. An existing file (a previous run's records) is an error, never appended to.
        self.path = Path(path)
        fd = os.open(self.path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_APPEND, 0o644)
        self._fh = os.fdopen(fd, "ab", buffering=0)
        self.terminal = False          # True once an "end" or "abort" record has been written
        self.broken = False            # True once a torn record could not be rolled back: nothing more is appended

    def _truncate(self, size: int) -> bool:
        """Remove everything after ``size``. Returns False, and marks the stream unusable, if that fails: a torn
        record may then still be in the file, so no further record may be appended behind it."""
        try:
            os.ftruncate(self._fh.fileno(), size)
            return True
        except OSError:
            self.broken = True
            logger.error("Could not roll back a failed stream record in %s; no further records will be "
                         "written to it.", self.path)
            return False

    def write(self, kind: str, payload: dict[str, Any]) -> None:
        if self.broken:
            raise OSError(f"row stream {self.path} is unusable after a failed rollback; refusing to append")
        if self.terminal or self.has_terminal():     # the file decides: the flag may be stale after an interrupt
            self.terminal = True
            raise OSError(f"row stream {self.path} already holds a terminal record; refusing to append")
        data = (json.dumps({"type": kind, **payload}, default=str) + "\n").encode("utf-8")
        fd = self._fh.fileno()
        start = os.fstat(fd).st_size
        try:
            view = memoryview(data)
            while view:
                n = self._fh.write(view)
                if not isinstance(n, int) or n <= 0 or n > len(view):
                    raise OSError(f"unbuffered write returned {n!r} for {len(view)} bytes")   # no endless loop
                view = view[n:]
            os.fsync(fd)
        except Exception:
            # Ordinary failure (disk full, fsync error): the record is not durable, so remove all of it.
            self._truncate(start)
            raise
        except BaseException:
            # KeyboardInterrupt / SystemExit: a complete line stays (it is a committed record and its score
            # file must be kept); a torn line is removed so it can never be mistaken for a record.
            try:
                size = os.fstat(fd).st_size
                if start < size < start + len(data):
                    self._truncate(start)
                elif size == start + len(data) and kind in ("end", "abort"):
                    self.terminal = True     # the whole terminal line is in the file: it is committed
            except OSError:
                self.broken = True       # cannot tell whether a torn line is there: append nothing more
            raise
        if kind in ("end", "abort"):
            self.terminal = True

    def has_row(self, scores_file: str) -> bool:
        """True if a complete ``row`` record naming ``scores_file`` is in the stream file. When the file cannot be
        read the answer is True: keeping a score file is always safe, deleting one a row refers to is not."""
        try:
            with open(self.path, "rb") as fh:
                for line in fh:
                    try:
                        rec = json.loads(line)
                    except ValueError:
                        continue
                    if isinstance(rec, dict) and rec.get("type") == "row" and rec.get("scores_file") == scores_file:
                        return True
        except OSError:
            return True
        return False

    def has_terminal(self) -> bool:
        """True if the stream FILE already holds a complete ``end`` or ``abort`` record. Like ``has_row`` the file,
        not the ``terminal`` flag, decides (an interrupt can arrive after a record is durable but before the flag
        is set). A complete JSON terminal object counts even without its trailing newline: appending behind it
        would merge two records into one unreadable line, so nothing may be appended. A torn (unparseable) tail
        does not count; ``write`` rolls those back. When the file cannot be read the answer is True: appending
        nothing is always safe."""
        try:
            with open(self.path, "rb") as fh:
                for line in fh:
                    try:
                        rec = json.loads(line)
                    except ValueError:
                        continue
                    if isinstance(rec, dict) and rec.get("type") in ("end", "abort"):
                        return True
        except OSError:
            return True
        return False

    def abort(self, exc: BaseException) -> None:
        """Best-effort terminal record for a run that stops on an exception (including Ctrl-C).

        Ordinary errors while writing it are swallowed so they cannot replace the exception already
        propagating. KeyboardInterrupt/SystemExit raised *during* this write are deliberately not caught: a
        second interrupt is a new request to stop and must not be hidden."""
        if self.terminal or self._fh.closed:
            return
        if self.has_terminal():            # a committed end/abort is already there: never write a second one
            self.terminal = True
            return
        try:
            self.write("abort", {"error": f"{type(exc).__name__}: {exc}"})
        except Exception:                  # never mask the exception that is already propagating
            pass

    def close(self) -> None:
        if not self._fh.closed:
            try:
                self._fh.close()           # every record was already flushed and fsynced by write()
            except OSError:
                logger.error("Could not close the verification row stream cleanly.")


def _flat_metrics(scores: np.ndarray, labels: np.ndarray, suffix: str) -> dict[str, Any]:
    m = compute_verification_metrics(scores.tolist(), labels.tolist(),
                                     far_operating_points=FAR_POINTS)
    out = {f"auc{suffix}": m["auc"], f"eer{suffix}": m["eer"]}
    for k, v in (m.get("frr_at_far") or {}).items():
        out[f"frr_at_far_{k}{suffix}"] = v
    return out


# ---------------------------------------------------------------------------
# main entry point
# ---------------------------------------------------------------------------

def run_fold0_verification(*, config, **kwargs) -> dict[str, Any]:
    """Run the Fold 0 verification protocol; return a summary dict (also written to disk).

    Any failure after the output directory exists leaves a ``verification_results.json`` with
    ``status: INCOMPLETE`` (written by the implementation when it has rows, here otherwise).
    """
    out_dir = Path(config.output_dir) / "verification" / config.run_id
    out_dir.mkdir(parents=True, exist_ok=False)
    stream = _RowStream(out_dir / "verification_rows.jsonl")
    try:
        return _run_fold0_verification(out_dir, config=config, stream=stream, **kwargs)
    except BaseException as exc:
        stream.abort(exc)
        if isinstance(exc, Exception):
            path = out_dir / "verification_results.json"
            try:
                if not path.exists():
                    _write_json_atomic(path, {"protocol": "lfw_fold0_verification", "status": "INCOMPLETE",
                                              "error": f"{type(exc).__name__}: {exc}", "n_rows": 0,
                                              "failures": [], "results": []})
            except Exception:              # never mask the exception that is already propagating
                logger.error("Could not write the INCOMPLETE fallback summary %s.", path)
        raise
    finally:
        stream.close()


def _run_fold0_verification(
    out_dir: Path,
    *,
    config,
    stream: _RowStream,
    records: list,
    raw_images: dict | None,
    model_names: list[str],
    trad_names: set[str],
    composer,
    specs: list,
    dataset_name: str,
    preprocess: Callable,
    adapt_hyperparameters: Callable,
    preprocess_chips: Callable | None = None,
    get_model: Callable | None = None,
    progress: Callable[[str, int, int], None] | None = None,
) -> dict[str, Any]:
    """Run the Fold 0 verification protocol; return a summary dict (also written to disk)."""
    from faceeval.core.types import SplitStrategy
    from faceeval.data.splitter import DatasetSplitter
    from faceeval.core.registry import model_registry
    from faceeval.preprocessing.pipeline import alignment_for_config, describe_preprocessing, pipeline_for_model

    progress = progress or (lambda stage, done, total: None)
    seed = config.random_seed

    # ---- 0. Validate the dataset actually used (the selected subset, not necessarily full LFW) ----
    ds_fingerprint = dataset_fingerprint(records)
    n_identities = len({r.subject_id for r in records})

    # ---- 1. Fold 0 split -------------------------------------------------
    fold_index = int(getattr(config, "verification_fold", FOLD_INDEX))
    if not 0 <= fold_index < K_FOLDS:
        raise OrchestratorError(f"verification_fold must be in [0, {K_FOLDS - 1}], got {fold_index}")
    # The reference split (seed 42, fold 0) keeps EVERY historical behaviour byte for byte, including the official
    # Revision 4.1 G/K acceptance counts on a verified full LFW. Any other (seed, fold) is a further identity-disjoint
    # split of the same protocol: same pair-generation parameters (cap, rho), pair seed = split seed, and the
    # acceptance counts are those of the split itself (leakage and pair-count invariants are checked instead).
    reference_split = (seed == REFERENCE_SEED and fold_index == FOLD_INDEX)
    splits = DatasetSplitter(seed=seed).all_k_folds(records, k=K_FOLDS, dataset_name=dataset_name)
    fold = splits[fold_index]
    by_id = {r.image_id: r for r in records}
    parts_ids = {"train": fold.train_ids, "validation": fold.val_ids, "test": fold.test_ids}
    parts_recs = {p: [by_id[i] for i in ids] for p, ids in parts_ids.items()}
    partitions = {p: [(r.subject_id, r.image_id) for r in rs] for p, rs in parts_recs.items()}

    counts_match = (n_identities, len(records)) == (FULL_LFW_IDENTITIES, FULL_LFW_IMAGES)
    full_lfw = counts_match and ds_fingerprint in OFFICIAL_LFW_FINGERPRINTS
    if counts_match and not full_lfw:
        logger.warning("Dataset has the official LFW counts but its fingerprint %s is not a verified "
                       "official copy: treated as a custom dataset.", ds_fingerprint)
    official_counts = full_lfw and reference_split
    expected_ids = expected_imgs = None
    if not official_counts:
        logger.info("Dataset has %d identities / %d images (not a verified official full LFW; "
                    "min_images_per_subject=%s, max_subjects=%s): Fold 0 is built on this dataset and "
                    "the Revision 4.1 G/K acceptance counts are not applied (reference_split=%s).",
                    n_identities, len(records), config.min_images_per_subject, config.max_subjects, reference_split)
        expected_ids = {p: len({i for i, _ in rows}) for p, rows in partitions.items()}
        expected_imgs = {p: len(rows) for p, rows in partitions.items()}

    # ---- 2. Manifest ----------------------------------------------------------
    metadata = {
        "dataset_name": dataset_name, "fold": str(fold_index), "seed": str(seed),
        "python_version": platform.python_version(), "numpy_version": np.__version__,
        "code_version": _code_version(),
        "shuffled_order_sha256": shuffled_order_sha256(records, seed),
    }
    manifest = sm.build_manifest(partitions, metadata, expected_ids, expected_imgs)
    manifest_path = out_dir / "split_manifest_fold0.tsv"
    manifest_sha = sm.write_manifest(manifest, str(manifest_path))
    shuffled_cmp = sm.compare_shuffled_order(manifest)
    logger.info("Fold 0 manifest written (%s) sha256=%s; shuffled-order vs V25: %s",
                manifest_path, manifest_sha, shuffled_cmp["status"])

    # ---- 3. Pair sets -------------------------------------------------------------
    pair_sets: dict[str, pairs_mod.PairSet] = {}
    pair_shas: dict[str, str] = {}
    for part in ("validation", "test"):
        code = PARTITION_CODE[part]
        rows = list(manifest.partitions[part])
        if official_counts:
            ps = pairs_mod.build_fold0_pairs(rows, code)
        elif reference_split:
            ps = pairs_mod.build_pairs(rows, partition_code=code, **pairs_mod.FOLD0_DEFAULTS)
        else:                                   # another split: same cap / rho, pair seed = split seed
            ps = pairs_mod.build_pairs(rows, partition_code=code, **{**pairs_mod.FOLD0_DEFAULTS, "seed": seed})
        pair_sets[part] = ps
        pair_shas[part] = pairs_mod.write_manifest(ps, str(out_dir / f"pairs_{part}_fold0.tsv"))
        logger.info("Pairs[%s]: G=%d K=%d balanced=%d sha256=%s", part,
                    len(ps.genuine), len(ps.pool), len(ps.balanced), pair_shas[part])

    def pair_arrays(ps):
        bal = ps.balanced_pairs()
        pool = ps.sorted_pool()
        gen = list(ps.genuine)
        return {
            "balanced": (gen + bal, np.r_[np.ones(len(gen), bool), np.zeros(len(bal), bool)]),
            "full": (gen + pool, np.r_[np.ones(len(gen), bool), np.zeros(len(pool), bool)]),
        }

    split_counts = None
    if not reference_split:
        from faceeval.evaluation.split_invariants import SplitInvariantError, check_split_and_pairs
        try:
            split_counts = check_split_and_pairs(partitions, pair_sets, rho=pairs_mod.FOLD0_DEFAULTS["rho"])
        except SplitInvariantError as exc:
            raise OrchestratorError(f"split invariants violated (seed={seed}, fold={fold_index}): {exc}") from exc
    pa = {p: pair_arrays(ps) for p, ps in pair_sets.items()}
    # PairSet.images is the sorted (identity, image_id) table; index -> record
    img_recs = {p: [by_id[im] for _, im in ps.images] for p, ps in pair_sets.items()}

    # ---- 4. Per-model evaluation -----------------------------------------------------
    rows_out: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    solver_used: dict[str, Any] = {}
    # Calibration (config.compute_calibration): fitted on CLEAN VALIDATION pairs only, additive artifacts in a sibling
    # directory, never able to fail a run. verification_results.csv and the run directory are not changed by it.
    cal_rec = CalibrationRecorder() if getattr(config, "compute_calibration", False) else None
    preprocessing: dict[str, dict[str, Any]] = {}   # exact preprocessing per model, written to the JSON
    total = len(model_names) * len(specs)
    done = 0
    fatal: OrchestratorError | None = None   # require_real_models violation: stop, but write results first
    progress("verification", 0, total)
    stream.write("plan", {"models": list(model_names), "conditions": [s.label for s in specs],
                          "n_rows_expected": total})

    if any(m not in trad_names for m in model_names) and len({s_.label for s_ in specs}) != len(specs):
        raise OrchestratorError("Two perturbation conditions share one label (type + severity to 2 decimals); "
                                "per-condition score files would collide. Use distinct severities.")

    def _fail(entry: dict[str, str]) -> None:
        failures.append(entry)
        stream.write("failure", entry)

    for model_name in model_names:
        paradigm = "traditional" if model_name in trad_names else "deep_learning"
        try:
            model = (get_model or model_registry.get)(model_name)
            from faceeval.models.traditional.base import apply_classical_rng
            apply_classical_rng(model, config)
            pipeline = pipeline_for_model(model_name, skip_detection_for_pre_cropped=True,
                                          alignment=alignment_for_config(model_name, config))
            prep = preprocessing[model_name] = describe_preprocessing(pipeline)
            is_deep = paradigm == "deep_learning"
            if is_deep:
                X_train, y_train = [], [r.subject_id for r in parts_recs["train"]]
            else:
                X_train = preprocess(pipeline, parts_recs["train"], raw_images)
                y_train = [r.subject_id for r in parts_recs["train"]]
                adapt_hyperparameters(model, model_name, len(X_train))
            X_val = preprocess(pipeline, img_recs["validation"], raw_images)
            test_chips = None
            test_image_ids = [r.image_id for r in img_recs["test"]]   # parallel to X_test; seeds per-image noise
            if preprocess_chips is not None and needs_chips(pipeline.normalizer):
                # Perturb the uint8 chip and normalise afterwards (see perturbed_inputs).
                test_chips = preprocess_chips(pipeline, img_recs["test"], raw_images)
                X_test = [pipeline.normalizer.normalize(c) for c in test_chips]
            else:
                X_test = preprocess(pipeline, img_recs["test"], raw_images)

            model.fit_embedding(X_train, y_train) if not is_deep else model.fit_embedding([], y_train)
            solver_used[model_name] = getattr(getattr(model, "_extractor", None), "_fit_svd_solver", None)
            backbone = "n/a"
            if is_deep:
                backbone = "STUB" if model.is_stub_backbone() else "real"
                if backbone == "STUB":
                    if config.require_real_models:
                        model.assert_real_backbone()          # raises EmbeddingError
                    logger.warning("[%s] running on a STUB backbone: results are meaningless.", model_name)

            miss = {"rate": float("nan")}

            def embed(arrs):
                bb = getattr(model, "_backbone", None)
                t0, m0 = getattr(bb, "n_total", 0), getattr(bb, "n_detector_miss", 0)
                out = _l2(np.asarray(model.extract_features(arrs), dtype=np.float64))
                dt = getattr(bb, "n_total", 0) - t0
                miss["rate"] = (getattr(bb, "n_detector_miss", 0) - m0) / dt if dt else float("nan")
                return out

            # threshold from CLEAN VALIDATION balanced pairs only
            v_pairs, v_lab = pa["validation"]["balanced"]
            val_emb = embed(X_val)
            val_valid = _valid_rows(val_emb)
            if is_deep:                                   # classical embeddings are scored exactly as before
                val_emb = _sanitize(val_emb, val_valid)
            v_scores = _scores(val_emb, v_pairs)
            val_miss = miss["rate"]
            _warn_if_not_unit(val_emb, val_valid, model_name)
            if val_miss == val_miss and val_miss > 0.05:
                logger.warning("[%s] detector missed %.1f%% of clean validation images "
                               "(scored as zero embeddings / non-matches).", model_name, 100 * val_miss)
            thr = best_accuracy_threshold(v_scores, v_lab)
            emb_dim = int(embed(X_val[:1]).shape[1])
            val_acc = _accuracy_at(v_scores, v_lab, thr)["accuracy"]
            if cal_rec is not None:
                cal_rec.fit(model_name, v_scores, v_lab, thr)
            logger.info("[%s] validation threshold=%.5f (balanced acc=%.4f)", model_name, thr, val_acc)
        except EmbeddingError as exc:
            _fail({"protocol": "verification", "model": model_name, "stage": "setup",
                             "error": f"EmbeddingError: {exc}"})
            if config.require_real_models:
                fatal = OrchestratorError(f"Model '{model_name}' has no real backbone and "
                                          f"require_real_models=true: {exc}")
                break
            done += len(specs); progress("verification", done, total); continue
        except Exception as exc:
            logger.warning("Verification skipped for '%s': %s", model_name, exc)
            _fail({"protocol": "verification", "model": model_name, "stage": "setup", "error": f"{type(exc).__name__}: {exc}"})
            if config.require_real_models and model_name not in trad_names:
                fatal = OrchestratorError(f"Model '{model_name}' unavailable and require_real_models=true: {exc}")
                fatal.__cause__ = exc
                break
            done += len(specs); progress("verification", done, total); continue

        if is_deep:
            (out_dir / "scores").mkdir(exist_ok=True)
            if not (out_dir / "scores" / "test_pairs.npz").exists():
                _atomic_savez(out_dir / "scores" / "test_pairs.npz",
                              image_ids=np.asarray(test_image_ids),
                              bal_pairs=np.asarray(pa["test"]["balanced"][0], dtype=np.int32),
                              bal_labels=np.asarray(pa["test"]["balanced"][1], dtype=bool),
                              full_pairs=np.asarray(pa["test"]["full"][0], dtype=np.int32),
                              full_labels=np.asarray(pa["test"]["full"][1], dtype=bool))
        for spec in specs:
            score_file = (f"{model_name}__{spec.perturbation_type.value}__sev{spec.severity:.2f}.npz"
                          if is_deep else None)
            try:
                emb = embed(perturb_batch(composer, spec, pipeline.normalizer, X_test, test_chips, test_image_ids))
                det_miss = miss["rate"]
                valid = _valid_rows(emb)
                if is_deep:
                    emb = _sanitize(emb, valid)
                row: dict[str, Any] = {
                    "model": model_name, "paradigm": paradigm, "backbone": backbone,
                    "perturbation": spec.perturbation_type.value, "severity": spec.severity,
                    "label": spec.label, "embedding_dim": emb_dim, "threshold": thr,
                    "detector_miss_rate": det_miss, "val_detector_miss_rate": val_miss,
                }
                t_pairs, t_lab = pa["test"]["balanced"]
                s_bal = _scores(emb, t_pairs)
                if is_deep:        # classical rows keep their original columns
                    row.update({"normalization_mode": prep["normalization_mode"], "alignment": prep["alignment"],
                                "crop_fraction": prep["crop_fraction"],
                                "val_balanced_accuracy": val_acc,
                                "val_n_failed_embeddings": int((~val_valid).sum()),
                                "n_test_images": int(valid.size),
                                "n_failed_embeddings": int((~valid).sum()),
                                "n_failed_pairs_balanced": _pairs_with_failure(valid, t_pairs)})
                row.update(_accuracy_at(s_bal, t_lab, thr))
                row["n_genuine"] = int(t_lab.sum()); row["n_impostor_balanced"] = int((~t_lab).sum())
                row.update(_flat_metrics(s_bal, t_lab, ""))
                f_pairs, f_lab = pa["test"]["full"]
                s_full = _scores(emb, f_pairs)
                row["n_impostor_full"] = int((~f_lab).sum())
                row.update(_flat_metrics(s_full, f_lab, "_full"))
                far_full = _accuracy_at(s_full, f_lab, thr)
                row["far_full"] = far_full["far"]
                if is_deep:
                    row["n_failed_pairs_full"] = _pairs_with_failure(valid, f_pairs)
                    row["scores_file"] = score_file
                    _atomic_savez(out_dir / "scores" / score_file,
                                  valid=valid, s_bal=s_bal, s_full=s_full, threshold=np.float64(thr))
                stream.write("row", row)       # recorded first: a row in rows_out is always in the stream
                rows_out.append(row)
                if cal_rec is not None and cal_rec.has(model_name):
                    cal_rec.record(model_name, spec.perturbation_type.value, spec.severity, spec.label, s_bal, t_lab)
            except Exception as exc:
                logger.warning("Verification condition %s/%s failed: %s", model_name, spec.label, exc)
                if score_file is not None and not stream.has_row(score_file):   # no file without its row
                    _discard_score_file(out_dir / "scores" / score_file)
                _fail({"protocol": "verification", "model": model_name, "stage": spec.label,
                                 "error": f"{type(exc).__name__}: {exc}"})
            except BaseException:
                # KeyboardInterrupt / SystemExit: never swallowed. The stream FILE decides (not a local flag, which
                # an interrupt could leave stale): this condition's score file is removed only if its row is not
                # in the stream, so no recorded row ever points at a deleted file. Earlier rows and files are
                # untouched; the interrupt then propagates unchanged and the wrapper records the abort.
                if score_file is not None and not stream.has_row(score_file):
                    _discard_score_file(out_dir / "scores" / score_file)
                raise
            done += 1
            progress("verification", done, total)

    # ---- 4b. Calibration artifacts (sibling directory; a failure here never changes the verification results) ----
    cal_info: dict[str, Any] | None = None
    if cal_rec is not None:
        try:
            cal_info = cal_rec.write(out_dir.with_name(out_dir.name + CAL_SUFFIX))
        except Exception as exc:
            logger.exception("Calibration artifacts could not be written")
            cal_info = {"status": "FAILED", "error": f"{type(exc).__name__}: {exc}"}

    # ---- 5. Write results ------------------------------------------------------
    fields: list[str] = []
    for r in rows_out:
        for k in r:
            if k not in fields:
                fields.append(k)
    with open(out_dir / "verification_results.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(rows_out)
    summary = {
        "protocol": "lfw_fold0_verification",
        "full_lfw": full_lfw,
        "status": "INCOMPLETE" if (failures or fatal or not rows_out or len(rows_out) != total) else "COMPLETE",
        "dataset": {
            "n_identities": n_identities, "n_images": len(records),
            "scope": "official_lfw" if full_lfw else "custom_or_subset",
            "counts_match_official_lfw": counts_match,
            "min_images_per_subject": config.min_images_per_subject,
            "max_subjects": config.max_subjects,
            "fingerprint_sha256": ds_fingerprint,
        },
        "content_sha256": {
            "split": split_content_sha256(manifest),
            "pairs": {p: pair_content_sha256(ps) for p, ps in pair_sets.items()},
        },
        "environment": {k: metadata[k] for k in ("python_version", "numpy_version", "code_version")},
        "provenance": {**run_provenance(model_names),
                       "classical_rng": {"protocol": getattr(config, "classical_rng", "global_seed"),
                                         "pca_svd_solver_resolved": {k: (v if v else "NOT_RECORDED") for k, v in solver_used.items()
                                                                     if k in trad_names}}},
        "fold": fold_index, "k_folds": K_FOLDS, "seed": seed,
        "manifest_path": str(manifest_path), "manifest_sha256": manifest_sha,
        "shuffled_order": shuffled_cmp,
        "pair_manifest_sha256": pair_shas,
        "pair_params": {p: ps.params for p, ps in pair_sets.items()},
        "noise_seeding": noise_seeding_metadata(),
        "threshold_protocol": "max-accuracy on clean validation balanced pairs; frozen for test",
        "preprocessing": preprocessing,
        "n_rows": len(rows_out), "n_rows_expected": total, "failures": failures,
        "stub_backbones": sorted({r["model"] for r in rows_out if r["backbone"] == "STUB"}),
        "results": rows_out,
    }
    if not reference_split:                     # only non-reference splits carry this block: reference runs are unchanged
        summary["split"] = {"reference_split": False, "seed": seed, "fold": fold_index, "k_folds": K_FOLDS,
                            "pair_params": {"seed": seed, "cap": pairs_mod.FOLD0_DEFAULTS["cap"],
                                            "rho": pairs_mod.FOLD0_DEFAULTS["rho"]},
                            "invariants_checked": True, "counts": split_counts}
    if cal_info is not None:
        summary["calibration"] = cal_info
    # Finalization order. A run is COMPLETE only when BOTH the JSON summary says COMPLETE AND the stream ends
    # with a successfully fsynced "end" record saying COMPLETE. So the summary is first written as INCOMPLETE,
    # then the end record is written, and only then is the summary promoted to its final status. Whichever step
    # fails, the summary on disk is INCOMPLETE; a COMPLETE summary can never exist without a finished stream.
    results_path = out_dir / "verification_results.json"
    final_status = summary["status"]
    summary["status"] = "INCOMPLETE"
    _write_json_atomic(results_path, summary)
    stream.write("end", {"status": final_status, "n_rows": len(rows_out), "n_rows_expected": total})
    if final_status != "INCOMPLETE":
        summary["status"] = final_status
        _write_json_atomic(results_path, summary)
    summary["output_dir"] = str(out_dir)
    if fatal is not None:
        raise fatal
    if not rows_out:
        raise OrchestratorError(f"Fold 0 verification produced no results; failures: {failures[:3]}")
    logger.info("Verification complete: %d rows -> %s", len(rows_out), out_dir)
    return summary

"""
faceeval.analysis.fold0_results
===============================
Reproducible analysis of the archived LFW Fold 0 verification results.

Reads ONLY archived ``verification_results.csv`` / ``verification_results.json`` files and writes tables and
figures to a separate, NEW output directory. It never runs an experiment, never recomputes a metric and never writes
into (or modifies) an archived run directory.

Inputs
------
* PRIMARY classical runs, under ``--results-root`` (default ``reports/experiment_results``): ``classical``
  (Fisherfaces, LBPH, HOG-SVM; 288 rows) and ``eigenfaces_final`` (Eigenfaces; 96 rows).
* PRIMARY deep runs, given as explicit directories (they live in a run tree, not under the results root):
  ``--facenet-run DIR`` and ``--arcface-run DIR``, each a directory that holds that run's
  ``verification_results.csv`` and ``verification_results.json`` (96 rows per model).
  With both deep runs the analysis covers 6 models x 16 perturbations x 6 severities = 576 rows; with none it is the
  original four-model, 384-row analysis (the default, unchanged).
* HISTORICAL run: ``eigenfaces_per_image_noise`` (an earlier Eigenfaces run on another code commit). It is read
  separately and is compared with ``eigenfaces_final`` in one dedicated table only. It is never mixed into the
  primary rows.

Run compatibility. Every run is validated on its own (COMPLETE, full LFW, no run-level failures, CSV rows == JSON rows,
exactly 96 rows per model on the full grid, constant frozen threshold, consistent clean baseline). Across runs the
analysis requires the SAME split content hash, validation and test pair hashes, dataset fingerprint, fold/seed and
noise protocol (version, master seed, approved stream match). It deliberately does NOT compare ``code_version`` or
``manifest_sha256`` between runs: separate model runs legitimately come from different commits. Each run's own code
version, manifest hash and environment are preserved in ``provenance.json``.

ArcFace failure semantics (read before interpreting any deep-model number)
--------------------------------------------------------------------------
The archived deep-model metrics are END-TO-END results. An image whose embedding could not be produced (for ArcFace,
mainly the internal InsightFace face detector finding no face) is replaced by a zero vector and every pair that
contains it is still scored; failed pairs are NOT removed. AUC/EER/TAR/FAR of such a condition therefore mix recognition
quality with face-acquisition failure and are NOT valid-pairs-only recognition performance. This module keeps those
archived values exactly as reported and adds failure diagnostics (``table12``/``table13``, figure 7):
``n_test_images``, ``n_failed_embeddings``, ``n_failed_pairs_balanced``, ``n_failed_pairs_full`` and the derived
rates with explicit denominators (image rate = failed images / ``n_test_images``; balanced pair rate = failed balanced
pairs / (``n_genuine`` + ``n_impostor_balanced``); full-pool pair rate = failed full-pool pairs / (``n_genuine`` +
``n_impostor_full``)). Counts summed across conditions are NOT unique failed images. Classical models have no failure
counters; their diagnostic fields are blank (``diagnostics_applicable = false``), never invented. The FaceNet and
ArcFace pipelines also differ in alignment, input size and detection (FaceNet: centre crop, no detector), so they are
end-to-end pipeline results, not a backbone-only comparison.

Metric definitions (all values come straight from the archived CSV; nothing is re-estimated)
---------------------------------------------------------------------------------------
Every row is one (model, perturbation, severity) condition scored on the 764 genuine + 764 impostor balanced
test pairs, using a threshold chosen on clean validation pairs and frozen before testing.

* ``accuracy`` fraction of the 1,528 balanced test pairs classified correctly at the frozen threshold (higher
  is better).
* ``tar``      true accept rate = accepted genuine pairs / 764 (higher is better).
* ``far``      false accept rate = accepted impostor pairs / 764 (lower is better).
* ``auc``      area under the ROC curve over the balanced pairs (threshold-free; higher is better).
* ``eer``      equal error rate, FAR = FRR (threshold-free; lower is better).

Clean baseline: the severity-0.0 row. It does not depend on the perturbation, so all 16 perturbations of a model
carry the same severity-0.0 values; this is verified, not assumed.

Degradation (``harm``): ``delta = value(severity) - value(clean)`` and
``harm = -delta`` for higher-is-better metrics, ``harm = +delta`` for lower-is-better metrics.
POSITIVE HARM ALWAYS MEANS WORSE THAN CLEAN; negative harm means the metric improved.

Uncertainty (INDICATIVE ONLY): see ``STATISTICAL_LIMITATIONS.md``. Intervals assume independent pairs, which is
not true (pairs share images and identities) and are computed from aggregates only. No significance test is
claimed.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

# --------------------------------------------------------------------------------------
# Study design constants
# --------------------------------------------------------------------------------------
MODEL_ORDER = ("eigenfaces", "fisherfaces", "lbph", "hog_svm", "facenet", "arcface")
MODEL_LABELS = {"eigenfaces": "Eigenfaces", "fisherfaces": "Fisherfaces", "lbph": "LBPH", "hog_svm": "HOG-SVM",
                "facenet": "FaceNet", "arcface": "ArcFace"}
CLASSICAL_MODELS = ("eigenfaces", "fisherfaces", "lbph", "hog_svm")
DEEP_MODELS = ("facenet", "arcface")
# Fixed Okabe-Ito colour per model (colour-blind safe; the four classical colours are unchanged, yellow is avoided).
MODEL_COLORS = {"eigenfaces": "#0072B2", "fisherfaces": "#E69F00", "lbph": "#009E73", "hog_svm": "#CC79A7",
                "facenet": "#D55E00", "arcface": "#56B4E9"}
PERTURBATION_ORDER = (
    "gaussian_blur", "motion_blur", "gaussian_noise", "salt_pepper_noise", "speckle_noise", "brightness",
    "contrast", "gamma", "rotation", "scaling", "cropping", "random_occlusion", "face_mask", "sunglasses",
    "jpeg_compression", "resolution_degradation",
)
NOISE_PERTURBATIONS = frozenset({"gaussian_noise", "salt_pepper_noise", "speckle_noise"})
SEVERITIES = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)
CLASSICAL_RUNS = {"classical": ("fisherfaces", "lbph", "hog_svm"), "eigenfaces_final": ("eigenfaces",)}
PRIMARY_RUNS = CLASSICAL_RUNS          # backward-compatible name: the runs that live under the results root
DEEP_RUNS = {"facenet": ("facenet",), "arcface": ("arcface",)}     # located by explicit directories, not the root
HISTORICAL_RUN = "eigenfaces_per_image_noise"
CONDITIONS_PER_MODEL = 96              # 16 perturbations x 6 severities

HIGHER_IS_BETTER = {"accuracy": True, "tar": True, "auc": True, "far": False, "eer": False}
METRICS = tuple(HIGHER_IS_BETTER)
RATE_COLUMNS = (
    "accuracy", "tar", "far", "auc", "eer", "auc_full", "eer_full", "far_full",
    "frr_at_far_0.001", "frr_at_far_0.01", "frr_at_far_0.1",
    "frr_at_far_0.001_full", "frr_at_far_0.01_full", "frr_at_far_0.1_full",
)
INTEGER_COLUMNS = ("n_genuine", "n_impostor_balanced", "n_impostor_full", "embedding_dim")
# Failure counters written by the verification stage for DEEP models only (classical rows do not have them).
DIAGNOSTIC_COLUMNS = ("n_test_images", "n_failed_embeddings", "n_failed_pairs_balanced", "n_failed_pairs_full")
OPTIONAL_DIAGNOSTIC_COLUMNS = ("val_n_failed_embeddings",)
# Content-level fields every compared run must agree on. code_version / manifest_sha256 are deliberately NOT here.
SHARED_PROVENANCE_KEYS = ("split_content_sha256", "pair_content_sha256", "dataset_fingerprint_sha256",
                          "noise_protocol_version", "noise_master_seed", "fold", "k_folds", "seed")
FAILURE_SEMANTICS = ("Deep-model metrics are archived end-to-end results: embeddings that failed are zero vectors and "
                     "their pairs are still scored (not removed). They are not valid-pairs-only recognition performance. "
                     "Failure counts summed across conditions are not unique failed images.")
OPTIONAL_NAN_COLUMNS = ("detector_miss_rate", "val_detector_miss_rate")   # NaN by design for models without a detector
INDICATIVE_Z = 1.959963984540054


class AnalysisError(ValueError):
    """The archived results are structurally unusable (missing/duplicate/inconsistent data)."""


class InvalidMetricError(AnalysisError):
    """A metric value is not a finite number inside its valid range."""


# --------------------------------------------------------------------------------------
# Loading and validation
# --------------------------------------------------------------------------------------
def _parse_float(text: str, column: str, where: str, allow_nan: bool) -> float:
    s = str(text).strip()
    if s == "" or s.lower() == "nan":
        if allow_nan:
            return math.nan
        raise InvalidMetricError(f"{where}: column '{column}' is empty/NaN")
    try:
        v = float(s)
    except ValueError as exc:
        raise InvalidMetricError(f"{where}: column '{column}' is not numeric: {text!r}") from exc
    if math.isinf(v):
        raise InvalidMetricError(f"{where}: column '{column}' is infinite")
    return v


def parse_row(raw: dict[str, str], where: str = "row", allow_nan_metrics: bool = False) -> dict[str, Any]:
    """Convert one CSV row to typed values. Rates must lie in [0, 1]. NaN in a rate column is an error unless
    ``allow_nan_metrics`` (NaN is then kept explicitly and every aggregate excludes it, reporting the count)."""
    row: dict[str, Any] = dict(raw)
    row["severity"] = round(_parse_float(raw["severity"], "severity", where, False), 6)
    for col in RATE_COLUMNS:
        if col in raw:
            v = _parse_float(raw[col], col, where, allow_nan_metrics)
            if not math.isnan(v) and not (0.0 <= v <= 1.0):
                raise InvalidMetricError(f"{where}: column '{col}' = {v} is outside [0, 1]")
            row[col] = v
    for col in INTEGER_COLUMNS:
        if col in raw:
            row[col] = int(_parse_float(raw[col], col, where, False))
    for col in OPTIONAL_NAN_COLUMNS + ("threshold",):
        if col in raw:
            row[col] = _parse_float(raw[col], col, where, col in OPTIONAL_NAN_COLUMNS)
    for col in DIAGNOSTIC_COLUMNS + OPTIONAL_DIAGNOSTIC_COLUMNS:
        if col in raw:
            v = _parse_float(raw[col], col, where, False)
            if v < 0 or v != int(v):
                raise InvalidMetricError(f"{where}: column '{col}' = {v} is not a non-negative integer")
            row[col] = int(v)
    return row


def load_run_dir(directory: Path, allow_nan_metrics: bool = False) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Read one archived run from an explicit directory: ``(typed CSV rows, parsed JSON)``. Both files must exist."""
    d = Path(directory)
    csv_path, json_path = d / "verification_results.csv", d / "verification_results.json"
    for p in (csv_path, json_path):
        if not p.is_file():
            raise AnalysisError(f"missing archived file: {p}")
    with open(csv_path, newline="") as fh:
        rows = [parse_row(r, f"{d.name}.csv line {i + 2}", allow_nan_metrics) for i, r in enumerate(csv.DictReader(fh))]
    with open(json_path) as fh:
        meta = json.load(fh)
    return rows, meta


def load_run(results_root: Path, run: str, allow_nan_metrics: bool = False) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Read one archived run under the results root: ``(typed CSV rows, parsed JSON)``. Both files must exist."""
    return load_run_dir(Path(results_root) / run, allow_nan_metrics)


def _file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _coerce_float(value: Any) -> float | None:
    """``value`` as a float, or None when it is not numeric (booleans are not numbers here; NaN/None/'' map to NaN)."""
    if isinstance(value, bool):
        return None
    if value is None or (isinstance(value, str) and value.strip().lower() in ("", "nan")):
        return math.nan
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _floats_agree(a: float, b: float, tol: float) -> bool:
    return (math.isnan(a) and math.isnan(b)) or (not math.isnan(a) and not math.isnan(b) and abs(a - b) <= tol)


def check_csv_json_agreement(rows: list[dict[str, Any]], meta: dict[str, Any], run: str, tol: float = 1e-9) -> None:
    """The archived JSON ``results`` must carry the same data as the CSV, row by row (same order of conditions).

    EVERY column of the CSV is reconciled with the JSON row, and the two column sets must be identical:
    * identity: ``model``, ``perturbation``, ``severity`` (and therefore the condition order and row count);
    * rate/metric columns (accuracy, TAR, FAR, AUC, EER, FRR@FAR, ``*_full``, ``threshold``, detector miss rates):
      numeric, absolute tolerance ``tol`` (default 1e-9); NaN agrees with NaN only;
    * integer columns (``n_genuine``, ``n_impostor_balanced``, ``n_impostor_full``, ``embedding_dim`` and the failure
      counters): exact equality, and a JSON value that is not an integer is a failure;
    * every other column (label, paradigm, backbone, normalization_mode, alignment, ``crop_fraction``, validation
      accuracy, ``scores_file``, ...): numeric tolerance when both sides are numbers, otherwise exact text.
    INTENTIONAL EXCLUSIONS: none of the row columns is skipped. Only representation is ignored (CSV text versus JSON
    numbers, ``nan``/``null``/empty for a missing value, column order). Missing, extra or malformed fields are
    ``AnalysisError`` and never a Python exception."""
    jr = meta.get("results")
    if not isinstance(jr, list):
        raise AnalysisError(f"{run}: the JSON has no 'results' list")
    if len(jr) != len(rows):
        raise AnalysisError(f"{run}: CSV has {len(rows)} rows but JSON has {len(jr)}")
    typed_float = set(RATE_COLUMNS) | set(OPTIONAL_NAN_COLUMNS) | {"threshold", "severity"}
    typed_int = set(INTEGER_COLUMNS) | set(DIAGNOSTIC_COLUMNS) | set(OPTIONAL_DIAGNOSTIC_COLUMNS)
    for i, (c, j) in enumerate(zip(rows, jr)):
        if not isinstance(j, dict):
            raise AnalysisError(f"{run}: row {i} of the JSON is not an object")
        for key in ("model", "perturbation"):
            if key not in j or str(c[key]) != str(j[key]):
                raise AnalysisError(f"{run}: row {i} {key} differs between CSV ({c[key]}) and JSON ({j.get(key, 'MISSING')})")
        for col in sorted(set(c) ^ set(j)):
            raise AnalysisError(f"{run}: row {i} column '{col}' is present in only one of CSV and JSON")
        for col in c:
            a, b = c[col], j[col]
            where = f"{run}: row {i} {col} differs between CSV ({a}) and JSON ({b})"
            if col in typed_float:
                bf = _coerce_float(b)
                if bf is None:
                    raise AnalysisError(f"{run}: row {i} column '{col}' is not numeric in the JSON: {b!r}")
                if not _floats_agree(a, bf, tol):
                    raise AnalysisError(where)
            elif col in typed_int:
                if isinstance(b, bool) or not isinstance(b, (int, float)) or b != int(b) or int(b) != a:
                    raise AnalysisError(where)
            else:
                af, bf = _coerce_float(a), _coerce_float(b)
                if af is not None and bf is not None:
                    if not _floats_agree(af, bf, tol):
                        raise AnalysisError(where)
                elif af is None and bf is None:
                    if str(a) != str(b):
                        raise AnalysisError(where)
                else:
                    raise AnalysisError(where)


def check_run_metadata(rows: list[dict[str, Any]], meta: dict[str, Any], run: str) -> None:
    """Cross-check the CSV rows against the run-level metadata of the same JSON, for the fields that are present.

    * pair denominators: ``n_genuine`` = pair_params.test.n_genuine, ``n_impostor_balanced`` = ``n_balanced`` and
      ``n_impostor_full`` = ``n_pool`` (the full pool is genuine + impostor pool, the balanced impostors are a sample of
      that pool);
    * deep models: ``n_test_images`` = pair_params.test.n_images (every test image is embedded) and the validation
      failure count cannot exceed pair_params.validation.n_images;
    * ``n_rows_expected`` (when recorded) equals ``n_rows`` and the CSV row count;
    * no stub backbone, ``dataset.counts_match_official_lfw`` not false, and a threshold protocol that names clean
      validation data. Fields that an older archive did not record are skipped, never invented."""
    pp = (meta.get("pair_params") or {})
    test, val = pp.get("test") or {}, pp.get("validation") or {}
    for r in rows:
        tag = f"{run}: {r['model']} {r['perturbation']}@{r['severity']}"
        for col, key in (("n_genuine", "n_genuine"), ("n_impostor_balanced", "n_balanced"), ("n_impostor_full", "n_pool")):
            if key in test and r[col] != test[key]:
                raise AnalysisError(f"{tag}: {col} = {r[col]} but the run's pair_params.test.{key} = {test[key]}")
        if "n_test_images" in r and "n_images" in test and r["n_test_images"] != test["n_images"]:
            raise AnalysisError(f"{tag}: n_test_images = {r['n_test_images']} but pair_params.test.n_images = {test['n_images']}")
        if "val_n_failed_embeddings" in r and "n_images" in val and r["val_n_failed_embeddings"] > val["n_images"]:
            raise AnalysisError(f"{tag}: val_n_failed_embeddings = {r['val_n_failed_embeddings']} exceeds the "
                                f"{val['n_images']} validation images")
    expected = meta.get("n_rows_expected")
    if expected is not None and not (expected == meta.get("n_rows") == len(rows)):
        raise AnalysisError(f"{run}: n_rows_expected {expected}, n_rows {meta.get('n_rows')} and CSV rows {len(rows)} differ")
    if meta.get("stub_backbones"):
        raise AnalysisError(f"{run}: the run used stub backbones {meta['stub_backbones']}")
    if (meta.get("dataset") or {}).get("counts_match_official_lfw") is False:
        raise AnalysisError(f"{run}: the dataset counts do not match official LFW")
    proto = meta.get("threshold_protocol")
    if proto is not None and "clean validation" not in str(proto):
        raise AnalysisError(f"{run}: the recorded threshold protocol does not use clean validation data: {proto!r}")


def _expected_grid(models: tuple[str, ...]) -> set[tuple[str, str, float]]:
    return {(m, p, s) for m in models for p in PERTURBATION_ORDER for s in SEVERITIES}


def _validate_diagnostics(rows: list[dict[str, Any]], model: str, where: str) -> None:
    """Failure-counter sanity for one DEEP model (counters are required for deep models). Archived values are only
    checked, never altered."""
    mine = [r for r in rows if r["model"] == model]
    for r in mine:
        for col in DIAGNOSTIC_COLUMNS:
            if col not in r:
                raise AnalysisError(f"{where}: {model} is missing failure diagnostic column '{col}'")
        if str(r.get("backbone", "")).upper() == "STUB":
            raise AnalysisError(f"{where}: {model} row uses a STUB backbone; only real pretrained models are analysed")
        n_img, f_img = r["n_test_images"], r["n_failed_embeddings"]
        n_bal, n_full = r["n_genuine"] + r["n_impostor_balanced"], r["n_genuine"] + r["n_impostor_full"]
        f_bal, f_full = r["n_failed_pairs_balanced"], r["n_failed_pairs_full"]
        tag = f"{where}: {model} {r['perturbation']}@{r['severity']}"
        if n_img <= 0 or f_img > n_img:
            raise AnalysisError(f"{tag}: failed embeddings {f_img} exceed test images {n_img}")
        if f_bal > n_bal or f_full > n_full or f_bal > f_full:
            raise AnalysisError(f"{tag}: inconsistent failed-pair counts (balanced {f_bal}/{n_bal}, full {f_full}/{n_full})")
        if f_img == 0 and (f_bal or f_full):
            raise AnalysisError(f"{tag}: failed pairs reported without any failed embedding")
        if f_img == n_img and (f_bal != n_bal or f_full != n_full):
            raise AnalysisError(f"{tag}: every image failed but not every pair is counted as failed")
        if "val_n_failed_embeddings" in r and r["val_n_failed_embeddings"] < 0:
            raise AnalysisError(f"{tag}: negative validation failure count")
    if len({r["n_test_images"] for r in mine}) != 1:
        raise AnalysisError(f"{where}: {model} n_test_images is not constant across conditions")
    for col in DIAGNOSTIC_COLUMNS:
        clean = {r[col] for r in mine if r["severity"] == 0.0}
        if len(clean) > 1:
            raise AnalysisError(f"{where}: {model} severity-0.0 '{col}' differs across perturbations: {sorted(clean)[:3]}")
    # The validation embeddings are computed once per model (clean validation images, before the test conditions), so
    # their failure count is one number for the whole run.
    val = {r["val_n_failed_embeddings"] for r in mine if "val_n_failed_embeddings" in r}
    if len(val) > 1:
        raise AnalysisError(f"{where}: {model} val_n_failed_embeddings is not constant across rows: {sorted(val)[:3]}")


def _validate_grid(rows: list[dict[str, Any]], models: tuple[str, ...], where: str) -> None:
    """Structural checks for ``models``: complete 16 x 6 grid per model (no duplicates, nothing extra), consistent
    clean baseline, one frozen threshold per model, and the failure counters of deep models."""
    seen: dict[tuple, int] = defaultdict(int)
    for r in rows:
        seen[(r["model"], r["perturbation"], r["severity"])] += 1
    dups = [k for k, n in seen.items() if n > 1]
    if dups:
        raise AnalysisError(f"{where}: duplicate conditions: {dups[:5]}")
    found = {r["model"] for r in rows}
    if found != set(models):
        raise AnalysisError(f"{where}: model coverage differs: found {sorted(found)}, expected {sorted(models)}")
    expected = _expected_grid(models)
    missing, extra = sorted(expected - set(seen)), sorted(set(seen) - expected)
    if missing or extra:
        raise AnalysisError(f"{where}: condition grid differs: missing {missing[:5]} ({len(missing)}), extra {extra[:5]} ({len(extra)})")
    if len(rows) != len(models) * CONDITIONS_PER_MODEL:
        raise AnalysisError(f"{where}: expected {len(models) * CONDITIONS_PER_MODEL} rows ({len(models)} model(s) x "
                            f"{CONDITIONS_PER_MODEL}), found {len(rows)}")
    for m in models:
        clean = [r for r in rows if r["model"] == m and r["severity"] == 0.0]
        for col in METRICS + ("threshold",):
            vals = {r[col] for r in clean if not math.isnan(r[col])}
            if len(vals) > 1:
                raise AnalysisError(f"{where}: {m}: severity-0.0 '{col}' differs across perturbations: {sorted(vals)[:3]}")
        thr = {r["threshold"] for r in rows if r["model"] == m}
        if len(thr) != 1:
            raise AnalysisError(f"{where}: {m}: the frozen threshold is not constant across rows: {sorted(thr)[:3]}")
        if m in DEEP_MODELS:
            _validate_diagnostics(rows, m, where)


def validate_run(rows: list[dict[str, Any]], run: str, models: tuple[str, ...]) -> None:
    """Validate ONE archived run on its own: exactly 96 rows per expected model on the full grid, etc."""
    _validate_grid(rows, tuple(models), run)


def validate_primary(rows: list[dict[str, Any]], models: tuple[str, ...] = CLASSICAL_MODELS) -> None:
    """Structural checks on the combined primary rows: the complete ``models`` x 16 x 6 design (576 rows for all six
    models, 384 for the four classical ones), no duplicates, consistent clean baselines, one threshold per model."""
    _validate_grid(rows, tuple(models), "primary rows")


def primary_run_dirs(results_root: Path, deep_runs: dict[str, Path] | None = None) -> dict[str, tuple[Path, tuple[str, ...]]]:
    """``{run name: (directory, expected models)}`` for the primary runs. Classical runs live under the results root;
    deep runs are given as explicit directories (``{"facenet": DIR, "arcface": DIR}``)."""
    runs: dict[str, tuple[Path, tuple[str, ...]]] = {
        run: (Path(results_root) / run, models) for run, models in CLASSICAL_RUNS.items()}
    for name, directory in (deep_runs or {}).items():
        if name not in DEEP_RUNS:
            raise AnalysisError(f"unknown deep run '{name}'; expected one of {sorted(DEEP_RUNS)}")
        runs[name] = (Path(directory), DEEP_RUNS[name])
    return runs


def load_primary(results_root: Path, allow_nan_metrics: bool = False,
                 deep_runs: dict[str, Path] | None = None) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Load, cross-check and validate the primary rows plus a provenance record.

    Without ``deep_runs``: the four classical models (384 rows). With ``deep_runs`` (``{"facenet": DIR, "arcface": DIR}``
    or either one): those models are added (576 rows for all six). Each run is validated independently before the
    cross-run compatibility check and the combined validation."""
    rows: list[dict[str, Any]] = []
    prov: dict[str, Any] = {"runs": {}, "input_files": {}, "run_directories": {}}
    for run, (directory, models) in primary_run_dirs(results_root, deep_runs).items():
        prov["run_directories"][run] = str(Path(directory).resolve())
        r, meta = load_run_dir(directory, allow_nan_metrics)
        check_csv_json_agreement(r, meta, run)
        found = {x["model"] for x in r}
        if found != set(models):
            raise AnalysisError(f"{run}: expected models {sorted(models)}, found {sorted(found)}")
        validate_run(r, run, models)
        check_run_metadata(r, meta, run)
        prov["runs"][run] = _provenance(meta, len(r))
        prov["input_files"][run] = {
            f: {"path": str((Path(directory) / f).resolve()), "sha256": _file_sha256(Path(directory) / f)}
            for f in ("verification_results.csv", "verification_results.json")}
        rows.extend(r)
    _check_primary_compatibility(prov["runs"])
    present = {r["model"] for r in rows}
    validate_primary(rows, tuple(m for m in MODEL_ORDER if m in present))
    return rows, prov


def _provenance(meta: dict[str, Any], n_rows: int) -> dict[str, Any]:
    ns = meta.get("noise_seeding") or {}
    return {
        "status": meta.get("status"), "full_lfw": meta.get("full_lfw"), "failures": len(meta.get("failures", [])),
        "n_rows_csv": n_rows, "n_rows_json": meta.get("n_rows"), "n_rows_expected": meta.get("n_rows_expected"),
        "counts_match_official_lfw": (meta.get("dataset") or {}).get("counts_match_official_lfw"),
        "code_version": (meta.get("environment") or {}).get("code_version"),
        "numpy_version": (meta.get("environment") or {}).get("numpy_version"),
        "python_version": (meta.get("environment") or {}).get("python_version"),
        "manifest_sha256": meta.get("manifest_sha256"),
        "split_content_sha256": (meta.get("content_sha256") or {}).get("split"),
        "pair_content_sha256": (meta.get("content_sha256") or {}).get("pairs"),
        "dataset_fingerprint_sha256": (meta.get("dataset") or {}).get("fingerprint_sha256"),
        "noise_protocol_version": ns.get("protocol_version"), "noise_master_seed": ns.get("master_seed"),
        "noise_stream_matches_approved": (ns.get("stream_fingerprint") or {}).get("matches_approved_reference"),
        "fold": meta.get("fold"), "k_folds": meta.get("k_folds"), "seed": meta.get("seed"),
        "pair_manifest_sha256": meta.get("pair_manifest_sha256"),
        "threshold_protocol": meta.get("threshold_protocol"),
        "stub_backbones": meta.get("stub_backbones"),
        "preprocessing": meta.get("preprocessing"),
        # Environment block written by newer verification runs (package versions, weight hashes, ...); older runs
        # did not record it.
        "run_provenance": meta.get("provenance", "not_recorded"),
    }


def _check_primary_compatibility(runs: dict[str, dict[str, Any]]) -> None:
    """Per-run completeness, then agreement of the content-level fields across runs.

    ``code_version`` and ``manifest_sha256`` are intentionally NOT compared: separate model runs may come from
    different commits (the manifest file embeds the code version). The split, pair and dataset content hashes and the
    noise protocol are what make the runs comparable."""
    for run, p in runs.items():
        if p["status"] != "COMPLETE" or p["failures"] or p["full_lfw"] is not True:
            raise AnalysisError(f"{run}: not a complete, failure-free, full-LFW run: {p}")
        if p["n_rows_csv"] != p["n_rows_json"]:
            raise AnalysisError(f"{run}: CSV rows {p['n_rows_csv']} != JSON n_rows {p['n_rows_json']}")
        if p["noise_stream_matches_approved"] is not True:
            raise AnalysisError(f"{run}: the noise stream does not match the approved reference "
                                f"({p['noise_stream_matches_approved']!r})")
    for key in SHARED_PROVENANCE_KEYS:
        values = {json.dumps(p[key], sort_keys=True) for p in runs.values()}
        if len(values) != 1:
            raise AnalysisError(f"primary runs disagree on '{key}': {sorted(values)}")


# --------------------------------------------------------------------------------------
# Statistics helpers
# --------------------------------------------------------------------------------------
def hanley_mcneil_se(auc: float, n_pos: int, n_neg: int) -> float:
    """Hanley & McNeil (1982) standard error of an AUC, assuming independent pairs. NaN for a NaN AUC."""
    if math.isnan(auc):
        return math.nan
    q1, q2 = auc / (2.0 - auc), 2.0 * auc * auc / (1.0 + auc)
    var = (auc * (1.0 - auc) + (n_pos - 1) * (q1 - auc * auc) + (n_neg - 1) * (q2 - auc * auc)) / (n_pos * n_neg)
    return math.sqrt(max(var, 0.0))


def wilson_interval(p: float, n: int, z: float = INDICATIVE_Z) -> tuple[float, float]:
    """Wilson score interval for a proportion from ``n`` independent trials."""
    if math.isnan(p) or n <= 0:
        return (math.nan, math.nan)
    den = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (centre - half, centre + half)


def harm(metric: str, value: float, clean: float) -> float:
    """Degradation relative to clean; positive = worse (see module docstring)."""
    delta = value - clean
    return -delta if HIGHER_IS_BETTER[metric] else delta


def _nanmean_count(values: list[float]) -> tuple[float, int]:
    v = [x for x in values if not math.isnan(x)]
    return (float(np.mean(v)) if v else math.nan, len(v))


# --------------------------------------------------------------------------------------
# Tables
# --------------------------------------------------------------------------------------
def _index(rows: list[dict[str, Any]]) -> dict[tuple[str, str, float], dict[str, Any]]:
    return {(r["model"], r["perturbation"], r["severity"]): r for r in rows}


def _models_in(items: list[dict[str, Any]]) -> tuple[str, ...]:
    """The models present in ``items`` (rows or long cells), in the canonical MODEL_ORDER."""
    present = {x["model"] for x in items}
    return tuple(m for m in MODEL_ORDER if m in present)


def clean_baseline(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """The severity-0.0 values per model (taken from the first perturbation; equality across all was validated)."""
    out: dict[str, dict[str, Any]] = {}
    for m in _models_in(rows):
        first = next(r for r in rows if r["model"] == m and r["severity"] == 0.0)
        out[m] = first
    return out


def table_clean_baseline(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    base = clean_baseline(rows)
    out = []
    for m in _models_in(rows):
        b = base[m]
        out.append({"model": m, "label": MODEL_LABELS[m], "accuracy": b["accuracy"], "tar": b["tar"], "far": b["far"],
                    "auc": b["auc"], "eer": b["eer"], "threshold": b["threshold"],
                    "auc_full_pool": b["auc_full"], "eer_full_pool": b["eer_full"]})
    for metric, reverse in (("auc", True), ("eer", False), ("accuracy", True)):
        order = sorted(out, key=lambda r: (-r[metric] if reverse else r[metric]))
        for rank, r in enumerate(order, 1):
            r[f"rank_{metric}"] = rank
    return out


def table_cells_long(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """One line per (model, perturbation, severity, metric) with clean value, delta and harm."""
    idx, base = _index(rows), clean_baseline(rows)
    out = []
    for m in _models_in(rows):
        for p in PERTURBATION_ORDER:
            for s in SEVERITIES:
                r = idx[(m, p, s)]
                for metric in METRICS:
                    v, c = r[metric], base[m][metric]
                    out.append({"model": m, "perturbation": p, "severity": s, "metric": metric, "value": v, "clean": c,
                                "delta": v - c, "harm": harm(metric, v, c), "higher_is_better": HIGHER_IS_BETTER[metric]})
    return out


def _harm_lookup(cells: list[dict[str, Any]], metric: str) -> dict[tuple[str, str, float], float]:
    return {(c["model"], c["perturbation"], c["severity"]): c["harm"] for c in cells if c["metric"] == metric}


def table_heatmap(cells: list[dict[str, Any]], metric: str = "auc", severity: float | None = 1.0) -> list[dict[str, Any]]:
    """Perturbation x model matrix of harm. ``severity=None`` averages over severities 0.2-1.0 (NaN-excluded)."""
    h = _harm_lookup(cells, metric)
    models = _models_in(cells)
    out = []
    for p in PERTURBATION_ORDER:
        row: dict[str, Any] = {"perturbation": p}
        for m in models:
            if severity is not None:
                row[m] = h[(m, p, severity)]
            else:
                row[m] = _nanmean_count([h[(m, p, s)] for s in SEVERITIES if s > 0])[0]
        out.append(row)
    return out


def table_auc_by_severity(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Mean/min/max AUC across the 16 perturbations per (model, severity); NaN-excluded with a count."""
    out = []
    for m in _models_in(rows):
        for s in SEVERITIES:
            vals = [r["auc"] for r in rows if r["model"] == m and r["severity"] == s]
            mean, n = _nanmean_count(vals)
            valid = [v for v in vals if not math.isnan(v)]
            out.append({"model": m, "severity": s, "mean_auc": mean, "min_auc": min(valid) if valid else math.nan,
                        "max_auc": max(valid) if valid else math.nan, "n_valid": n})
    return out


def table_rankings(cells: list[dict[str, Any]], metric: str = "auc") -> list[dict[str, Any]]:
    """Per model: perturbations ranked 1 = most damaging, by mean harm over severities 0.2-1.0 and by harm at 1.0."""
    h = _harm_lookup(cells, metric)
    out = []
    for m in _models_in(cells):
        stats = []
        for p in PERTURBATION_ORDER:
            mean, n = _nanmean_count([h[(m, p, s)] for s in SEVERITIES if s > 0])
            stats.append({"model": m, "perturbation": p, "mean_harm": mean, "harm_sev1": h[(m, p, 1.0)], "n_valid": n})
        for key, name in (("mean_harm", "rank_mean"), ("harm_sev1", "rank_sev1")):
            order = sorted(stats, key=lambda r: (math.isnan(r[key]), -r[key] if not math.isnan(r[key]) else 0.0, r["perturbation"]))
            for rank, r in enumerate(order, 1):
                r[name] = rank
        out.extend(sorted(stats, key=lambda r: r["rank_mean"]))
    for r in out:
        r["metric"] = metric
    return out


def table_degradation_summary(rows: list[dict[str, Any]], cells: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Per (model, metric): average, severity-1.0 and worst-case harm over the 16 x 5 perturbed conditions."""
    out = []
    for metric in ("auc", "eer", "accuracy"):
        h = _harm_lookup(cells, metric)
        base = clean_baseline(rows)
        idx = _index(rows)
        for m in _models_in(rows):
            keys = [(m, p, s) for p in PERTURBATION_ORDER for s in SEVERITIES if s > 0]
            vals = [h[k] for k in keys]
            mean_all, n_valid = _nanmean_count(vals)
            sev1 = _nanmean_count([h[(m, p, 1.0)] for p in PERTURBATION_ORDER])[0]
            valid = [(h[k], k) for k in keys if not math.isnan(h[k])]
            worst = max(valid, key=lambda t: (t[0], t[1][1], t[1][2])) if valid else (math.nan, (m, "", math.nan))
            best = min(valid, key=lambda t: (t[0], t[1][1], t[1][2])) if valid else (math.nan, (m, "", math.nan))
            clean = base[m][metric]
            out.append({
                "model": m, "metric": metric, "clean": clean, "mean_harm_all": mean_all,
                "mean_harm_pct_of_clean": 100.0 * mean_all / clean if clean else math.nan,
                "mean_harm_sev1": sev1, "worst_harm": worst[0], "worst_perturbation": worst[1][1],
                "worst_severity": worst[1][2], "worst_value": idx[worst[1]][metric] if valid else math.nan,
                "best_harm": best[0], "best_perturbation": best[1][1], "best_severity": best[1][2],
                "n_conditions": len(keys), "n_valid": n_valid, "n_improved": sum(1 for v, _ in valid if v < 0),
            })
    return out


def table_perturbation_across_models(cells: list[dict[str, Any]], metric: str = "auc") -> list[dict[str, Any]]:
    """Mean harm of each perturbation averaged over the models present (severity 1.0 and over 0.2-1.0), ranked.
    ArcFace's value is end-to-end (failed embeddings scored as zero vectors); see the failure diagnostics tables."""
    h = _harm_lookup(cells, metric)
    models = _models_in(cells)
    out = []
    for p in PERTURBATION_ORDER:
        sev1 = _nanmean_count([h[(m, p, 1.0)] for m in models])[0]
        allsev = _nanmean_count([h[(m, p, s)] for m in models for s in SEVERITIES if s > 0])[0]
        out.append({"perturbation": p, "mean_harm_sev1": sev1, "mean_harm_all_severities": allsev,
                    "n_models_harmed_sev1": sum(1 for m in models if h[(m, p, 1.0)] > 0)})
    for key, name in (("mean_harm_all_severities", "rank_all_severities"), ("mean_harm_sev1", "rank_sev1")):
        order = sorted(out, key=lambda r: (-r[key], r["perturbation"]))
        for rank, r in enumerate(order, 1):
            r[name] = rank
    for r in out:
        r["n_models"] = len(models)
    return sorted(out, key=lambda r: r["rank_all_severities"])


def table_clean_intervals(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """INDICATIVE 95% intervals for the clean metrics (independent-pair assumption; see the limitations report)."""
    base = clean_baseline(rows)
    out = []
    for m in _models_in(rows):
        b = base[m]
        n_pos, n_neg = b["n_genuine"], b["n_impostor_balanced"]
        se = hanley_mcneil_se(b["auc"], n_pos, n_neg)
        acc_lo, acc_hi = wilson_interval(b["accuracy"], n_pos + n_neg)
        tar_lo, tar_hi = wilson_interval(b["tar"], n_pos)
        far_lo, far_hi = wilson_interval(b["far"], n_neg)
        out.append({"model": m, "auc": b["auc"], "auc_se_hanley_mcneil": se,
                    "auc_lo": b["auc"] - INDICATIVE_Z * se, "auc_hi": b["auc"] + INDICATIVE_Z * se,
                    "accuracy": b["accuracy"], "accuracy_lo": acc_lo, "accuracy_hi": acc_hi,
                    "tar": b["tar"], "tar_lo": tar_lo, "tar_hi": tar_hi, "far": b["far"], "far_lo": far_lo, "far_hi": far_hi,
                    "eer": b["eer"], "eer_interval": "not computed (no closed form from aggregates)",
                    "n_genuine": n_pos, "n_impostor_balanced": n_neg})
    return out


def table_auc_cells_indicative(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Per condition: AUC, its indicative SE, and harm against the indicative scale of the difference.

    ``scale = z * sqrt(se_clean^2 + se_cell^2)``. It is a descriptive yardstick, NOT a significance test: the clean and
    perturbed scores come from the same pairs (positively correlated, which makes the independent scale too large) while
    pairs also share images (which makes every SE too small). The two biases pull in opposite directions."""
    base = clean_baseline(rows)
    out = []
    for r in rows:
        if r["severity"] == 0.0:
            continue
        m = r["model"]
        b = base[m]
        se_c = hanley_mcneil_se(b["auc"], b["n_genuine"], b["n_impostor_balanced"])
        se = hanley_mcneil_se(r["auc"], r["n_genuine"], r["n_impostor_balanced"])
        scale = INDICATIVE_Z * math.sqrt(se_c ** 2 + se ** 2)
        hm = harm("auc", r["auc"], b["auc"])
        out.append({"model": m, "perturbation": r["perturbation"], "severity": r["severity"], "auc": r["auc"],
                    "auc_clean": b["auc"], "harm": hm, "se_cell": se, "indicative_scale_95": scale,
                    "abs_harm_exceeds_scale": (abs(hm) > scale) if not math.isnan(hm) else ""})
    key = {m: i for i, m in enumerate(MODEL_ORDER)}
    pk = {p: i for i, p in enumerate(PERTURBATION_ORDER)}
    return sorted(out, key=lambda c: (key[c["model"]], pk[c["perturbation"]], c["severity"]))


def table_historical_vs_final(results_root: Path, allow_nan_metrics: bool = False) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Eigenfaces ``eigenfaces_final`` vs the HISTORICAL ``eigenfaces_per_image_noise`` run, one row per condition.

    The historical run used an earlier code commit, so its AUC/EER come from the older grid-based estimate: any
    AUC/EER difference mixes the metric method with the run difference. ``accuracy``/``tar``/``far`` (exact counts at
    the frozen threshold) are the clean signal; for the 13 non-noise perturbations they should be identical."""
    final, fmeta = load_run(results_root, "eigenfaces_final", allow_nan_metrics)
    hist, hmeta = load_run(results_root, HISTORICAL_RUN, allow_nan_metrics)
    check_csv_json_agreement(hist, hmeta, HISTORICAL_RUN)
    f, h = _index(final), _index(hist)
    if set(f) != set(h):
        raise AnalysisError("eigenfaces_final and the historical run cover different conditions")
    out = []
    for p in PERTURBATION_ORDER:
        for s in SEVERITIES:
            a, b = f[("eigenfaces", p, s)], h[("eigenfaces", p, s)]
            row: dict[str, Any] = {"perturbation": p, "severity": s, "is_noise_perturbation": p in NOISE_PERTURBATIONS}
            for metric in METRICS:
                row[f"{metric}_final"], row[f"{metric}_historical"] = a[metric], b[metric]
                row[f"{metric}_diff"] = b[metric] - a[metric]
            out.append(row)
    prov = {"eigenfaces_final": _provenance(fmeta, len(final)), HISTORICAL_RUN: _provenance(hmeta, len(hist))}
    return out, prov


# --------------------------------------------------------------------------------------
# Failure diagnostics (deep models; classical models have no failure counters)
# --------------------------------------------------------------------------------------
def safe_rate(numerator: float, denominator: float) -> float:
    """``numerator / denominator``, NaN (written blank) when the denominator is not positive."""
    return numerator / denominator if denominator and denominator > 0 else math.nan


def table_failure_diagnostics(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """One line per (model, perturbation, severity) with the failure counters, their explicit denominators and rates.

    * ``image_failure_rate``          = ``n_failed_embeddings`` / ``n_test_images``
    * ``balanced_pair_failure_rate``  = ``n_failed_pairs_balanced`` / ``n_balanced_pairs`` (``n_genuine`` + ``n_impostor_balanced``)
    * ``full_pair_failure_rate``      = ``n_failed_pairs_full`` / ``n_full_pairs`` (``n_genuine`` + ``n_impostor_full``)

    A pair counts as failed when at least one of its two images has no valid embedding, so a pair rate is NOT the image
    rate. The archived end-to-end ``auc``/``eer``/``accuracy`` of the same condition are repeated for reference; they
    still include the zero-filled failures. Classical rows have no counters: ``diagnostics_applicable`` is false and
    every count, denominator and rate is blank (nothing is invented)."""
    idx = _index(rows)
    out = []
    for m in _models_in(rows):
        for p in PERTURBATION_ORDER:
            for s in SEVERITIES:
                r = idx[(m, p, s)]
                line: dict[str, Any] = {"model": m, "label": MODEL_LABELS[m], "perturbation": p, "severity": s,
                                        "is_clean_condition": s == 0.0}
                applicable = all(c in r for c in DIAGNOSTIC_COLUMNS)
                line["diagnostics_applicable"] = applicable
                blank = {k: "" for k in ("n_test_images", "n_failed_embeddings", "image_failure_rate", "n_balanced_pairs",
                                         "n_failed_pairs_balanced", "balanced_pair_failure_rate", "n_full_pairs",
                                         "n_failed_pairs_full", "full_pair_failure_rate", "val_n_failed_embeddings")}
                if applicable:
                    n_bal = r["n_genuine"] + r["n_impostor_balanced"]
                    n_full = r["n_genuine"] + r["n_impostor_full"]
                    blank.update({
                        "n_test_images": r["n_test_images"], "n_failed_embeddings": r["n_failed_embeddings"],
                        "image_failure_rate": safe_rate(r["n_failed_embeddings"], r["n_test_images"]),
                        "n_balanced_pairs": n_bal, "n_failed_pairs_balanced": r["n_failed_pairs_balanced"],
                        "balanced_pair_failure_rate": safe_rate(r["n_failed_pairs_balanced"], n_bal),
                        "n_full_pairs": n_full, "n_failed_pairs_full": r["n_failed_pairs_full"],
                        "full_pair_failure_rate": safe_rate(r["n_failed_pairs_full"], n_full),
                        "val_n_failed_embeddings": r.get("val_n_failed_embeddings", "")})
                line.update(blank)
                line.update({"auc_end_to_end": r["auc"], "eer_end_to_end": r["eer"], "accuracy_end_to_end": r["accuracy"]})
                out.append(line)
    return out


def table_failure_model_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Per model: failure totals and rates, separately for the 16 clean (severity 0.0) rows and the 80 perturbed rows.

    All sums are summed ACROSS CONDITIONS (the same image can fail in many conditions), so they are not unique failed
    images; a pooled rate divides by the matching summed denominator. The condition-level variation (maximum,
    number of conditions with any failure, number with every image failed) is kept next to the totals; the per-condition
    values are in ``table12``. Classical models: ``diagnostics_applicable`` false, everything else blank."""
    idx = _index(rows)
    out = []
    for m in _models_in(rows):
        mine = [idx[(m, p, s)] for p in PERTURBATION_ORDER for s in SEVERITIES]
        line: dict[str, Any] = {"model": m, "label": MODEL_LABELS[m]}
        if not all(c in r for r in mine for c in DIAGNOSTIC_COLUMNS):
            line["diagnostics_applicable"] = False
            out.append(line)
            continue
        line["diagnostics_applicable"] = True
        line["n_test_images_per_condition"] = mine[0]["n_test_images"]
        for tag, subset in (("clean", [r for r in mine if r["severity"] == 0.0]),
                            ("perturbed", [r for r in mine if r["severity"] > 0.0])):
            n_img = sum(r["n_test_images"] for r in subset)
            f_img = sum(r["n_failed_embeddings"] for r in subset)
            n_bal = sum(r["n_genuine"] + r["n_impostor_balanced"] for r in subset)
            f_bal = sum(r["n_failed_pairs_balanced"] for r in subset)
            n_full = sum(r["n_genuine"] + r["n_impostor_full"] for r in subset)
            f_full = sum(r["n_failed_pairs_full"] for r in subset)
            rates = [safe_rate(r["n_failed_embeddings"], r["n_test_images"]) for r in subset]
            worst = max(subset, key=lambda r: (safe_rate(r["n_failed_embeddings"], r["n_test_images"]), r["perturbation"], r["severity"]))
            line.update({
                f"{tag}_n_conditions": len(subset),
                f"{tag}_sum_failed_embeddings": f_img, f"{tag}_sum_image_denominator": n_img,
                f"{tag}_pooled_image_failure_rate": safe_rate(f_img, n_img),
                f"{tag}_sum_failed_pairs_balanced": f_bal, f"{tag}_sum_balanced_pair_denominator": n_bal,
                f"{tag}_pooled_balanced_pair_failure_rate": safe_rate(f_bal, n_bal),
                f"{tag}_sum_failed_pairs_full": f_full, f"{tag}_sum_full_pair_denominator": n_full,
                f"{tag}_pooled_full_pair_failure_rate": safe_rate(f_full, n_full),
                f"{tag}_min_condition_image_failure_rate": min(rates), f"{tag}_max_condition_image_failure_rate": max(rates),
                f"{tag}_max_rate_perturbation": worst["perturbation"], f"{tag}_max_rate_severity": worst["severity"],
                f"{tag}_n_conditions_with_any_failure": sum(1 for r in subset if r["n_failed_embeddings"] > 0),
                f"{tag}_n_conditions_all_images_failed": sum(1 for r in subset if r["n_failed_embeddings"] == r["n_test_images"]),
            })
        line["all_rows_sum_failed_embeddings"] = sum(r["n_failed_embeddings"] for r in mine)
        line["note"] = "sums are across conditions, not unique failed images; rates are pooled sum/sum"
        out.append(line)
    return out


# --------------------------------------------------------------------------------------
# Writing
# --------------------------------------------------------------------------------------
def _fmt(v: Any) -> str:
    if isinstance(v, float):
        return "" if math.isnan(v) else format(v, ".10g")
    if isinstance(v, bool):
        return "true" if v else "false"
    return str(v)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise AnalysisError(f"refusing to write an empty table: {path}")
    cols: list[str] = []
    for r in rows:
        for k in r:
            if k not in cols:
                cols.append(k)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh, lineterminator="\n")
        w.writerow(cols)
        for r in rows:
            w.writerow([_fmt(r.get(c, "")) for c in cols])




# --------------------------------------------------------------------------------------
# Figures
# --------------------------------------------------------------------------------------
ARCFACE_NOTE = ("ArcFace: archived end-to-end values (failed embeddings are zero vectors, failed pairs are kept); "
                "see table12/table13 and fig7 for failure rates.")


def make_figures(rows: list[dict[str, Any]], cells: list[dict[str, Any]], out_dir: Path) -> list[Path]:
    """Draw the figures for the models present in ``rows`` (4 or 6): model order, colours, tick positions, legends and
    panel grids all follow that set. Figure 7 (failure rates) is drawn only when a deep model with counters is present."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from faceeval.visualization.style import apply_style

    apply_style()
    out_dir.mkdir(parents=True, exist_ok=True)
    models = _models_in(rows)
    n = len(models)
    colors = {m: MODEL_COLORS[m] for m in models}
    paths: list[Path] = []

    def save(fig, name: str) -> None:
        if "arcface" in models:
            fig.text(0.01, -0.015, ARCFACE_NOTE, fontsize=5.5, ha="left", va="top", style="italic")
        p = out_dir / name
        fig.savefig(p, dpi=200, bbox_inches="tight")
        plt.close(fig)
        paths.append(p)

    # 1. clean baseline (bars with indicative intervals)
    iv = {r["model"]: r for r in table_clean_intervals(rows)}
    base = {r["model"]: r for r in table_clean_baseline(rows)}
    fig, axes = plt.subplots(1, 3, figsize=(8.0 if n <= 4 else 11.0, 3.0))
    for ax, metric, title in zip(axes, ("auc", "eer", "accuracy"), ("AUC (higher is better)", "EER (lower is better)", "Accuracy at frozen threshold")):
        vals = [base[m][metric] for m in models]
        err = None
        if metric in ("auc", "accuracy"):
            lo = [vals[i] - iv[m][f"{metric}_lo"] for i, m in enumerate(models)]
            hi = [iv[m][f"{metric}_hi"] - vals[i] for i, m in enumerate(models)]
            err = [lo, hi]
        ax.bar(range(n), vals, color=[colors[m] for m in models], yerr=err, capsize=3)
        ax.set_xticks(range(n))
        ax.set_xticklabels([MODEL_LABELS[m] for m in models], rotation=30, ha="right", fontsize=7)
        ax.set_ylim(0.0, min(max(vals) * 1.25, 1.1))          # a rate or AUC axis never runs far past 1.0
        ax.axhline(0.5, color="grey", lw=0.6, ls=":")
        ax.set_title(title, fontsize=8)
        for i, v in enumerate(vals):
            top = v + (err[1][i] if err is not None else 0.0)
            ax.text(i, top + 0.012, f"{v:.3f}", ha="center", fontsize=6)
    fig.suptitle("Clean baseline, LFW Fold 0 (bars: indicative 95% intervals, independent-pair assumption; dotted line: chance level)", fontsize=7)
    fig.tight_layout()
    save(fig, "fig1_clean_baseline.png")

    # 2/3. heatmaps (positive = AUC lost); annotation colour follows the cell luminance
    for severity, name, title in ((1.0, "fig2_auc_harm_heatmap_severity1.png", "AUC lost at severity 1.0 (positive = worse than clean)"),
                                  (None, "fig3_auc_harm_heatmap_mean_severity.png", "Mean AUC lost over severities 0.2-1.0 (positive = worse)")):
        t = table_heatmap(cells, "auc", severity)
        mat = np.array([[r[m] for m in models] for r in t], dtype=float)
        lim = float(np.nanmax(np.abs(mat)))
        cmap = plt.get_cmap("coolwarm")
        norm = matplotlib.colors.Normalize(vmin=-lim, vmax=lim)
        fig, ax = plt.subplots(figsize=(6.0 if n <= 4 else 8.0, 6.0))
        im = ax.imshow(mat, cmap=cmap, norm=norm, aspect="auto")
        cb = fig.colorbar(im, ax=ax, shrink=0.8)
        cb.set_label("AUC lost (clean - perturbed)", fontsize=8)
        ax.set_xticks(range(n))
        ax.set_xticklabels([MODEL_LABELS[m] for m in models], fontsize=8, rotation=0 if n <= 4 else 30, ha="center" if n <= 4 else "right")
        ax.set_yticks(range(len(t)))
        ax.set_yticklabels([r["perturbation"] for r in t], fontsize=7)
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                rgba = cmap(norm(mat[i, j]))
                lum = 0.299 * rgba[0] + 0.587 * rgba[1] + 0.114 * rgba[2]
                ax.text(j, i, f"{mat[i, j]:+.3f}", ha="center", va="center", fontsize=6.5, color="white" if lum < 0.45 else "black")
        ax.set_title(title, fontsize=9)
        ax.grid(False)
        fig.tight_layout()
        save(fig, name)

    # 4. mean AUC vs severity
    t = table_auc_by_severity(rows)
    fig, ax = plt.subplots(figsize=(5.0, 3.5))
    for m in models:
        r = [x for x in t if x["model"] == m]
        ax.plot([x["severity"] for x in r], [x["mean_auc"] for x in r], marker="o", ms=3, color=colors[m], label=MODEL_LABELS[m])
        ax.fill_between([x["severity"] for x in r], [x["min_auc"] for x in r], [x["max_auc"] for x in r], color=colors[m], alpha=0.10)
    ax.axhline(0.5, color="grey", lw=0.6, ls=":")
    ax.set_xlabel("severity")
    ax.set_ylabel("AUC (mean over 16 perturbations; band = min-max)")
    if n <= 4:
        ax.legend(fontsize=7)
    else:                                          # six lines: keep the legend off the data
        ax.legend(fontsize=7, loc="center left", bbox_to_anchor=(1.02, 0.5))
    ax.set_title("Severity-wise AUC", fontsize=9)
    fig.tight_layout()
    save(fig, "fig4_auc_severity_mean.png")

    # 5. AUC vs severity per perturbation
    idx = _index(rows)
    fig, axes = plt.subplots(4, 4, figsize=(10.0, 8.0), sharex=True, sharey=True)
    for ax, p in zip(axes.ravel(), PERTURBATION_ORDER):
        for m in models:
            ax.plot(SEVERITIES, [idx[(m, p, s)]["auc"] for s in SEVERITIES], marker="o", ms=2.5, lw=1.0, color=colors[m], label=MODEL_LABELS[m])
        ax.axhline(0.5, color="grey", lw=0.5, ls=":")
        ax.set_title(p, fontsize=7)
        ax.tick_params(labelsize=6)
    fig.supxlabel("severity", fontsize=8)
    fig.supylabel("AUC", fontsize=8)
    if n <= 4:
        axes[0, 0].legend(fontsize=6)
        fig.tight_layout()
    else:                                          # the first panel's lines (FaceNet/ArcFace, AUC ~1) sit under an inner legend
        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.tight_layout(rect=(0.0, 0.0, 0.9, 1.0))
        fig.legend(handles, labels, loc="center right", fontsize=7, title="model", title_fontsize=7)
    save(fig, "fig5_auc_severity_by_perturbation.png")

    # 6. per-model degradation rankings (mean harm over severities), one panel per model
    rk = table_rankings(cells, "auc")
    ncols = n if n <= 4 else 3
    nrows = math.ceil(n / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(11.0 if n <= 4 else 3.7 * ncols, 4.2 * nrows), sharex=True, squeeze=False)
    for ax, m in zip(axes.ravel(), models):
        r = [x for x in rk if x["model"] == m][::-1]
        ax.barh([x["perturbation"] for x in r], [x["mean_harm"] for x in r], color=colors[m])
        ax.axvline(0, color="black", lw=0.6)
        ax.set_title(MODEL_LABELS[m], fontsize=8)
        ax.tick_params(labelsize=6)
    for ax in axes.ravel()[n:]:
        ax.set_visible(False)
    for ax in axes[:, 0]:
        ax.set_ylabel("perturbation (top = most damaging)", fontsize=7)
    fig.supxlabel("mean AUC lost over severities 0.2-1.0 (negative = AUC gained)", fontsize=8)
    fig.tight_layout()
    save(fig, "fig6_degradation_rankings.png")

    # 7. image failure rate of the deep models (only when failure counters exist)
    diag = [r for r in table_failure_diagnostics(rows) if r["diagnostics_applicable"]]
    dmodels = tuple(m for m in models if any(r["model"] == m for r in diag))
    if dmodels:
        fig, axes = plt.subplots(1, len(dmodels), figsize=(3.6 * len(dmodels) + 0.8, 5.6), squeeze=False)
        cmap = plt.get_cmap("magma_r")
        for ax, m in zip(axes[0], dmodels):
            lut = {(r["perturbation"], r["severity"]): r["image_failure_rate"] for r in diag if r["model"] == m}
            mat = np.array([[lut[(p, s)] for s in SEVERITIES] for p in PERTURBATION_ORDER], dtype=float)
            im = ax.imshow(mat, cmap=cmap, vmin=0.0, vmax=1.0, aspect="auto")
            ax.set_xticks(range(len(SEVERITIES)))
            ax.set_xticklabels([f"{s:.1f}" for s in SEVERITIES], fontsize=7)
            ax.set_yticks(range(len(PERTURBATION_ORDER)))
            ax.set_yticklabels(PERTURBATION_ORDER if ax is axes[0][0] else [], fontsize=7)
            for i in range(mat.shape[0]):
                for j in range(mat.shape[1]):
                    rgba = cmap(mat[i, j])
                    lum = 0.299 * rgba[0] + 0.587 * rgba[1] + 0.114 * rgba[2]
                    ax.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center", fontsize=6, color="white" if lum < 0.45 else "black")
            ax.set_title(f"{MODEL_LABELS[m]}: failed-embedding rate", fontsize=8)
            ax.set_xlabel("severity", fontsize=8)
            ax.grid(False)
        fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.8).set_label("failed embeddings / test images", fontsize=8)
        save(fig, "fig7_failure_rates.png")
    return paths


# --------------------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------------------
def _is_inside(path: Path, parent: Path) -> bool:
    return path == parent or parent in path.parents


def run_analysis(results_root: Path, out_dir: Path, figures: bool = True,
                 deep_runs: dict[str, Path] | None = None, overwrite: bool = False) -> dict[str, Path]:
    """Run the whole analysis. Writes only under ``out_dir``.

    ``out_dir`` must be a NEW (or empty) directory: it may not lie inside, or contain, any input run directory, and an
    existing non-empty directory is refused unless ``overwrite`` is true (so an earlier analysis is never replaced by
    accident). ``deep_runs`` maps ``"facenet"``/``"arcface"`` to the directory holding that run's results files; without
    it the original four-model analysis is produced."""
    results_root, out_dir = Path(results_root).resolve(), Path(out_dir).resolve()
    deep = {k: Path(v).resolve() for k, v in (deep_runs or {}).items()}
    inputs = [(results_root / run).resolve() for run in (*CLASSICAL_RUNS, HISTORICAL_RUN)] + list(deep.values())
    for d in inputs:
        if _is_inside(out_dir, d):
            raise AnalysisError(f"refusing to write inside the archived run directory {d}")
        if _is_inside(d, out_dir):
            raise AnalysisError(f"refusing to write into {out_dir}: it contains the archived run directory {d}")
    if out_dir.exists() and not out_dir.is_dir():
        raise AnalysisError(f"output path exists and is not a directory: {out_dir}")
    if out_dir.is_dir() and any(out_dir.iterdir()) and not overwrite:
        raise AnalysisError(f"refusing to write into the non-empty directory {out_dir} (use a new directory, or --overwrite)")
    rows, prov = load_primary(results_root, deep_runs=deep)
    cells = table_cells_long(rows)
    hist_rows, hist_prov = table_historical_vs_final(results_root)
    tables = out_dir / "tables"
    written: dict[str, Path] = {}

    def put(name: str, data: list[dict[str, Any]]) -> None:
        write_csv(tables / f"{name}.csv", data)
        written[name] = tables / f"{name}.csv"

    put("table1_clean_baseline", table_clean_baseline(rows))
    put("table2_clean_intervals_indicative", table_clean_intervals(rows))
    put("table3_auc_harm_heatmap_severity1", table_heatmap(cells, "auc", 1.0))
    put("table4_auc_harm_heatmap_mean_severity", table_heatmap(cells, "auc", None))
    put("table5_auc_by_severity", table_auc_by_severity(rows))
    put("table6_degradation_rankings_auc", table_rankings(cells, "auc"))
    put("table7_degradation_summary", table_degradation_summary(rows, cells))
    put("table8_perturbation_across_models_auc", table_perturbation_across_models(cells, "auc"))
    put("table9_auc_cells_indicative_scale", table_auc_cells_indicative(rows))
    put("table10_all_cells_long", cells)
    put("table11_historical_vs_final_eigenfaces", hist_rows)
    put("table12_failure_diagnostics_by_condition", table_failure_diagnostics(rows))
    put("table13_failure_diagnostics_model_summary", table_failure_model_summary(rows))
    if figures:
        for p in make_figures(rows, cells, out_dir / "figures"):
            written[p.stem] = p
    models = list(_models_in(rows))
    # With --overwrite nothing is deleted, so files of an earlier analysis that this run did not rewrite would stay
    # behind. They are listed here instead of being silently mixed in or removed.
    rewritten = {p.resolve() for p in written.values()}
    stale = sorted(q.relative_to(out_dir).as_posix() for q in out_dir.rglob("*")
                   if q.is_file() and q.resolve() not in rewritten and q.name != "provenance.json")
    provenance = {"primary_runs": prov["runs"], "historical_run": {HISTORICAL_RUN: hist_prov[HISTORICAL_RUN]},
                  "input_files": prov["input_files"], "run_directories": prov["run_directories"],
                  "stale_files_not_rewritten": stale,
                  "n_primary_rows": len(rows), "models": models, "n_perturbations": len(PERTURBATION_ORDER),
                  "severities": list(SEVERITIES), "source": str(results_root),
                  "deep_run_directories": {k: str(v) for k, v in sorted(deep.items())},
                  "cross_run_checks": {"required_equal": list(SHARED_PROVENANCE_KEYS),
                                       "recorded_not_compared": ["code_version", "manifest_sha256", "numpy_version",
                                                                 "python_version", "run_provenance"]},
                  "failure_semantics": FAILURE_SEMANTICS}
    (out_dir / "provenance.json").write_text(json.dumps(provenance, indent=2, sort_keys=True) + "\n")
    written["provenance"] = out_dir / "provenance.json"
    return written


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description=("Analyse the archived LFW Fold 0 verification results (no experiment is run, no metric is "
                     "recomputed). Without deep-run options: the four classical models (384 rows). With --facenet-run "
                     "and --arcface-run: all six models (576 rows). Deep-model values are archived END-TO-END results: "
                     "failed embeddings are zero vectors and their pairs are still scored, so ArcFace AUC/EER are not "
                     "valid-pairs-only recognition performance; see tables 12-13 and figure 7 for the failure rates."),
        epilog=("Example (six models, new output directory):\n"
                "  python -m faceeval.analysis.fold0_results --results-root reports/experiment_results \\\n"
                "      --facenet-run <facenet verification run dir> --arcface-run <arcface verification run dir> \\\n"
                "      --out reports/experiment_results/analysis_20261119_six_models"),
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results-root", default="reports/experiment_results", type=Path,
                    help="folder holding the classical/, eigenfaces_final/ and eigenfaces_per_image_noise/ archives")
    ap.add_argument("--facenet-run", default=None, type=Path, metavar="DIR",
                    help="directory containing the FaceNet verification_results.csv and .json (not under the results root)")
    ap.add_argument("--arcface-run", default=None, type=Path, metavar="DIR",
                    help="directory containing the ArcFace verification_results.csv and .json (not under the results root)")
    ap.add_argument("--out", default=None, type=Path,
                    help="NEW output directory (default: <results-root>/analysis); an existing non-empty one is refused")
    ap.add_argument("--overwrite", action="store_true", help="allow writing into an existing non-empty output directory")
    ap.add_argument("--no-figures", action="store_true")
    return ap


def main(argv: list[str] | None = None) -> int:
    a = build_parser().parse_args(argv)
    out = a.out or (a.results_root / "analysis")
    deep = {k: v for k, v in (("facenet", a.facenet_run), ("arcface", a.arcface_run)) if v is not None}
    try:
        written = run_analysis(a.results_root, out, figures=not a.no_figures, deep_runs=deep, overwrite=a.overwrite)
    except AnalysisError as exc:
        print(f"ANALYSIS REFUSED: {exc}")
        return 2
    for k, v in written.items():
        print(f"{k}: {v}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

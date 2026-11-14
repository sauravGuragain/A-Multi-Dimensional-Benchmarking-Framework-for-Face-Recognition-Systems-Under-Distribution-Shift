"""
faceeval.analysis.fold0_results
===============================
Reproducible analysis of the archived LFW Fold 0 verification results.

Reads ONLY the archived ``verification_results.csv`` / ``verification_results.json`` files under
``reports/experiment_results/<run>/`` and writes tables and figures to a separate output directory. It never
runs an experiment and never writes into (or modifies) the archived run directories.

Inputs
------
* PRIMARY runs (the 384 comparison rows): ``classical`` (Fisherfaces, LBPH, HOG-SVM) and ``eigenfaces_final``
  (Eigenfaces). They must share one Fold 0 manifest, one code commit and one split hash.
* HISTORICAL run: ``eigenfaces_per_image_noise`` (an earlier Eigenfaces run on another code commit). It is read
  separately and is compared with ``eigenfaces_final`` in one dedicated table only. It is never mixed into the
  primary rows.

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
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

# --------------------------------------------------------------------------------------
# Study design constants
# --------------------------------------------------------------------------------------
MODEL_ORDER = ("eigenfaces", "fisherfaces", "lbph", "hog_svm")
MODEL_LABELS = {"eigenfaces": "Eigenfaces", "fisherfaces": "Fisherfaces", "lbph": "LBPH", "hog_svm": "HOG-SVM"}
PERTURBATION_ORDER = (
    "gaussian_blur", "motion_blur", "gaussian_noise", "salt_pepper_noise", "speckle_noise", "brightness",
    "contrast", "gamma", "rotation", "scaling", "cropping", "random_occlusion", "face_mask", "sunglasses",
    "jpeg_compression", "resolution_degradation",
)
NOISE_PERTURBATIONS = frozenset({"gaussian_noise", "salt_pepper_noise", "speckle_noise"})
SEVERITIES = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)
PRIMARY_RUNS = {"classical": ("fisherfaces", "lbph", "hog_svm"), "eigenfaces_final": ("eigenfaces",)}
HISTORICAL_RUN = "eigenfaces_per_image_noise"

HIGHER_IS_BETTER = {"accuracy": True, "tar": True, "auc": True, "far": False, "eer": False}
METRICS = tuple(HIGHER_IS_BETTER)
RATE_COLUMNS = (
    "accuracy", "tar", "far", "auc", "eer", "auc_full", "eer_full", "far_full",
    "frr_at_far_0.001", "frr_at_far_0.01", "frr_at_far_0.1",
    "frr_at_far_0.001_full", "frr_at_far_0.01_full", "frr_at_far_0.1_full",
)
INTEGER_COLUMNS = ("n_genuine", "n_impostor_balanced", "n_impostor_full", "embedding_dim")
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
    return row


def load_run(results_root: Path, run: str, allow_nan_metrics: bool = False) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Read one archived run: ``(typed CSV rows, parsed JSON)``. Both files must exist."""
    d = Path(results_root) / run
    csv_path, json_path = d / "verification_results.csv", d / "verification_results.json"
    for p in (csv_path, json_path):
        if not p.is_file():
            raise AnalysisError(f"missing archived file: {p}")
    with open(csv_path, newline="") as fh:
        rows = [parse_row(r, f"{run}.csv line {i + 2}", allow_nan_metrics) for i, r in enumerate(csv.DictReader(fh))]
    with open(json_path) as fh:
        meta = json.load(fh)
    return rows, meta


def check_csv_json_agreement(rows: list[dict[str, Any]], meta: dict[str, Any], run: str, tol: float = 1e-9) -> None:
    """The archived JSON ``results`` must carry the same numbers as the CSV (same order of conditions)."""
    jr = meta.get("results", [])
    if len(jr) != len(rows):
        raise AnalysisError(f"{run}: CSV has {len(rows)} rows but JSON has {len(jr)}")
    for i, (c, j) in enumerate(zip(rows, jr)):
        for key in ("model", "perturbation"):
            if str(c[key]) != str(j[key]):
                raise AnalysisError(f"{run}: row {i} {key} differs between CSV ({c[key]}) and JSON ({j[key]})")
        for col in METRICS + ("threshold",):
            a, b = c[col], float(j[col])
            if not (math.isnan(a) and math.isnan(b)) and abs(a - b) > tol:
                raise AnalysisError(f"{run}: row {i} {col} differs between CSV ({a}) and JSON ({b})")


def validate_primary(rows: list[dict[str, Any]]) -> None:
    """Structural checks on the primary rows: complete 4 x 16 x 6 design, no duplicates, consistent clean baseline."""
    seen: dict[tuple, int] = defaultdict(int)
    for r in rows:
        seen[(r["model"], r["perturbation"], r["severity"])] += 1
    dups = [k for k, n in seen.items() if n > 1]
    if dups:
        raise AnalysisError(f"duplicate conditions: {dups[:5]}")
    models = {r["model"] for r in rows}
    if models != set(MODEL_ORDER):
        raise AnalysisError(f"model coverage differs: found {sorted(models)}, expected {sorted(MODEL_ORDER)}")
    expected = {(m, p, s) for m in MODEL_ORDER for p in PERTURBATION_ORDER for s in SEVERITIES}
    missing, extra = sorted(expected - set(seen)), sorted(set(seen) - expected)
    if missing or extra:
        raise AnalysisError(f"condition grid differs: missing {missing[:5]} ({len(missing)}), extra {extra[:5]} ({len(extra)})")
    if len(rows) != len(MODEL_ORDER) * len(PERTURBATION_ORDER) * len(SEVERITIES):
        raise AnalysisError(f"expected 384 primary rows, found {len(rows)}")
    for m in MODEL_ORDER:
        clean = [r for r in rows if r["model"] == m and r["severity"] == 0.0]
        for col in METRICS + ("threshold",):
            vals = {r[col] for r in clean if not math.isnan(r[col])}
            if len(vals) > 1:
                raise AnalysisError(f"{m}: severity-0.0 '{col}' differs across perturbations: {sorted(vals)[:3]}")
        thr = {r["threshold"] for r in rows if r["model"] == m}
        if len(thr) != 1:
            raise AnalysisError(f"{m}: the frozen threshold is not constant across rows: {sorted(thr)[:3]}")


def load_primary(results_root: Path, allow_nan_metrics: bool = False) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Load, cross-check and validate the 384 primary rows plus a provenance record."""
    rows: list[dict[str, Any]] = []
    prov: dict[str, Any] = {"runs": {}}
    for run, models in PRIMARY_RUNS.items():
        r, meta = load_run(results_root, run, allow_nan_metrics)
        check_csv_json_agreement(r, meta, run)
        found = {x["model"] for x in r}
        if found != set(models):
            raise AnalysisError(f"{run}: expected models {sorted(models)}, found {sorted(found)}")
        prov["runs"][run] = _provenance(meta, len(r))
        rows.extend(r)
    _check_primary_compatibility(prov["runs"])
    validate_primary(rows)
    return rows, prov


def _provenance(meta: dict[str, Any], n_rows: int) -> dict[str, Any]:
    ns = meta.get("noise_seeding") or {}
    return {
        "status": meta.get("status"), "full_lfw": meta.get("full_lfw"), "failures": len(meta.get("failures", [])),
        "n_rows_csv": n_rows, "n_rows_json": meta.get("n_rows"),
        "code_version": (meta.get("environment") or {}).get("code_version"),
        "numpy_version": (meta.get("environment") or {}).get("numpy_version"),
        "python_version": (meta.get("environment") or {}).get("python_version"),
        "manifest_sha256": meta.get("manifest_sha256"),
        "split_content_sha256": (meta.get("content_sha256") or {}).get("split"),
        "pair_content_sha256": (meta.get("content_sha256") or {}).get("pairs"),
        "dataset_fingerprint_sha256": (meta.get("dataset") or {}).get("fingerprint_sha256"),
        "noise_protocol_version": ns.get("protocol_version"), "noise_master_seed": ns.get("master_seed"),
        "noise_stream_matches_approved": (ns.get("stream_fingerprint") or {}).get("matches_approved_reference"),
    }


def _check_primary_compatibility(runs: dict[str, dict[str, Any]]) -> None:
    for run, p in runs.items():
        if p["status"] != "COMPLETE" or p["failures"] or p["full_lfw"] is not True:
            raise AnalysisError(f"{run}: not a complete, failure-free, full-LFW run: {p}")
        if p["n_rows_csv"] != p["n_rows_json"]:
            raise AnalysisError(f"{run}: CSV rows {p['n_rows_csv']} != JSON n_rows {p['n_rows_json']}")
    for key in ("code_version", "manifest_sha256", "split_content_sha256", "pair_content_sha256",
                "dataset_fingerprint_sha256", "noise_protocol_version", "noise_master_seed"):
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


def clean_baseline(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """The severity-0.0 values per model (taken from the first perturbation; equality across all was validated)."""
    out: dict[str, dict[str, Any]] = {}
    for m in MODEL_ORDER:
        first = next(r for r in rows if r["model"] == m and r["severity"] == 0.0)
        out[m] = first
    return out


def table_clean_baseline(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    base = clean_baseline(rows)
    out = []
    for m in MODEL_ORDER:
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
    for m in MODEL_ORDER:
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
    out = []
    for p in PERTURBATION_ORDER:
        row: dict[str, Any] = {"perturbation": p}
        for m in MODEL_ORDER:
            if severity is not None:
                row[m] = h[(m, p, severity)]
            else:
                row[m] = _nanmean_count([h[(m, p, s)] for s in SEVERITIES if s > 0])[0]
        out.append(row)
    return out


def table_auc_by_severity(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Mean/min/max AUC across the 16 perturbations per (model, severity); NaN-excluded with a count."""
    out = []
    for m in MODEL_ORDER:
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
    for m in MODEL_ORDER:
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
        for m in MODEL_ORDER:
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
    """Mean harm of each perturbation averaged over the four models (severity 1.0 and over 0.2-1.0), ranked."""
    h = _harm_lookup(cells, metric)
    out = []
    for p in PERTURBATION_ORDER:
        sev1 = _nanmean_count([h[(m, p, 1.0)] for m in MODEL_ORDER])[0]
        allsev = _nanmean_count([h[(m, p, s)] for m in MODEL_ORDER for s in SEVERITIES if s > 0])[0]
        out.append({"perturbation": p, "mean_harm_sev1": sev1, "mean_harm_all_severities": allsev,
                    "n_models_harmed_sev1": sum(1 for m in MODEL_ORDER if h[(m, p, 1.0)] > 0)})
    for key, name in (("mean_harm_all_severities", "rank_all_severities"), ("mean_harm_sev1", "rank_sev1")):
        order = sorted(out, key=lambda r: (-r[key], r["perturbation"]))
        for rank, r in enumerate(order, 1):
            r[name] = rank
    return sorted(out, key=lambda r: r["rank_all_severities"])


def table_clean_intervals(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """INDICATIVE 95% intervals for the clean metrics (independent-pair assumption; see the limitations report)."""
    base = clean_baseline(rows)
    out = []
    for m in MODEL_ORDER:
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
def make_figures(rows: list[dict[str, Any]], cells: list[dict[str, Any]], out_dir: Path) -> list[Path]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from faceeval.visualization.style import OKABE_ITO, apply_style

    apply_style()
    out_dir.mkdir(parents=True, exist_ok=True)
    colors = {m: OKABE_ITO[i] for i, m in enumerate(MODEL_ORDER)}
    paths: list[Path] = []

    def save(fig, name: str) -> None:
        p = out_dir / name
        fig.savefig(p, dpi=200, bbox_inches="tight")
        plt.close(fig)
        paths.append(p)

    # 1. clean baseline (bars with indicative intervals)
    iv = {r["model"]: r for r in table_clean_intervals(rows)}
    base = {r["model"]: r for r in table_clean_baseline(rows)}
    fig, axes = plt.subplots(1, 3, figsize=(8.0, 3.0))
    for ax, metric, title in zip(axes, ("auc", "eer", "accuracy"), ("AUC (higher is better)", "EER (lower is better)", "Accuracy at frozen threshold")):
        vals = [base[m][metric] for m in MODEL_ORDER]
        err = None
        if metric in ("auc", "accuracy"):
            lo = [vals[i] - iv[m][f"{metric}_lo"] for i, m in enumerate(MODEL_ORDER)]
            hi = [iv[m][f"{metric}_hi"] - vals[i] for i, m in enumerate(MODEL_ORDER)]
            err = [lo, hi]
        ax.bar(range(4), vals, color=[colors[m] for m in MODEL_ORDER], yerr=err, capsize=3)
        ax.set_xticks(range(4))
        ax.set_xticklabels([MODEL_LABELS[m] for m in MODEL_ORDER], rotation=30, ha="right", fontsize=7)
        ax.set_ylim(0.0, max(vals) * 1.25)
        ax.axhline(0.5 if metric != "eer" else 0.5, color="grey", lw=0.6, ls=":")
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
        mat = np.array([[r[m] for m in MODEL_ORDER] for r in t], dtype=float)
        lim = float(np.nanmax(np.abs(mat)))
        cmap = plt.get_cmap("coolwarm")
        norm = matplotlib.colors.Normalize(vmin=-lim, vmax=lim)
        fig, ax = plt.subplots(figsize=(6.0, 6.0))
        im = ax.imshow(mat, cmap=cmap, norm=norm, aspect="auto")
        cb = fig.colorbar(im, ax=ax, shrink=0.8)
        cb.set_label("AUC lost (clean - perturbed)", fontsize=8)
        ax.set_xticks(range(4))
        ax.set_xticklabels([MODEL_LABELS[m] for m in MODEL_ORDER], fontsize=8)
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
    for m in MODEL_ORDER:
        r = [x for x in t if x["model"] == m]
        ax.plot([x["severity"] for x in r], [x["mean_auc"] for x in r], marker="o", ms=3, color=colors[m], label=MODEL_LABELS[m])
        ax.fill_between([x["severity"] for x in r], [x["min_auc"] for x in r], [x["max_auc"] for x in r], color=colors[m], alpha=0.10)
    ax.axhline(0.5, color="grey", lw=0.6, ls=":")
    ax.set_xlabel("severity")
    ax.set_ylabel("AUC (mean over 16 perturbations; band = min-max)")
    ax.legend(fontsize=7)
    ax.set_title("Severity-wise AUC", fontsize=9)
    fig.tight_layout()
    save(fig, "fig4_auc_severity_mean.png")

    # 5. AUC vs severity per perturbation
    idx = _index(rows)
    fig, axes = plt.subplots(4, 4, figsize=(10.0, 8.0), sharex=True, sharey=True)
    for ax, p in zip(axes.ravel(), PERTURBATION_ORDER):
        for m in MODEL_ORDER:
            ax.plot(SEVERITIES, [idx[(m, p, s)]["auc"] for s in SEVERITIES], marker="o", ms=2.5, lw=1.0, color=colors[m], label=MODEL_LABELS[m])
        ax.axhline(0.5, color="grey", lw=0.5, ls=":")
        ax.set_title(p, fontsize=7)
        ax.tick_params(labelsize=6)
    axes[0, 0].legend(fontsize=6)
    fig.supxlabel("severity", fontsize=8)
    fig.supylabel("AUC", fontsize=8)
    fig.tight_layout()
    save(fig, "fig5_auc_severity_by_perturbation.png")

    # 6. per-model degradation rankings (mean harm over severities)
    rk = table_rankings(cells, "auc")
    fig, axes = plt.subplots(1, 4, figsize=(11.0, 4.2), sharex=True)
    for ax, m in zip(axes, MODEL_ORDER):
        r = [x for x in rk if x["model"] == m][::-1]
        ax.barh([x["perturbation"] for x in r], [x["mean_harm"] for x in r], color=colors[m])
        ax.axvline(0, color="black", lw=0.6)
        ax.set_title(MODEL_LABELS[m], fontsize=8)
        ax.tick_params(labelsize=6)
    axes[0].set_ylabel("perturbation (top = most damaging)", fontsize=7)
    fig.supxlabel("mean AUC lost over severities 0.2-1.0 (negative = AUC gained)", fontsize=8)
    fig.tight_layout()
    save(fig, "fig6_degradation_rankings.png")
    return paths


# --------------------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------------------
def run_analysis(results_root: Path, out_dir: Path, figures: bool = True) -> dict[str, Path]:
    """Run the whole analysis. Writes only under ``out_dir`` (which must not be inside an archived run directory)."""
    results_root, out_dir = Path(results_root).resolve(), Path(out_dir).resolve()
    for run in (*PRIMARY_RUNS, HISTORICAL_RUN):
        d = (results_root / run).resolve()
        if out_dir == d or d in out_dir.parents:
            raise AnalysisError(f"refusing to write inside the archived run directory {d}")
    rows, prov = load_primary(results_root)
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
    if figures:
        for p in make_figures(rows, cells, out_dir / "figures"):
            written[p.stem] = p
    provenance = {"primary_runs": prov["runs"], "historical_run": {HISTORICAL_RUN: hist_prov[HISTORICAL_RUN]},
                  "n_primary_rows": len(rows), "models": list(MODEL_ORDER), "n_perturbations": len(PERTURBATION_ORDER),
                  "severities": list(SEVERITIES), "source": str(results_root)}
    (out_dir / "provenance.json").write_text(json.dumps(provenance, indent=2, sort_keys=True) + "\n")
    written["provenance"] = out_dir / "provenance.json"
    return written


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Analyse the archived LFW Fold 0 verification results (no experiment is run).")
    ap.add_argument("--results-root", default="reports/experiment_results", type=Path)
    ap.add_argument("--out", default=None, type=Path, help="default: <results-root>/analysis")
    ap.add_argument("--no-figures", action="store_true")
    a = ap.parse_args(argv)
    out = a.out or (a.results_root / "analysis")
    written = run_analysis(a.results_root, out, figures=not a.no_figures)
    for k, v in written.items():
        print(f"{k}: {v}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""
faceeval.analysis.multi_split
=============================
Aggregate several identity-disjoint verification runs (different seed and/or fold of the SAME protocol) into
per-(model, perturbation, severity) mean / SD / t-based confidence interval tables.

Rules
-----
* Input: run directories that each hold ``verification_results.csv`` + ``verification_results.json``. Runs are only READ.
* Every run must be COMPLETE, use the same models/perturbations/severities, the same ``pair_params`` cap/rho and the
  same dataset fingerprint; otherwise the aggregation is refused. Runs with an identical (seed, fold) are refused as
  duplicates. The reference run (seed 42, fold 0) may be included; it is just one more split.
* Historical results are never mixed with these tables silently: the output records every run's seed, fold and
  manifest/content hashes.
* Comparability is CHECKED, not assumed. Besides the dataset fingerprint, pair parameters and condition set, every run
  must agree on: protocol and threshold protocol, dataset scope, per-model preprocessing/alignment record, the classical
  random-state protocol (``provenance.classical_rng.protocol``, default ``global_seed`` for runs that predate it) and the
  SHA-256 of each model's weight files (``provenance.weights``). A run with any STUB backbone is refused. Seed and fold
  are the only things allowed to differ. Every run must contain exactly the same set of models (a model missing from, or
  extra in, any run is refused and named). Weights recorded as NOT_RECORDED, or not recorded at all, in EVERY run are
  allowed and listed under ``comparability.weights_unverified``; recorded in some runs but not others is refused unless
  ``--allow-unverified-weights`` is given; differing recorded hashes are ALWAYS refused. Only the forms written by the
  runner are accepted for ``provenance.weights.<model>``: the string ``NOT_RECORDED`` or a non-empty object mapping file
  names to 64-hex SHA-256 digests; an empty object, any other string or any other type is refused as malformed. A
  duplicated (model, perturbation, severity) row inside one run is refused (it would be counted as an extra split).
  Comparison fields that no run records at all (older result files) are accepted but listed under
  ``comparability.fields_not_recorded`` so that nothing is silently assumed.
* The across-split SD is the sample SD (ddof=1); the "naive" interval is the Student-t interval of the mean over
  splits. Splits of one dataset share images, so they are NOT independent samples; the naive interval is a description
  of split-to-split spread, never a population confidence interval. With n<2 splits SD and CI are NaN (never invented).
* The kind of split set is classified (see ``design_report``): folds of ONE seed (test identity sets disjoint, training
  partitions overlap), several seeds (test identity sets overlap), or a mix. Every row carries ``model_kind``
  (``pretrained``: weights fixed, nothing is trained on a split; ``trained_per_split``: fitted on each split's training
  partition) and ``interval_status`` saying how its interval may be read:
    - ``spread_across_disjoint_test_identities``: pretrained model, folds of one seed, >= 5 splits. Different people in
      every test set, no training variance, so the SD is a fair split-to-split spread (still one dataset; not a claim
      about other populations).
    - ``corrected_interval_only``: trained model, folds of one seed, >= 5 splits. Training partitions of different folds
      overlap, so the naive interval is TOO NARROW. The ``*_corrected_ci_*`` columns use the Nadeau-Bengio corrected
      resampled t: variance of the mean = (1/n + n_test/n_train) * s^2, with n_test/n_train taken from the manifests.
      This is an approximation (it assumes the correlation between splits is about n_test/(n_test+n_train)) and is still
      not a population interval.
    - ``descriptive_only``: several seeds, overlapping or unverifiable test identities, or an unknown model. The test
      sets share people and the models share training data; no interval is a valid confidence interval. Read SD/range only.
    - ``too_few_splits``: fewer than 5 splits (otherwise as above). Intervals from so few splits are unreliable.
    - ``single_split``: n = 1; no spread exists.
* Pass the output to a NEW directory (never inside a run directory).
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path
from typing import Sequence

import numpy as np

KEYS = ("model", "perturbation", "severity")
METRICS = ("accuracy", "tar", "far", "auc", "eer")
OUT_TABLE = "multi_split_by_condition.csv"
OUT_RUNS = "multi_split_runs.csv"
OUT_PROV = "multi_split_provenance.json"
CAVEAT = ("Splits are different identity-disjoint partitions of the same dataset; they overlap in images, so they are "
          "not independent samples. SD is across splits (ddof=1). The naive Student-t interval measures sensitivity to the "
          "split and is too narrow for models trained on overlapping training partitions; see each row's interval_status. "
          "No interval is a population confidence interval.")


PRETRAINED_MODELS = frozenset({"facenet", "arcface", "insightface", "dlib_fr"})
TRAINED_MODELS = frozenset({"eigenfaces", "fisherfaces", "pca_svm", "lbph", "hog_svm", "knn"})
MIN_SPLITS_FOR_INTERVAL = 5
NOT_RECORDED = "NOT_RECORDED"
RNG_PROTOCOLS = ("global_seed", "isolated_seed")      # core.config: Literal["global_seed", "isolated_seed"]
_SHA256_HEX = re.compile(r"[0-9a-fA-F]{64}")
STATUS_SINGLE, STATUS_DESCRIPTIVE, STATUS_FEW = "single_split", "descriptive_only", "too_few_splits"
STATUS_PRETRAINED_OK = "spread_across_disjoint_test_identities"
STATUS_CORRECTED = "corrected_interval_only"


def model_kind(name: str) -> str:
    return "pretrained" if name in PRETRAINED_MODELS else ("trained_per_split" if name in TRAINED_MODELS else "unknown")


class MultiSplitError(RuntimeError):
    pass


def t_crit(n: int, conf: float = 0.95) -> float:
    from scipy import stats
    return float(stats.t.ppf(0.5 + conf / 2.0, n - 1))


def _load(run: Path) -> tuple[dict, list[dict]]:
    j, c = run / "verification_results.json", run / "verification_results.csv"
    if not j.is_file() or not c.is_file():
        raise MultiSplitError(f"{run}: verification_results.json/.csv missing")
    summary = json.loads(j.read_text())
    if summary.get("status") != "COMPLETE":
        raise MultiSplitError(f"{run}: status is {summary.get('status')!r}, not COMPLETE")
    with open(c, newline="") as fh:
        rows = list(csv.DictReader(fh))
    return summary, rows


def _partition_identities(summary: dict, part: str) -> set[str] | None:
    """Identities of one partition from the run's canonical manifest, or None if it cannot be read/verified."""
    import hashlib
    try:
        raw = Path(summary["manifest_path"]).read_bytes()
        if hashlib.sha256(raw).hexdigest() != summary["manifest_sha256"]:
            return None
        _head, *lines = raw.decode("utf-8").splitlines()
        return {ln.split("|")[1] for ln in lines if ln.startswith(part + "|")}
    except (OSError, KeyError, ValueError, IndexError):
        return None


def _test_identities(summary: dict) -> set[str] | None:
    return _partition_identities(summary, "test")


def _train_test_ratio(loaded: list) -> float | str:
    """Mean n_test / n_train (identities) over the splits, or NOT_RECORDED if a manifest cannot be read."""
    ratios = []
    for s, _ in loaded:
        te, tr = _partition_identities(s, "test"), _partition_identities(s, "train")
        if not te or not tr:
            return "NOT_RECORDED"
        ratios.append(len(te) / len(tr))
    return float(np.mean(ratios)) if ratios else "NOT_RECORDED"


PREPROCESSING_KEYS = ("alignment", "normalization_mode", "crop_fraction", "chip_size", "padding", "detector",
                      "face_selection", "requested_alignment")


def _prep_signature(summary: dict) -> dict:
    out = {}
    for model, rec in sorted((summary.get("preprocessing") or {}).items()):
        out[model] = {k: rec.get(k) for k in PREPROCESSING_KEYS if isinstance(rec, dict) and k in rec}
    return out


def _model_set(run: Path, rows: list[dict]) -> set[str]:
    models = {x.get("model") for x in rows}
    if None in models or "" in models:
        raise MultiSplitError(f"{run}: a result row has no 'model' value; the file is not a valid verification_results.csv.")
    return models


def _check_model_sets(runs: Sequence[Path], loaded: list) -> set[str]:
    """Every run must contain exactly the models of the first run. Returns that model set."""
    want = _model_set(runs[0], loaded[0][1])
    for r, (_s, rows) in zip(runs, loaded):
        got = _model_set(r, rows)
        if got != want:
            raise MultiSplitError(
                f"{r}: model set differs from the first run ({runs[0]}): missing {sorted(want - got)}, extra {sorted(got - want)}. "
                f"Every split must contain exactly the same models, otherwise some models would be summarised over fewer "
                f"splits than others. Restore or re-run the missing models, or aggregate only runs that share one model set.")
    return want


def _check_rows(run: Path, rows: list[dict]) -> None:
    """Row-level sanity of one run's CSV: non-empty, key columns present, numeric severity/metrics, no duplicate condition."""
    if not rows:
        raise MultiSplitError(f"{run}: verification_results.csv has no result rows.")
    seen: dict[tuple, int] = {}
    for line, x in enumerate(rows, start=2):                      # CSV line 1 is the header
        for k in KEYS:
            if x.get(k) in (None, ""):
                raise MultiSplitError(f"{run}: verification_results.csv line {line} has no {k!r} value.")
        try:
            sev = str(float(x["severity"]))
        except ValueError:
            raise MultiSplitError(f"{run}: verification_results.csv line {line}: severity {x['severity']!r} is not a number.") from None
        for met in METRICS:
            try:
                float(x[met])
            except (KeyError, TypeError, ValueError):
                raise MultiSplitError(f"{run}: verification_results.csv line {line}: column {met!r} is missing or not numeric "
                                      f"({x.get(met)!r}).") from None
        cond = (x["model"], x["perturbation"], sev)
        if cond in seen:
            raise MultiSplitError(f"{run}: condition {cond} appears more than once (CSV lines {seen[cond]} and {line}). Each "
                                  f"(model, perturbation, severity) must occur exactly once per run; a duplicated row would be "
                                  f"counted as an extra split and bias the mean and SD.")
        seen[cond] = line


def _provenance(run: Path, summary: dict) -> dict:
    """The run's ``provenance`` object ({} for older runs that have none). A present but non-object value is malformed."""
    if "provenance" not in summary:
        return {}
    p = summary["provenance"]
    if not isinstance(p, dict):
        raise MultiSplitError(f"{run}: 'provenance' must be an object, found {type(p).__name__}; the result file is malformed.")
    return p


def _rng_protocol(run: Path, summary: dict) -> str:
    """Classical random-state protocol. Absent (runs that predate it) means the historical ``global_seed``; a present value
    must be one of the protocols the runner can write."""
    rng = _provenance(run, summary).get("classical_rng", {})
    if not isinstance(rng, dict):
        raise MultiSplitError(f"{run}: provenance.classical_rng must be an object, found {type(rng).__name__}.")
    protocol = rng.get("protocol", "global_seed")
    if protocol not in RNG_PROTOCOLS:
        raise MultiSplitError(f"{run}: provenance.classical_rng.protocol is {protocol!r}; expected one of {list(RNG_PROTOCOLS)}.")
    return protocol


def _weight_signature(run: Path, summary: dict, model: str) -> dict | None:
    """The recorded weight signature of ``model`` in one run: ``{file name: lower-case SHA-256}``, or None if the run does not
    record it (older run without ``provenance.weights``, model absent from it, or the explicit string ``NOT_RECORDED``).
    Anything else is malformed and refused; in particular an empty object is NOT a signature and NOT a legitimate
    "not recorded". An empty ``weights`` container is fine: the runner writes it for runs without pretrained models."""
    weights = _provenance(run, summary).get("weights", {})
    where = f"{run}: provenance.weights"
    if not isinstance(weights, dict):
        raise MultiSplitError(f"{where} must be an object, found {type(weights).__name__}; the result file is malformed.")
    if model not in weights:
        return None
    w = weights[model]
    if isinstance(w, str):
        if w == NOT_RECORDED:
            return None
        raise MultiSplitError(f"{where}.{model} is the string {w!r}; the only accepted string is {NOT_RECORDED!r}.")
    if not isinstance(w, dict):
        raise MultiSplitError(f"{where}.{model} is a {type(w).__name__}; expected {NOT_RECORDED!r} or an object of "
                              f"{{file name: SHA-256}}.")
    if not w:
        raise MultiSplitError(f"{where}.{model} is an empty object. It is neither a weight signature nor an explicit "
                              f"{NOT_RECORDED!r}; the run's provenance is malformed or incomplete. Fix or re-record it.")
    sig = {}
    for name, digest in w.items():
        if not isinstance(name, str) or not name or not isinstance(digest, str) or not _SHA256_HEX.fullmatch(digest):
            raise MultiSplitError(f"{where}.{model}[{name!r}] = {digest!r} is not a 64-hex-digit SHA-256; the result file is malformed.")
        sig[name] = digest.lower()
    return sig


def check_comparable(runs: Sequence[Path], loaded: list, allow_unverified_weights: bool = False) -> dict:
    """Refuse runs that differ in anything other than seed/fold. Returns a record of what was verified."""
    ref_run, ref = runs[0], loaded[0][0]
    models = _check_model_sets(runs, loaded)

    def prov(s: dict, *path, default=None):
        cur: object = s
        for k in path:
            if not isinstance(cur, dict) or k not in cur:
                return default
            cur = cur[k]
        return cur

    not_recorded: dict[str, list[str]] = {}
    for r, (s, _rows) in zip(runs, loaded):
        if "stub_backbones" not in s:
            not_recorded.setdefault("stub_backbones", []).append(str(r))
            continue
        stubs = s["stub_backbones"]
        if not isinstance(stubs, (list, tuple)):
            raise MultiSplitError(f"{r}: stub_backbones must be a list, found {type(stubs).__name__}; the result file is malformed.")
        if stubs:
            raise MultiSplitError(f"{r}: run used STUB backbone(s) {sorted(stubs)}; stub results are meaningless and "
                                  f"cannot be aggregated. Re-run those models with real weights.")
    checks = {"protocol": lambda r, s: s.get("protocol"), "threshold_protocol": lambda r, s: s.get("threshold_protocol"),
              "dataset scope": lambda r, s: prov(s, "dataset", "scope"), "full_lfw": lambda r, s: s.get("full_lfw"),
              "classical RNG protocol": _rng_protocol,
              "preprocessing/alignment": lambda r, s: _prep_signature(s)}
    for name, fn in checks.items():
        want = fn(ref_run, ref)
        for r, (s, _rows) in zip(runs, loaded):
            got = fn(r, s)
            if got != want:
                raise MultiSplitError(f"{r}: {name} differs from {ref_run}: {got!r} vs {want!r}. Runs that differ in "
                                      f"{name} are different protocols and must not be aggregated; only seed and fold may differ.")
        if want is None or want == {}:                       # equal only because NO run records it (older result files)
            not_recorded.setdefault(name, []).extend(str(r) for r in runs)
    unverified = set()
    for m in sorted(models):
        sigs = [_weight_signature(r, s, m) for r, (s, _rows) in zip(runs, loaded)]
        recorded = [x for x in sigs if x is not None]
        if not recorded:
            if m in PRETRAINED_MODELS:
                unverified.add(m)
            continue
        if any(x != recorded[0] for x in recorded[1:]):
            files = sorted({f for x in recorded for f in set(x) | set(recorded[0]) if x.get(f) != recorded[0].get(f)})
            raise MultiSplitError(f"model {m}: weight file hashes differ between runs (files: {files}); different weights are "
                                  f"different models and are never aggregated, even with --allow-unverified-weights.")
        if len(recorded) != len(sigs):
            if not allow_unverified_weights:
                missing = [str(r) for r, x in zip(runs, sigs) if x is None]
                raise MultiSplitError(f"model {m}: weight hashes are recorded in some runs but not others (not recorded in: {missing}); "
                                      f"cannot verify that the same weights were used. If those runs are known to use the same weights "
                                      f"(e.g. an older reference run made before provenance existed), pass --allow-unverified-weights; "
                                      f"the model is then listed under comparability.weights_unverified.")
            unverified.add(m)
    return {"checked": sorted(checks) + ["identical model set", "stub_backbones empty", "weights sha256 (where recorded)"],
            "weights_unverified": sorted(unverified), "fields_not_recorded": {k: sorted(set(v)) for k, v in sorted(not_recorded.items())}}


def design_report(loaded: list) -> dict:
    """What the set of splits actually is, so that the interval can be read correctly."""
    seeds = sorted({s.get("seed") for s, _ in loaded}, key=str)
    folds = sorted({s.get("fold") for s, _ in loaded}, key=str)
    sets = [_test_identities(s) for s, _ in loaded]
    if any(x is None for x in sets):
        overlap, disjoint = "NOT_RECORDED", "NOT_RECORDED"
    else:
        worst = 0.0
        for i in range(len(sets)):
            for j in range(i + 1, len(sets)):
                union = len(sets[i] | sets[j])
                worst = max(worst, len(sets[i] & sets[j]) / union if union else 0.0)
        overlap, disjoint = worst, worst == 0.0
    if disjoint is True and len(seeds) == 1:
        meaning = ("Test identity sets are disjoint (folds of ONE seed). Fixed pretrained models are evaluated on separate "
                   "identities, so the fold-to-fold SD is a reasonable spread; for models trained here the training "
                   "partitions overlap, which makes the interval too narrow.")
    elif disjoint is True:
        meaning = "Test identity sets are disjoint but come from different seeds; read the spread as descriptive."
    else:
        meaning = ("Test identity sets OVERLAP (or could not be checked): the splits are not independent, so the interval "
                   "describes sensitivity to the split only and understates uncertainty about a new population.")
    n = len(loaded)
    same_seed_pairs = sum(1 for i in range(n) for j in range(i + 1, n) if loaded[i][0].get("seed") == loaded[j][0].get("seed"))
    total_pairs = n * (n - 1) // 2
    if n < 2:
        kind = "single_split"
    elif len(seeds) == 1:
        kind = "folds_within_one_seed"
    elif same_seed_pairs == 0:
        kind = "different_seeds"
    else:
        kind = "mixed_seeds_and_folds"
    return {"design_kind": kind, "n_splits": n, "seeds": seeds, "folds": folds,
            "pairs_same_seed_different_fold": same_seed_pairs, "pairs_different_seed": total_pairs - same_seed_pairs,
            "max_pairwise_test_identity_jaccard": overlap, "test_identity_sets_disjoint": disjoint,
            "train_test_identity_ratio": _train_test_ratio(loaded), "interval_interpretation": meaning,
            "assumptions": ["Splits come from one dataset (one source, one collection process); nothing here generalises beyond it.",
                            "Folds of ONE seed: test identities are disjoint, training partitions overlap (each fold's training set contains the other folds' test people).",
                            "Different seeds: independent shuffles of the same people; test identity sets overlap, so results are positively correlated.",
                            "Pretrained models are not fitted on a split, so only test-set sampling varies; trained models also vary with their training partition.",
                            "No interval here is a population confidence interval."]}


def aggregate(runs: Sequence[Path], out_dir: Path, conf: float = 0.95, overwrite: bool = False,
              allow_unverified_weights: bool = False) -> dict:
    runs = [Path(r) for r in runs]
    out_dir = Path(out_dir)
    if not runs:
        raise MultiSplitError("no runs given")
    for r in runs:
        if out_dir.resolve() == r.resolve() or r.resolve() in out_dir.resolve().parents:
            raise MultiSplitError("output directory may not be inside a run directory")
    if out_dir.exists() and any(out_dir.iterdir()) and not overwrite:
        raise MultiSplitError(f"{out_dir} exists and is not empty (use --overwrite)")
    loaded = [_load(r) for r in runs]
    for r, (_s, rows) in zip(runs, loaded):
        _check_rows(r, rows)

    ident = {}
    for r, (s, rows) in zip(runs, loaded):
        key = (s.get("seed"), s.get("fold"))
        if key in ident:
            raise MultiSplitError(f"duplicate split (seed, fold)={key}: {r} and {ident[key]}")
        ident[key] = r
    ref = loaded[0][0]
    _check_model_sets(runs, loaded)
    ref_cond = {(x["model"], x["perturbation"], str(float(x["severity"]))) for x in loaded[0][1]}
    for r, (s, rows) in zip(runs, loaded):
        cond = {(x["model"], x["perturbation"], str(float(x["severity"]))) for x in rows}
        if cond != ref_cond:
            missing, extra = sorted(ref_cond - cond), sorted(cond - ref_cond)
            raise MultiSplitError(f"{r}: condition set differs from the first run ({runs[0]}): {len(missing)} missing, {len(extra)} extra "
                                  f"(e.g. missing {missing[:3]}, extra {extra[:3]}). Every run must contain the same "
                                  f"(model, perturbation, severity) conditions.")
        if s.get("k_folds") != ref.get("k_folds"):
            raise MultiSplitError(f"{r}: k_folds differs")
        if s["dataset"]["fingerprint_sha256"] != ref["dataset"]["fingerprint_sha256"]:
            raise MultiSplitError(f"{r}: dataset fingerprint differs (different data than the first run)")
        for part in ("validation", "test"):
            a, b = s["pair_params"][part], ref["pair_params"][part]
            if (a["cap"], a["rho"]) != (b["cap"], b["rho"]):
                raise MultiSplitError(f"{r}: pair cap/rho differ in {part}")

    comparability = check_comparable(runs, loaded, allow_unverified_weights)
    n = len(runs)
    grouped: dict[tuple, list[dict]] = {}
    for _s, rows in loaded:
        for x in rows:
            grouped.setdefault((x["model"], x["perturbation"], str(float(x["severity"]))), []).append(x)
    tc = t_crit(n, conf) if n >= 2 else float("nan")
    design = design_report(loaded)
    ratio = design["train_test_identity_ratio"]
    clean_design = design["design_kind"] == "folds_within_one_seed" and design["test_identity_sets_disjoint"] is True
    table = []
    for (m, p, sev), xs in sorted(grouped.items()):
        kind = model_kind(m)
        if len(xs) < 2:
            status = STATUS_SINGLE
        elif not clean_design or kind == "unknown":
            status = STATUS_DESCRIPTIVE
        elif len(xs) < MIN_SPLITS_FOR_INTERVAL:
            status = STATUS_FEW
        elif kind == "pretrained":
            status = STATUS_PRETRAINED_OK
        else:
            status = STATUS_CORRECTED if isinstance(ratio, float) else STATUS_DESCRIPTIVE
        row = {"model": m, "perturbation": p, "severity": float(sev), "n_splits": len(xs),
               "model_kind": kind, "interval_status": status}
        for met in METRICS:
            v = np.array([float(x[met]) for x in xs], dtype=float)
            mean = float(np.mean(v)) if len(v) else float("nan")
            sd = float(np.std(v, ddof=1)) if len(v) >= 2 else float("nan")
            half = tc * sd / math.sqrt(len(v)) if len(v) >= 2 else float("nan")
            row.update({f"{met}_mean": mean, f"{met}_sd": sd, f"{met}_ci_lo": mean - half, f"{met}_ci_hi": mean + half})
            # Nadeau-Bengio corrected resampled t: only for trained models in a clean one-seed fold design
            if status == STATUS_CORRECTED and len(v) >= 2:
                chalf = tc * sd * math.sqrt(1.0 / len(v) + ratio)
                row.update({f"{met}_corrected_ci_lo": mean - chalf, f"{met}_corrected_ci_hi": mean + chalf})
            else:
                row.update({f"{met}_corrected_ci_lo": float("nan"), f"{met}_corrected_ci_hi": float("nan")})
        table.append(row)

    out_dir.mkdir(parents=True, exist_ok=True)
    cols = list(table[0])
    with open(out_dir / OUT_TABLE, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows(table)
    run_rows = [{"run": str(r), "seed": s.get("seed"), "fold": s.get("fold"),
                 "reference_split": "split" not in s,
                 "content_sha256": json.dumps(s.get("content_sha256"), sort_keys=True)}
                for r, (s, _rows) in zip(runs, loaded)]
    with open(out_dir / OUT_RUNS, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(run_rows[0]))
        w.writeheader()
        w.writerows(run_rows)
    prov = {"n_splits": n, "confidence": conf, "t_critical": tc, "metrics": list(METRICS), "caveat": CAVEAT,
            "design": design, "comparability": comparability, "min_splits_for_interval": MIN_SPLITS_FOR_INTERVAL,
            "interval_status_counts": {k: sum(1 for r in table if r["interval_status"] == k)
                                       for k in sorted({r["interval_status"] for r in table})},
            "runs": run_rows, "n_conditions": len(table)}
    (out_dir / OUT_PROV).write_text(json.dumps(prov, indent=2, sort_keys=True, default=str) + "\n")
    return prov


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Aggregate identity-disjoint verification splits (mean/SD, interval with an explicit reliability status).")
    ap.add_argument("--runs", nargs="+", required=True, help="run directories (each with verification_results.*)")
    ap.add_argument("--out", required=True, help="NEW output directory")
    ap.add_argument("--confidence", type=float, default=0.95)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--allow-unverified-weights", action="store_true",
                    help="accept runs whose weight hashes were not recorded in every run (listed as unverified in the provenance)")
    a = ap.parse_args(argv)
    try:
        prov = aggregate([Path(r) for r in a.runs], Path(a.out), a.confidence, a.overwrite, a.allow_unverified_weights)
    except MultiSplitError as exc:
        print(f"REFUSED: {exc}")
        return 2
    print(f"OK: {prov['n_splits']} splits, {prov['n_conditions']} conditions -> {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

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


def aggregate(runs: Sequence[Path], out_dir: Path, conf: float = 0.95, overwrite: bool = False) -> dict:
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

    ident = {}
    for r, (s, rows) in zip(runs, loaded):
        key = (s.get("seed"), s.get("fold"))
        if key in ident:
            raise MultiSplitError(f"duplicate split (seed, fold)={key}: {r} and {ident[key]}")
        ident[key] = r
    ref = loaded[0][0]
    ref_cond = {(x["model"], x["perturbation"], str(float(x["severity"]))) for x in loaded[0][1]}
    for r, (s, rows) in zip(runs, loaded):
        cond = {(x["model"], x["perturbation"], str(float(x["severity"]))) for x in rows}
        if cond != ref_cond:
            raise MultiSplitError(f"{r}: condition set differs from the first run")
        if s.get("k_folds") != ref.get("k_folds"):
            raise MultiSplitError(f"{r}: k_folds differs")
        if s["dataset"]["fingerprint_sha256"] != ref["dataset"]["fingerprint_sha256"]:
            raise MultiSplitError(f"{r}: dataset fingerprint differs (different data than the first run)")
        for part in ("validation", "test"):
            a, b = s["pair_params"][part], ref["pair_params"][part]
            if (a["cap"], a["rho"]) != (b["cap"], b["rho"]):
                raise MultiSplitError(f"{r}: pair cap/rho differ in {part}")

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
            "design": design, "min_splits_for_interval": MIN_SPLITS_FOR_INTERVAL,
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
    a = ap.parse_args(argv)
    try:
        prov = aggregate([Path(r) for r in a.runs], Path(a.out), a.confidence, a.overwrite)
    except MultiSplitError as exc:
        print(f"REFUSED: {exc}")
        return 2
    print(f"OK: {prov['n_splits']} splits, {prov['n_conditions']} conditions -> {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

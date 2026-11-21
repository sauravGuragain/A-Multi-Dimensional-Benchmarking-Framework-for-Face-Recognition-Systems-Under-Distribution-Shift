"""
faceeval.analysis.valid_pairs
=============================
Valid-pairs-only re-analysis of ONE archived deep-model LFW Fold 0 verification run (FaceNet or ArcFace), computed
from the per-condition score files the run saved. It answers one question the archived (end-to-end) numbers cannot:
"how well does the model separate the pairs for which BOTH images actually produced an embedding?"

What it is, and is not
----------------------
* The archived metrics are END-TO-END: an image without an embedding (for ArcFace, mainly a detector miss) becomes a
  zero vector and every pair containing it is still scored. Those archived values are never changed and are always
  reported next to the valid-pairs-only values (``e2e_*`` columns).
* ``valid_*`` columns drop every balanced test pair in which at least one image has no valid embedding, and
  evaluate the rest at the SAME frozen validation threshold (the threshold is never re-tuned). AUC and EER are
  recomputed with the same function that produced the archived values (``compute_verification_metrics``).
* The valid-pairs-only value is a CONDITIONAL result. Images that fail detection are not a random sample (they are
  likely the hard ones), and the surviving pair set differs between conditions and between models. Never present
  ``valid_*`` as "the" recognition performance, and never compare it across models or across conditions as if the
  pair sets were the same.
* ``common_*`` columns evaluate every condition of the run on ONE fixed pair set: the pairs in which both images are
  valid in EVERY condition of the run. That set is identical across conditions, so ``common_*`` is the fair
  within-model severity curve (it still excludes the hardest images, so it is optimistic).
* No confidence interval and no significance test is computed (pairs share images and identities).

Integrity checks (a run that fails any of them is refused, nothing is written)
------------------------------------------------------------------------------
For every condition the archived CSV row is reproduced from the saved scores (accuracy, TAR, FAR, AUC, EER, balanced
and full-pool) and the failure counts are recomputed from the saved ``valid`` mask and pair indices
(``n_failed_embeddings``, ``n_failed_pairs_balanced``, ``n_failed_pairs_full``). The saved threshold must equal the
CSV threshold and be constant over the run. ``--allow-archive-mismatch`` downgrades only the numeric reproduction check
to a recorded warning.

A run directory normally holds exactly one model. A COMBINED run (several models in one verification run, e.g. the
archived ``deep_core`` run with FaceNet and ArcFace) is analysed one model at a time with ``--model NAME``: only that
model's rows and score files are read and every check above is applied to them unchanged. Without ``--model`` a
multi-model run is still refused.

Inputs are only READ. Output goes to a NEW (or empty) directory that may not lie inside the run directory.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from faceeval.evaluation.metrics import compute_verification_metrics

MIN_PER_CLASS = 10                      # fewer valid genuine or impostor pairs than this -> value reported as NaN
ABS_TOL = 1e-9                          # reproduction tolerance (CSV round-trip of a double)
SEMANTICS = ("Archived (e2e_*) values are end-to-end: failed embeddings are zero vectors and their pairs are still "
             "scored. valid_* drops pairs with a failed image and keeps the frozen validation threshold; it is a "
             "conditional, selection-biased result and not comparable across models or conditions. common_* uses one "
             "fixed pair set (valid in every condition of the run). No confidence intervals or significance tests.")
BY_CONDITION = "valid_pairs_by_condition.csv"
MODEL_SUMMARY = "valid_pairs_model_summary.csv"
PROVENANCE = "valid_pairs_provenance.json"


class ValidPairsError(RuntimeError):
    """The run cannot be analysed; nothing was written."""


# ---------------------------------------------------------------- helpers
def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _is_inside(path: Path, parent: Path) -> bool:
    return path == parent or parent in path.parents


def _f(x: Any) -> float:
    return float(x)


def _close(a: float, b: float) -> bool:
    if math.isnan(a) and math.isnan(b):
        return True
    return abs(a - b) <= ABS_TOL


def _metrics(scores: np.ndarray, labels: np.ndarray, thr: float) -> dict[str, float]:
    """accuracy / TAR / FAR at ``thr`` (accept iff score >= thr) plus AUC and EER; NaN when a class is too small."""
    nan = {"accuracy": math.nan, "tar": math.nan, "far": math.nan, "auc": math.nan, "eer": math.nan}
    if int(labels.sum()) < MIN_PER_CLASS or int((~labels).sum()) < MIN_PER_CLASS:
        return nan
    pred = scores >= thr
    out = {"accuracy": float((pred == labels).mean()), "tar": float(pred[labels].mean()),
           "far": float(pred[~labels].mean())}
    m = compute_verification_metrics(scores.tolist(), labels.tolist())
    out["auc"], out["eer"] = float(m["auc"]), float(m["eer"])
    return out


def _pair_mask(valid: np.ndarray, pairs: np.ndarray) -> np.ndarray:
    return valid[pairs[:, 0]] & valid[pairs[:, 1]]


def _read_csv(path: Path) -> list[dict[str, str]]:
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def _require(cond: bool, msg: str) -> None:
    if not cond:
        raise ValidPairsError(msg)


# ---------------------------------------------------------------- analysis
def analyse_run(run_dir: Path, allow_archive_mismatch: bool = False, model: str | None = None
                ) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    run_dir = Path(run_dir)
    csv_path, json_path = run_dir / "verification_results.csv", run_dir / "verification_results.json"
    pairs_path, scores_dir = run_dir / "scores" / "test_pairs.npz", run_dir / "scores"
    for p in (csv_path, json_path, pairs_path):
        _require(p.is_file(), f"missing input file: {p}")
    summary = json.loads(json_path.read_text())
    _require(summary.get("status") == "COMPLETE", f"run status is {summary.get('status')!r}, not COMPLETE")
    rows = _read_csv(csv_path)
    _require(len(rows) > 0, "the CSV has no rows")
    _require(all((r.get("scores_file") or "").strip() for r in rows),
             "a row has no scores_file: this is not a deep-model run (classical runs save no score files)")
    labels_seen = [r["label"] + "|" + r["model"] for r in rows]
    _require(len(set(labels_seen)) == len(labels_seen), "duplicate (model, label) rows in the CSV")
    models_in_run = sorted({r["model"] for r in rows})
    n_rows_in_run = len(rows)
    if model is not None:
        _require(model in models_in_run, f"--model {model!r} is not in this run (models in the CSV: {models_in_run})")
        rows = [r for r in rows if r["model"] == model]
    _require(len(models_in_run) == 1 or model is not None,
             f"the run directory must hold exactly one model, found {models_in_run}; to analyse one model of a "
             f"multi-model run pass --model NAME")

    tp = np.load(pairs_path, allow_pickle=False)
    bal_pairs, bal_lab = np.asarray(tp["bal_pairs"], dtype=np.int64), np.asarray(tp["bal_labels"], dtype=bool)
    full_pairs, full_lab = np.asarray(tp["full_pairs"], dtype=np.int64), np.asarray(tp["full_labels"], dtype=bool)
    n_images = int(len(tp["image_ids"]))
    for name, arr in (("bal_pairs", bal_pairs), ("full_pairs", full_pairs)):
        _require(arr.ndim == 2 and arr.shape[1] == 2, f"{name} must have shape (n, 2)")
        _require(arr.size > 0 and int(arr.min()) >= 0 and int(arr.max()) < n_images,
                 f"{name} indexes outside the {n_images} test images")
    _require(len(bal_lab) == len(bal_pairs) and len(full_lab) == len(full_pairs), "pair/label length mismatch")

    per: list[dict[str, Any]] = []
    valids: list[np.ndarray] = []
    thresholds: set[float] = set()
    mismatches: list[str] = []
    file_hashes: dict[str, str] = {}
    for r in rows:
        sf = scores_dir / r["scores_file"]
        _require(sf.parent == scores_dir and sf.is_file(), f"missing score file {r['scores_file']}")
        z = np.load(sf, allow_pickle=False)
        valid = np.asarray(z["valid"], dtype=bool)
        s_bal, s_full = np.asarray(z["s_bal"], dtype=np.float64), np.asarray(z["s_full"], dtype=np.float64)
        thr = float(z["threshold"])
        tag = r["label"]
        _require(valid.shape == (n_images,), f"{tag}: valid mask has {valid.size} entries, expected {n_images}")
        _require(s_bal.shape == (len(bal_pairs),) and s_full.shape == (len(full_pairs),),
                 f"{tag}: score arrays do not match the pair arrays")
        _require(np.isfinite(s_bal).all() and np.isfinite(s_full).all(), f"{tag}: non-finite scores")
        _require(_close(thr, _f(r["threshold"])), f"{tag}: saved threshold {thr} != CSV threshold {r['threshold']}")
        thresholds.add(thr)
        # failure counts recomputed from the saved mask must equal the archived counts
        nf_img = int((~valid).sum())
        nf_bal = int((~_pair_mask(valid, bal_pairs)).sum())
        nf_full = int((~_pair_mask(valid, full_pairs)).sum())
        for col, val in (("n_test_images", n_images), ("n_failed_embeddings", nf_img),
                         ("n_failed_pairs_balanced", nf_bal), ("n_failed_pairs_full", nf_full)):
            _require(int(float(r[col])) == val, f"{tag}: archived {col}={r[col]} but the score file gives {val}")
        # the archived end-to-end values must be reproducible from the saved scores
        e2e = _metrics(s_bal, bal_lab, thr)
        e2e_full = _metrics(s_full, full_lab, thr)
        for k in ("accuracy", "tar", "far", "auc", "eer"):
            if not _close(e2e[k], _f(r[k])):
                mismatches.append(f"{tag}: {k} archived {r[k]} vs recomputed {e2e[k]!r}")
        for k in ("auc", "eer"):
            if not _close(e2e_full[k], _f(r[f"{k}_full"])):
                mismatches.append(f"{tag}: {k}_full archived {r[k + '_full']} vs recomputed {e2e_full[k]!r}")
        per.append({"row": r, "valid": valid, "s_bal": s_bal, "s_full": s_full, "thr": thr, "e2e": e2e})
        valids.append(valid)
        file_hashes[r["scores_file"]] = _sha256(sf)
    _require(len(thresholds) == 1, f"the frozen threshold is not constant over the run: {sorted(thresholds)}")
    if mismatches and not allow_archive_mismatch:
        raise ValidPairsError("archived values are not reproducible from the saved scores ("
                              f"{len(mismatches)} differences, first: {mismatches[0]}); use "
                              "--allow-archive-mismatch only to inspect, never to report")

    common_valid = np.logical_and.reduce(valids)
    common_mask = _pair_mask(common_valid, bal_pairs)
    out_rows: list[dict[str, Any]] = []
    for p in per:
        r, valid, thr = p["row"], p["valid"], p["thr"]
        m = _pair_mask(valid, bal_pairs)
        v = _metrics(p["s_bal"][m], bal_lab[m], thr)
        c = _metrics(p["s_bal"][common_mask], bal_lab[common_mask], thr)
        mf = _pair_mask(valid, full_pairs)
        vf = _metrics(p["s_full"][mf], full_lab[mf], thr)
        e2e = p["e2e"]
        ok = not math.isnan(v["auc"])
        out_rows.append({
            "model": r["model"], "perturbation": r["perturbation"], "severity": _f(r["severity"]), "label": r["label"],
            "threshold": thr, "n_test_images": n_images, "n_failed_embeddings": int((~valid).sum()),
            "n_pairs_balanced": len(bal_pairs), "n_valid_pairs_balanced": int(m.sum()),
            "valid_pair_rate_balanced": float(m.mean()),
            "n_valid_genuine": int((m & bal_lab).sum()), "n_valid_impostor": int((m & ~bal_lab).sum()),
            "e2e_accuracy": e2e["accuracy"], "e2e_tar": e2e["tar"], "e2e_far": e2e["far"],
            "e2e_auc": e2e["auc"], "e2e_eer": e2e["eer"],
            "valid_accuracy": v["accuracy"], "valid_tar": v["tar"], "valid_far": v["far"],
            "valid_auc": v["auc"], "valid_eer": v["eer"],
            "valid_minus_e2e_auc": v["auc"] - e2e["auc"] if ok else math.nan,
            "valid_minus_e2e_eer": v["eer"] - e2e["eer"] if ok else math.nan,
            "n_common_valid_pairs": int(common_mask.sum()),
            "common_accuracy": c["accuracy"], "common_tar": c["tar"], "common_far": c["far"],
            "common_auc": c["auc"], "common_eer": c["eer"],
            "n_valid_pairs_full": int(mf.sum()), "valid_auc_full": vf["auc"], "valid_eer_full": vf["eer"],
            "valid_status": "ok" if ok else f"too_few_valid_pairs(<{MIN_PER_CLASS} per class)",
            "archive_reproduced": not any(x.startswith(r["label"] + ":") for x in mismatches),
        })

    def mean(rows_: list[dict[str, Any]], k: str) -> float:
        xs = [x[k] for x in rows_ if not math.isnan(x[k])]
        return float(np.mean(xs)) if xs else math.nan

    model_filter = model
    model = out_rows[0]["model"]
    summary_rows = [{
        "model": model, "n_conditions": len(out_rows),
        "n_conditions_valid_status_ok": sum(1 for x in out_rows if x["valid_status"] == "ok"),
        "failed_embeddings_summed_over_conditions_NOT_UNIQUE_IMAGES": sum(x["n_failed_embeddings"] for x in out_rows),
        "n_conditions_with_failures": sum(1 for x in out_rows if x["n_failed_embeddings"] > 0),
        "min_valid_pair_rate_balanced": min(x["valid_pair_rate_balanced"] for x in out_rows),
        "n_images_valid_in_every_condition": int(common_valid.sum()), "n_common_valid_pairs": int(common_mask.sum()),
        "mean_e2e_auc": mean(out_rows, "e2e_auc"), "mean_valid_auc": mean(out_rows, "valid_auc"),
        "mean_common_auc": mean(out_rows, "common_auc"),
        "mean_e2e_eer": mean(out_rows, "e2e_eer"), "mean_valid_eer": mean(out_rows, "valid_eer"),
        "mean_common_eer": mean(out_rows, "common_eer"),
    }]
    prov = {
        "run_directory": str(run_dir.resolve()), "model": model, "n_conditions": len(out_rows),
        "model_filter": model_filter, "models_in_run": models_in_run, "n_rows_in_run": n_rows_in_run,
        "expected_full_grid_conditions": 96, "is_full_grid": len(out_rows) == 96,
        "frozen_threshold": next(iter(thresholds)), "min_valid_pairs_per_class": MIN_PER_CLASS,
        "archive_mismatches_allowed": bool(allow_archive_mismatch), "archive_mismatches": mismatches,
        "input_sha256": {"verification_results.csv": _sha256(csv_path), "verification_results.json": _sha256(json_path),
                         "scores/test_pairs.npz": _sha256(pairs_path)},
        "score_file_sha256": file_hashes,
        "run_json": {k: summary.get(k) for k in ("protocol", "full_lfw", "status", "fold", "seed", "manifest_sha256",
                                                  "threshold_protocol", "n_rows", "provenance")},
        "semantics": SEMANTICS,
    }
    return out_rows, summary_rows, prov


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def run_valid_pairs_analysis(run_dir: Path, out_dir: Path, overwrite: bool = False,
                             allow_archive_mismatch: bool = False, model: str | None = None) -> dict[str, Path]:
    run_dir, out_dir = Path(run_dir).resolve(), Path(out_dir).resolve()
    if _is_inside(out_dir, run_dir):
        raise ValidPairsError(f"refusing to write inside the archived run directory {run_dir}")
    if _is_inside(run_dir, out_dir):
        raise ValidPairsError(f"refusing to write into {out_dir}: it contains the archived run directory")
    if out_dir.exists() and not out_dir.is_dir():
        raise ValidPairsError(f"output path exists and is not a directory: {out_dir}")
    if out_dir.is_dir() and any(out_dir.iterdir()) and not overwrite:
        raise ValidPairsError(f"refusing to write into the non-empty directory {out_dir} (new directory or --overwrite)")
    rows, summary, prov = analyse_run(run_dir, allow_archive_mismatch, model)      # raises before anything is written
    _write_csv(out_dir / BY_CONDITION, rows)
    _write_csv(out_dir / MODEL_SUMMARY, summary)
    (out_dir / PROVENANCE).write_text(json.dumps(prov, indent=2, sort_keys=True, default=str) + "\n")
    return {"by_condition": out_dir / BY_CONDITION, "model_summary": out_dir / MODEL_SUMMARY,
            "provenance": out_dir / PROVENANCE}


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description=("Valid-pairs-only re-analysis of ONE archived deep-model Fold 0 verification run, from its saved "
                     "score files. Archived end-to-end values are reported unchanged next to the valid-pairs-only "
                     "values. No experiment is run; the run directory is only read."),
        epilog=("Example:\n  python -m faceeval.analysis.valid_pairs --run <ARCFACE_VERIFICATION_RUN_DIR> "
                "--out <NEW_OUTPUT_DIRECTORY>"),
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True, type=Path, metavar="DIR",
                    help="verification run directory (holds verification_results.csv/.json and scores/)")
    ap.add_argument("--model", default=None, metavar="NAME",
                    help="analyse only this model of a multi-model (combined) run, e.g. facenet or arcface; without it a run "
                         "must hold exactly one model")
    ap.add_argument("--out", required=True, type=Path, help="NEW output directory (non-empty is refused)")
    ap.add_argument("--overwrite", action="store_true", help="allow a non-empty output directory")
    ap.add_argument("--allow-archive-mismatch", action="store_true",
                    help="record (do not refuse) numeric differences between archived and recomputed end-to-end values")
    return ap


def main(argv: list[str] | None = None) -> int:
    a = build_parser().parse_args(argv)
    try:
        written = run_valid_pairs_analysis(a.run, a.out, overwrite=a.overwrite,
                                           allow_archive_mismatch=a.allow_archive_mismatch, model=a.model)
    except ValidPairsError as exc:
        print(f"VALID-PAIRS ANALYSIS REFUSED: {exc}")
        return 2
    for k, v in written.items():
        print(f"{k}: {v}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""
faceeval.analysis.fairness
==========================
Subgroup verification metrics and disparities from the saved score files of ONE deep-model verification run and an
EXTERNAL attribute file supplied by the user.

What this is
------------
* Reproducible data interface: a CSV with an ``image_id`` column (the ids in ``scores/test_pairs.npz``) and one column
  holding the group of each image (default column name ``group``). Nothing is read from image content, names or
  filenames, and NOTHING IS INFERRED: an image without a row has an unknown group and is excluded and counted.
* A pair is assigned to a group only when BOTH images carry the same group value (within-group pairs). Pairs whose two
  images belong to different groups, or to an unknown group, are counted but not scored per group.
* Per group and condition, at the run's FROZEN validation threshold: accuracy, TAR, FAR (and AUC/EER when enough
  pairs). Groups with fewer than ``min_pairs`` genuine or impostor pairs are reported as NaN with ``sufficient=False``.
* Disparities over the sufficient groups: max-min and max/min for TAR and FAR, max-min for accuracy.
* Coverage (how many test images/pairs carry a label, per group) is always written.

What it is not
--------------
* The public LFW distribution used here carries NO demographic attribute labels. This module therefore cannot
  produce a fairness result for LFW by itself, and no result is claimed. It is validated only on SYNTHETIC labelled
  data in the test suite. A real analysis needs an attribute file with a documented, licensed provenance.
* Metrics are end-to-end (failed embeddings are zero vectors, as archived). Pair counts per group are small on a
  single fold: treat differences as descriptive, not as statistical evidence. No test or interval is computed.
* Group metrics at one global threshold are the standard operational notion of disparity (equal threshold, unequal
  error rates). They say nothing about the cause of a difference.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from faceeval.analysis.valid_pairs import _metrics

MIN_PAIRS = 30
BY_GROUP = "fairness_by_group.csv"
DISPARITY = "fairness_disparities.csv"
COVERAGE = "fairness_coverage.json"
CAVEAT = ("Within-group pairs only; frozen global threshold; end-to-end metrics; no inference of any attribute; "
          "descriptive, small-sample differences are not statistical evidence.")


class FairnessError(RuntimeError):
    pass


def load_attributes(path: Path, column: str = "group") -> dict[str, str]:
    with open(path, newline="") as fh:
        rd = csv.DictReader(fh)
        if rd.fieldnames is None or "image_id" not in rd.fieldnames or column not in rd.fieldnames:
            raise FairnessError(f"{path}: needs columns 'image_id' and '{column}' (found {rd.fieldnames})")
        out: dict[str, str] = {}
        for r in rd:
            k, v = (r["image_id"] or "").strip(), (r[column] or "").strip()
            if not k:
                raise FairnessError(f"{path}: empty image_id")
            if k in out and out[k] != v:
                raise FairnessError(f"{path}: image_id {k!r} has conflicting groups")
            if v:
                out[k] = v
    if not out:
        raise FairnessError(f"{path}: no labelled images")
    return out


def analyse(run_dir: Path, attributes: dict[str, str], min_pairs: int = MIN_PAIRS):
    run_dir = Path(run_dir)
    summary = json.loads((run_dir / "verification_results.json").read_text())
    if summary.get("status") != "COMPLETE":
        raise FairnessError(f"run status is {summary.get('status')!r}, not COMPLETE")
    with open(run_dir / "verification_results.csv", newline="") as fh:
        rows = list(csv.DictReader(fh))
    if not rows or not all((r.get("scores_file") or "").strip() for r in rows):
        raise FairnessError("no saved score files: only deep-model runs save per-pair scores")
    tp = np.load(run_dir / "scores" / "test_pairs.npz", allow_pickle=False)
    ids = [str(x) for x in tp["image_ids"]]
    pairs = np.asarray(tp["bal_pairs"], dtype=np.int64)
    lab = np.asarray(tp["bal_labels"], dtype=bool)
    grp = [attributes.get(i) for i in ids]
    known = [g is not None for g in grp]
    unmatched_attr = sorted(set(attributes) - set(ids))
    ga = np.array([g if g is not None else "" for g in grp], dtype=object)
    same = np.array([bool(ga[a]) and ga[a] == ga[b] for a, b in pairs], dtype=bool)
    groups = sorted({g for g in grp if g is not None})
    if not groups:
        raise FairnessError("no row of the attribute file matches any test image id of this run "
                            f"(attribute rows: {len(attributes)}, test images: {len(ids)}); the image_id column must hold "
                            "the ids stored in scores/test_pairs.npz. Nothing was written.")
    cov = {"n_test_images": len(ids), "n_images_with_group": int(sum(known)),
           "image_coverage": float(sum(known)) / max(1, len(ids)), "n_attribute_rows_not_in_test_set": len(unmatched_attr),
           "n_pairs": int(len(pairs)), "n_within_group_pairs": int(same.sum()),
           "n_cross_or_unknown_pairs": int((~same).sum()), "min_pairs": min_pairs, "groups": {}}
    for g in groups:
        m = same & (ga[pairs[:, 0]] == g)
        cov["groups"][g] = {"n_images": int(sum(1 for x in grp if x == g)), "n_genuine_pairs": int((m & lab).sum()),
                            "n_impostor_pairs": int((m & ~lab).sum()),
                            "sufficient": bool((m & lab).sum() >= min_pairs and (m & ~lab).sum() >= min_pairs)}
    out, disp = [], []
    for r in rows:
        z = np.load(run_dir / r["scores_file"], allow_pickle=False)
        s, thr = np.asarray(z["s_bal"], dtype=float), float(z["threshold"])
        per = {}
        for g in groups:
            m = same & (ga[pairs[:, 0]] == g)
            ok = cov["groups"][g]["sufficient"]
            met = _metrics(s[m], lab[m], thr) if ok else {k: math.nan for k in ("accuracy", "tar", "far", "auc", "eer")}
            per[g] = met
            out.append({"model": r["model"], "perturbation": r["perturbation"], "severity": float(r["severity"]),
                        "group": g, "threshold": thr, "sufficient": ok,
                        "n_genuine": cov["groups"][g]["n_genuine_pairs"], "n_impostor": cov["groups"][g]["n_impostor_pairs"], **met})
        suff = [g for g in groups if cov["groups"][g]["sufficient"]]
        d = {"model": r["model"], "perturbation": r["perturbation"], "severity": float(r["severity"]), "n_groups_used": len(suff)}
        for k in ("accuracy", "tar", "far"):
            v = [per[g][k] for g in suff]
            d[f"{k}_max_minus_min"] = (max(v) - min(v)) if len(v) >= 2 else math.nan
            if k != "accuracy":
                d[f"{k}_max_over_min"] = (max(v) / min(v)) if len(v) >= 2 and min(v) > 0 else math.nan
        disp.append(d)
    return out, disp, cov


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def _sha256(path: Path) -> str:
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def run(run_dir: Path, attributes_csv: Path, out_dir: Path, column: str = "group", min_pairs: int = MIN_PAIRS,
        overwrite: bool = False, source: str = "NOT_RECORDED") -> dict:
    run_dir, out_dir = Path(run_dir).resolve(), Path(out_dir).resolve()
    if out_dir == run_dir or run_dir in out_dir.parents:
        raise FairnessError("output directory may not be inside the run directory")
    if out_dir.exists() and any(out_dir.iterdir()) and not overwrite:
        raise FairnessError(f"{out_dir} exists and is not empty (use --overwrite)")
    by_group, disp, cov = analyse(run_dir, load_attributes(Path(attributes_csv), column), min_pairs)
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(out_dir / BY_GROUP, by_group)
    _write_csv(out_dir / DISPARITY, disp)
    cov["caveat"] = CAVEAT
    cov["attribute_file"] = {"path": str(Path(attributes_csv).resolve()), "sha256": _sha256(Path(attributes_csv)),
                             "column": column, "source_note": source,
                             "run_directory": str(run_dir)}
    (out_dir / COVERAGE).write_text(json.dumps(cov, indent=2, sort_keys=True) + "\n")
    return cov


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Subgroup verification metrics from saved scores + an external attribute CSV.")
    ap.add_argument("--run", required=True)
    ap.add_argument("--attributes", required=True, help="CSV with image_id and a group column")
    ap.add_argument("--column", default="group")
    ap.add_argument("--min-pairs", type=int, default=MIN_PAIRS)
    ap.add_argument("--out", required=True)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--source", default="NOT_RECORDED",
                    help="where the labels come from (provenance/licence note); recorded in fairness_coverage.json")
    a = ap.parse_args(argv)
    try:
        cov = run(a.run, a.attributes, a.out, a.column, a.min_pairs, a.overwrite, a.source)
    except (FairnessError, FileNotFoundError, KeyError) as exc:
        print(f"REFUSED: {exc}")
        return 2
    print(f"OK: coverage {cov['image_coverage']:.1%}, groups {sorted(cov['groups'])} -> {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

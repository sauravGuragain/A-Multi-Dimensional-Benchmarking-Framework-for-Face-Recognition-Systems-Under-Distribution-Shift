"""
faceeval.analysis.evidence_index
================================
Build ONE machine-readable and human-readable index of the evidence products a thesis cites, WITHOUT recomputing or
copying any of them. Each product is a directory written by one of the analysis tools; the index records whether the
expected files are present, their SHA-256, CSV row counts and the tool's own status fields.

Status per product
    PRESENT       every expected file exists and parses
    INCOMPLETE    the directory exists but an expected file is missing or does not parse
    NOT_PROVIDED  no directory was given (nothing is claimed about it)

The index never turns a missing product into a result, never reads a run directory for numbers, and refuses an output
directory that lies inside an input directory.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

PRODUCTS: dict[str, dict[str, Any]] = {
    "fold0_analysis": {"files": ["provenance.json"], "csv": [], "what": "six-model Fold-0 tables and figures (faceeval.analysis.fold0_results)"},
    "valid_pairs": {"files": ["valid_pairs_provenance.json", "valid_pairs_by_condition.csv", "valid_pairs_model_summary.csv"],
                    "csv": ["valid_pairs_by_condition.csv"], "what": "valid-pairs-only re-analysis of one deep run"},
    "calibration": {"files": ["calibration_summary.json", "calibration_results.csv"], "csv": ["calibration_results.csv"],
                    "what": "validation-fitted Platt calibration sibling of one verification run"},
    "efficiency": {"files": ["efficiency_results.csv"], "csv": ["efficiency_results.csv"], "what": "per-model runtime and peak memory"},
    "fairness": {"files": ["fairness_by_group.csv", "fairness_disparities.csv", "fairness_coverage.json"],
                 "csv": ["fairness_by_group.csv"], "what": "subgroup metrics from an external attribute file"},
    "multi_split": {"files": ["multi_split_by_condition.csv", "multi_split_runs.csv", "multi_split_provenance.json"],
                    "csv": ["multi_split_by_condition.csv", "multi_split_runs.csv"], "what": "multi-split mean/SD/CI"},
}
TAXONOMY = ("PRESENT = expected files exist and parse; INCOMPLETE = directory given but a file is missing or unreadable; "
            "NOT_PROVIDED = no directory given, nothing is claimed.")


class IndexError_(RuntimeError):
    pass


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def inspect(kind: str, directory: Path | None) -> dict[str, Any]:
    spec = PRODUCTS[kind]
    if directory is None:
        return {"kind": kind, "status": "NOT_PROVIDED", "what": spec["what"]}
    d = Path(directory)
    entry: dict[str, Any] = {"kind": kind, "what": spec["what"], "directory": str(d), "files": {}, "problems": []}
    if not d.is_dir():
        entry.update(status="INCOMPLETE", problems=["directory does not exist"])
        return entry
    for name in spec["files"]:
        p = d / name
        if not p.is_file():
            entry["problems"].append(f"missing {name}")
            continue
        info: dict[str, Any] = {"sha256": _sha(p), "bytes": p.stat().st_size}
        try:
            if name.endswith(".json"):
                json.loads(p.read_text())
            if name in spec["csv"]:
                with open(p, newline="") as fh:
                    info["rows"] = sum(1 for _ in csv.DictReader(fh))
        except (ValueError, csv.Error) as exc:
            entry["problems"].append(f"{name} does not parse: {exc}")
        entry["files"][name] = info
    entry["status"] = "PRESENT" if not entry["problems"] else "INCOMPLETE"
    return entry


def build(dirs: dict[str, Path | None], out_dir: Path, overwrite: bool = False) -> dict[str, Any]:
    out_dir = Path(out_dir).resolve()
    for d in dirs.values():
        if d is not None and (out_dir == Path(d).resolve() or Path(d).resolve() in out_dir.parents):
            raise IndexError_("output directory may not be inside an input directory")
    if out_dir.exists() and any(out_dir.iterdir()) and not overwrite:
        raise IndexError_(f"{out_dir} exists and is not empty (use --overwrite)")
    products = [inspect(k, dirs.get(k)) for k in PRODUCTS]
    index = {"taxonomy": TAXONOMY, "products": products,
             "summary": {s: sum(1 for p in products if p["status"] == s) for s in ("PRESENT", "INCOMPLETE", "NOT_PROVIDED")}}
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "evidence_index.json").write_text(json.dumps(index, indent=2, sort_keys=True) + "\n")
    lines = ["# Evidence index", "", TAXONOMY, "", "| product | status | files | problems |", "|---|---|---|---|"]
    for p in products:
        lines.append(f"| {p['kind']} | {p['status']} | {len(p.get('files', {}))} | {'; '.join(p.get('problems', [])) or '-'} |")
    (out_dir / "evidence_index.md").write_text("\n".join(lines) + "\n")
    return index


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Index the evidence products a thesis cites (no recomputation).")
    for k in PRODUCTS:
        ap.add_argument("--" + k.replace("_", "-"), default=None, help=PRODUCTS[k]["what"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--overwrite", action="store_true")
    a = ap.parse_args(argv)
    dirs = {k: (Path(getattr(a, k)) if getattr(a, k) else None) for k in PRODUCTS}
    try:
        idx = build(dirs, Path(a.out), a.overwrite)
    except IndexError_ as exc:
        print(f"REFUSED: {exc}")
        return 2
    print("OK:", idx["summary"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

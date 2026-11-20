"""
faceeval.analysis.figures
=========================
Reproducible figures drawn ONLY from result files that already exist. Nothing is simulated, smoothed or filled in.

    python -m faceeval.analysis.figures calibration --input <CAL_DIR or calibration_results.csv> --out <NEW_DIR>
    python -m faceeval.analysis.figures multi_split --input <multi_split_by_condition.csv or its dir> --out <NEW_DIR>
    python -m faceeval.analysis.figures valid_pairs --input <valid_pairs_by_condition.csv or its dir> --out <NEW_DIR>
    python -m faceeval.analysis.figures efficiency  --input <efficiency_results.json or its dir> --out <NEW_DIR>

Inputs are the outputs of ``verification_calibration`` (calibration_results.csv), ``multi_split`` (multi_split_by_condition.csv),
``valid_pairs`` (valid_pairs_by_condition.csv) and ``efficiency`` (efficiency_results.json). Each figure is written as
``<kind>.png`` plus ``<kind>_data.csv`` (exactly the numbers plotted) and ``<kind>_provenance.json`` (input path and
SHA-256, row count, plotted columns, matplotlib version). The output directory must be new or empty (``--overwrite`` to
replace); inputs are only read. A missing file or column, an empty table or a non-numeric value stops the run with a
message and writes nothing. Rendering is deterministic for a fixed matplotlib version (Agg backend, fixed size and dpi).

What the figures do NOT say (also written on the figure where it matters)
-------------------------------------------------------------------------
* calibration: ECE before/after Platt scaling fitted on clean validation pairs; balanced pairs (50% genuine base rate).
* multi_split: error bars are the across-split SD, not a confidence interval; each model's ``interval_status`` is printed.
* valid_pairs: valid-pair-only AUC is conditional on pairs that survived preprocessing and must not be compared across
  models or conditions; the figure always shows the end-to-end AUC next to it.
* efficiency: a figure of rows measured under different conditions carries a NOT COMPARABLE banner; synthetic self-test
  rows are refused unless ``--allow-synthetic`` (and are then watermarked).
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any, Callable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

PALETTE = ("#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9", "#000000", "#F0E442")   # Okabe-Ito
FIG_SIZE, DPI = (9.0, 5.2), 130
FILES = {"calibration": "calibration_results.csv", "multi_split": "multi_split_by_condition.csv",
         "valid_pairs": "valid_pairs_by_condition.csv", "efficiency": "efficiency_results.json"}
REQUIRED = {
    "calibration": ["model", "perturbation", "severity", "label", "n_pairs", "ece_before", "ece_after"],
    "multi_split": ["model", "perturbation", "severity", "n_splits", "model_kind", "interval_status", "auc_mean", "auc_sd"],
    "valid_pairs": ["model", "perturbation", "severity", "label", "valid_pair_rate_balanced", "e2e_auc", "valid_auc"],
}


class FigureError(RuntimeError):
    pass


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def _resolve(kind: str, p: Path) -> Path:
    p = Path(p)
    q = p / FILES[kind] if p.is_dir() else p
    if not q.is_file():
        raise FigureError(f"{q} not found (expected {FILES[kind]})")
    return q


def _num(x: Any, where: str) -> float:
    try:
        v = float(x)
    except (TypeError, ValueError):
        raise FigureError(f"non-numeric value {x!r} in {where}") from None
    return v


def _read_csv(kind: str, path: Path) -> list[dict[str, str]]:
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise FigureError(f"{path} has no data rows")
    missing = [c for c in REQUIRED[kind] if c not in rows[0]]
    if missing:
        raise FigureError(f"{path} lacks column(s) {missing}; is it the output of the right analysis?")
    return rows


def _models(rows: list[dict]) -> list[str]:
    return list(dict.fromkeys(r["model"] for r in rows))


def _cond(r: dict) -> str:
    return r.get("label") or f"{r['perturbation']}@{r['severity']}"


# ------------------------------------------------------------------------------------------------- figures
def fig_calibration(rows: list[dict], **_: Any) -> tuple[Any, list[dict]]:
    """ECE per condition as MARKERS (before: open circle, after Platt: filled square). Conditions are categorical and not
    ordered, so nothing is drawn between them; the x axis carries the condition names."""
    fig, ax = plt.subplots(figsize=FIG_SIZE)
    conds = list(dict.fromkeys(_cond(r) for r in rows))
    pos = {c: i for i, c in enumerate(conds)}
    models = _models(rows)
    data = []
    for i, m in enumerate(models):
        rs = [r for r in rows if r["model"] == m]
        c = PALETTE[i % len(PALETTE)]
        dx = (i - (len(models) - 1) / 2.0) * 0.12
        x = [pos[_cond(r)] + dx for r in rs]
        b = [_num(r["ece_before"], "ece_before") for r in rs]
        a = [_num(r["ece_after"], "ece_after") for r in rs]
        ax.plot(x, b, linestyle="None", marker="o", mfc="none", color=c, ms=6, label=f"{m}: before (p=(cos+1)/2)")
        ax.plot(x, a, linestyle="None", marker="s", color=c, ms=5, label=f"{m}: after Platt")
        data += [{"model": m, "condition": _cond(r), "ece_before": bb, "ece_after": aa, "n_pairs": r["n_pairs"]}
                 for r, bb, aa in zip(rs, b, a)]
    ax.set_xticks(range(len(conds)))
    ax.set_xticklabels(conds, rotation=45, ha="right", fontsize=6)
    ax.set_xlabel("condition (categorical; no ordering implied)")
    ax.set_ylabel("expected calibration error (15 bins)")
    ax.set_title("Calibration before/after Platt scaling (fit on clean validation pairs; balanced pairs, 50% genuine)", fontsize=9)
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)
    return fig, data


def fig_multi_split(rows: list[dict], metric: str = "auc", perturbation: str = "clean", severity: float = 0.0, **_: Any):
    sel = [r for r in rows if r["perturbation"] == perturbation and math.isclose(_num(r["severity"], "severity"), severity, abs_tol=1e-9)]
    if not sel:
        raise FigureError(f"no rows for perturbation={perturbation!r}, severity={severity}")
    for c in (f"{metric}_mean", f"{metric}_sd"):
        if c not in sel[0]:
            raise FigureError(f"column {c} missing (metric {metric!r} not in the table)")
    fig, ax = plt.subplots(figsize=FIG_SIZE)
    data = []
    for i, r in enumerate(sel):
        mean, sd = _num(r[f"{metric}_mean"], "mean"), _num(r[f"{metric}_sd"], "sd")
        c = PALETTE[0] if r["model_kind"] == "pretrained" else PALETTE[1]
        ax.errorbar(i, mean, yerr=None if math.isnan(sd) else sd, fmt="o", color=c, capsize=4)
        ax.annotate(r["interval_status"].replace("_", " "), (i, mean), textcoords="offset points", xytext=(0, 12), ha="center", fontsize=6)
        data.append({"model": r["model"], "model_kind": r["model_kind"], f"{metric}_mean": mean, f"{metric}_sd_across_splits": sd,
                     "n_splits": r["n_splits"], "interval_status": r["interval_status"]})
    ax.set_xticks(range(len(sel)))
    ax.set_xticklabels([r["model"] for r in sel], rotation=30, ha="right")
    ax.set_ylabel(metric.upper())
    ax.set_title(f"{metric.upper()} across splits at {perturbation} {severity:g}: mean +/- SD across splits (NOT a confidence interval)\n"
                 f"blue = pretrained, orange = trained per split; label = how far the spread may be trusted", fontsize=8)
    ax.grid(alpha=0.3)
    return fig, data


def fig_valid_pairs(rows: list[dict], **_: Any):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(FIG_SIZE[0] * 1.25, FIG_SIZE[1]))
    data = []
    for i, m in enumerate(_models(rows)):
        rs = [r for r in rows if r["model"] == m]
        c = PALETTE[i % len(PALETTE)]
        rate = [_num(r["valid_pair_rate_balanced"], "valid_pair_rate_balanced") for r in rs]
        e2e = [_num(r["e2e_auc"], "e2e_auc") for r in rs]
        val = [_num(r["valid_auc"], "valid_auc") for r in rs]
        a1.plot(range(len(rs)), rate, linestyle="None", marker="o", ms=4, color=c, label=m)
        a2.scatter(e2e, val, s=14, color=c, label=m)
        data += [{"model": m, "condition": _cond(r), "valid_pair_rate_balanced": a, "e2e_auc": b, "valid_auc": v}
                 for r, a, b, v in zip(rs, rate, e2e, val)]
    finite = [d["e2e_auc"] for d in data if not math.isnan(d["e2e_auc"])]
    if not finite or all(math.isnan(d["valid_auc"]) for d in data):
        plt.close(fig)
        raise FigureError("no finite AUC values in e2e_auc/valid_auc (every condition had too few valid pairs per class); nothing to plot")
    lim = [min(finite), 1.0]
    a2.plot(lim, lim, ":", color="grey")
    a1.set_xlabel("condition (file order; categorical, not connected)"); a1.set_ylabel("share of balanced pairs with both embeddings valid"); a1.set_ylim(0, 1.02)
    a2.set_xlabel("end-to-end AUC (failures scored as zero vectors)"); a2.set_ylabel("valid-pairs-only AUC (conditional)")
    a1.set_title("Valid-pair rate per condition", fontsize=9)
    a2.set_title("Valid-pairs-only vs end-to-end AUC\n(conditional on surviving pairs; not comparable across models)", fontsize=8)
    for a in (a1, a2):
        a.legend(fontsize=7); a.grid(alpha=0.3)
    return fig, data


def fig_efficiency(obj: dict, allow_synthetic: bool = False, **_: Any):
    res = obj["results"]
    if any(r.get("synthetic") for r in res) and not allow_synthetic:
        raise FigureError("the file holds synthetic self-test rows; refusing to plot them (use --allow-synthetic for a watermarked plot)")
    cmp_ = obj.get("comparability", {})
    fig, axes = plt.subplots(1, 4, figsize=(FIG_SIZE[0] * 1.5, FIG_SIZE[1]))
    spec = [("load_s", "initialisation (s)"), ("embed_ms_mean", "embedding, batch=1 (ms/image)"),
            ("images_per_s", "preprocess+embed throughput, batch=1 (images/s)"), ("rss_peak_mb", "process peak RSS (MB)")]
    data = []
    labels = [f"{r['model']}\n[{r.get('device_kind', '?')}]" for r in res]
    for ax, (k, t) in zip(axes, spec):
        vals = [_num(r[k], k) for r in res]
        ax.bar(range(len(res)), vals, color=[PALETTE[i % len(PALETTE)] for i in range(len(res))])
        ax.set_xticks(range(len(res))); ax.set_xticklabels(labels, fontsize=7); ax.set_title(t, fontsize=8); ax.grid(alpha=0.3, axis="y")
    for r in res:
        data.append({"model": r["model"], "device_kind": r.get("device_kind"), "load_s": r["load_s"], "embed_ms_mean": r["embed_ms_mean"],
                     "images_per_s": r["images_per_s"], "rss_peak_mb": r["rss_peak_mb"], "n_images": r["n_images"]})
    if cmp_ and not cmp_.get("all_same_conditions", True):
        fig.suptitle("NOT COMPARABLE: rows differ in " + ", ".join(cmp_.get("differing_fields", [])), color="#D55E00", fontsize=10, weight="bold")
    else:
        fig.suptitle("One machine, one run; no confidence intervals", fontsize=9)
    if any(r.get("synthetic") for r in res):
        fig.text(0.5, 0.5, "SYNTHETIC SELF-TEST - NOT A RESULT", ha="center", va="center", fontsize=28, color="red", alpha=0.25, rotation=20)
    return fig, data


BUILDERS: dict[str, Callable] = {"calibration": fig_calibration, "multi_split": fig_multi_split,
                                 "valid_pairs": fig_valid_pairs, "efficiency": fig_efficiency}


def make_figure(kind: str, input_path: Path, out_dir: Path, overwrite: bool = False, **opts: Any) -> dict[str, Path]:
    if kind not in BUILDERS:
        raise FigureError(f"unknown figure kind {kind!r}")
    src = _resolve(kind, input_path)
    out_dir = Path(out_dir)
    if out_dir.is_dir() and any(out_dir.iterdir()) and not overwrite:
        raise FigureError(f"refusing to write into the non-empty directory {out_dir} (new directory or --overwrite)")
    if kind == "efficiency":
        try:
            payload: Any = json.loads(src.read_text())
            payload["results"][0]
        except (ValueError, KeyError, IndexError, TypeError) as exc:
            raise FigureError(f"{src} is not an efficiency_results.json: {exc}") from None
        n_rows = len(payload["results"])
    else:
        payload = _read_csv(kind, src)
        n_rows = len(payload)
    fig, data = BUILDERS[kind](payload, **opts)          # raises before anything is written
    fig.tight_layout(rect=(0, 0, 1, 0.95) if kind == "efficiency" else None)
    out_dir.mkdir(parents=True, exist_ok=True)
    png, csv_p, prov_p = out_dir / f"{kind}.png", out_dir / f"{kind}_data.csv", out_dir / f"{kind}_provenance.json"
    fig.savefig(png, dpi=DPI, metadata={"Software": None})
    plt.close(fig)
    with open(csv_p, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(data[0]))
        w.writeheader()
        w.writerows(data)
    prov_p.write_text(json.dumps({"kind": kind, "input": str(src), "input_sha256": _sha256(src), "input_rows": n_rows,
                                  "plotted_rows": len(data), "options": {k: v for k, v in opts.items()},
                                  "matplotlib": matplotlib.__version__, "dpi": DPI,
                                  "note": "Drawn only from the input file; <kind>_data.csv holds exactly the plotted values."},
                                 indent=2, sort_keys=True, default=str) + "\n")
    return {"png": png, "data": csv_p, "provenance": prov_p}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Draw a figure from an existing result file (no simulation, no data invented).")
    ap.add_argument("kind", choices=sorted(BUILDERS))
    ap.add_argument("--input", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path, help="NEW output directory")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--metric", default="auc", help="multi_split: auc | accuracy | tar | far | eer")
    ap.add_argument("--perturbation", default="clean", help="multi_split: condition to plot")
    ap.add_argument("--severity", type=float, default=0.0, help="multi_split: severity to plot")
    ap.add_argument("--allow-synthetic", action="store_true", help="efficiency: plot harness self-test rows (watermarked)")
    a = ap.parse_args(argv)
    opts: dict[str, Any] = {}
    if a.kind == "multi_split":
        opts = {"metric": a.metric, "perturbation": a.perturbation, "severity": a.severity}
    if a.kind == "efficiency":
        opts = {"allow_synthetic": a.allow_synthetic}
    try:
        out = make_figure(a.kind, a.input, a.out, a.overwrite, **opts)
    except FigureError as exc:
        print(f"REFUSED: {exc}")
        return 2
    print(f"OK: {out['png']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

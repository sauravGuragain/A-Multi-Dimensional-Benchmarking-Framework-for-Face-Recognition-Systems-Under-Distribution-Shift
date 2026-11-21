"""
faceeval.evaluation.verification_calibration
============================================
Calibration of verification similarity scores, integrated into the Fold-0 verification runner.

Question answered: "when the model's cosine score says two photos are the same person, how trustworthy is that as a
probability, and does that trust survive the perturbation?"

Protocol (never changes the frozen evaluation protocol)
-------------------------------------------------------
* Probabilities are P(same person) for a PAIR. The "before calibration" probability is the fixed, label-free mapping
  ``p0 = (cosine + 1) / 2``. The "after calibration" probability is Platt scaling ``p = sigmoid(a * cosine + b)``.
* ``a`` and ``b`` are fitted ONCE per model on the CLEAN VALIDATION balanced pairs only (the same pairs the frozen
  threshold is chosen on). Test labels are never used to fit anything. The fitted calibrator is NOT refitted per
  perturbation, so the per-condition test numbers show how calibration degrades under distribution shift.
* Metrics (before and after) on the balanced test pairs of every condition: ECE (equal-width bins), MCE, Brier score and
  negative log-likelihood. Balanced pairs have a 50% genuine base rate, so the calibrated probabilities refer to that
  prior, NOT to a deployment prior in which impostor comparisons vastly outnumber genuine ones.
* End-to-end semantics are kept: a pair containing an image without an embedding is still scored (zero vector,
  cosine 0), exactly as in the archived metrics.
* Failures never stop a run: any error while fitting or scoring is recorded under ``errors`` and the verification
  results are unaffected. The artifacts are written to a SIBLING directory ``<run_id>.calibration`` so that the run
  directory itself (checked strictly by ``tools/run_integrity_checker``) is unchanged.
"""
from __future__ import annotations

import csv
import json
import logging
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

N_BINS = 15
MIN_PER_CLASS = 10
EPS = 1e-12
METHOD = "platt_scaling_on_clean_validation_balanced_pairs"
BEFORE_DEFINITION = "p0 = (cosine + 1) / 2 (fixed, label-free)"
CAL_SUFFIX = ".calibration"
RESULT_COLUMNS = ["model", "perturbation", "severity", "label", "n_pairs", "base_rate_genuine",
                  "ece_before", "ece_after", "mce_before", "mce_after", "brier_before", "brier_after",
                  "nll_before", "nll_after", "mean_p_before", "mean_p_after"]


def naive_probability(scores: np.ndarray) -> np.ndarray:
    return np.clip((np.asarray(scores, dtype=np.float64) + 1.0) / 2.0, 0.0, 1.0)


@dataclass(frozen=True)
class PlattCalibrator:
    a: float
    b: float
    n_fit: int

    def predict(self, scores: np.ndarray) -> np.ndarray:
        z = self.a * np.asarray(scores, dtype=np.float64) + self.b
        return 1.0 / (1.0 + np.exp(-np.clip(z, -500.0, 500.0)))


def fit_platt(scores: np.ndarray, labels: np.ndarray) -> PlattCalibrator:
    """Fit ``sigmoid(a * s + b)`` by (nearly) unregularised logistic regression. Raises ``ValueError`` when a class
    has fewer than ``MIN_PER_CLASS`` pairs or a score is not finite."""
    from sklearn.linear_model import LogisticRegression
    s = np.asarray(scores, dtype=np.float64).reshape(-1)
    y = np.asarray(labels, dtype=bool).reshape(-1)
    if s.shape != y.shape or s.size == 0:
        raise ValueError("scores and labels must be non-empty and the same length")
    if not np.isfinite(s).all():
        raise ValueError("scores must be finite")
    if int(y.sum()) < MIN_PER_CLASS or int((~y).sum()) < MIN_PER_CLASS:
        raise ValueError(f"need at least {MIN_PER_CLASS} genuine and {MIN_PER_CLASS} impostor pairs to calibrate")
    lr = LogisticRegression(C=1e4, solver="lbfgs", max_iter=2000)
    lr.fit(s.reshape(-1, 1), y.astype(int))
    return PlattCalibrator(a=float(lr.coef_[0, 0]), b=float(lr.intercept_[0]), n_fit=int(s.size))


def calibration_metrics(p: np.ndarray, y: np.ndarray, n_bins: int = N_BINS) -> dict[str, float]:
    """ECE, MCE, Brier and NLL of probabilities ``p`` of the positive (same-person) class against labels ``y``."""
    p = np.asarray(p, dtype=np.float64).reshape(-1)
    y = np.asarray(y, dtype=bool).reshape(-1).astype(np.float64)
    if p.size == 0 or p.shape != y.shape or not np.isfinite(p).all() or p.min() < 0 or p.max() > 1:
        raise ValueError("probabilities must be finite, in [0, 1], and match the labels")
    idx = np.minimum((p * n_bins).astype(int), n_bins - 1)
    ece = mce = 0.0
    for b in range(n_bins):
        m = idx == b
        if m.any():
            gap = abs(float(y[m].mean()) - float(p[m].mean()))
            ece += gap * float(m.mean())
            mce = max(mce, gap)
    q = np.clip(p, EPS, 1.0 - EPS)
    return {"ece": float(ece), "mce": float(mce), "brier": float(np.mean((p - y) ** 2)),
            "nll": float(-np.mean(y * np.log(q) + (1.0 - y) * np.log(1.0 - q))), "mean_p": float(p.mean())}


def _atomic_text(path: Path, text: str) -> None:
    tmp = path.with_name(path.name + ".tmp")
    try:
        with open(tmp, "w", newline="", encoding="utf-8") as f:
            f.write(text)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


class CalibrationRecorder:
    """Collects calibration results during a verification run; never raises into the caller."""

    def __init__(self) -> None:
        self.models: dict[str, dict[str, Any]] = {}
        self._cal: dict[str, PlattCalibrator] = {}
        self._val: dict[str, tuple[np.ndarray, np.ndarray, float]] = {}
        self.rows: list[dict[str, Any]] = []
        self.errors: list[dict[str, str]] = []

    def _err(self, where: str, exc: BaseException) -> None:
        logger.warning("calibration (%s) skipped: %s", where, exc)
        self.errors.append({"where": where, "error": f"{type(exc).__name__}: {exc}"})

    def fit(self, model: str, val_scores: np.ndarray, val_labels: np.ndarray, threshold: float) -> bool:
        """Fit on CLEAN VALIDATION balanced pairs only. Returns False (and records the error) on failure."""
        try:
            cal = fit_platt(val_scores, val_labels)
            before = calibration_metrics(naive_probability(val_scores), val_labels)
            after = calibration_metrics(cal.predict(val_scores), val_labels)
        except Exception as exc:
            self._err(f"{model}/fit", exc)
            return False
        self._cal[model] = cal
        self._val[model] = (np.asarray(val_scores, dtype=np.float64), np.asarray(val_labels, dtype=bool), float(threshold))
        self.models[model] = {"platt_a": cal.a, "platt_b": cal.b, "n_validation_pairs": cal.n_fit,
                              "validation_in_sample": {"before": before, "after": after}}
        return True

    def has(self, model: str) -> bool:
        return model in self._cal

    def record(self, model: str, perturbation: str, severity: float, label: str,
               test_scores: np.ndarray, test_labels: np.ndarray) -> None:
        """Calibration metrics of one condition on the balanced TEST pairs, with the already-fitted calibrator."""
        try:
            cal = self._cal[model]
            y = np.asarray(test_labels, dtype=bool)
            b = calibration_metrics(naive_probability(test_scores), y)
            a = calibration_metrics(cal.predict(test_scores), y)
            self.rows.append({
                "model": model, "perturbation": perturbation, "severity": float(severity), "label": label,
                "n_pairs": int(y.size), "base_rate_genuine": float(y.mean()),
                "ece_before": b["ece"], "ece_after": a["ece"], "mce_before": b["mce"], "mce_after": a["mce"],
                "brier_before": b["brier"], "brier_after": a["brier"], "nll_before": b["nll"], "nll_after": a["nll"],
                "mean_p_before": b["mean_p"], "mean_p_after": a["mean_p"]})
        except Exception as exc:
            self._err(f"{model}/{label}", exc)

    def write(self, out_dir: Path) -> dict[str, Any]:
        """Write the artifacts into ``out_dir`` (created) and return a small pointer for the run summary."""
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        import io
        buf = io.StringIO()
        w = csv.DictWriter(buf, fieldnames=RESULT_COLUMNS)
        w.writeheader()
        w.writerows(self.rows)
        _atomic_text(out_dir / "calibration_results.csv", buf.getvalue())
        for m, (s, y, thr) in self._val.items():
            tmp = out_dir / f"validation_scores__{m}.tmp.npz"
            np.savez_compressed(tmp, scores=s, labels=y, threshold=np.float64(thr))
            os.replace(tmp, out_dir / f"validation_scores__{m}.npz")
        summary = {
            "method": METHOD, "before_definition": BEFORE_DEFINITION, "n_bins": N_BINS,
            "fit_data": "clean validation balanced pairs only; test labels are never used to fit; "
                        "calibrator is not refitted per perturbation",
            "base_rate_note": "balanced pairs have a 50% genuine base rate; calibrated probabilities refer to that "
                              "prior, not to a deployment prior",
            "models": self.models, "n_condition_rows": len(self.rows), "errors": self.errors}
        _atomic_text(out_dir / "calibration_summary.json", json.dumps(summary, indent=2, sort_keys=True, default=str) + "\n")
        return {"status": "OK" if not self.errors else "PARTIAL", "directory": out_dir.name,
                "n_condition_rows": len(self.rows), "n_errors": len(self.errors)}


def is_finite_number(x: Any) -> bool:
    return isinstance(x, (int, float)) and math.isfinite(x)

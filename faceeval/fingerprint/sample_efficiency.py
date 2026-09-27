from __future__ import annotations

import logging
import random
from typing import Any

import numpy as np

from faceeval.core.types import (
    ModelName,
    PerturbationSpec,
    SampleEfficiencyCurve,
    SubjectID,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Sample efficiency computation
# ---------------------------------------------------------------------------

def compute_sample_efficiency_curve(
    model_name: ModelName,
    fit_fn: Any,             # callable(X_train, y_train) → None
    predict_fn: Any,         # callable(X_test) → list[PredictionResult]
    X_train: list[np.ndarray],
    y_train: list[SubjectID],
    X_test: list[np.ndarray],
    y_test: list[SubjectID],
    training_sizes: list[int] | None = None,
    n_repeats: int = 3,
    seed: int = 42,
    perturbation_spec: PerturbationSpec | None = None,
) -> SampleEfficiencyCurve:
    """
    Compute sample efficiency by refitting the model at multiple training sizes.

    Parameters
    ----------
    model_name:
        Name of the model being profiled.
    fit_fn:
        Callable that trains the model: ``fit_fn(X_sub, y_sub) → None``.
    predict_fn:
        Callable that runs inference: ``predict_fn(X_test) → list[PredictionResult]``.
    X_train, y_train:
        Full training pool.
    X_test, y_test:
        Fixed test split (unchanged across sizes).
    training_sizes:
        List of training-set sizes to evaluate.  Auto-generated as 10 log-spaced
        points from ``min_per_class`` to ``len(X_train)`` if ``None``.
    n_repeats:
        Number of random subsampling repeats per size for variance estimation.
    seed:
        Random seed for reproducible subsampling.
    perturbation_spec:
        Perturbation condition (if evaluating under distribution shift).

    Returns
    -------
    SampleEfficiencyCurve
    """
    n_subjects = len(set(y_train))
    sizes = _resolve_training_sizes(training_sizes, len(X_train), n_subjects)

    logger.info(
        "[%s] Computing sample efficiency curve over %d sizes (repeats=%d)…",
        model_name, len(sizes), n_repeats,
    )

    mean_accuracies: list[float] = []
    rng = random.Random(seed)

    for size in sizes:
        repeat_accs: list[float] = []
        for _ in range(n_repeats):
            X_sub, y_sub = _stratified_subsample(X_train, y_train, size, rng)
            try:
                fit_fn(X_sub, y_sub)
                preds = predict_fn(X_test)
                acc = sum(
                    1 for p, t in zip(preds, y_test)
                    if p.predicted_subject_id == t
                ) / len(y_test)
                repeat_accs.append(acc)
            except Exception as exc:
                logger.warning(
                    "[%s] size=%d repeat failed: %s", model_name, size, exc
                )

        mean_acc = float(np.mean(repeat_accs)) if repeat_accs else 0.0
        mean_accuracies.append(mean_acc)
        logger.debug(
            "[%s] size=%d → acc=%.4f (±%.4f over %d repeats)",
            model_name, size, mean_acc,
            float(np.std(repeat_accs)) if repeat_accs else 0.0,
            len(repeat_accs),
        )

    return SampleEfficiencyCurve(
        model_name=model_name,
        training_sizes=sizes,
        accuracies=mean_accuracies,
        perturbation_spec=perturbation_spec,
    )


# ---------------------------------------------------------------------------
# Summary metrics derived from the curve
# ---------------------------------------------------------------------------

def saturation_accuracy(curve: SampleEfficiencyCurve) -> float:
    """Accuracy at the largest training size (saturation point)."""
    return curve.accuracies[-1] if curve.accuracies else 0.0


def half_saturation_size(curve: SampleEfficiencyCurve) -> int | None:
    """
    Number of training samples needed to reach 50% of the saturation accuracy.

    A lower value indicates a more sample-efficient model.
    Returns ``None`` if the model never reaches 50% of saturation.
    """
    if not curve.accuracies:
        return None
    target = saturation_accuracy(curve) * 0.5
    for size, acc in zip(curve.training_sizes, curve.accuracies):
        if acc >= target:
            return size
    return None


def sample_efficiency_auc(curve: SampleEfficiencyCurve) -> float:
    """
    Area under the normalised sample efficiency curve.

    Normalise training sizes to [0, 1] and compute the integral of
    accuracy over normalised size.  Higher = more sample-efficient.
    """
    if len(curve.accuracies) < 2:
        return curve.accuracies[0] if curve.accuracies else 0.0

    sizes = np.array(curve.training_sizes, dtype=np.float64)
    accs  = np.array(curve.accuracies,     dtype=np.float64)

    # Normalise sizes to [0, 1]
    size_range = sizes[-1] - sizes[0]
    if size_range < 1e-9:
        return float(accs.mean())

    norm_sizes = (sizes - sizes[0]) / size_range
    return float(np.trapezoid(accs, norm_sizes))


def sample_efficiency_summary(
    curves: list[SampleEfficiencyCurve],
) -> list[dict[str, Any]]:
    """
    Build a flat summary table of sample efficiency metrics for all models.
    """
    rows: list[dict[str, Any]] = []
    for curve in sorted(curves, key=lambda c: c.model_name):
        rows.append({
            "model_name": curve.model_name,
            "saturation_accuracy": round(saturation_accuracy(curve), 4),
            "half_saturation_size": half_saturation_size(curve),
            "sample_efficiency_auc": round(sample_efficiency_auc(curve), 4),
            "training_sizes": curve.training_sizes,
            "accuracies": [round(a, 4) for a in curve.accuracies],
        })
    return rows


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _resolve_training_sizes(
    requested: list[int] | None,
    max_size: int,
    n_subjects: int,
) -> list[int]:
    """
    Build a list of training sizes.

    If ``requested`` is None, generates 10 log-spaced sizes from
    ``n_subjects`` (minimum 1 sample per subject) to ``max_size``.
    """
    if requested is not None:
        return [min(s, max_size) for s in sorted(requested)]

    min_size = max(n_subjects, 1)
    if min_size >= max_size:
        return [max_size]

    log_sizes = np.logspace(
        np.log10(min_size),
        np.log10(max_size),
        num=10,
    )
    sizes = sorted(set(int(round(s)) for s in log_sizes))
    return [min(s, max_size) for s in sizes]


def _stratified_subsample(
    X: list[np.ndarray],
    y: list[SubjectID],
    target_size: int,
    rng: random.Random,
) -> tuple[list[np.ndarray], list[SubjectID]]:
    """
    Draw a stratified random subsample of size ``target_size``.

    Ensures at least 1 sample per subject.  If ``target_size`` exceeds the
    dataset size, returns the full dataset.
    """
    if target_size >= len(X):
        return X, y

    # Group indices by subject
    by_subject: dict[SubjectID, list[int]] = {}
    for i, label in enumerate(y):
        by_subject.setdefault(label, []).append(i)

    n_subjects = len(by_subject)
    per_subject = max(1, target_size // n_subjects)
    remainder   = target_size - per_subject * n_subjects

    selected_indices: list[int] = []
    for subj, indices in by_subject.items():
        shuffled = list(indices)
        rng.shuffle(shuffled)
        selected_indices.extend(shuffled[:per_subject])

    # Fill remainder from random subjects
    if remainder > 0:
        all_remaining = [
            idx for subj, indices in by_subject.items()
            for idx in indices
            if idx not in set(selected_indices)
        ]
        rng.shuffle(all_remaining)
        selected_indices.extend(all_remaining[:remainder])

    X_sub = [X[i] for i in selected_indices]
    y_sub = [y[i] for i in selected_indices]
    return X_sub, y_sub

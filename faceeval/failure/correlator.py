from __future__ import annotations

import logging
from collections import defaultdict
from typing import Any

import numpy as np

from faceeval.core.types import (
    CrossParadigmFailureCorrelation,
    FailureCase,
    ModelName,
    ModelParadigm,
    PerturbationType,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Core overlap computation
# ---------------------------------------------------------------------------

def compute_cross_paradigm_correlation(
    cases: list[FailureCase],
    traditional_model: ModelName,
    dl_model: ModelName,
    perturbation_type: PerturbationType | None = None,
) -> CrossParadigmFailureCorrelation:
    """
    Compute failure overlap between one traditional and one DL model.

    Parameters
    ----------
    cases:
        All failure cases from the experiment (any models, any perturbations).
    traditional_model:
        Name of the traditional ML model.
    dl_model:
        Name of the deep learning model.
    perturbation_type:
        If specified, only compare failures under this perturbation type.
        If ``None``, compare across all perturbation types.

    Returns
    -------
    CrossParadigmFailureCorrelation
    """
    trad_cases = _filter_cases(cases, traditional_model, perturbation_type)
    dl_cases   = _filter_cases(cases, dl_model,          perturbation_type)

    trad_ids = {c.image_id for c in trad_cases}
    dl_ids   = {c.image_id for c in dl_cases}

    shared_ids     = sorted(trad_ids & dl_ids)
    trad_only_ids  = sorted(trad_ids - dl_ids)
    dl_only_ids    = sorted(dl_ids   - trad_ids)
    union_ids      = trad_ids | dl_ids

    overlap = len(shared_ids) / len(union_ids) if union_ids else 0.0

    pt = perturbation_type or _most_common_perturbation(cases)

    logger.info(
        "Cross-paradigm correlation [%s vs %s, %s]: "
        "overlap=%.4f, shared=%d, trad_only=%d, dl_only=%d",
        traditional_model, dl_model,
        pt.value if pt else "all",
        overlap, len(shared_ids), len(trad_only_ids), len(dl_only_ids),
    )

    return CrossParadigmFailureCorrelation(
        traditional_model=traditional_model,
        dl_model=dl_model,
        perturbation_type=pt or PerturbationType.GAUSSIAN_BLUR,
        overlap_fraction=float(overlap),
        traditional_only_ids=trad_only_ids,
        dl_only_ids=dl_only_ids,
        shared_ids=shared_ids,
    )


def compute_all_correlations(
    cases: list[FailureCase],
    traditional_models: list[ModelName],
    dl_models: list[ModelName],
    per_perturbation: bool = True,
) -> list[CrossParadigmFailureCorrelation]:
    """
    Compute cross-paradigm correlations for all (traditional × DL) model pairs.

    Parameters
    ----------
    cases:
        All failure cases.
    traditional_models, dl_models:
        Lists of model names by paradigm.
    per_perturbation:
        If ``True``, compute one correlation per perturbation type.
        If ``False``, compute one aggregate correlation per pair.

    Returns
    -------
    list[CrossParadigmFailureCorrelation]
    """
    correlations: list[CrossParadigmFailureCorrelation] = []

    for t_model in traditional_models:
        for d_model in dl_models:
            if per_perturbation:
                pt_values = _unique_perturbation_types(cases)
                for pt in pt_values:
                    corr = compute_cross_paradigm_correlation(
                        cases, t_model, d_model, pt
                    )
                    correlations.append(corr)
            else:
                corr = compute_cross_paradigm_correlation(
                    cases, t_model, d_model, None
                )
                correlations.append(corr)

    return correlations


# ---------------------------------------------------------------------------
# Overlap analysis utilities
# ---------------------------------------------------------------------------

def overlap_by_severity(
    cases: list[FailureCase],
    traditional_model: ModelName,
    dl_model: ModelName,
    perturbation_type: PerturbationType,
) -> dict[float, float]:
    """
    Compute failure overlap at each individual severity level.

    Returns dict: severity → overlap_fraction.
    Reveals whether traditional and DL models diverge in failure patterns
    as severity increases.
    """
    severities = sorted({
        c.perturbation_spec.severity
        for c in cases
        if c.perturbation_spec and c.perturbation_spec.perturbation_type == perturbation_type
    })

    result: dict[float, float] = {}
    for sev in severities:
        sev_cases = [
            c for c in cases
            if c.perturbation_spec
            and c.perturbation_spec.perturbation_type == perturbation_type
            and abs(c.perturbation_spec.severity - sev) < 1e-6
        ]
        t_ids = {c.image_id for c in sev_cases if c.model_name == traditional_model}
        d_ids = {c.image_id for c in sev_cases if c.model_name == dl_model}
        union = t_ids | d_ids
        overlap = len(t_ids & d_ids) / len(union) if union else 0.0
        result[sev] = float(overlap)

    return result


def paradigm_failure_profile(
    correlations: list[CrossParadigmFailureCorrelation],
) -> dict[str, Any]:
    """
    Aggregate overlap statistics across all (model pair × perturbation) correlations.

    Returns:
        mean_overlap, max_overlap, min_overlap,
        per_perturbation_mean_overlap,
        most_complementary_pair (lowest overlap),
        most_redundant_pair (highest overlap).
    """
    if not correlations:
        return {}

    overlaps = [c.overlap_fraction for c in correlations]

    by_pt: dict[str, list[float]] = defaultdict(list)
    for c in correlations:
        by_pt[c.perturbation_type.value].append(c.overlap_fraction)

    # Most complementary pair (lowest overlap → best ensemble candidate)
    min_corr = min(correlations, key=lambda c: c.overlap_fraction)
    max_corr = max(correlations, key=lambda c: c.overlap_fraction)

    return {
        "mean_overlap": float(np.mean(overlaps)),
        "max_overlap":  float(np.max(overlaps)),
        "min_overlap":  float(np.min(overlaps)),
        "per_perturbation_mean_overlap": {
            pt: float(np.mean(vals)) for pt, vals in sorted(by_pt.items())
        },
        "most_complementary_pair": {
            "traditional_model": min_corr.traditional_model,
            "dl_model": min_corr.dl_model,
            "perturbation_type": min_corr.perturbation_type.value,
            "overlap": min_corr.overlap_fraction,
        },
        "most_redundant_pair": {
            "traditional_model": max_corr.traditional_model,
            "dl_model": max_corr.dl_model,
            "perturbation_type": max_corr.perturbation_type.value,
            "overlap": max_corr.overlap_fraction,
        },
        "n_correlations": len(correlations),
    }


def correlation_matrix(
    correlations: list[CrossParadigmFailureCorrelation],
    traditional_models: list[ModelName],
    dl_models: list[ModelName],
) -> tuple[np.ndarray, list[str], list[str]]:
    """
    Build a (n_traditional × n_dl) overlap matrix, averaged over perturbation types.

    Returns
    -------
    (matrix, traditional_model_names, dl_model_names)
    """
    n_t = len(traditional_models)
    n_d = len(dl_models)
    matrix = np.zeros((n_t, n_d), dtype=np.float64)
    counts = np.zeros((n_t, n_d), dtype=int)

    t_idx = {m: i for i, m in enumerate(traditional_models)}
    d_idx = {m: i for i, m in enumerate(dl_models)}

    for corr in correlations:
        ti = t_idx.get(corr.traditional_model)
        di = d_idx.get(corr.dl_model)
        if ti is not None and di is not None:
            matrix[ti, di] += corr.overlap_fraction
            counts[ti, di] += 1

    # Average
    mask = counts > 0
    matrix[mask] /= counts[mask]

    return matrix, traditional_models, dl_models


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _filter_cases(
    cases: list[FailureCase],
    model_name: ModelName,
    perturbation_type: PerturbationType | None,
) -> list[FailureCase]:
    result = [c for c in cases if c.model_name == model_name]
    if perturbation_type is not None:
        result = [
            c for c in result
            if c.perturbation_spec
            and c.perturbation_spec.perturbation_type == perturbation_type
        ]
    return result


def _unique_perturbation_types(cases: list[FailureCase]) -> list[PerturbationType]:
    return sorted(
        {c.perturbation_spec.perturbation_type for c in cases if c.perturbation_spec},
        key=lambda pt: pt.value,
    )


def _most_common_perturbation(cases: list[FailureCase]) -> PerturbationType | None:
    pts = [c.perturbation_spec.perturbation_type for c in cases if c.perturbation_spec]
    if not pts:
        return None
    from collections import Counter
    return Counter(pts).most_common(1)[0][0]

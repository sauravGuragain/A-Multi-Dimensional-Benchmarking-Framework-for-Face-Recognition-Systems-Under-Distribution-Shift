from __future__ import annotations

import logging
from typing import Any

import numpy as np

from faceeval.core.types import (
    DeploymentScore,
    DeploymentWeights,
    ModelName,
)

logger = logging.getLogger(__name__)

_WEIGHT_DIMENSIONS = [
    "accuracy", "robustness", "calibration",
    "fairness", "latency", "memory", "computational_cost",
]


# ---------------------------------------------------------------------------
# Analytical sensitivity (first-order partial derivatives)
# ---------------------------------------------------------------------------

def compute_sensitivity(
    scores: list[DeploymentScore],
) -> dict[ModelName, dict[str, float]]:
    """
    Compute ∂S/∂w_d for each model and each weight dimension.

    Since S = Σ w_d × c_d, we have ∂S/∂w_d = c_d (component score).
    This gives the rate of change of total score per unit increase in weight.

    Returns
    -------
    dict: model_name → {dimension: sensitivity_value}
    """
    return {
        score.model_name: dict(score.component_scores)
        for score in scores
    }


def sensitivity_summary(
    scores: list[DeploymentScore],
) -> dict[str, Any]:
    """
    Summarise sensitivity across all models.

    Returns for each dimension:
      - which model is most sensitive to that weight
      - mean sensitivity across models
      - variance (high variance = weight is discriminating)
    """
    sensitivities = compute_sensitivity(scores)
    summary: dict[str, Any] = {}

    for dim in _WEIGHT_DIMENSIONS:
        dim_vals = {
            model: sens.get(dim, 0.0)
            for model, sens in sensitivities.items()
        }
        if not dim_vals:
            continue
        vals = list(dim_vals.values())
        most_sensitive = max(dim_vals, key=lambda m: dim_vals[m])
        least_sensitive = min(dim_vals, key=lambda m: dim_vals[m])
        summary[dim] = {
            "mean_sensitivity": float(np.mean(vals)),
            "variance": float(np.var(vals)),
            "most_sensitive_model": most_sensitive,
            "least_sensitive_model": least_sensitive,
            "model_values": dim_vals,
        }

    return summary


# ---------------------------------------------------------------------------
# Monte Carlo ranking robustness
# ---------------------------------------------------------------------------

def monte_carlo_ranking_robustness(
    scores: list[DeploymentScore],
    n_samples: int = 1000,
    perturbation_std: float = 0.05,
    random_state: int = 42,
) -> dict[str, Any]:
    """
    Estimate ranking robustness via random weight perturbations.

    For each of ``n_samples`` random weight vectors (drawn from a Dirichlet
    distribution centred on the baseline weights), re-compute total scores
    and record the ranking.

    Parameters
    ----------
    scores:
        Baseline deployment scores (already computed).
    n_samples:
        Number of Monte Carlo draws.
    perturbation_std:
        Standard deviation of weight perturbation (controls how far from
        baseline we explore).
    random_state:
        Reproducibility seed.

    Returns
    -------
    dict with:
        top_model_stability — fraction of draws where baseline rank-1 model
                              stays rank-1
        pairwise_stability  — for each model pair, fraction of draws where
                              their relative order matches the baseline
        ranking_entropy     — Shannon entropy of rank-1 model distribution
                              (0 = fully stable, log(n) = fully random)
    """
    if not scores:
        return {}

    rng = np.random.default_rng(random_state)
    baseline_weights = _weights_to_array(scores[0].weights)
    n_dims = len(baseline_weights)
    n_models = len(scores)

    # Component score matrix: (n_models, n_dims)
    comp_matrix = np.array([
        [score.component_scores.get(d, 0.0) for d in _WEIGHT_DIMENSIONS]
        for score in scores
    ], dtype=np.float64)

    model_names = [s.model_name for s in scores]
    baseline_top = scores[0].model_name   # rank-1 model in baseline

    rank1_counts: dict[str, int] = {m: 0 for m in model_names}
    pairwise_agree: dict[tuple[str, str], int] = {}
    for i in range(n_models):
        for j in range(i + 1, n_models):
            pairwise_agree[(model_names[i], model_names[j])] = 0

    for _ in range(n_samples):
        # Perturb weights via Dirichlet (concentration = baseline / perturbation_std²)
        concentration = np.maximum(baseline_weights / (perturbation_std ** 2 + 1e-9), 0.1)
        perturbed = rng.dirichlet(concentration)

        # Compute total scores under perturbed weights
        totals = comp_matrix @ perturbed   # (n_models,)
        order  = np.argsort(totals)[::-1]

        rank1_counts[model_names[order[0]]] += 1

        # Pairwise agreement
        for i in range(n_models):
            for j in range(i + 1, n_models):
                pair = (model_names[i], model_names[j])
                baseline_order = scores.index(
                    next(s for s in scores if s.model_name == model_names[i])
                )
                perturbed_i_better = totals[i] > totals[j]
                baseline_i_better  = (
                    next(s.total_score for s in scores if s.model_name == model_names[i])
                    > next(s.total_score for s in scores if s.model_name == model_names[j])
                )
                if perturbed_i_better == baseline_i_better:
                    pairwise_agree[pair] += 1

    top_model_stability = rank1_counts[baseline_top] / n_samples

    pairwise_stability = {
        f"{a}_vs_{b}": count / n_samples
        for (a, b), count in pairwise_agree.items()
    }

    # Ranking entropy
    probs = np.array([rank1_counts[m] / n_samples for m in model_names])
    probs = probs[probs > 0]
    entropy = float(-np.sum(probs * np.log(probs))) if len(probs) > 0 else 0.0

    return {
        "top_model_stability": top_model_stability,
        "top_model": baseline_top,
        "rank1_distribution": {m: c / n_samples for m, c in rank1_counts.items()},
        "pairwise_stability": pairwise_stability,
        "ranking_entropy": entropy,
        "n_samples": n_samples,
        "perturbation_std": perturbation_std,
        "interpretation": (
            "Stable" if top_model_stability > 0.80
            else ("Moderate" if top_model_stability > 0.50 else "Unstable")
        ),
    }


# ---------------------------------------------------------------------------
# Stability interval
# ---------------------------------------------------------------------------

def find_stability_interval(
    scores: list[DeploymentScore],
    model_a: ModelName,
    model_b: ModelName,
    dimension: str,
    n_steps: int = 100,
) -> dict[str, Any]:
    """
    Find the range of weight values for ``dimension`` where model_a stays
    above model_b, holding all other weights proportionally constant.

    Uses a grid search over the dimension weight from 0.0 to 1.0.

    Returns
    -------
    dict with:
        stable_range      — (w_min, w_max) where A > B
        baseline_weight   — current weight for dimension
        crossover_weight  — approximate weight where A == B (or None)
    """
    if dimension not in _WEIGHT_DIMENSIONS:
        raise ValueError(f"Unknown dimension '{dimension}'.")

    score_a = next((s for s in scores if s.model_name == model_a), None)
    score_b = next((s for s in scores if s.model_name == model_b), None)

    if score_a is None or score_b is None:
        raise ValueError(f"Model(s) not found in scores: {model_a}, {model_b}")

    baseline_weights = _weights_to_array(score_a.weights)
    comp_a = np.array([score_a.component_scores.get(d, 0.0) for d in _WEIGHT_DIMENSIONS])
    comp_b = np.array([score_b.component_scores.get(d, 0.0) for d in _WEIGHT_DIMENSIONS])
    dim_idx = _WEIGHT_DIMENSIONS.index(dimension)

    w_values = np.linspace(0.0, 1.0, n_steps)
    a_wins: list[bool] = []
    crossover: float | None = None

    for w_dim in w_values:
        # Redistribute remaining weight proportionally to other dimensions
        w_vec = baseline_weights.copy()
        remaining = 1.0 - w_dim
        other_sum = w_vec.sum() - w_vec[dim_idx]
        if other_sum < 1e-9:
            w_vec = np.full(len(_WEIGHT_DIMENSIONS), (1.0 - w_dim) / (len(_WEIGHT_DIMENSIONS) - 1))
        else:
            scale = remaining / other_sum
            w_vec = w_vec * scale
        w_vec[dim_idx] = w_dim

        total_a = float(comp_a @ w_vec)
        total_b = float(comp_b @ w_vec)
        a_wins.append(total_a > total_b)

    # Find crossover
    for i in range(len(a_wins) - 1):
        if a_wins[i] != a_wins[i + 1]:
            crossover = float((w_values[i] + w_values[i + 1]) / 2.0)
            break

    stable_min = float(w_values[next((i for i, v in enumerate(a_wins) if v), 0)])
    stable_max = float(w_values[next((
        len(a_wins) - 1 - i for i, v in enumerate(reversed(a_wins)) if v
    ), 0)])

    return {
        "model_a": model_a,
        "model_b": model_b,
        "dimension": dimension,
        "stable_range": (stable_min, stable_max),
        "baseline_weight": float(baseline_weights[dim_idx]),
        "crossover_weight": crossover,
        "a_wins_at_baseline": bool(a_wins[int(n_steps * float(baseline_weights[dim_idx]))]) if n_steps > 0 else None,
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _weights_to_array(weights: DeploymentWeights) -> np.ndarray:
    return np.array([
        weights.accuracy,
        weights.robustness,
        weights.calibration,
        weights.fairness,
        weights.latency,
        weights.memory,
        weights.computational_cost,
    ], dtype=np.float64)

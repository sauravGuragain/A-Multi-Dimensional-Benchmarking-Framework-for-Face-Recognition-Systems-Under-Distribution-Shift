from __future__ import annotations

import logging
from typing import Any

import numpy as np

from faceeval.core.types import (
    BehavioralFingerprint,
    ModelName,
    ModelParadigm,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Distance functions
# ---------------------------------------------------------------------------

def euclidean_distance(fp_a: BehavioralFingerprint, fp_b: BehavioralFingerprint) -> float:
    """L2 distance between two fingerprint vectors."""
    return fp_a.distance_to(fp_b)


def cosine_distance(fp_a: BehavioralFingerprint, fp_b: BehavioralFingerprint) -> float:
    """1 - cosine_similarity between two fingerprint vectors."""
    _check_same_dimensions(fp_a, fp_b)
    va = np.array(fp_a.vector, dtype=np.float64)
    vb = np.array(fp_b.vector, dtype=np.float64)
    na, nb = np.linalg.norm(va), np.linalg.norm(vb)
    if na < 1e-9 or nb < 1e-9:
        return 1.0
    return float(1.0 - np.dot(va, vb) / (na * nb))


def manhattan_distance(fp_a: BehavioralFingerprint, fp_b: BehavioralFingerprint) -> float:
    """L1 distance between two fingerprint vectors."""
    _check_same_dimensions(fp_a, fp_b)
    return float(sum(abs(a - b) for a, b in zip(fp_a.vector, fp_b.vector)))


# ---------------------------------------------------------------------------
# Pairwise distance matrix
# ---------------------------------------------------------------------------

def pairwise_distance_matrix(
    fingerprints: list[BehavioralFingerprint],
    metric: str = "euclidean",
) -> tuple[np.ndarray, list[ModelName]]:
    """
    Compute the (N × N) pairwise distance matrix for N fingerprints.

    Parameters
    ----------
    fingerprints:
        List of fingerprints (must all share the same dimension names).
    metric:
        Distance metric: ``"euclidean"`` (default), ``"cosine"``, or ``"manhattan"``.

    Returns
    -------
    (matrix, model_names)
        ``matrix[i, j]`` = distance between fingerprints[i] and fingerprints[j].
        ``model_names`` gives the row/column labels.
    """
    n = len(fingerprints)
    matrix = np.zeros((n, n), dtype=np.float64)
    model_names = [fp.model_name for fp in fingerprints]

    dist_fn = {
        "euclidean": euclidean_distance,
        "cosine":    cosine_distance,
        "manhattan": manhattan_distance,
    }.get(metric)
    if dist_fn is None:
        raise ValueError(f"Unknown metric '{metric}'. Choose: euclidean, cosine, manhattan")

    for i in range(n):
        for j in range(i + 1, n):
            d = dist_fn(fingerprints[i], fingerprints[j])
            matrix[i, j] = d
            matrix[j, i] = d

    return matrix, model_names


# ---------------------------------------------------------------------------
# Paradigm comparison report
# ---------------------------------------------------------------------------

def cross_paradigm_comparison(
    fingerprints: list[BehavioralFingerprint],
) -> dict[str, Any]:
    """
    Compare traditional ML vs. DL fingerprints dimension-by-dimension.

    Returns a summary dict with:
        - Mean fingerprint vector per paradigm
        - Per-dimension gap (DL_mean - Traditional_mean)
        - Dimensions where DL is clearly better / clearly worse
        - Overall similarity between paradigm centroids
    """
    traditional = [fp for fp in fingerprints if fp.paradigm == ModelParadigm.TRADITIONAL]
    deep        = [fp for fp in fingerprints if fp.paradigm == ModelParadigm.DEEP_LEARNING]

    if not traditional or not deep:
        logger.warning("cross_paradigm_comparison requires both TRADITIONAL and DEEP_LEARNING fingerprints.")
        return {}

    _validate_same_dims(fingerprints)
    dim_names = fingerprints[0].dimension_names

    trad_matrix = np.array([fp.vector for fp in traditional], dtype=np.float64)
    deep_matrix = np.array([fp.vector for fp in deep],        dtype=np.float64)

    trad_mean = trad_matrix.mean(axis=0)
    deep_mean = deep_matrix.mean(axis=0)
    gap       = deep_mean - trad_mean   # positive → DL is better

    # Dimensions where one paradigm is clearly dominant (|gap| > 0.1)
    dl_better  = [dim_names[i] for i, g in enumerate(gap) if g >  0.1]
    trad_better= [dim_names[i] for i, g in enumerate(gap) if g < -0.1]

    # Overall cosine similarity between paradigm centroids
    na = np.linalg.norm(trad_mean)
    nb = np.linalg.norm(deep_mean)
    centroid_similarity = float(np.dot(trad_mean, deep_mean) / (na * nb)) if na > 0 and nb > 0 else 0.0

    return {
        "traditional_models": [fp.model_name for fp in traditional],
        "deep_learning_models": [fp.model_name for fp in deep],
        "dimension_names": dim_names,
        "traditional_mean_vector": trad_mean.tolist(),
        "deep_learning_mean_vector": deep_mean.tolist(),
        "dimension_gaps": dict(zip(dim_names, gap.tolist())),
        "dimensions_where_dl_better": dl_better,
        "dimensions_where_traditional_better": trad_better,
        "paradigm_centroid_similarity": centroid_similarity,
        "n_traditional": len(traditional),
        "n_deep_learning": len(deep),
    }


# ---------------------------------------------------------------------------
# Hierarchical clustering
# ---------------------------------------------------------------------------

def cluster_fingerprints(
    fingerprints: list[BehavioralFingerprint],
    n_clusters: int = 3,
    metric: str = "euclidean",
) -> dict[ModelName, int]:
    """
    Cluster fingerprints via hierarchical agglomerative clustering.

    Parameters
    ----------
    fingerprints:
        List of fingerprints to cluster.
    n_clusters:
        Number of clusters to cut the dendrogram at.
    metric:
        Distance metric to use.

    Returns
    -------
    dict mapping model_name → cluster_id (0-indexed).
    """
    if len(fingerprints) < n_clusters:
        logger.warning(
            "n_clusters=%d > n_fingerprints=%d; assigning each model its own cluster.",
            n_clusters, len(fingerprints),
        )
        return {fp.model_name: i for i, fp in enumerate(fingerprints)}

    try:
        from scipy.cluster.hierarchy import linkage, fcluster
        from scipy.spatial.distance import squareform

        matrix, model_names = pairwise_distance_matrix(fingerprints, metric)

        # Convert to condensed form for scipy
        condensed = squareform(matrix, checks=False)
        Z = linkage(condensed, method="ward")
        labels = fcluster(Z, t=n_clusters, criterion="maxclust")

        return {name: int(label - 1) for name, label in zip(model_names, labels)}

    except ImportError:
        logger.warning("scipy not installed — falling back to simple k-means clustering.")
        return _simple_kmeans_cluster(fingerprints, n_clusters)


def _simple_kmeans_cluster(
    fingerprints: list[BehavioralFingerprint],
    k: int,
) -> dict[ModelName, int]:
    """Minimal k-means clustering for when scipy is unavailable."""
    vectors = np.array([fp.vector for fp in fingerprints], dtype=np.float64)
    n = len(vectors)

    # Initialise centroids as first k fingerprints
    centroids = vectors[:k].copy()

    for _ in range(50):  # max iterations
        # Assign
        dists = np.array([
            [np.linalg.norm(v - c) for c in centroids]
            for v in vectors
        ])
        assignments = np.argmin(dists, axis=1)

        # Update
        new_centroids = np.array([
            vectors[assignments == ci].mean(axis=0) if (assignments == ci).any() else centroids[ci]
            for ci in range(k)
        ])
        if np.allclose(centroids, new_centroids):
            break
        centroids = new_centroids

    return {fp.model_name: int(assignments[i]) for i, fp in enumerate(fingerprints)}


# ---------------------------------------------------------------------------
# Most / least similar pair
# ---------------------------------------------------------------------------

def most_similar_pair(
    fingerprints: list[BehavioralFingerprint],
    metric: str = "euclidean",
) -> tuple[ModelName, ModelName, float]:
    """Return the (model_a, model_b, distance) pair with the smallest distance."""
    return _extreme_pair(fingerprints, metric, find_min=True)


def most_different_pair(
    fingerprints: list[BehavioralFingerprint],
    metric: str = "euclidean",
) -> tuple[ModelName, ModelName, float]:
    """Return the (model_a, model_b, distance) pair with the largest distance."""
    return _extreme_pair(fingerprints, metric, find_min=False)


def _extreme_pair(
    fingerprints: list[BehavioralFingerprint],
    metric: str,
    find_min: bool,
) -> tuple[ModelName, ModelName, float]:
    matrix, names = pairwise_distance_matrix(fingerprints, metric)
    np.fill_diagonal(matrix, np.inf if find_min else -np.inf)
    idx = np.argmin(matrix) if find_min else np.argmax(matrix)
    i, j = np.unravel_index(idx, matrix.shape)
    dist = float(matrix[int(i), int(j)])
    return names[int(i)], names[int(j)], dist


# ---------------------------------------------------------------------------
# Dimension-level analysis
# ---------------------------------------------------------------------------

def dimension_rankings(
    fingerprints: list[BehavioralFingerprint],
) -> dict[str, list[tuple[ModelName, float]]]:
    """
    For every fingerprint dimension, rank all models from best to worst.

    Returns dict: dimension_name → [(model_name, score), ...] sorted descending.
    """
    if not fingerprints:
        return {}
    _validate_same_dims(fingerprints)
    dim_names = fingerprints[0].dimension_names
    rankings: dict[str, list[tuple[ModelName, float]]] = {}
    for i, dim in enumerate(dim_names):
        scored = [(fp.model_name, fp.vector[i]) for fp in fingerprints]
        rankings[dim] = sorted(scored, key=lambda x: x[1], reverse=True)
    return rankings


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _check_same_dimensions(fp_a: BehavioralFingerprint, fp_b: BehavioralFingerprint) -> None:
    if fp_a.dimension_names != fp_b.dimension_names:
        raise ValueError(
            f"Fingerprints have different dimensions: "
            f"{fp_a.model_name}={fp_a.dimension_names} vs "
            f"{fp_b.model_name}={fp_b.dimension_names}"
        )


def _validate_same_dims(fingerprints: list[BehavioralFingerprint]) -> None:
    if not fingerprints:
        return
    ref_dims = fingerprints[0].dimension_names
    for fp in fingerprints[1:]:
        if fp.dimension_names != ref_dims:
            raise ValueError(
                f"Fingerprint for '{fp.model_name}' has different dimensions than "
                f"'{fingerprints[0].model_name}'"
            )

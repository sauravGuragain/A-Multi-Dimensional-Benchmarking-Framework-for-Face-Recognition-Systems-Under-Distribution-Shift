from __future__ import annotations

import logging
from collections import Counter
from typing import Any

import numpy as np

from faceeval.core.types import (
    FailureCase,
    FailureCluster,
    FailureMode,
    ModelName,
    PerturbationCategory,
    PerturbationType,
)

logger = logging.getLogger(__name__)

_UMAP_AVAILABLE: bool | None = None
_MIN_CLUSTER_SIZE = 3


def _try_import_umap() -> Any | None:
    global _UMAP_AVAILABLE
    if _UMAP_AVAILABLE is None:
        try:
            import umap  # noqa: F401
            _UMAP_AVAILABLE = True
        except ImportError:
            _UMAP_AVAILABLE = False
            logger.debug("umap-learn not installed — using PCA fallback for 2D projections.")
    return __import__("umap") if _UMAP_AVAILABLE else None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def cluster_failures(
    cases: list[FailureCase],
    model_name: ModelName,
    n_clusters: int = 5,
    use_umap: bool = True,
    random_state: int = 42,
) -> list[FailureCluster]:
    """
    Cluster failure cases and return annotated ``FailureCluster`` objects.

    Parameters
    ----------
    cases:
        Failure cases to cluster (from ``extract_failures``).
    model_name:
        The model whose failures are being analysed.
    n_clusters:
        Target number of clusters.  Auto-reduced if fewer cases exist.
    use_umap:
        Whether to attempt UMAP for 2D projection (falls back to PCA).
    random_state:
        Random seed for reproducibility.

    Returns
    -------
    list[FailureCluster]
        One cluster object per discovered cluster, with 2D centroid and
        dominant failure mode / perturbation type annotated.
    """
    # Filter to requested model
    model_cases = [c for c in cases if c.model_name == model_name]
    if len(model_cases) < _MIN_CLUSTER_SIZE:
        logger.warning(
            "[%s] Too few failures to cluster (%d < %d).",
            model_name, len(model_cases), _MIN_CLUSTER_SIZE,
        )
        return []

    # Cap clusters to n_cases
    k = min(n_clusters, len(model_cases))

    # Build feature matrix
    feature_matrix = _build_feature_matrix(model_cases)

    # 2D projection
    projections_2d = _project_to_2d(feature_matrix, use_umap, random_state)

    # Cluster in 2D space
    labels = _hierarchical_cluster(projections_2d, k, random_state)

    # Attach 2D coordinates to cases (we need mutable copies)
    annotated_cases = _attach_projections(model_cases, projections_2d, labels)

    # Build FailureCluster objects
    clusters: list[FailureCluster] = []
    for cluster_id in range(k):
        cluster_indices = [i for i, lbl in enumerate(labels) if lbl == cluster_id]
        if not cluster_indices:
            continue

        cluster_cases = [annotated_cases[i] for i in cluster_indices]
        centroid = projections_2d[cluster_indices].mean(axis=0)

        dom_mode = _dominant_mode(cluster_cases)
        dom_pt   = _dominant_perturbation_type(cluster_cases)
        desc     = _describe_cluster(cluster_cases, dom_mode, dom_pt, cluster_id)

        clusters.append(FailureCluster(
            cluster_id=cluster_id,
            model_name=model_name,
            cases=cluster_cases,
            dominant_failure_mode=dom_mode,
            dominant_perturbation_type=dom_pt,
            centroid_2d=(float(centroid[0]), float(centroid[1])),
            description=desc,
        ))

    logger.info(
        "[%s] Clustered %d failures into %d clusters.",
        model_name, len(model_cases), len(clusters),
    )
    return clusters


def cluster_all_models(
    cases: list[FailureCase],
    model_names: list[ModelName],
    n_clusters: int = 5,
    use_umap: bool = True,
    random_state: int = 42,
) -> dict[ModelName, list[FailureCluster]]:
    """
    Cluster failures for every model in ``model_names``.

    Returns dict mapping model_name → list of FailureCluster.
    """
    return {
        name: cluster_failures(cases, name, n_clusters, use_umap, random_state)
        for name in model_names
    }


# ---------------------------------------------------------------------------
# Cluster summary
# ---------------------------------------------------------------------------

def cluster_summary(clusters: list[FailureCluster]) -> list[dict[str, Any]]:
    """
    Build a flat summary table of cluster statistics.
    One row per cluster, suitable for CSV export and LaTeX tables.
    """
    rows: list[dict[str, Any]] = []
    for cl in clusters:
        confs = [c.confidence for c in cl.cases if c.confidence is not None]
        sevs  = [
            c.perturbation_spec.severity
            for c in cl.cases
            if c.perturbation_spec is not None
        ]
        rows.append({
            "cluster_id": cl.cluster_id,
            "model_name": cl.model_name,
            "size": cl.size,
            "dominant_failure_mode": cl.dominant_failure_mode.value,
            "dominant_perturbation_type": (
                cl.dominant_perturbation_type.value
                if cl.dominant_perturbation_type else None
            ),
            "mean_confidence_at_failure": round(float(np.mean(confs)), 4) if confs else None,
            "mean_severity": round(float(np.mean(sevs)), 4) if sevs else None,
            "centroid_2d": cl.centroid_2d,
            "description": cl.description,
        })
    return rows


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------

def _build_feature_matrix(cases: list[FailureCase]) -> np.ndarray:
    """
    Build a (N, D) feature matrix for clustering.

    Features per case:
      [0]   confidence (or 0.5 if missing)
      [1]   perturbation severity (or 0.0 if clean)
      [2]   failure mode one-hot (4 dims)
      [2:6] perturbation category one-hot (7 dims)
    """
    failure_modes = list(FailureMode)
    categories    = list(PerturbationCategory)

    rows: list[list[float]] = []
    for case in cases:
        conf = case.confidence if case.confidence is not None else 0.5
        sev  = (case.perturbation_spec.severity
                if case.perturbation_spec else 0.0)

        # Failure mode one-hot
        mode_oh = [1.0 if m == case.failure_mode else 0.0 for m in failure_modes]

        # Category one-hot
        if case.perturbation_spec:
            cat = case.perturbation_spec.category
        else:
            cat = None
        cat_oh = [1.0 if c == cat else 0.0 for c in categories]

        rows.append([conf, sev] + mode_oh + cat_oh)

    return np.array(rows, dtype=np.float32)


def _project_to_2d(
    features: np.ndarray,
    use_umap: bool,
    random_state: int,
) -> np.ndarray:
    """Reduce features to 2D for visualisation and clustering."""
    if features.shape[0] < 3:
        return features[:, :2] if features.shape[1] >= 2 else np.hstack([
            features, np.zeros((features.shape[0], 2 - features.shape[1]))
        ])

    if use_umap:
        umap_mod = _try_import_umap()
        if umap_mod is not None:
            try:
                reducer = umap_mod.UMAP(
                    n_components=2,
                    n_neighbors=min(15, features.shape[0] - 1),
                    random_state=random_state,
                    verbose=False,
                )
                return reducer.fit_transform(features).astype(np.float32)
            except Exception as exc:
                logger.debug("UMAP failed (%s); falling back to PCA.", exc)

    # PCA fallback
    return _pca_2d(features)


def _pca_2d(features: np.ndarray) -> np.ndarray:
    """Pure-NumPy PCA to 2D."""
    X = features.astype(np.float64)
    X -= X.mean(axis=0)
    cov = np.cov(X.T) if X.shape[0] > 1 else np.eye(X.shape[1])
    if cov.ndim == 0:
        cov = np.array([[float(cov)]])
    try:
        vals, vecs = np.linalg.eigh(cov)
    except np.linalg.LinAlgError:
        return np.zeros((X.shape[0], 2), dtype=np.float32)
    # Take top-2 eigenvectors (eigh returns ascending order)
    top2 = vecs[:, -2:]
    projected = (X @ top2).astype(np.float32)
    if projected.shape[1] < 2:
        projected = np.hstack([projected, np.zeros((projected.shape[0], 1), dtype=np.float32)])
    return projected


def _hierarchical_cluster(
    coords: np.ndarray,
    k: int,
    random_state: int,
) -> np.ndarray:
    """Ward-linkage hierarchical clustering; fallback to k-means."""
    try:
        from scipy.cluster.hierarchy import linkage, fcluster
        from scipy.spatial.distance import pdist

        dists = pdist(coords.astype(np.float64))
        Z = linkage(dists, method="ward")
        labels = fcluster(Z, t=k, criterion="maxclust")
        return labels - 1   # zero-indexed
    except ImportError:
        return _kmeans_cluster(coords, k, random_state)


def _kmeans_cluster(
    coords: np.ndarray,
    k: int,
    random_state: int,
) -> np.ndarray:
    """Minimal k-means as scipy fallback."""
    rng = np.random.default_rng(random_state)
    n = len(coords)
    indices = rng.choice(n, size=min(k, n), replace=False)
    centroids = coords[indices].astype(np.float64)

    labels = np.zeros(n, dtype=int)
    for _ in range(100):
        dists = np.array([[np.linalg.norm(c - ct) for ct in centroids] for c in coords])
        new_labels = np.argmin(dists, axis=1)
        if np.array_equal(new_labels, labels):
            break
        labels = new_labels
        for ci in range(k):
            members = coords[labels == ci]
            if len(members):
                centroids[ci] = members.mean(axis=0)
    return labels


# ---------------------------------------------------------------------------
# Annotation helpers
# ---------------------------------------------------------------------------

def _attach_projections(
    cases: list[FailureCase],
    projections: np.ndarray,
    labels: np.ndarray,
) -> list[FailureCase]:
    """Return new FailureCase objects with projection_2d and cluster_id filled in."""
    annotated: list[FailureCase] = []
    for case, proj, lbl in zip(cases, projections, labels):
        # FailureCase is frozen; rebuild with updated fields
        annotated.append(FailureCase(
            case_id=case.case_id,
            image_id=case.image_id,
            subject_id=case.subject_id,
            model_name=case.model_name,
            perturbation_spec=case.perturbation_spec,
            failure_mode=case.failure_mode,
            predicted_subject_id=case.predicted_subject_id,
            confidence=case.confidence,
            similarity_score=case.similarity_score,
            true_subject_id=case.true_subject_id,
            projection_2d=(float(proj[0]), float(proj[1])),
            cluster_id=int(lbl),
        ))
    return annotated


def _dominant_mode(cases: list[FailureCase]) -> FailureMode:
    counts = Counter(c.failure_mode for c in cases)
    return counts.most_common(1)[0][0]


def _dominant_perturbation_type(
    cases: list[FailureCase],
) -> PerturbationType | None:
    types = [
        c.perturbation_spec.perturbation_type
        for c in cases
        if c.perturbation_spec
    ]
    if not types:
        return None
    return Counter(types).most_common(1)[0][0]


def _describe_cluster(
    cases: list[FailureCase],
    dom_mode: FailureMode,
    dom_pt: PerturbationType | None,
    cluster_id: int,
) -> str:
    sevs = [
        c.perturbation_spec.severity
        for c in cases
        if c.perturbation_spec
    ]
    mean_sev = float(np.mean(sevs)) if sevs else 0.0
    confs = [c.confidence for c in cases if c.confidence is not None]
    mean_conf = float(np.mean(confs)) if confs else 0.0

    pt_str = dom_pt.value if dom_pt else "clean"
    return (
        f"Cluster {cluster_id}: {len(cases)} cases | "
        f"mode={dom_mode.value} | "
        f"perturbation={pt_str} | "
        f"mean_severity={mean_sev:.2f} | "
        f"mean_confidence={mean_conf:.3f}"
    )

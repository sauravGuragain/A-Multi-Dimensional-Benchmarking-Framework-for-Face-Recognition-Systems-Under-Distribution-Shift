from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Run summaries and listings
# ---------------------------------------------------------------------------

class RunSummary(BaseModel):
    run_id: str
    experiment_name: str
    status: str
    started_at: str | None = None
    completed_at: str | None = None


class RunListResponse(BaseModel):
    runs: list[RunSummary]
    total: int
    limit: int
    offset: int


class RunDetailResponse(BaseModel):
    run_id: str
    status: str
    config: dict[str, Any]
    started_at: str | None = None
    completed_at: str | None = None
    duration_seconds: float | None = None
    error_message: str = ""
    num_evaluation_results: int
    num_fingerprints: int
    num_deployment_rankings: int


# ---------------------------------------------------------------------------
# Evaluation results
# ---------------------------------------------------------------------------

class EvaluationResultResponse(BaseModel):
    model_name: str
    model_paradigm: str
    dataset_name: str
    perturbation_type: str | None = None
    severity: float | None = None
    # None means "not recorded" - missing values are never reported as 0.
    accuracy: float | None = None
    precision: float | None = None
    recall: float | None = None
    f1_score: float | None = None
    auc: float | None = None
    eer: float | None = None
    num_samples: int | None = None


class EvaluationResultListResponse(BaseModel):
    results: list[EvaluationResultResponse]
    total: int


# ---------------------------------------------------------------------------
# Fingerprints
# ---------------------------------------------------------------------------

class FingerprintResponse(BaseModel):
    model_name: str
    paradigm: str
    dimension_names: list[str]
    vector: list[float]
    scores: dict[str, float]


class FingerprintListResponse(BaseModel):
    fingerprints: list[FingerprintResponse]
    skipped: int = 0   # stored entries that were malformed and not shown


class FingerprintComparisonResponse(BaseModel):
    model_a: str
    model_b: str
    euclidean_distance: float
    cosine_distance: float
    dimension_gaps: dict[str, float]


# ---------------------------------------------------------------------------
# Deployment scoring
# ---------------------------------------------------------------------------

class DeploymentScoreResponse(BaseModel):
    model_name: str
    paradigm: str
    rank: int | None = None
    total_score: float | None = None
    component_scores: dict[str, float]
    weighted_components: dict[str, float]
    recommendation: str


class DeploymentRankingResponse(BaseModel):
    scenario: str
    weights: dict[str, float]
    ranked_scores: list[DeploymentScoreResponse]


class DecisionGuideResponse(BaseModel):
    title: str
    scenarios: dict[str, Any]


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

class FigureInfo(BaseModel):
    figure_id: str
    figure_type: str
    title: str
    caption: str
    png_url: str | None = None
    svg_url: str | None = None


class FigureListResponse(BaseModel):
    figures: list[FigureInfo]


# ---------------------------------------------------------------------------
# Models / config introspection
# ---------------------------------------------------------------------------

class ModelInfoResponse(BaseModel):
    name: str
    paradigm: str
    model_type: str
    description: str


class ModelListResponse(BaseModel):
    traditional: list[ModelInfoResponse]
    deep_learning: list[ModelInfoResponse]


class PerturbationInfoResponse(BaseModel):
    name: str
    category: str
    description: str


class PerturbationListResponse(BaseModel):
    perturbations: list[PerturbationInfoResponse]


class ScenarioInfoResponse(BaseModel):
    scenario: str
    display_name: str
    description: str
    weights: dict[str, float]


class ScenarioListResponse(BaseModel):
    scenarios: list[ScenarioInfoResponse]


# ---------------------------------------------------------------------------
# Generic
# ---------------------------------------------------------------------------

class HealthResponse(BaseModel):
    status: str
    framework_version: str


class ErrorResponse(BaseModel):
    detail: str


class StatusResponse(BaseModel):
    """What the API can currently see. Used by the dashboard for empty states."""
    db_path: str
    db_exists: bool
    db_readable: bool
    db_error: str | None = None
    run_count: int | None = None
    figures_dir: str
    figures_dir_exists: bool
    reports_dir: str
    reports_dir_exists: bool
    read_only: bool = True

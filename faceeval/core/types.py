from __future__ import annotations

import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum, auto
from typing import Any


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------

class ModelParadigm(str, Enum):
    """High-level family a recognition model belongs to."""
    TRADITIONAL = "traditional"
    DEEP_LEARNING = "deep_learning"


class TraditionalModelType(str, Enum):
    EIGENFACES = "eigenfaces"
    FISHERFACES = "fisherfaces"
    LBPH = "lbph"
    PCA_SVM = "pca_svm"
    HOG_SVM = "hog_svm"
    KNN = "knn"


class DeepModelType(str, Enum):
    FACENET = "facenet"
    ARCFACE = "arcface"
    INSIGHTFACE = "insightface"
    DLIB_FR = "dlib_fr"


class PerturbationCategory(str, Enum):
    BLUR = "blur"
    NOISE = "noise"
    PHOTOMETRIC = "photometric"
    GEOMETRIC = "geometric"
    OCCLUSION = "occlusion"
    COMPRESSION = "compression"
    RESOLUTION = "resolution"


class PerturbationType(str, Enum):
    # Blur
    GAUSSIAN_BLUR = "gaussian_blur"
    MOTION_BLUR = "motion_blur"
    # Noise
    GAUSSIAN_NOISE = "gaussian_noise"
    SALT_PEPPER_NOISE = "salt_pepper_noise"
    SPECKLE_NOISE = "speckle_noise"
    # Photometric
    BRIGHTNESS = "brightness"
    CONTRAST = "contrast"
    GAMMA = "gamma"
    # Geometric
    ROTATION = "rotation"
    SCALING = "scaling"
    CROPPING = "cropping"
    # Occlusion
    RANDOM_OCCLUSION = "random_occlusion"
    FACE_MASK = "face_mask"
    SUNGLASSES = "sunglasses"
    # Compression / resolution
    JPEG_COMPRESSION = "jpeg_compression"
    RESOLUTION_DEGRADATION = "resolution_degradation"


class SplitStrategy(str, Enum):
    RANDOM = "random"
    STRATIFIED = "stratified"
    SUBJECT_DISJOINT = "subject_disjoint"   # no subject in both train and test
    K_FOLD = "k_fold"


class DistanceMetric(str, Enum):
    COSINE = "cosine"
    EUCLIDEAN = "euclidean"
    L2 = "l2"


class ExperimentStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class DeploymentScenario(str, Enum):
    BORDER_CONTROL = "border_control"
    ACCESS_CONTROL = "access_control"
    SURVEILLANCE = "surveillance"
    MOBILE_AUTHENTICATION = "mobile_authentication"
    RESEARCH_BENCHMARK = "research_benchmark"
    CUSTOM = "custom"


class FailureMode(str, Enum):
    FALSE_ACCEPT = "false_accept"
    FALSE_REJECT = "false_reject"
    LOW_CONFIDENCE_CORRECT = "low_confidence_correct"
    HIGH_CONFIDENCE_WRONG = "high_confidence_wrong"


# ---------------------------------------------------------------------------
# Primitive typed aliases
# ---------------------------------------------------------------------------

ImagePath = str          # absolute or relative path to an image file
SubjectID = str          # opaque identity label (e.g. "s001", "subject_42")
ModelName = str          # registered model name string
RunID = str              # UUID4 hex string for an experiment run
Severity = float         # perturbation severity in [0.0, 1.0]
Probability = float      # value in [0.0, 1.0]
EmbeddingVector = list[float]


# ---------------------------------------------------------------------------
# Dataset types
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ImageRecord:
    """One image in the dataset with its ground-truth labels."""
    image_id: str
    image_path: ImagePath
    subject_id: SubjectID
    dataset_name: str
    split: str                          # "train" | "test" | "val"
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class DatasetInfo:
    """
    Descriptor for a registered face dataset.

    ``subject_attributes`` captures per-subject demographic metadata
    (gender, age-range, ethnicity) for fairness evaluation.  Keys are
    subject IDs; values are free-form attribute dicts.
    """
    name: str
    root_path: str
    num_subjects: int
    num_images: int
    image_size: tuple[int, int]         # (height, width)
    color_mode: str                     # "rgb" | "grayscale"
    subject_attributes: dict[SubjectID, dict[str, Any]] = field(default_factory=dict)
    split_strategy: SplitStrategy = SplitStrategy.SUBJECT_DISJOINT
    description: str = ""
    license: str = ""
    citation: str = ""

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["split_strategy"] = self.split_strategy.value
        return d


@dataclass(frozen=True)
class DataSplit:
    """Train / validation / test partitions for one dataset run."""
    dataset_name: str
    train_ids: list[str]        # image_id list
    val_ids: list[str]
    test_ids: list[str]
    split_strategy: SplitStrategy
    random_seed: int
    k_fold_index: int | None = None
    k_fold_total: int | None = None

    @property
    def train_size(self) -> int:
        return len(self.train_ids)

    @property
    def test_size(self) -> int:
        return len(self.test_ids)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["split_strategy"] = self.split_strategy.value
        return d


# ---------------------------------------------------------------------------
# Model types
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ModelInfo:
    """
    Static descriptor for a registered recognition model.

    ``hyperparameters`` holds model-specific training config (e.g. PCA
    components for Eigenfaces, margin for ArcFace) so every run is
    self-documenting.
    """
    name: ModelName
    paradigm: ModelParadigm
    model_type: str                     # TraditionalModelType | DeepModelType value
    version: str
    embedding_dim: int | None           # None for non-embedding models (e.g. LBPH)
    distance_metric: DistanceMetric
    requires_gpu: bool
    model_size_mb: float | None = None
    hyperparameters: dict[str, Any] = field(default_factory=dict)
    description: str = ""
    paper_url: str = ""

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["paradigm"] = self.paradigm.value
        d["distance_metric"] = self.distance_metric.value
        return d


# ---------------------------------------------------------------------------
# Perturbation types
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PerturbationSpec:
    """
    Fully describes one perturbation at one severity level.

    ``raw_params`` stores the actual parameter values that were applied
    (e.g. ``{"sigma": 3.2}`` for Gaussian blur at severity 0.6).  These
    are recorded in the run log so figures can display human-readable axes.
    """
    perturbation_type: PerturbationType
    category: PerturbationCategory
    severity: Severity                  # normalised [0, 1]
    raw_params: dict[str, Any] = field(default_factory=dict)
    description: str = ""

    def __post_init__(self) -> None:
        if not 0.0 <= self.severity <= 1.0:
            raise ValueError(
                f"Severity must be in [0, 1], got {self.severity}"
            )

    @property
    def label(self) -> str:
        return f"{self.perturbation_type.value}_s{self.severity:.2f}"

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["perturbation_type"] = self.perturbation_type.value
        d["category"] = self.category.value
        return d


@dataclass(frozen=True)
class PerturbedBatch:
    """
    A batch of images after one perturbation has been applied.

    ``image_tensors`` is intentionally typed as ``list[Any]`` so the type
    layer does not import NumPy or PyTorch.  Concrete implementations pass
    ``np.ndarray`` or ``torch.Tensor`` objects.
    """
    original_ids: list[str]             # image_id for each item
    subject_ids: list[SubjectID]
    image_tensors: list[Any]            # shape (H, W, C) per image
    perturbation_spec: PerturbationSpec
    dataset_name: str

    def __post_init__(self) -> None:
        n = len(self.original_ids)
        if len(self.subject_ids) != n or len(self.image_tensors) != n:
            raise ValueError(
                "original_ids, subject_ids, and image_tensors must have the same length"
            )

    @property
    def batch_size(self) -> int:
        return len(self.original_ids)


# ---------------------------------------------------------------------------
# Prediction / inference types
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PredictionResult:
    """
    Raw output from one model on one image (or image pair for verification).

    For *identification* tasks ``predicted_subject_id`` and ``confidence``
    are populated.  For *verification* tasks ``is_same_person`` and
    ``similarity_score`` are populated.  Both may be populated when a model
    supports both modes.
    """
    image_id: str
    model_name: ModelName
    perturbation_spec: PerturbationSpec | None

    # Identification output
    predicted_subject_id: SubjectID | None = None
    confidence: Probability | None = None
    top_k_predictions: list[tuple[SubjectID, Probability]] = field(default_factory=list)

    # Verification output
    probe_image_id: str | None = None
    gallery_image_id: str | None = None
    is_same_person: bool | None = None
    similarity_score: float | None = None

    # Embedding (stored for downstream analysis)
    embedding: EmbeddingVector | None = None

    # Timing
    inference_time_ms: float = 0.0
    embedding_time_ms: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        if self.perturbation_spec is not None:
            d["perturbation_spec"] = self.perturbation_spec.to_dict()
        return d


# ---------------------------------------------------------------------------
# Resource / profiling types
# ---------------------------------------------------------------------------

@dataclass
class ResourceUsage:
    """
    Hardware resource snapshot for one model inference pass.

    Fields are optional because not all hardware supports all counters
    (e.g. GPU fields are None when running CPU-only).
    """
    model_name: ModelName
    batch_size: int
    inference_time_ms: float
    embedding_time_ms: float
    peak_memory_mb: float
    model_size_mb: float
    cpu_utilization_pct: float | None = None
    gpu_utilization_pct: float | None = None
    gpu_memory_mb: float | None = None
    throughput_fps: float | None = None     # images per second
    device: str = "cpu"                     # "cpu" | "cuda:0" | …

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# Evaluation metric types
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ROCCurve:
    """Receiver operating characteristic curve data."""
    fpr: list[float]        # false positive rates
    tpr: list[float]        # true positive rates
    thresholds: list[float]
    auc: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class PRCurve:
    """Precision-recall curve data."""
    precision: list[float]
    recall: list[float]
    thresholds: list[float]
    average_precision: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class CalibrationMetrics:
    """
    Calibration quality metrics for a model's confidence outputs.

    ``reliability_diagram_bins`` stores (mean_confidence, mean_accuracy,
    bin_count) triples for plotting the reliability diagram.
    ``ece`` is the Expected Calibration Error (lower = better calibrated).
    ``mce`` is the Maximum Calibration Error.
    ``brier_score`` measures probabilistic forecast quality.
    """
    model_name: ModelName
    perturbation_spec: PerturbationSpec | None
    ece: float                          # expected calibration error
    mce: float                          # maximum calibration error
    brier_score: float
    overconfidence_rate: float          # fraction of bins where conf > acc
    underconfidence_rate: float
    reliability_diagram_bins: list[tuple[float, float, int]]   # (conf, acc, n)
    num_samples: int

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        if self.perturbation_spec is not None:
            d["perturbation_spec"] = self.perturbation_spec.to_dict()
        return d


@dataclass(frozen=True)
class EvaluationResult:
    """
    Complete evaluation output for one (model × perturbation × split) triple.

    This is the central result object that flows from the evaluation layer
    to fingerprinting, failure analysis, deployment scoring, and reporting.
    """
    # Identity
    result_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    run_id: RunID = ""
    model_name: ModelName = ""
    model_paradigm: ModelParadigm = ModelParadigm.TRADITIONAL
    dataset_name: str = ""
    perturbation_spec: PerturbationSpec | None = None

    # Classification metrics
    accuracy: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    f1_score: float = 0.0
    top_k_accuracy: dict[int, float] = field(default_factory=dict)   # {1: 0.93, 5: 0.98}

    # Verification metrics
    auc: float = 0.0
    eer: float = 0.0                    # equal error rate
    far_at_thresholds: dict[float, float] = field(default_factory=dict)  # {0.001: far_val}
    frr_at_thresholds: dict[float, float] = field(default_factory=dict)
    roc_curve: ROCCurve | None = None
    pr_curve: PRCurve | None = None

    # Robustness
    baseline_accuracy: float | None = None   # accuracy at severity=0.0
    degradation_auc: float | None = None     # area under accuracy-vs-severity curve

    # Resource usage
    resource_usage: ResourceUsage | None = None

    # Calibration
    calibration: CalibrationMetrics | None = None

    # Per-subject breakdown for fairness
    per_subject_accuracy: dict[SubjectID, float] = field(default_factory=dict)

    # Subgroup fairness (keyed by attribute string, e.g. "gender:female")
    subgroup_metrics: dict[str, dict[str, float]] = field(default_factory=dict)

    # Confusion matrix (flattened row-major, n_classes × n_classes)
    confusion_matrix: list[list[int]] = field(default_factory=list)
    class_labels: list[str] = field(default_factory=list)

    # Metadata
    num_samples: int = 0
    num_subjects: int = 0
    evaluated_at: datetime = field(default_factory=datetime.utcnow)
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["model_paradigm"] = self.model_paradigm.value
        d["evaluated_at"] = self.evaluated_at.isoformat()
        if self.perturbation_spec is not None:
            d["perturbation_spec"] = self.perturbation_spec.to_dict()
        if self.roc_curve is not None:
            d["roc_curve"] = self.roc_curve.to_dict()
        if self.pr_curve is not None:
            d["pr_curve"] = self.pr_curve.to_dict()
        if self.calibration is not None:
            d["calibration"] = self.calibration.to_dict()
        if self.resource_usage is not None:
            d["resource_usage"] = self.resource_usage.to_dict()
        return d


# ---------------------------------------------------------------------------
# Behavioral fingerprint types
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ADCPoint:
    """One point on an Accuracy-Degradation Curve."""
    severity: Severity
    accuracy: float
    perturbation_type: PerturbationType

    def to_dict(self) -> dict[str, Any]:
        return {
            "severity": self.severity,
            "accuracy": self.accuracy,
            "perturbation_type": self.perturbation_type.value,
        }


@dataclass(frozen=True)
class ADCCurve:
    """
    Full Accuracy-Degradation Curve for one (model × perturbation type) pair.

    ``area`` is the integral of accuracy over severity — higher means the
    model degrades more gracefully.  It is used as a single summary scalar
    in radar charts.
    """
    model_name: ModelName
    perturbation_type: PerturbationType
    points: list[ADCPoint]
    area: float                         # AUC of accuracy vs severity in [0, 1]²

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_name": self.model_name,
            "perturbation_type": self.perturbation_type.value,
            "points": [p.to_dict() for p in self.points],
            "area": self.area,
        }


@dataclass(frozen=True)
class SampleEfficiencyCurve:
    """Accuracy as a function of training-set size (for traditional ML models)."""
    model_name: ModelName
    training_sizes: list[int]
    accuracies: list[float]
    perturbation_spec: PerturbationSpec | None = None

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        if self.perturbation_spec is not None:
            d["perturbation_spec"] = self.perturbation_spec.to_dict()
        return d


@dataclass(frozen=True)
class BehavioralFingerprint:
    """
    Fixed-length characterisation vector for one recognition model.

    The fingerprint is a dict of named scalar scores, each in [0, 1],
    computed from the full evaluation suite.  Entries include:

    - ``robustness_{perturbation_type}`` — ADC area for each perturbation
    - ``calibration_quality`` — 1 - ECE
    - ``sample_efficiency`` — area under the sample efficiency curve
    - ``fairness_gap`` — 1 - max subgroup accuracy gap
    - ``speed_score`` — normalised throughput
    - ``memory_efficiency`` — normalised inverse peak memory

    The ``vector`` field is the ordered values matching ``dimension_names``
    for use in distance computations and radar chart rendering.
    """
    model_name: ModelName
    paradigm: ModelParadigm
    dimension_names: list[str]
    vector: list[float]                 # parallel to dimension_names
    scores: dict[str, float]            # named access to the same data
    adc_curves: list[ADCCurve] = field(default_factory=list)
    sample_efficiency: SampleEfficiencyCurve | None = None
    computed_at: datetime = field(default_factory=datetime.utcnow)

    def __post_init__(self) -> None:
        if len(self.dimension_names) != len(self.vector):
            raise ValueError(
                "dimension_names and vector must have the same length; "
                f"got {len(self.dimension_names)} names and {len(self.vector)} values"
            )

    def distance_to(self, other: "BehavioralFingerprint") -> float:
        """Euclidean distance between two fingerprint vectors."""
        if self.dimension_names != other.dimension_names:
            raise ValueError("Cannot compare fingerprints with different dimensions")
        return float(
            sum((a - b) ** 2 for a, b in zip(self.vector, other.vector)) ** 0.5
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_name": self.model_name,
            "paradigm": self.paradigm.value,
            "dimension_names": self.dimension_names,
            "vector": self.vector,
            "scores": self.scores,
            "adc_curves": [c.to_dict() for c in self.adc_curves],
            "computed_at": self.computed_at.isoformat(),
        }


# ---------------------------------------------------------------------------
# Failure analysis types
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class FailureCase:
    """
    One misclassification or missed verification captured for failure analysis.

    ``failure_mode`` distinguishes false accepts, false rejects, and
    confidence-calibration failures so the failure analysis module can
    cluster them independently.
    """
    case_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    image_id: str = ""
    subject_id: SubjectID = ""
    model_name: ModelName = ""
    perturbation_spec: PerturbationSpec | None = None
    failure_mode: FailureMode = FailureMode.FALSE_REJECT

    # What the model predicted
    predicted_subject_id: SubjectID | None = None
    confidence: Probability | None = None
    similarity_score: float | None = None

    # Ground truth
    true_subject_id: SubjectID = ""

    # Low-dimensional projection for clustering (e.g. 2-D UMAP coords)
    projection_2d: tuple[float, float] | None = None

    # Cluster assignment (populated by clustering step)
    cluster_id: int | None = None

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["failure_mode"] = self.failure_mode.value
        if self.perturbation_spec is not None:
            d["perturbation_spec"] = self.perturbation_spec.to_dict()
        return d


@dataclass(frozen=True)
class FailureCluster:
    """A cluster of structurally similar failure cases."""
    cluster_id: int
    model_name: ModelName
    cases: list[FailureCase]
    dominant_failure_mode: FailureMode
    dominant_perturbation_type: PerturbationType | None
    centroid_2d: tuple[float, float] | None
    description: str = ""

    @property
    def size(self) -> int:
        return len(self.cases)

    def to_dict(self) -> dict[str, Any]:
        return {
            "cluster_id": self.cluster_id,
            "model_name": self.model_name,
            "size": self.size,
            "dominant_failure_mode": self.dominant_failure_mode.value,
            "dominant_perturbation_type": (
                self.dominant_perturbation_type.value
                if self.dominant_perturbation_type else None
            ),
            "centroid_2d": self.centroid_2d,
            "description": self.description,
        }


@dataclass(frozen=True)
class CrossParadigmFailureCorrelation:
    """
    Overlap analysis between failure cases of a traditional and a DL model.

    ``overlap_fraction`` is |traditional_failures ∩ dl_failures| /
    |traditional_failures ∪ dl_failures|.  High overlap means both paradigms
    fail on the same inputs — a fundamental dataset/perturbation challenge.
    Low overlap means the failures are complementary — an ensemble could help.
    """
    traditional_model: ModelName
    dl_model: ModelName
    perturbation_type: PerturbationType
    overlap_fraction: float
    traditional_only_ids: list[str]     # image_ids where only trad fails
    dl_only_ids: list[str]
    shared_ids: list[str]

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["perturbation_type"] = self.perturbation_type.value
        return d


# ---------------------------------------------------------------------------
# Deployment scoring types
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class DeploymentWeights:
    """
    Context-specific importance weights for the deployment score.

    All weights should sum to 1.0.  Each weight controls how much the
    corresponding metric dimension contributes to the final score.
    """
    accuracy: float = 0.25
    robustness: float = 0.20
    calibration: float = 0.10
    fairness: float = 0.10
    latency: float = 0.15
    memory: float = 0.10
    computational_cost: float = 0.10

    def __post_init__(self) -> None:
        total = (
            self.accuracy + self.robustness + self.calibration
            + self.fairness + self.latency + self.memory
            + self.computational_cost
        )
        if abs(total - 1.0) > 1e-6:
            raise ValueError(
                f"DeploymentWeights must sum to 1.0, got {total:.6f}"
            )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class DeploymentScore:
    """
    Composite deployment readiness score for one model.

    ``component_scores`` holds each dimension's normalised contribution
    (before weighting) so the score can be explained.
    ``weighted_components`` holds each dimension's weighted contribution
    (component_score × weight) so they sum to ``total_score``.
    """
    model_name: ModelName
    paradigm: ModelParadigm
    scenario: DeploymentScenario
    weights: DeploymentWeights

    # Component scores (each in [0, 1], higher = better)
    component_scores: dict[str, float] = field(default_factory=dict)
    weighted_components: dict[str, float] = field(default_factory=dict)
    total_score: float = 0.0

    # Sensitivity: how much total_score changes per unit change in each weight
    weight_sensitivity: dict[str, float] = field(default_factory=dict)

    # Rank among all evaluated models (1 = best)
    rank: int | None = None

    # Human-readable recommendation
    recommendation: str = ""

    computed_at: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["paradigm"] = self.paradigm.value
        d["scenario"] = self.scenario.value
        d["weights"] = self.weights.to_dict()
        d["computed_at"] = self.computed_at.isoformat()
        return d


@dataclass
class DeploymentRanking:
    """Ordered ranking of all evaluated models for one scenario."""
    scenario: DeploymentScenario
    weights: DeploymentWeights
    ranked_scores: list[DeploymentScore]        # index 0 = rank 1

    @property
    def best_model(self) -> ModelName:
        return self.ranked_scores[0].model_name if self.ranked_scores else ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "scenario": self.scenario.value,
            "weights": self.weights.to_dict(),
            "ranking": [s.to_dict() for s in self.ranked_scores],
        }


# ---------------------------------------------------------------------------
# Experiment configuration types
# ---------------------------------------------------------------------------

@dataclass
class PerturbationSchedule:
    """
    Specifies which perturbations to apply and at which severity levels.

    ``severity_levels`` is the list of normalised severities to evaluate.
    Setting it to ``[0.0, 0.2, 0.4, 0.6, 0.8, 1.0]`` gives 6-point curves.
    """
    perturbation_types: list[PerturbationType]
    severity_levels: list[Severity] = field(
        default_factory=lambda: [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    )
    compose_perturbations: bool = False     # apply multiple perturbations at once

    def __post_init__(self) -> None:
        for s in self.severity_levels:
            if not 0.0 <= s <= 1.0:
                raise ValueError(f"Severity level {s} not in [0, 1]")

    @property
    def total_conditions(self) -> int:
        return len(self.perturbation_types) * len(self.severity_levels)

    def to_dict(self) -> dict[str, Any]:
        return {
            "perturbation_types": [p.value for p in self.perturbation_types],
            "severity_levels": self.severity_levels,
            "compose_perturbations": self.compose_perturbations,
        }


@dataclass
class ExperimentConfig:
    """
    Top-level configuration object loaded from a YAML file.

    This is the single source of truth for one experiment run.  Every
    field is recorded in MLflow so runs are fully reproducible from config.

    ``run_id`` is generated at runtime if not provided in the YAML.
    ``config_hash`` is the SHA256 of the serialised config (set by the
    orchestrator) and used to detect when a run has been re-submitted
    with identical config.
    """
    # Run identity
    experiment_name: str
    run_id: RunID = field(default_factory=lambda: uuid.uuid4().hex)
    description: str = ""

    # Reproducibility
    random_seed: int = 42
    deterministic_mode: bool = True     # sets CUDA deterministic ops

    # Data
    dataset_names: list[str] = field(default_factory=list)
    data_root: str = "data/"
    # Optional dataset-size controls. Defaults preserve the full dataset.
    min_images_per_subject: int = 0
    max_subjects: int | None = None
    split_strategy: SplitStrategy = SplitStrategy.SUBJECT_DISJOINT
    test_fraction: float = 0.3
    val_fraction: float = 0.1
    k_folds: int | None = None

    # Models
    traditional_models: list[TraditionalModelType] = field(default_factory=list)
    deep_models: list[DeepModelType] = field(default_factory=list)

    # Perturbations
    perturbation_schedule: PerturbationSchedule | None = None

    # Evaluation
    compute_calibration: bool = True
    compute_fairness: bool = True
    compute_fingerprints: bool = True
    compute_failure_analysis: bool = True
    compute_deployment_scores: bool = True
    deployment_scenarios: list[DeploymentScenario] = field(default_factory=list)
    custom_deployment_weights: dict[str, DeploymentWeights] = field(default_factory=dict)

    # Resource profiling
    profile_resources: bool = True
    profile_batch_size: int = 32

    # Feature caching
    use_feature_cache: bool = True
    feature_cache_dir: str = "experiments/cache/"

    # Output
    output_dir: str = "experiments/"
    report_dir: str = "reports/"
    export_svg: bool = True
    export_png: bool = True
    export_csv: bool = True
    export_json: bool = True
    generate_pdf_report: bool = True

    # Tracking
    mlflow_tracking_uri: str = "sqlite:///experiments/mlflow.db"
    mlflow_experiment_name: str | None = None       # defaults to experiment_name

    # Hardware
    device: str = "auto"            # "auto" | "cpu" | "cuda" | "cuda:0"

    # Internal (set by orchestrator, not from YAML)
    config_hash: str = ""
    git_commit: str = ""
    framework_version: str = "0.1.0"
    created_at: datetime = field(default_factory=datetime.utcnow)

    @property
    def all_model_names(self) -> list[str]:
        return (
            [m.value for m in self.traditional_models]
            + [m.value for m in self.deep_models]
        )

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["split_strategy"] = self.split_strategy.value
        d["traditional_models"] = [m.value for m in self.traditional_models]
        d["deep_models"] = [m.value for m in self.deep_models]
        d["deployment_scenarios"] = [s.value for s in self.deployment_scenarios]
        d["created_at"] = self.created_at.isoformat()
        if self.perturbation_schedule is not None:
            d["perturbation_schedule"] = self.perturbation_schedule.to_dict()
        d["custom_deployment_weights"] = {
            k: v.to_dict() for k, v in self.custom_deployment_weights.items()
        }
        return d


# ---------------------------------------------------------------------------
# Experiment run state types
# ---------------------------------------------------------------------------

@dataclass
class ExperimentRun:
    """
    Live state record for one experiment run.

    Written to the result store and updated as the run progresses.
    ``evaluation_results`` accumulates one ``EvaluationResult`` per
    (model × perturbation × severity) triple as they complete.
    """
    run_id: RunID
    config: ExperimentConfig
    status: ExperimentStatus = ExperimentStatus.PENDING
    evaluation_results: list[EvaluationResult] = field(default_factory=list)
    fingerprints: list[BehavioralFingerprint] = field(default_factory=list)
    failure_clusters: list[FailureCluster] = field(default_factory=list)
    deployment_rankings: list[DeploymentRanking] = field(default_factory=list)
    started_at: datetime | None = None
    completed_at: datetime | None = None
    error_message: str = ""

    @property
    def duration_seconds(self) -> float | None:
        if self.started_at and self.completed_at:
            return (self.completed_at - self.started_at).total_seconds()
        return None

    @property
    def num_conditions_completed(self) -> int:
        return len(self.evaluation_results)

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "status": self.status.value,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "duration_seconds": self.duration_seconds,
            "num_conditions_completed": self.num_conditions_completed,
            "error_message": self.error_message,
        }


# ---------------------------------------------------------------------------
# Fairness types
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SubgroupMetrics:
    """
    Evaluation metrics computed for one demographic subgroup.

    ``attribute_key`` is a string like ``"gender:female"`` or
    ``"age_group:senior"``.
    """
    model_name: ModelName
    attribute_key: str
    attribute_value: str
    num_samples: int
    accuracy: float
    eer: float
    far: float
    frr: float
    auc: float

    @property
    def attribute_label(self) -> str:
        return f"{self.attribute_key}:{self.attribute_value}"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class FairnessReport:
    """
    Cross-subgroup fairness summary for one model.

    ``max_accuracy_gap`` is the largest accuracy difference between any two
    subgroups.  ``equal_opportunity_gap`` compares true positive rates.
    These map directly to Chapter 10 subgroup disparity analysis.
    """
    model_name: ModelName
    perturbation_spec: PerturbationSpec | None
    subgroup_metrics: list[SubgroupMetrics]
    max_accuracy_gap: float
    max_eer_gap: float
    equal_opportunity_gap: float
    demographic_parity_gap: float

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        if self.perturbation_spec is not None:
            d["perturbation_spec"] = self.perturbation_spec.to_dict()
        d["subgroup_metrics"] = [m.to_dict() for m in self.subgroup_metrics]
        return d


# ---------------------------------------------------------------------------
# Reporting types
# ---------------------------------------------------------------------------

@dataclass
class FigureSpec:
    """Descriptor for one generated figure (before it is rendered)."""
    figure_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    figure_type: str = ""               # "roc", "adc", "radar", "heatmap", …
    title: str = ""
    caption: str = ""                   # LaTeX-safe caption for thesis
    output_path_svg: str = ""
    output_path_png: str = ""
    chapter: int | None = None          # which thesis chapter uses this figure
    models_included: list[ModelName] = field(default_factory=list)
    perturbations_included: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ExperimentReport:
    """
    Complete structured report assembled from one finished ExperimentRun.

    This object drives both the PDF report generator and the React dashboard.
    """
    run_id: RunID
    experiment_name: str
    config: ExperimentConfig
    evaluation_results: list[EvaluationResult]
    fingerprints: list[BehavioralFingerprint]
    failure_clusters: list[FailureCluster]
    deployment_rankings: list[DeploymentRanking]
    fairness_reports: list[FairnessReport]
    figures: list[FigureSpec]
    generated_at: datetime = field(default_factory=datetime.utcnow)
    summary_text: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "experiment_name": self.experiment_name,
            "generated_at": self.generated_at.isoformat(),
            "summary_text": self.summary_text,
            "num_evaluation_results": len(self.evaluation_results),
            "num_models": len({r.model_name for r in self.evaluation_results}),
            "num_perturbations": len({
                r.perturbation_spec.perturbation_type.value
                for r in self.evaluation_results
                if r.perturbation_spec
            }),
            "figures": [f.to_dict() for f in self.figures],
        }


# ---------------------------------------------------------------------------
# Public re-exports (everything a downstream module needs to import)
# ---------------------------------------------------------------------------

__all__ = [
    # Enums
    "ModelParadigm",
    "TraditionalModelType",
    "DeepModelType",
    "PerturbationCategory",
    "PerturbationType",
    "SplitStrategy",
    "DistanceMetric",
    "ExperimentStatus",
    "DeploymentScenario",
    "FailureMode",
    # Aliases
    "ImagePath",
    "SubjectID",
    "ModelName",
    "RunID",
    "Severity",
    "Probability",
    "EmbeddingVector",
    # Dataset
    "ImageRecord",
    "DatasetInfo",
    "DataSplit",
    # Model
    "ModelInfo",
    # Perturbation
    "PerturbationSpec",
    "PerturbedBatch",
    # Prediction
    "PredictionResult",
    # Resource
    "ResourceUsage",
    # Evaluation
    "ROCCurve",
    "PRCurve",
    "CalibrationMetrics",
    "EvaluationResult",
    # Fingerprint
    "ADCPoint",
    "ADCCurve",
    "SampleEfficiencyCurve",
    "BehavioralFingerprint",
    # Failure
    "FailureCase",
    "FailureCluster",
    "CrossParadigmFailureCorrelation",
    # Deployment
    "DeploymentWeights",
    "DeploymentScore",
    "DeploymentRanking",
    # Config
    "PerturbationSchedule",
    "ExperimentConfig",
    "ExperimentRun",
    # Fairness
    "SubgroupMetrics",
    "FairnessReport",
    # Reporting
    "FigureSpec",
    "ExperimentReport",
]

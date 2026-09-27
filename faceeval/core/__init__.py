from faceeval.core.types import (
    # Enums
    ModelParadigm,
    TraditionalModelType,
    DeepModelType,
    PerturbationCategory,
    PerturbationType,
    SplitStrategy,
    DistanceMetric,
    ExperimentStatus,
    DeploymentScenario,
    FailureMode,
    # Primitives
    ImagePath,
    SubjectID,
    ModelName,
    RunID,
    Severity,
    Probability,
    EmbeddingVector,
    # Dataset
    ImageRecord,
    DatasetInfo,
    DataSplit,
    # Model
    ModelInfo,
    # Perturbation
    PerturbationSpec,
    PerturbedBatch,
    # Prediction
    PredictionResult,
    # Resource
    ResourceUsage,
    # Evaluation
    ROCCurve,
    PRCurve,
    CalibrationMetrics,
    EvaluationResult,
    # Fingerprint
    ADCPoint,
    ADCCurve,
    SampleEfficiencyCurve,
    BehavioralFingerprint,
    # Failure
    FailureCase,
    FailureCluster,
    CrossParadigmFailureCorrelation,
    # Deployment
    DeploymentWeights,
    DeploymentScore,
    DeploymentRanking,
    # Config dataclass
    PerturbationSchedule,
    ExperimentConfig,
    ExperimentRun,
    # Fairness
    SubgroupMetrics,
    FairnessReport,
    # Reporting
    FigureSpec,
    ExperimentReport,
)

from faceeval.core.config import (
    ConfigLoader,
    load_config,
    make_default_config,
)

from faceeval.core.registry import (
    ModelRegistry,
    PerturbationRegistry,
    DatasetRegistry,
    model_registry,
    perturbation_registry,
    dataset_registry,
    register_model,
    register_perturbation,
    register_dataset,
)

from faceeval.core.exceptions import (
    FaceEvalError,
    ConfigError,
    MissingConfigKeyError,
    RegistrationError,
    UnknownPluginError,
    DataError,
    DatasetNotFoundError,
    ImageLoadError,
    SplitError,
    CacheError,
    PreprocessingError,
    FaceDetectionError,
    AlignmentError,
    PerturbationError,
    InvalidSeverityError,
    ModelError,
    ModelNotTrainedError,
    ModelCheckpointError,
    InferenceError,
    EmbeddingError,
    EvaluationError,
    InsufficientSamplesError,
    CalibrationError,
    FairnessError,
    FingerprintError,
    FailureAnalysisError,
    DeploymentScoringError,
    StorageError,
    RunNotFoundError,
    DuplicateRunError,
    VisualizationError,
    ReportError,
    OrchestratorError,
    ReproducibilityError,
)

__version__ = "0.1.0"

__all__ = [
    # Version
    "__version__",
    # Enums
    "ModelParadigm", "TraditionalModelType", "DeepModelType",
    "PerturbationCategory", "PerturbationType", "SplitStrategy",
    "DistanceMetric", "ExperimentStatus", "DeploymentScenario", "FailureMode",
    # Primitives
    "ImagePath", "SubjectID", "ModelName", "RunID",
    "Severity", "Probability", "EmbeddingVector",
    # Dataset
    "ImageRecord", "DatasetInfo", "DataSplit",
    # Model
    "ModelInfo",
    # Perturbation
    "PerturbationSpec", "PerturbedBatch",
    # Prediction
    "PredictionResult",
    # Resource
    "ResourceUsage",
    # Evaluation
    "ROCCurve", "PRCurve", "CalibrationMetrics", "EvaluationResult",
    # Fingerprint
    "ADCPoint", "ADCCurve", "SampleEfficiencyCurve", "BehavioralFingerprint",
    # Failure
    "FailureCase", "FailureCluster", "CrossParadigmFailureCorrelation",
    # Deployment
    "DeploymentWeights", "DeploymentScore", "DeploymentRanking",
    # Config
    "PerturbationSchedule", "ExperimentConfig", "ExperimentRun",
    # Fairness
    "SubgroupMetrics", "FairnessReport",
    # Reporting
    "FigureSpec", "ExperimentReport",
    # Config utilities
    "ConfigLoader", "load_config", "make_default_config",
    # Registry singletons
    "ModelRegistry", "PerturbationRegistry", "DatasetRegistry",
    "model_registry", "perturbation_registry", "dataset_registry",
    "register_model", "register_perturbation", "register_dataset",
    # Exceptions
    "FaceEvalError", "ConfigError", "MissingConfigKeyError",
    "RegistrationError", "UnknownPluginError",
    "DataError", "DatasetNotFoundError", "ImageLoadError", "SplitError", "CacheError",
    "PreprocessingError", "FaceDetectionError", "AlignmentError",
    "PerturbationError", "InvalidSeverityError",
    "ModelError", "ModelNotTrainedError", "ModelCheckpointError",
    "InferenceError", "EmbeddingError",
    "EvaluationError", "InsufficientSamplesError", "CalibrationError", "FairnessError",
    "FingerprintError", "FailureAnalysisError", "DeploymentScoringError",
    "StorageError", "RunNotFoundError", "DuplicateRunError",
    "VisualizationError", "ReportError",
    "OrchestratorError", "ReproducibilityError",
]

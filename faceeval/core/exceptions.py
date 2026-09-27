from __future__ import annotations

from typing import Any


# ---------------------------------------------------------------------------
# Base
# ---------------------------------------------------------------------------

class FaceEvalError(Exception):
    """Root of the FaceEval-X exception hierarchy."""

    def __init__(self, message: str, context: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.context: dict[str, Any] = context or {}

    def __str__(self) -> str:
        base = super().__str__()
        if self.context:
            ctx_str = ", ".join(f"{k}={v!r}" for k, v in self.context.items())
            return f"{base} [{ctx_str}]"
        return base


# ---------------------------------------------------------------------------
# Configuration errors
# ---------------------------------------------------------------------------

class ConfigError(FaceEvalError):
    """Raised when a configuration file is malformed or fails validation."""


class MissingConfigKeyError(ConfigError):
    """A required configuration key is absent."""

    def __init__(self, key: str, config_path: str = "") -> None:
        super().__init__(
            f"Required config key '{key}' is missing.",
            context={"key": key, "config_path": config_path},
        )
        self.key = key


# ---------------------------------------------------------------------------
# Registry errors
# ---------------------------------------------------------------------------

class RegistrationError(FaceEvalError):
    """Raised when a duplicate key is registered in a registry."""


class UnknownPluginError(FaceEvalError):
    """Raised when a requested plugin key is not found in a registry."""


# ---------------------------------------------------------------------------
# Data layer errors
# ---------------------------------------------------------------------------

class DataError(FaceEvalError):
    """Base class for data-layer errors."""


class DatasetNotFoundError(DataError):
    """A named dataset is not registered or its root path does not exist."""

    def __init__(self, dataset_name: str, path: str = "") -> None:
        super().__init__(
            f"Dataset '{dataset_name}' not found.",
            context={"dataset_name": dataset_name, "path": path},
        )


class ImageLoadError(DataError):
    """An image could not be loaded or decoded."""

    def __init__(self, image_path: str, reason: str = "") -> None:
        super().__init__(
            f"Failed to load image '{image_path}': {reason}",
            context={"image_path": image_path, "reason": reason},
        )


class SplitError(DataError):
    """Dataset splitting failed (e.g. too few subjects for k-fold)."""


class CacheError(DataError):
    """Feature cache read or write failed."""


# ---------------------------------------------------------------------------
# Preprocessing errors
# ---------------------------------------------------------------------------

class PreprocessingError(FaceEvalError):
    """Base class for preprocessing-layer errors."""


class FaceDetectionError(PreprocessingError):
    """No face detected in an image that was expected to contain one."""

    def __init__(self, image_path: str, detector: str = "") -> None:
        super().__init__(
            f"No face detected in '{image_path}' using detector '{detector}'.",
            context={"image_path": image_path, "detector": detector},
        )


class AlignmentError(PreprocessingError):
    """Landmark detection or affine alignment failed."""


# ---------------------------------------------------------------------------
# Perturbation errors
# ---------------------------------------------------------------------------

class PerturbationError(FaceEvalError):
    """Base class for perturbation-engine errors."""


class InvalidSeverityError(PerturbationError):
    """A severity value outside [0, 1] was supplied."""

    def __init__(self, severity: float) -> None:
        super().__init__(
            f"Severity {severity} is not in [0.0, 1.0].",
            context={"severity": severity},
        )


# ---------------------------------------------------------------------------
# Model errors
# ---------------------------------------------------------------------------

class ModelError(FaceEvalError):
    """Base class for recognition model errors."""


class ModelNotTrainedError(ModelError):
    """Inference was attempted on a model that has not been trained or loaded."""

    def __init__(self, model_name: str) -> None:
        super().__init__(
            f"Model '{model_name}' is not trained. Call .fit() or .load() first.",
            context={"model_name": model_name},
        )


class ModelCheckpointError(ModelError):
    """A model checkpoint file could not be loaded."""

    def __init__(self, model_name: str, checkpoint_path: str, reason: str = "") -> None:
        super().__init__(
            f"Failed to load checkpoint for '{model_name}' from '{checkpoint_path}': {reason}",
            context={
                "model_name": model_name,
                "checkpoint_path": checkpoint_path,
                "reason": reason,
            },
        )


class InferenceError(ModelError):
    """An unrecoverable error occurred during model inference."""

    def __init__(self, model_name: str, image_id: str = "", reason: str = "") -> None:
        super().__init__(
            f"Inference failed for model '{model_name}' on image '{image_id}': {reason}",
            context={"model_name": model_name, "image_id": image_id, "reason": reason},
        )


class EmbeddingError(ModelError):
    """Embedding extraction failed or produced an invalid vector."""


# ---------------------------------------------------------------------------
# Evaluation errors
# ---------------------------------------------------------------------------

class EvaluationError(FaceEvalError):
    """Base class for evaluation-layer errors."""


class InsufficientSamplesError(EvaluationError):
    """Not enough samples to compute a requested metric reliably."""

    def __init__(self, metric: str, required: int, available: int) -> None:
        super().__init__(
            f"Metric '{metric}' requires at least {required} samples; "
            f"only {available} available.",
            context={"metric": metric, "required": required, "available": available},
        )


class CalibrationError(EvaluationError):
    """Calibration computation failed."""


class FairnessError(EvaluationError):
    """Fairness metric computation failed (e.g. missing attribute labels)."""


# ---------------------------------------------------------------------------
# Fingerprint / failure / deployment errors
# ---------------------------------------------------------------------------

class FingerprintError(FaceEvalError):
    """Behavioral fingerprint computation failed."""


class FailureAnalysisError(FaceEvalError):
    """Failure analysis (clustering, correlation) failed."""


class DeploymentScoringError(FaceEvalError):
    """Deployment score computation failed."""

    def __init__(self, model_name: str, reason: str = "") -> None:
        super().__init__(
            f"Deployment scoring failed for '{model_name}': {reason}",
            context={"model_name": model_name, "reason": reason},
        )


# ---------------------------------------------------------------------------
# Storage / persistence errors
# ---------------------------------------------------------------------------

class StorageError(FaceEvalError):
    """Base class for result-store errors."""


class RunNotFoundError(StorageError):
    """A requested run ID is not in the result store."""

    def __init__(self, run_id: str) -> None:
        super().__init__(
            f"Run '{run_id}' not found in the result store.",
            context={"run_id": run_id},
        )


class DuplicateRunError(StorageError):
    """A run with the same config hash already exists."""

    def __init__(self, config_hash: str, existing_run_id: str) -> None:
        super().__init__(
            f"A run with config hash '{config_hash}' already exists: '{existing_run_id}'.",
            context={"config_hash": config_hash, "existing_run_id": existing_run_id},
        )


# ---------------------------------------------------------------------------
# Visualisation / reporting errors
# ---------------------------------------------------------------------------

class VisualizationError(FaceEvalError):
    """Figure generation failed."""


class ReportError(FaceEvalError):
    """Report assembly or export failed."""


# ---------------------------------------------------------------------------
# Orchestration errors
# ---------------------------------------------------------------------------

class OrchestratorError(FaceEvalError):
    """Experiment run orchestration encountered an unrecoverable error."""


class ReproducibilityError(OrchestratorError):
    """
    Raised when the orchestrator detects conditions that would compromise
    run reproducibility (e.g. non-deterministic ops on CUDA without a fixed
    seed, or missing git commit hash in a production run).
    """

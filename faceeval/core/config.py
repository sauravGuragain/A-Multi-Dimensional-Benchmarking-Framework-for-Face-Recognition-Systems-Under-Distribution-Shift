from __future__ import annotations

import hashlib
import json
import os
import subprocess
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field, field_validator, model_validator

from faceeval.core.types import (
    DeepModelType,
    DeploymentScenario,
    DeploymentWeights,
    ExperimentConfig,
    PerturbationSchedule,
    PerturbationType,
    SplitStrategy,
    TraditionalModelType,
)


# ---------------------------------------------------------------------------
# Pydantic validation models (internal to this module)
# ---------------------------------------------------------------------------

class _PerturbationScheduleSchema(BaseModel):
    perturbation_types: list[str]
    severity_levels: list[float] = Field(
        default=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    )
    compose_perturbations: bool = False

    @field_validator("severity_levels")
    @classmethod
    def _validate_severities(cls, v: list[float]) -> list[float]:
        for s in v:
            if not 0.0 <= s <= 1.0:
                raise ValueError(f"Severity level {s} is not in [0.0, 1.0]")
        return sorted(set(v))

    @field_validator("perturbation_types")
    @classmethod
    def _validate_perturbation_types(cls, v: list[str]) -> list[str]:
        valid = {pt.value for pt in PerturbationType}
        bad = [x for x in v if x not in valid]
        if bad:
            raise ValueError(
                f"Unknown perturbation_types: {bad}. Valid values: {sorted(valid)}"
            )
        return v


class _DeploymentWeightsSchema(BaseModel):
    accuracy: float = 0.25
    robustness: float = 0.20
    calibration: float = 0.10
    fairness: float = 0.10
    latency: float = 0.15
    memory: float = 0.10
    computational_cost: float = 0.10

    @model_validator(mode="after")
    def _weights_sum_to_one(self) -> "_DeploymentWeightsSchema":
        total = (
            self.accuracy + self.robustness + self.calibration
            + self.fairness + self.latency + self.memory
            + self.computational_cost
        )
        if abs(total - 1.0) > 1e-4:
            raise ValueError(
                f"DeploymentWeights must sum to 1.0, got {total:.6f}"
            )
        return self


class _ExperimentConfigSchema(BaseModel):
    # Run identity
    experiment_name: str
    description: str = ""

    # Reproducibility
    random_seed: int = 42
    deterministic_mode: bool = True

    # Data
    dataset_names: list[str] = Field(default_factory=list)
    data_root: str = "data/"
    split_strategy: str = SplitStrategy.SUBJECT_DISJOINT.value
    test_fraction: float = Field(default=0.3, ge=0.0, lt=1.0)
    val_fraction: float = Field(default=0.1, ge=0.0, lt=1.0)
    k_folds: int | None = Field(default=None, ge=2)

    # Models
    traditional_models: list[str] = Field(default_factory=list)
    deep_models: list[str] = Field(default_factory=list)

    # Perturbations
    perturbation_schedule: _PerturbationScheduleSchema | None = None

    # Evaluation flags
    compute_calibration: bool = True
    compute_fairness: bool = True
    compute_fingerprints: bool = True
    compute_failure_analysis: bool = True
    compute_deployment_scores: bool = True
    deployment_scenarios: list[str] = Field(default_factory=list)
    custom_deployment_weights: dict[str, _DeploymentWeightsSchema] = Field(
        default_factory=dict
    )

    # Resource profiling
    profile_resources: bool = True
    profile_batch_size: int = Field(default=32, ge=1)

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

    # MLflow tracking
    mlflow_tracking_uri: str = "sqlite:///experiments/mlflow.db"
    mlflow_experiment_name: str | None = None

    # Hardware
    device: str = "auto"

    @field_validator("split_strategy")
    @classmethod
    def _validate_split_strategy(cls, v: str) -> str:
        valid = {s.value for s in SplitStrategy}
        if v not in valid:
            raise ValueError(f"split_strategy '{v}' not in {sorted(valid)}")
        return v

    @field_validator("traditional_models")
    @classmethod
    def _validate_traditional_models(cls, v: list[str]) -> list[str]:
        valid = {m.value for m in TraditionalModelType}
        bad = [x for x in v if x not in valid]
        if bad:
            raise ValueError(
                f"Unknown traditional_models: {bad}. Valid: {sorted(valid)}"
            )
        return v

    @field_validator("deep_models")
    @classmethod
    def _validate_deep_models(cls, v: list[str]) -> list[str]:
        valid = {m.value for m in DeepModelType}
        bad = [x for x in v if x not in valid]
        if bad:
            raise ValueError(
                f"Unknown deep_models: {bad}. Valid: {sorted(valid)}"
            )
        return v

    @field_validator("deployment_scenarios")
    @classmethod
    def _validate_scenarios(cls, v: list[str]) -> list[str]:
        valid = {s.value for s in DeploymentScenario}
        bad = [x for x in v if x not in valid]
        if bad:
            raise ValueError(
                f"Unknown deployment_scenarios: {bad}. Valid: {sorted(valid)}"
            )
        return v

    @field_validator("device")
    @classmethod
    def _validate_device(cls, v: str) -> str:
        if v not in {"auto", "cpu", "cuda"} and not v.startswith("cuda:"):
            raise ValueError(
                f"device must be 'auto', 'cpu', 'cuda', or 'cuda:N', got '{v}'"
            )
        return v

    @model_validator(mode="after")
    def _fractions_fit(self) -> "_ExperimentConfigSchema":
        if self.test_fraction + self.val_fraction >= 1.0:
            raise ValueError(
                f"test_fraction ({self.test_fraction}) + val_fraction "
                f"({self.val_fraction}) must be < 1.0"
            )
        return self

    @model_validator(mode="after")
    def _has_at_least_one_model(self) -> "_ExperimentConfigSchema":
        if not self.traditional_models and not self.deep_models:
            raise ValueError(
                "At least one model must be specified in traditional_models "
                "or deep_models"
            )
        return self


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

class ConfigLoader:
    """
    Loads, validates, and hashes FaceEval-X experiment configurations.

    Usage::

        loader = ConfigLoader()
        config = loader.load("configs/lfw_full_benchmark.yaml")
        # config is a fully validated ExperimentConfig dataclass instance
    """

    def load(self, path: str | Path) -> ExperimentConfig:
        """
        Parse a YAML config file and return a validated ``ExperimentConfig``.

        Parameters
        ----------
        path:
            Path to the YAML configuration file.

        Returns
        -------
        ExperimentConfig
            Validated, hashed, and ready-to-use config object.

        Raises
        ------
        FileNotFoundError
            If the YAML file does not exist.
        pydantic.ValidationError
            If any field fails validation.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")

        with path.open("r", encoding="utf-8") as fh:
            raw: dict[str, Any] = yaml.safe_load(fh) or {}

        return self._build(raw)

    def loads(self, yaml_text: str) -> ExperimentConfig:
        """Parse a YAML string directly (useful for tests and programmatic use)."""
        raw: dict[str, Any] = yaml.safe_load(yaml_text) or {}
        return self._build(raw)

    def from_dict(self, data: dict[str, Any]) -> ExperimentConfig:
        """Build an ``ExperimentConfig`` from a plain dictionary."""
        return self._build(data)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build(self, raw: dict[str, Any]) -> ExperimentConfig:
        schema = _ExperimentConfigSchema(**raw)
        config = self._schema_to_dataclass(schema)
        config.config_hash = self._hash(config)
        config.git_commit = self._git_commit()
        return config

    def _schema_to_dataclass(
        self, schema: _ExperimentConfigSchema
    ) -> ExperimentConfig:
        """Convert the validated Pydantic model to the plain dataclass."""

        perturbation_schedule: PerturbationSchedule | None = None
        if schema.perturbation_schedule is not None:
            ps = schema.perturbation_schedule
            perturbation_schedule = PerturbationSchedule(
                perturbation_types=[
                    PerturbationType(t) for t in ps.perturbation_types
                ],
                severity_levels=ps.severity_levels,
                compose_perturbations=ps.compose_perturbations,
            )

        custom_weights: dict[str, DeploymentWeights] = {}
        for scenario_key, ws in schema.custom_deployment_weights.items():
            custom_weights[scenario_key] = DeploymentWeights(
                accuracy=ws.accuracy,
                robustness=ws.robustness,
                calibration=ws.calibration,
                fairness=ws.fairness,
                latency=ws.latency,
                memory=ws.memory,
                computational_cost=ws.computational_cost,
            )

        return ExperimentConfig(
            experiment_name=schema.experiment_name,
            description=schema.description,
            random_seed=schema.random_seed,
            deterministic_mode=schema.deterministic_mode,
            dataset_names=schema.dataset_names,
            data_root=schema.data_root,
            split_strategy=SplitStrategy(schema.split_strategy),
            test_fraction=schema.test_fraction,
            val_fraction=schema.val_fraction,
            k_folds=schema.k_folds,
            traditional_models=[
                TraditionalModelType(m) for m in schema.traditional_models
            ],
            deep_models=[DeepModelType(m) for m in schema.deep_models],
            perturbation_schedule=perturbation_schedule,
            compute_calibration=schema.compute_calibration,
            compute_fairness=schema.compute_fairness,
            compute_fingerprints=schema.compute_fingerprints,
            compute_failure_analysis=schema.compute_failure_analysis,
            compute_deployment_scores=schema.compute_deployment_scores,
            deployment_scenarios=[
                DeploymentScenario(s) for s in schema.deployment_scenarios
            ],
            custom_deployment_weights=custom_weights,
            profile_resources=schema.profile_resources,
            profile_batch_size=schema.profile_batch_size,
            use_feature_cache=schema.use_feature_cache,
            feature_cache_dir=schema.feature_cache_dir,
            output_dir=schema.output_dir,
            report_dir=schema.report_dir,
            export_svg=schema.export_svg,
            export_png=schema.export_png,
            export_csv=schema.export_csv,
            export_json=schema.export_json,
            generate_pdf_report=schema.generate_pdf_report,
            mlflow_tracking_uri=schema.mlflow_tracking_uri,
            mlflow_experiment_name=(
                schema.mlflow_experiment_name or schema.experiment_name
            ),
            device=schema.device,
        )

    @staticmethod
    def _hash(config: ExperimentConfig) -> str:
        """
        Return a SHA256 hex digest of the config's serialised form.

        Fields that vary per run (run_id, created_at, config_hash itself,
        git_commit) are excluded so identical experimental setups produce
        identical hashes even across separate invocations.
        """
        d = config.to_dict()
        for key in ("run_id", "created_at", "config_hash", "git_commit"):
            d.pop(key, None)
        canonical = json.dumps(d, sort_keys=True, default=str)
        return hashlib.sha256(canonical.encode()).hexdigest()

    @staticmethod
    def _git_commit() -> str:
        """Return the current HEAD commit hash, or empty string if unavailable."""
        try:
            result = subprocess.run(
                ["git", "rev-parse", "--short", "HEAD"],
                capture_output=True,
                text=True,
                timeout=3,
            )
            return result.stdout.strip() if result.returncode == 0 else ""
        except Exception:
            return ""


def load_config(path: str | Path) -> ExperimentConfig:
    """
    Module-level convenience function.

    Equivalent to ``ConfigLoader().load(path)``.
    """
    return ConfigLoader().load(path)


def make_default_config(experiment_name: str, **overrides: Any) -> ExperimentConfig:
    """
    Construct a minimal valid config programmatically (useful for tests).

    Defaults to Eigenfaces + FaceNet on LFW with all perturbations at
    five severity levels.  Pass keyword arguments to override any field.
    """
    defaults: dict[str, Any] = {
        "experiment_name": experiment_name,
        "dataset_names": ["lfw"],
        "traditional_models": [TraditionalModelType.EIGENFACES.value],
        "deep_models": [DeepModelType.FACENET.value],
        "perturbation_schedule": {
            "perturbation_types": [
                PerturbationType.GAUSSIAN_BLUR.value,
                PerturbationType.GAUSSIAN_NOISE.value,
                PerturbationType.BRIGHTNESS.value,
            ],
            "severity_levels": [0.0, 0.25, 0.5, 0.75, 1.0],
        },
    }
    defaults.update(overrides)
    return ConfigLoader().from_dict(defaults)

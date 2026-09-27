from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_MLFLOW_AVAILABLE: bool | None = None


def _mlflow() -> Any | None:
    global _MLFLOW_AVAILABLE
    if _MLFLOW_AVAILABLE is None:
        try:
            import mlflow
            _MLFLOW_AVAILABLE = True
        except ImportError:
            _MLFLOW_AVAILABLE = False
            logger.warning("mlflow not installed — tracking disabled.")
    return __import__("mlflow") if _MLFLOW_AVAILABLE else None


class ExperimentTracker:
    """
    MLflow-backed experiment tracker.

    Parameters
    ----------
    tracking_uri:
        MLflow tracking server URI (SQLite file or remote server).
    experiment_name:
        MLflow experiment name (group of related runs).
    """

    def __init__(
        self,
        tracking_uri: str = "sqlite:///experiments/mlflow.db",
        experiment_name: str = "faceeval_x",
    ) -> None:
        self._tracking_uri = tracking_uri
        self._experiment_name = experiment_name
        self._run_id: str | None = None
        self._active = False
        self._mlflow = _mlflow()

    # ------------------------------------------------------------------
    # Run lifecycle
    # ------------------------------------------------------------------

    def start_run(
        self,
        run_name: str,
        tags: dict[str, str] | None = None,
    ) -> str | None:
        """Start an MLflow run. Returns the MLflow run_id or None."""
        if self._mlflow is None:
            return None
        try:
            self._mlflow.set_tracking_uri(self._tracking_uri)
            self._mlflow.set_experiment(self._experiment_name)
            run = self._mlflow.start_run(run_name=run_name, tags=tags or {})
            self._run_id = run.info.run_id
            self._active = True
            logger.info("MLflow run started: %s (%s)", run_name, self._run_id)
            return self._run_id
        except Exception as exc:
            logger.warning("MLflow start_run failed: %s", exc)
            return None

    def end_run(self, status: str = "FINISHED") -> None:
        if self._mlflow and self._active:
            try:
                self._mlflow.end_run(status=status)
                self._active = False
                logger.info("MLflow run ended (status=%s).", status)
            except Exception as exc:
                logger.warning("MLflow end_run failed: %s", exc)

    # ------------------------------------------------------------------
    # Logging methods
    # ------------------------------------------------------------------

    def log_params(self, params: dict[str, Any]) -> None:
        """Log a flat dict of parameters."""
        if not self._mlflow or not self._active:
            return
        flat = _flatten_dict(params)
        try:
            # MLflow limits: 500 params per batch, key ≤ 250 chars, value ≤ 6000 chars
            for key, val in flat.items():
                safe_key = str(key)[:250]
                safe_val = str(val)[:6000]
                self._mlflow.log_param(safe_key, safe_val)
        except Exception as exc:
            logger.debug("MLflow log_params partial failure: %s", exc)

    def log_metric(
        self,
        key: str,
        value: float,
        step: int | None = None,
    ) -> None:
        if not self._mlflow or not self._active:
            return
        try:
            self._mlflow.log_metric(key, float(value), step=step)
        except Exception as exc:
            logger.debug("MLflow log_metric failed for '%s': %s", key, exc)

    def log_metrics(
        self,
        metrics: dict[str, float],
        step: int | None = None,
    ) -> None:
        if not self._mlflow or not self._active:
            return
        try:
            self._mlflow.log_metrics(
                {k: float(v) for k, v in metrics.items()},
                step=step,
            )
        except Exception as exc:
            logger.debug("MLflow log_metrics failed: %s", exc)

    def log_artifact(self, local_path: str, artifact_path: str = "") -> None:
        if not self._mlflow or not self._active:
            return
        if not Path(local_path).exists():
            return
        try:
            if artifact_path:
                self._mlflow.log_artifact(local_path, artifact_path)
            else:
                self._mlflow.log_artifact(local_path)
        except Exception as exc:
            logger.debug("MLflow log_artifact failed for '%s': %s", local_path, exc)

    def log_dict(self, data: dict[str, Any], artifact_filename: str) -> None:
        """Log a dict as a JSON artifact."""
        if not self._mlflow or not self._active:
            return
        try:
            self._mlflow.log_dict(data, artifact_filename)
        except Exception as exc:
            logger.debug("MLflow log_dict failed: %s", exc)

    def set_tag(self, key: str, value: str) -> None:
        if not self._mlflow or not self._active:
            return
        try:
            self._mlflow.set_tag(key, str(value)[:5000])
        except Exception as exc:
            logger.debug("MLflow set_tag failed: %s", exc)

    def log_evaluation_result(
        self,
        result: Any,  # EvaluationResult
        step: int | None = None,
    ) -> None:
        """Log all scalar metrics from one EvaluationResult."""
        prefix = f"{result.model_name}"
        if result.perturbation_spec:
            prefix += f".{result.perturbation_spec.label}"

        metrics: dict[str, float] = {
            f"{prefix}.accuracy":   result.accuracy,
            f"{prefix}.auc":        result.auc,
            f"{prefix}.eer":        result.eer,
            f"{prefix}.f1":         result.f1_score,
        }
        if result.calibration:
            metrics[f"{prefix}.ece"] = result.calibration.ece
        self.log_metrics(metrics, step=step)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def run_id(self) -> str | None:
        return self._run_id

    @property
    def is_active(self) -> bool:
        return self._active


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _flatten_dict(
    d: dict[str, Any],
    parent_key: str = "",
    sep: str = ".",
) -> dict[str, Any]:
    """Recursively flatten a nested dict."""
    items: list[tuple[str, Any]] = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(_flatten_dict(v, new_key, sep=sep).items())
        elif isinstance(v, (list, tuple)):
            items.append((new_key, json.dumps(v, default=str)[:5000]))
        else:
            items.append((new_key, v))
    return dict(items)

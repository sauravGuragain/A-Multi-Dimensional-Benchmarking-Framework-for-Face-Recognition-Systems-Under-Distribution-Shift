from __future__ import annotations

import gc
import logging
import os
import time
import tracemalloc
from pathlib import Path
from typing import Any

import numpy as np

from faceeval.core.types import ModelName, ResourceUsage

logger = logging.getLogger(__name__)

_PSUTIL_AVAILABLE: bool | None = None
_NVML_AVAILABLE:   bool | None = None


def _psutil() -> Any | None:
    global _PSUTIL_AVAILABLE
    if _PSUTIL_AVAILABLE is None:
        try:
            import psutil
            _PSUTIL_AVAILABLE = True
        except ImportError:
            _PSUTIL_AVAILABLE = False
            logger.debug("psutil not installed — CPU utilisation unavailable.")
    return __import__("psutil") if _PSUTIL_AVAILABLE else None


def _nvml() -> Any | None:
    global _NVML_AVAILABLE
    if _NVML_AVAILABLE is None:
        try:
            import pynvml
            pynvml.nvmlInit()
            _NVML_AVAILABLE = True
        except Exception:
            _NVML_AVAILABLE = False
    return __import__("pynvml") if _NVML_AVAILABLE else None


# ---------------------------------------------------------------------------
# Profiler
# ---------------------------------------------------------------------------

class ModelProfiler:
    """
    Wraps a model inference call to measure latency, memory, and throughput.

    Parameters
    ----------
    model_name:
        Name of the model being profiled.
    device:
        Device string (``"cpu"`` or ``"cuda:N"``).
    warmup_runs:
        Number of warm-up passes before timing begins (to avoid cold-start
        JIT / weight-loading latency being counted).
    n_repeats:
        Number of timed repetitions for averaging.
    model_path:
        If provided, model size on disk is measured.
    """

    def __init__(
        self,
        model_name: ModelName,
        device: str = "cpu",
        warmup_runs: int = 2,
        n_repeats: int = 5,
        model_path: str | None = None,
    ) -> None:
        self._model_name = model_name
        self._device = device
        self._warmup = warmup_runs
        self._n_repeats = n_repeats
        self._model_path = model_path

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def profile_inference(
        self,
        predict_fn: Any,  # callable: (X: list[np.ndarray]) → list
        X_batch: list[np.ndarray],
    ) -> ResourceUsage:
        """
        Profile ``predict_fn`` on ``X_batch`` across ``n_repeats`` runs.

        Parameters
        ----------
        predict_fn:
            Callable matching ``BaseRecognizer.predict`` signature.
        X_batch:
            Batch of preprocessed face arrays.

        Returns
        -------
        ResourceUsage
            Averaged timing and resource measurements.
        """
        batch_size = len(X_batch)

        # Warm up
        for _ in range(self._warmup):
            predict_fn(X_batch)

        gc.collect()

        # Measure CPU utilisation baseline
        cpu_before = self._cpu_percent()
        gpu_mem_before, _ = self._gpu_stats()

        # Timed runs with memory tracking
        tracemalloc.start()
        latencies: list[float] = []

        for _ in range(self._n_repeats):
            t0 = time.perf_counter()
            predict_fn(X_batch)
            latencies.append((time.perf_counter() - t0) * 1000)

        _, peak_bytes = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        cpu_after = self._cpu_percent()
        gpu_mem_after, gpu_util = self._gpu_stats()

        avg_ms = float(np.mean(latencies))
        per_image_ms = avg_ms / batch_size
        throughput = 1000.0 / per_image_ms if per_image_ms > 0 else 0.0
        peak_mb = peak_bytes / (1024 ** 2)
        model_size_mb = self._model_size_mb()

        cpu_util = None
        if cpu_before is not None and cpu_after is not None:
            cpu_util = (cpu_before + cpu_after) / 2.0

        gpu_mb = None
        if gpu_mem_before is not None and gpu_mem_after is not None:
            gpu_mb = max(gpu_mem_after - gpu_mem_before, 0.0)

        return ResourceUsage(
            model_name=self._model_name,
            batch_size=batch_size,
            inference_time_ms=per_image_ms,
            embedding_time_ms=per_image_ms,   # approximation; override if embed is separate
            peak_memory_mb=peak_mb,
            model_size_mb=model_size_mb or 0.0,
            cpu_utilization_pct=cpu_util,
            gpu_utilization_pct=gpu_util,
            gpu_memory_mb=gpu_mb,
            throughput_fps=throughput,
            device=self._device,
        )

    def profile_embedding(
        self,
        extract_fn: Any,  # callable: (X: list[np.ndarray]) → list
        X_batch: list[np.ndarray],
    ) -> dict[str, float]:
        """
        Profile embedding extraction time separately from classification.

        Returns dict with ``embedding_time_ms`` and ``throughput_fps``.
        """
        batch_size = len(X_batch)

        for _ in range(self._warmup):
            extract_fn(X_batch)

        latencies: list[float] = []
        for _ in range(self._n_repeats):
            t0 = time.perf_counter()
            extract_fn(X_batch)
            latencies.append((time.perf_counter() - t0) * 1000)

        avg_ms = float(np.mean(latencies))
        per_image_ms = avg_ms / batch_size

        return {
            "embedding_time_ms": per_image_ms,
            "throughput_fps": 1000.0 / per_image_ms if per_image_ms > 0 else 0.0,
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _cpu_percent(self) -> float | None:
        ps = _psutil()
        if ps is None:
            return None
        try:
            return float(ps.cpu_percent(interval=0.1))
        except Exception:
            return None

    def _gpu_stats(self) -> tuple[float | None, float | None]:
        """Returns (gpu_memory_mb, gpu_utilization_pct) or (None, None)."""
        pynvml = _nvml()
        if pynvml is None:
            return None, None
        try:
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            util = pynvml.nvmlDeviceGetUtilizationRates(handle)
            mem_mb = mem_info.used / (1024 ** 2)
            gpu_util = float(util.gpu)
            return mem_mb, gpu_util
        except Exception:
            return None, None

    def _model_size_mb(self) -> float | None:
        if self._model_path is None:
            return None
        path = Path(self._model_path)
        if not path.exists():
            return None
        return path.stat().st_size / (1024 ** 2)


# ---------------------------------------------------------------------------
# Lightweight timing decorator for per-call measurement
# ---------------------------------------------------------------------------

class TimingContext:
    """
    Context manager for measuring wall-clock time of a code block.

    Usage::

        with TimingContext() as tc:
            result = model.predict(batch)
        print(f"Took {tc.elapsed_ms:.1f} ms")
    """

    def __init__(self) -> None:
        self.elapsed_ms: float = 0.0
        self._start: float = 0.0

    def __enter__(self) -> "TimingContext":
        self._start = time.perf_counter()
        return self

    def __exit__(self, *_: Any) -> None:
        self.elapsed_ms = (time.perf_counter() - self._start) * 1000.0


# ---------------------------------------------------------------------------
# Batch resource summary
# ---------------------------------------------------------------------------

def summarise_resource_usages(usages: list[ResourceUsage]) -> dict[str, Any]:
    """
    Aggregate ``ResourceUsage`` records across multiple evaluation conditions.

    Returns mean, std, min, max for each numeric field.
    """
    if not usages:
        return {}

    fields = [
        "inference_time_ms", "embedding_time_ms", "peak_memory_mb",
        "model_size_mb", "throughput_fps",
    ]
    summary: dict[str, Any] = {"model_name": usages[0].model_name}
    for field in fields:
        vals = [getattr(u, field) for u in usages if getattr(u, field) is not None]
        if vals:
            summary[field] = {
                "mean": float(np.mean(vals)),
                "std":  float(np.std(vals)),
                "min":  float(np.min(vals)),
                "max":  float(np.max(vals)),
            }

    # Optional fields
    for field in ["cpu_utilization_pct", "gpu_utilization_pct", "gpu_memory_mb"]:
        vals = [getattr(u, field) for u in usages if getattr(u, field) is not None]
        if vals:
            summary[field] = {"mean": float(np.mean(vals))}

    return summary

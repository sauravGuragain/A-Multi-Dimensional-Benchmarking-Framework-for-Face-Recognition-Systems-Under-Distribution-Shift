from __future__ import annotations

import hashlib
import logging
import os
import platform
import random
import subprocess
import sys
from datetime import datetime
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Seed control
# ---------------------------------------------------------------------------

def set_global_seeds(seed: int, deterministic: bool = True) -> None:
    """
    Set random seeds for all relevant libraries.

    Parameters
    ----------
    seed:
        Integer seed value from ``ExperimentConfig.random_seed``.
    deterministic:
        If True, enables PyTorch deterministic algorithms and disables
        benchmark mode.  May reduce GPU throughput but guarantees
        bit-for-bit identical results across runs.
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            try:
                torch.use_deterministic_algorithms(True)
            except AttributeError:
                pass  # older PyTorch versions
        logger.debug("PyTorch seeds set (seed=%d, deterministic=%s).", seed, deterministic)
    except ImportError:
        logger.debug("PyTorch not available — skipping torch seed setting.")

    logger.info("Global seeds set: seed=%d, deterministic=%s.", seed, deterministic)


# ---------------------------------------------------------------------------
# Hardware / software snapshot
# ---------------------------------------------------------------------------

def capture_environment_snapshot() -> dict[str, Any]:
    """
    Capture a full environment snapshot for provenance.

    Returns a dict that is logged to MLflow and embedded in every
    ExperimentRun record.
    """
    snapshot: dict[str, Any] = {
        "timestamp": datetime.utcnow().isoformat(),
        "python_version": sys.version,
        "platform": platform.platform(),
        "cpu": _cpu_info(),
        "ram_gb": _ram_gb(),
        "gpu": _gpu_info(),
        "packages": _key_package_versions(),
        "git_commit": _git_commit(),
        "git_branch": _git_branch(),
        "git_dirty": _git_is_dirty(),
    }
    return snapshot


def _cpu_info() -> str:
    try:
        import psutil
        freq = psutil.cpu_freq()
        return (
            f"{platform.processor()} "
            f"({psutil.cpu_count(logical=False)} physical cores, "
            f"{psutil.cpu_count(logical=True)} logical, "
            f"{freq.max:.0f} MHz max)"
        )
    except Exception:
        return platform.processor() or "unknown"


def _ram_gb() -> float:
    try:
        import psutil
        return round(psutil.virtual_memory().total / (1024 ** 3), 2)
    except Exception:
        return -1.0


def _gpu_info() -> list[dict[str, Any]]:
    gpus: list[dict[str, Any]] = []
    try:
        import torch
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                gpus.append({
                    "index": i,
                    "name": props.name,
                    "memory_gb": round(props.total_memory / (1024 ** 3), 2),
                    "compute_capability": f"{props.major}.{props.minor}",
                })
    except ImportError:
        pass
    return gpus


def _key_package_versions() -> dict[str, str]:
    packages = [
        "numpy", "sklearn", "torch", "torchvision", "cv2",
        "matplotlib", "scipy", "mlflow", "pydantic",
    ]
    versions: dict[str, str] = {}
    for pkg in packages:
        try:
            mod = __import__(pkg if pkg != "sklearn" else "sklearn")
            versions[pkg] = getattr(mod, "__version__", "unknown")
        except ImportError:
            versions[pkg] = "not installed"
    return versions


def _git_commit() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=3,
        )
        return result.stdout.strip() if result.returncode == 0 else ""
    except Exception:
        return ""


def _git_branch() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            capture_output=True, text=True, timeout=3,
        )
        return result.stdout.strip() if result.returncode == 0 else ""
    except Exception:
        return ""


def _git_is_dirty() -> bool:
    try:
        result = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True, text=True, timeout=3,
        )
        return bool(result.stdout.strip()) if result.returncode == 0 else False
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Config hash verification
# ---------------------------------------------------------------------------

def verify_config_hash(config_dict: dict[str, Any], expected_hash: str) -> bool:
    """
    Verify that a config dict produces the expected hash.
    Used to detect config drift between the stored hash and a re-loaded config.
    """
    import json
    for key in ("run_id", "created_at", "config_hash", "git_commit"):
        config_dict.pop(key, None)
    canonical = json.dumps(config_dict, sort_keys=True, default=str)
    actual = hashlib.sha256(canonical.encode()).hexdigest()
    return actual == expected_hash

"""
faceeval.data.cache
====================
Disk-backed feature cache for embeddings and traditional ML feature vectors.

Design notes
------------
* Cache key: ``(image_id, model_name, perturbation_label)`` → a stable
  directory-safe string used as a filename.
* Storage: one ``.npy`` file per entry under ``cache_dir/``.
  NumPy's native format is fast, compact, and handles arbitrary float32/64
  arrays without serialisation overhead.
* Thread safety: write uses atomic rename (write to ``.tmp`` → rename) so
  concurrent processes writing the same key don't corrupt each other.
* Eviction: not implemented (cache is append-only during one thesis
  experiment cycle).  Total cache size is bounded by the number of
  unique (image, model, perturbation) triples, which is tractable.
* The cache is optional: if disabled in ``ExperimentConfig.use_feature_cache``
  the ``FeatureCache`` methods are no-ops that return ``None``.
"""

from __future__ import annotations

import hashlib
import logging
import os
from pathlib import Path
from typing import Any

import numpy as np

from faceeval.core.exceptions import CacheError

logger = logging.getLogger(__name__)


class FeatureCache:
    """
    Persistent on-disk cache mapping ``(image_id, model_name, perturbation_label)``
    to NumPy arrays (embeddings or feature vectors).

    Parameters
    ----------
    cache_dir:
        Root directory for cache files.  Created on first write.
    enabled:
        When ``False``, all reads return ``None`` and writes are skipped.
    """

    def __init__(self, cache_dir: str = "experiments/cache/", enabled: bool = True) -> None:
        self._root = Path(cache_dir)
        self._enabled = enabled
        self._hits = 0
        self._misses = 0
        if enabled:
            self._root.mkdir(parents=True, exist_ok=True)
            logger.debug("FeatureCache initialised at '%s'", self._root)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get(
        self,
        image_id: str,
        model_name: str,
        perturbation_label: str = "clean",
    ) -> np.ndarray | None:
        """
        Retrieve a cached feature array, or ``None`` on cache miss.

        Parameters
        ----------
        image_id:
            Unique image identifier from ``ImageRecord.image_id``.
        model_name:
            Registered model name (e.g. ``"facenet"``).
        perturbation_label:
            ``PerturbationSpec.label`` string, or ``"clean"`` for unperturbed.

        Returns
        -------
        np.ndarray or None
        """
        if not self._enabled:
            return None
        path = self._key_path(image_id, model_name, perturbation_label)
        if path.exists():
            self._hits += 1
            try:
                return np.load(str(path))
            except Exception as exc:
                logger.warning("Cache read failed for '%s': %s", path, exc)
                return None
        self._misses += 1
        return None

    def put(
        self,
        image_id: str,
        model_name: str,
        perturbation_label: str,
        array: np.ndarray,
    ) -> None:
        """
        Store a feature array in the cache.

        Uses atomic write-then-rename to avoid partial writes.
        """
        if not self._enabled:
            return
        path = self._key_path(image_id, model_name, perturbation_label)
        # np.save appends ".npy" automatically, so use "stem.tmp.npy" as tmp.
        tmp_path = path.parent / (path.stem + ".tmp.npy")
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            np.save(str(tmp_path), array)
            os.replace(str(tmp_path), str(path))
        except Exception as exc:
            tmp_path.unlink(missing_ok=True)
            raise CacheError(
                f"Failed to write cache entry for image_id='{image_id}': {exc}"
            ) from exc

    def get_or_compute(
        self,
        image_id: str,
        model_name: str,
        perturbation_label: str,
        compute_fn: Any,  # callable[[], np.ndarray]
    ) -> np.ndarray:
        """
        Return cached array if present, otherwise call ``compute_fn()``,
        store the result, and return it.

        This is the preferred API for hot paths — callers never need to
        manually check the cache.
        """
        cached = self.get(image_id, model_name, perturbation_label)
        if cached is not None:
            return cached
        result: np.ndarray = compute_fn()
        self.put(image_id, model_name, perturbation_label, result)
        return result

    def invalidate(
        self,
        image_id: str,
        model_name: str,
        perturbation_label: str,
    ) -> bool:
        """Delete one cache entry.  Returns True if the entry existed."""
        if not self._enabled:
            return False
        path = self._key_path(image_id, model_name, perturbation_label)
        if path.exists():
            path.unlink()
            return True
        return False

    def clear_model(self, model_name: str) -> int:
        """Delete all cache entries for one model.  Returns count deleted."""
        if not self._enabled:
            return 0
        model_dir = self._root / self._safe(model_name)
        if not model_dir.exists():
            return 0
        count = 0
        for f in model_dir.rglob("*.npy"):
            f.unlink()
            count += 1
        return count

    def clear_all(self) -> int:
        """Delete every cache entry.  Returns count deleted."""
        if not self._enabled:
            return 0
        count = 0
        for f in self._root.rglob("*.npy"):
            f.unlink()
            count += 1
        return count

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------

    @property
    def hit_count(self) -> int:
        return self._hits

    @property
    def miss_count(self) -> int:
        return self._misses

    @property
    def hit_rate(self) -> float:
        total = self._hits + self._misses
        return self._hits / total if total > 0 else 0.0

    def stats(self) -> dict[str, Any]:
        return {
            "enabled": self._enabled,
            "cache_dir": str(self._root),
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": round(self.hit_rate, 4),
        }

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _key_path(
        self, image_id: str, model_name: str, perturbation_label: str
    ) -> Path:
        """
        Map a (image_id, model_name, perturbation_label) triple to a file path.

        Structure: ``<cache_dir>/<model_name>/<perturbation_label>/<hash>.npy``

        The image_id hash avoids filesystem issues with long or path-like IDs.
        """
        id_hash = hashlib.md5(image_id.encode()).hexdigest()
        return (
            self._root
            / self._safe(model_name)
            / self._safe(perturbation_label)
            / f"{id_hash}.npy"
        )

    @staticmethod
    def _safe(s: str) -> str:
        """Replace characters that are unsafe in directory names."""
        return s.replace("/", "_").replace("\\", "_").replace(":", "_")

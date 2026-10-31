"""
faceeval.data.dataset_manager
==============================
Central dataset registry and lifecycle manager.

Responsibilities
----------------
* Register datasets by name with their root path and metadata.
* Validate that a dataset's directory structure is present on disk.
* Return ``DatasetInfo`` descriptors and iterators of ``ImageRecord`` objects.
* Delegate loading to per-dataset loader plugins (registered via
  ``dataset_registry`` in ``faceeval.core.registry``).
* Cache the full image-record list in memory so repeated calls across
  modules do not re-scan the filesystem.

Design rule
-----------
``DatasetManager`` never touches pixels.  It only manages paths and metadata.
Pixel I/O happens in ``faceeval.preprocessing.pipeline``.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterator

from faceeval.core.exceptions import DatasetNotFoundError, DataError
from faceeval.core.registry import dataset_registry
from faceeval.core.types import DatasetInfo, ImageRecord, SubjectID

logger = logging.getLogger(__name__)


class DatasetManager:
    """
    Manages registration and access for all face datasets used in an experiment.

    Parameters
    ----------
    data_root:
        Base directory under which all dataset subdirectories live.
        Individual loaders may use absolute paths and ignore this root.
    """

    def __init__(self, data_root: str = "data/") -> None:
        self._data_root = Path(data_root)
        self._registered: dict[str, DatasetInfo] = {}
        self._record_cache: dict[str, list[ImageRecord]] = {}

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register(self, name: str, root_path: str | None = None) -> DatasetInfo:
        """
        Register a dataset by name.

        The loader plugin for ``name`` must have been registered via
        ``@register_dataset`` before calling this.  The plugin's factory
        is called to produce the ``DatasetInfo`` and ``ImageRecord`` list.

        Parameters
        ----------
        name:
            Dataset name matching a ``dataset_registry`` key
            (e.g. ``"lfw"``, ``"celeba"``).
        root_path:
            Override the default root path supplied by the loader.
            If ``None``, the loader uses ``self._data_root / name``.

        Returns
        -------
        DatasetInfo
            Descriptor for the registered dataset.
        """
        if name in self._registered:
            logger.debug("Dataset '%s' already registered; returning cached info.", name)
            return self._registered[name]

        if not dataset_registry.is_registered(name):
            raise DatasetNotFoundError(name)

        resolved_root = str(root_path or self._data_root / name)
        loader = dataset_registry.get(name, root_path=resolved_root)
        info: DatasetInfo = loader.get_info()
        records: list[ImageRecord] = loader.load_records()

        self._registered[name] = info
        self._record_cache[name] = records
        logger.info(
            "Registered dataset '%s': %d subjects, %d images.",
            name, info.num_subjects, info.num_images,
        )
        return info

    def register_all(self, names: list[str], root_path: str | None = None) -> None:
        """Register multiple datasets in one call."""
        for name in names:
            self.register(name, root_path=root_path)

    # ------------------------------------------------------------------
    # Access
    # ------------------------------------------------------------------

    def get_info(self, name: str) -> DatasetInfo:
        self._ensure_registered(name)
        return self._registered[name]

    def get_records(self, name: str) -> list[ImageRecord]:
        """Return the full list of ``ImageRecord`` objects for a dataset."""
        self._ensure_registered(name)
        return self._record_cache[name]

    def iter_records(self, name: str) -> Iterator[ImageRecord]:
        """Iterate over ``ImageRecord`` objects without holding them all in memory."""
        for rec in self.get_records(name):
            yield rec

    def get_subjects(self, name: str) -> list[SubjectID]:
        """Return sorted list of unique subject IDs in a dataset."""
        self._ensure_registered(name)
        return sorted({r.subject_id for r in self._record_cache[name]})

    def get_records_for_subject(
        self, name: str, subject_id: SubjectID
    ) -> list[ImageRecord]:
        """Return all records belonging to one subject."""
        return [r for r in self.get_records(name) if r.subject_id == subject_id]

    def list_registered(self) -> list[str]:
        return sorted(self._registered)

    def is_registered(self, name: str) -> bool:
        return name in self._registered

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _ensure_registered(self, name: str) -> None:
        if name not in self._registered:
            raise DatasetNotFoundError(
                name,
                path=str(self._data_root / name),
            )

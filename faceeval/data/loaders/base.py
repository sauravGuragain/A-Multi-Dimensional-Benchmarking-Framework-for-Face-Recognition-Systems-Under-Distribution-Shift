"""
faceeval.data.loaders.base
===========================
Abstract base class for all dataset loader plugins.

Every concrete loader (LFW, CelebA, VGGFace2, …) must subclass
``BaseDatasetLoader`` and implement ``get_info()`` and ``load_records()``.
The loader is then registered via the ``@register_dataset`` decorator so
``DatasetManager`` can instantiate it by name.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from faceeval.core.types import DatasetInfo, ImageRecord


class BaseDatasetLoader(ABC):
    """
    Abstract loader for one face recognition dataset.

    Subclasses receive the dataset root path at construction time and must
    implement two methods:

    ``get_info()``
        Return a ``DatasetInfo`` descriptor (no filesystem I/O beyond stat
        calls).

    ``load_records()``
        Walk the dataset directory tree and return a list of ``ImageRecord``
        objects.  This is the only place in the framework where the dataset
        directory structure is assumed — all other layers consume
        ``ImageRecord`` objects.

    Parameters
    ----------
    root_path:
        Absolute or relative path to the dataset root directory.
    """

    SUPPORTED_EXTENSIONS: tuple[str, ...] = (".jpg", ".jpeg", ".png", ".bmp", ".tiff")

    def __init__(self, root_path: str) -> None:
        self.root = Path(root_path)

    @abstractmethod
    def get_info(self) -> DatasetInfo:
        """Return the ``DatasetInfo`` descriptor for this dataset."""
        ...

    @abstractmethod
    def load_records(self) -> list[ImageRecord]:
        """
        Scan the dataset directory and return all ``ImageRecord`` objects.

        Must be deterministic: the same directory content must always
        produce the same list in the same order (sort by image path).
        """
        ...

    # ------------------------------------------------------------------
    # Shared helpers available to all concrete loaders
    # ------------------------------------------------------------------

    def _iter_images(self, root: Path | None = None) -> list[Path]:
        """
        Return sorted list of image paths under ``root`` (or ``self.root``).

        Only files with extensions in ``SUPPORTED_EXTENSIONS`` are returned.
        """
        base = root or self.root
        paths = [
            p for p in sorted(base.rglob("*"))
            if p.suffix.lower() in self.SUPPORTED_EXTENSIONS
        ]
        return paths

    def _validate_root(self) -> None:
        """Raise ``FileNotFoundError`` if the dataset root does not exist."""
        if not self.root.exists():
            raise FileNotFoundError(
                f"Dataset root not found: '{self.root}'. "
                "Download the dataset and point data_root to its parent directory."
            )

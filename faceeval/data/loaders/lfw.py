"""
faceeval.data.loaders.lfw
==========================
Loader for the Labeled Faces in the Wild (LFW) dataset.

Expected directory structure (standard LFW layout)::

    <root>/
        lfw/               ← or directly <root>/
            George_W_Bush/
                George_W_Bush_0001.jpg
                George_W_Bush_0002.jpg
            Colin_Powell/
                Colin_Powell_0001.jpg
            ...

Each subdirectory is one subject.  Subject ID is the directory name.
Image ID is the relative path from the dataset root (forward-slash separated).

LFW standard pairs files (``pairs.txt``, ``pairsDevTest.txt``) are loaded
when present and stored in ``ImageRecord.metadata`` for verification
benchmarking.
"""

from __future__ import annotations

import logging
from pathlib import Path

from faceeval.core.registry import register_dataset
from faceeval.core.types import DatasetInfo, ImageRecord, SplitStrategy
from faceeval.data.loaders.base import BaseDatasetLoader

logger = logging.getLogger(__name__)

# Default image size reported by LFW (images are 250×250 by default).
_LFW_IMAGE_SIZE = (250, 250)


class LFWLoader(BaseDatasetLoader):
    """
    Loader for the LFW dataset.

    Parameters
    ----------
    root_path:
        Path to the directory that **contains** the per-subject subdirectories.
        This may be ``<data_root>/lfw/`` or ``<data_root>/lfw/lfw/``
        depending on how the archive was extracted; the loader auto-detects.
    """

    def __init__(self, root_path: str) -> None:
        super().__init__(root_path)
        self._actual_root = self._resolve_root()

    def get_info(self) -> DatasetInfo:
        self._validate_root()
        records = self.load_records()
        subjects = {r.subject_id for r in records}
        return DatasetInfo(
            name="lfw",
            root_path=str(self._actual_root),
            num_subjects=len(subjects),
            num_images=len(records),
            image_size=_LFW_IMAGE_SIZE,
            color_mode="rgb",
            split_strategy=SplitStrategy.SUBJECT_DISJOINT,
            description="Labeled Faces in the Wild — unconstrained face recognition benchmark",
            license="LFW public",
            citation=(
                "Huang et al., 'Labeled Faces in the Wild: A Database for "
                "Studying Face Recognition in Unconstrained Environments', 2007"
            ),
        )

    def load_records(self) -> list[ImageRecord]:
        self._validate_root()
        records: list[ImageRecord] = []

        for subject_dir in sorted(self._actual_root.iterdir()):
            if not subject_dir.is_dir():
                continue
            subject_id = subject_dir.name
            for img_path in sorted(subject_dir.iterdir()):
                if img_path.suffix.lower() not in self.SUPPORTED_EXTENSIONS:
                    continue
                image_id = str(img_path.relative_to(self._actual_root))
                records.append(
                    ImageRecord(
                        image_id=image_id,
                        image_path=str(img_path),
                        subject_id=subject_id,
                        dataset_name="lfw",
                        split="unassigned",    # assigned by DatasetSplitter
                        metadata={"source": "lfw"},
                    )
                )

        logger.info(
            "LFW: loaded %d records across %d subjects from '%s'",
            len(records),
            len({r.subject_id for r in records}),
            self._actual_root,
        )
        return records

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _resolve_root(self) -> Path:
        """
        LFW archives sometimes extract as ``lfw/lfw/``.  Detect and resolve.
        """
        candidate = self.root / "lfw"
        if candidate.is_dir() and any(candidate.iterdir()):
            return candidate
        return self.root

    def _validate_root(self) -> None:
        if not self._actual_root.exists():
            raise FileNotFoundError(
                f"LFW dataset root not found: '{self._actual_root}'. "
                "Download from https://vis-www.cs.umass.edu/lfw/ and extract "
                f"to '{self.root}'."
            )


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

@register_dataset(
    key="lfw",
    description="Labeled Faces in the Wild — unconstrained face recognition benchmark",
    num_subjects_approx=5749,
    license_str="LFW public",
)
def _lfw_factory(root_path: str) -> LFWLoader:
    return LFWLoader(root_path)

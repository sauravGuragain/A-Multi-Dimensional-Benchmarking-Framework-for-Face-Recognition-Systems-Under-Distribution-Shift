"""
faceeval.data.loaders.celeba
=============================
Loaders for CelebA and VGGFace2 datasets.

CelebA directory structure::

    <root>/
        img_align_celeba/
            000001.jpg
            000002.jpg
            ...
        identity_CelebA.txt      ← "image_filename subject_id" per line
        list_attr_celeba.txt     ← per-image attribute labels (optional)

VGGFace2 directory structure (same as LFW: subject-named subdirs)::

    <root>/
        train/
            n000001/
                0001_01.jpg
            n000002/
                ...
        test/
            n000003/
                ...
"""

from __future__ import annotations

import logging
from pathlib import Path

from faceeval.core.registry import register_dataset
from faceeval.core.types import DatasetInfo, ImageRecord, SplitStrategy
from faceeval.data.loaders.base import BaseDatasetLoader

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# CelebA
# ---------------------------------------------------------------------------

class CelebALoader(BaseDatasetLoader):
    """
    Loader for the CelebA (Large-scale CelebFaces Attributes) dataset.

    Subject IDs are read from ``identity_CelebA.txt``.
    Attribute labels (gender, glasses, smile, etc.) are parsed from
    ``list_attr_celeba.txt`` when present and stored in
    ``ImageRecord.metadata`` for fairness evaluation.
    """

    _IDENTITY_FILE = "identity_CelebA.txt"
    _ATTR_FILE = "list_attr_celeba.txt"
    _IMG_DIR = "img_align_celeba"

    def get_info(self) -> DatasetInfo:
        self._validate_root()
        records = self.load_records()
        subjects = {r.subject_id for r in records}
        return DatasetInfo(
            name="celeba",
            root_path=str(self.root),
            num_subjects=len(subjects),
            num_images=len(records),
            image_size=(218, 178),
            color_mode="rgb",
            split_strategy=SplitStrategy.SUBJECT_DISJOINT,
            description="CelebA — large-scale face attributes dataset",
            license="Non-commercial research only",
            citation=(
                "Liu et al., 'Deep Learning Face Attributes in the Wild', ICCV 2015"
            ),
        )

    def load_records(self) -> list[ImageRecord]:
        self._validate_root()
        identity_file = self.root / self._IDENTITY_FILE
        img_dir = self.root / self._IMG_DIR

        if not identity_file.exists():
            raise FileNotFoundError(
                f"CelebA identity file not found: '{identity_file}'"
            )

        # Parse subject identity mapping
        filename_to_subject: dict[str, str] = {}
        with identity_file.open("r") as fh:
            for line in fh:
                parts = line.strip().split()
                if len(parts) == 2:
                    filename_to_subject[parts[0]] = parts[1]

        # Parse attribute labels (optional)
        attributes: dict[str, dict[str, int]] = {}
        attr_file = self.root / self._ATTR_FILE
        if attr_file.exists():
            attributes = self._parse_attributes(attr_file)

        records: list[ImageRecord] = []
        for filename, subject_id in sorted(filename_to_subject.items()):
            img_path = img_dir / filename
            if not img_path.exists():
                continue
            meta: dict = {"source": "celeba"}
            if filename in attributes:
                meta["attributes"] = attributes[filename]
            records.append(
                ImageRecord(
                    image_id=filename,
                    image_path=str(img_path),
                    subject_id=subject_id,
                    dataset_name="celeba",
                    split="unassigned",
                    metadata=meta,
                )
            )

        logger.info(
            "CelebA: loaded %d records across %d subjects.",
            len(records), len({r.subject_id for r in records}),
        )
        return records

    @staticmethod
    def _parse_attributes(attr_file: Path) -> dict[str, dict[str, int]]:
        """
        Parse the CelebA attribute file.

        Returns a dict mapping filename → {attr_name: +1/-1}.
        """
        result: dict[str, dict[str, int]] = {}
        with attr_file.open("r") as fh:
            lines = fh.readlines()
        if len(lines) < 3:
            return result
        # Line 0: count; line 1: attribute names; line 2+: data
        attr_names = lines[1].strip().split()
        for line in lines[2:]:
            parts = line.strip().split()
            if len(parts) != len(attr_names) + 1:
                continue
            filename = parts[0]
            result[filename] = {
                attr_names[i]: int(parts[i + 1])
                for i in range(len(attr_names))
            }
        return result


@register_dataset(
    key="celeba",
    description="CelebA — large-scale face attributes dataset",
    num_subjects_approx=10177,
    license_str="Non-commercial research only",
)
def _celeba_factory(root_path: str) -> CelebALoader:
    return CelebALoader(root_path)


# ---------------------------------------------------------------------------
# VGGFace2
# ---------------------------------------------------------------------------

class VGGFace2Loader(BaseDatasetLoader):
    """
    Loader for the VGGFace2 dataset.

    Expects the standard ``train/`` and ``test/`` subdirectories with
    subject-named sub-subdirectories (``n000001/``, ``n000002/``, …).
    Both splits are loaded together; the framework's ``DatasetSplitter``
    handles train/test reassignment.
    """

    def get_info(self) -> DatasetInfo:
        self._validate_root()
        records = self.load_records()
        subjects = {r.subject_id for r in records}
        return DatasetInfo(
            name="vggface2",
            root_path=str(self.root),
            num_subjects=len(subjects),
            num_images=len(records),
            image_size=(224, 224),
            color_mode="rgb",
            split_strategy=SplitStrategy.SUBJECT_DISJOINT,
            description="VGGFace2 — large-scale face recognition dataset",
            license="Non-commercial research only",
            citation=(
                "Cao et al., 'VGGFace2: A dataset for recognising faces "
                "across pose and age', FG 2018"
            ),
        )

    def load_records(self) -> list[ImageRecord]:
        self._validate_root()
        records: list[ImageRecord] = []

        for split_name in ("train", "test"):
            split_dir = self.root / split_name
            if not split_dir.exists():
                continue
            for subject_dir in sorted(split_dir.iterdir()):
                if not subject_dir.is_dir():
                    continue
                subject_id = subject_dir.name
                for img_path in sorted(subject_dir.iterdir()):
                    if img_path.suffix.lower() not in self.SUPPORTED_EXTENSIONS:
                        continue
                    image_id = str(img_path.relative_to(self.root))
                    records.append(
                        ImageRecord(
                            image_id=image_id,
                            image_path=str(img_path),
                            subject_id=subject_id,
                            dataset_name="vggface2",
                            split="unassigned",
                            metadata={"source": "vggface2", "original_split": split_name},
                        )
                    )

        logger.info(
            "VGGFace2: loaded %d records across %d subjects.",
            len(records), len({r.subject_id for r in records}),
        )
        return records


@register_dataset(
    key="vggface2",
    description="VGGFace2 — large-scale face recognition dataset",
    num_subjects_approx=9131,
    license_str="Non-commercial research only",
)
def _vggface2_factory(root_path: str) -> VGGFace2Loader:
    return VGGFace2Loader(root_path)


# ---------------------------------------------------------------------------
# Generic folder loader (fallback for any subject-per-directory layout)
# ---------------------------------------------------------------------------

class FolderDatasetLoader(BaseDatasetLoader):
    """
    Generic loader for any dataset organised as::

        <root>/
            <subject_id>/
                <image_file>
                ...

    Use this for custom or private datasets without a dedicated loader.
    Register it with a custom key via ``@register_dataset``.
    """

    def __init__(self, root_path: str, dataset_name: str = "custom") -> None:
        super().__init__(root_path)
        self._dataset_name = dataset_name

    def get_info(self) -> DatasetInfo:
        self._validate_root()
        records = self.load_records()
        subjects = {r.subject_id for r in records}
        return DatasetInfo(
            name=self._dataset_name,
            root_path=str(self.root),
            num_subjects=len(subjects),
            num_images=len(records),
            image_size=(0, 0),          # unknown without reading pixels
            color_mode="rgb",
            split_strategy=SplitStrategy.SUBJECT_DISJOINT,
            description=f"Custom folder dataset: {self._dataset_name}",
        )

    def load_records(self) -> list[ImageRecord]:
        self._validate_root()
        records: list[ImageRecord] = []
        for subject_dir in sorted(self.root.iterdir()):
            if not subject_dir.is_dir():
                continue
            subject_id = subject_dir.name
            for img_path in sorted(subject_dir.iterdir()):
                if img_path.suffix.lower() not in self.SUPPORTED_EXTENSIONS:
                    continue
                image_id = str(img_path.relative_to(self.root))
                records.append(
                    ImageRecord(
                        image_id=image_id,
                        image_path=str(img_path),
                        subject_id=subject_id,
                        dataset_name=self._dataset_name,
                        split="unassigned",
                        metadata={"source": self._dataset_name},
                    )
                )
        return records

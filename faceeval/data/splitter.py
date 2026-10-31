"""
faceeval.data.splitter
=======================
Train / validation / test splitting for face recognition datasets.

All strategies guarantee that no subject appears in more than one partition
when ``SplitStrategy.SUBJECT_DISJOINT`` is chosen — which is the thesis
default, because identity leakage between train and test invalidates
closed-set recognition benchmarks.

Supported strategies
--------------------
SUBJECT_DISJOINT  — subjects are divided first; images follow.  No identity
                    overlap between splits.  Required for valid closed-set FR.
STRATIFIED        — images split image-by-image with class stratification.
                    Subjects appear in both train and test (open-set only).
RANDOM            — simple random shuffle of all image IDs, no stratification.
K_FOLD            — subject-disjoint k-fold cross-validation.  Returns one
                    fold at a time; the caller iterates over k folds.
"""

from __future__ import annotations

import logging
import random
from collections import defaultdict

from faceeval.core.exceptions import SplitError
from faceeval.core.types import DataSplit, ImageRecord, SplitStrategy, SubjectID

logger = logging.getLogger(__name__)


class DatasetSplitter:
    """
    Produces ``DataSplit`` objects from a list of ``ImageRecord`` instances.

    Parameters
    ----------
    seed:
        Random seed for reproducibility.  Must match ``ExperimentConfig.random_seed``.
    test_fraction:
        Fraction of subjects (or images) to assign to the test set.
    val_fraction:
        Fraction of subjects (or images) to assign to the validation set.
    """

    def __init__(
        self,
        seed: int = 42,
        test_fraction: float = 0.3,
        val_fraction: float = 0.1,
    ) -> None:
        if test_fraction + val_fraction >= 1.0:
            raise SplitError(
                f"test_fraction ({test_fraction}) + val_fraction ({val_fraction}) "
                "must be less than 1.0"
            )
        self._seed = seed
        self._test_frac = test_fraction
        self._val_frac = val_fraction

    # ------------------------------------------------------------------
    # Public entry points
    # ------------------------------------------------------------------

    def split(
        self,
        records: list[ImageRecord],
        strategy: SplitStrategy,
        dataset_name: str = "",
        k_fold_index: int | None = None,
        k_folds: int | None = None,
    ) -> DataSplit:
        """
        Produce one ``DataSplit`` from ``records`` using ``strategy``.

        For ``K_FOLD`` strategy, ``k_fold_index`` (0-based) and ``k_folds``
        must be supplied.
        """
        if strategy == SplitStrategy.SUBJECT_DISJOINT:
            return self._subject_disjoint(records, dataset_name)
        if strategy == SplitStrategy.STRATIFIED:
            return self._stratified(records, dataset_name)
        if strategy == SplitStrategy.RANDOM:
            return self._random(records, dataset_name)
        if strategy == SplitStrategy.K_FOLD:
            if k_fold_index is None or k_folds is None:
                raise SplitError(
                    "k_fold_index and k_folds must be supplied for K_FOLD strategy."
                )
            return self._k_fold(records, dataset_name, k_folds, k_fold_index)
        raise SplitError(f"Unknown SplitStrategy: {strategy}")

    def all_k_folds(
        self,
        records: list[ImageRecord],
        k: int,
        dataset_name: str = "",
    ) -> list[DataSplit]:
        """Return all ``k`` folds as a list of ``DataSplit`` objects."""
        return [
            self._k_fold(records, dataset_name, k, i) for i in range(k)
        ]

    # ------------------------------------------------------------------
    # Strategy implementations
    # ------------------------------------------------------------------

    def _subject_disjoint(
        self, records: list[ImageRecord], dataset_name: str
    ) -> DataSplit:
        """
        Divide subjects randomly, then assign all their images accordingly.

        No subject crosses partition boundaries.
        """
        subjects = self._sorted_subjects(records)
        rng = random.Random(self._seed)
        rng.shuffle(subjects)

        n = len(subjects)
        n_test = max(1, round(n * self._test_frac))
        n_val = max(1, round(n * self._val_frac))
        n_train = n - n_test - n_val

        if n_train < 1:
            raise SplitError(
                f"Not enough subjects ({n}) for subject-disjoint split with "
                f"test={self._test_frac:.0%}, val={self._val_frac:.0%}. "
                "Reduce fractions or use a larger dataset."
            )

        train_subjects = set(subjects[:n_train])
        val_subjects = set(subjects[n_train: n_train + n_val])
        test_subjects = set(subjects[n_train + n_val:])

        train_ids, val_ids, test_ids = [], [], []
        for rec in records:
            if rec.subject_id in train_subjects:
                train_ids.append(rec.image_id)
            elif rec.subject_id in val_subjects:
                val_ids.append(rec.image_id)
            else:
                test_ids.append(rec.image_id)

        logger.info(
            "[%s] Subject-disjoint split — train: %d imgs (%d subj), "
            "val: %d imgs (%d subj), test: %d imgs (%d subj)",
            dataset_name,
            len(train_ids), len(train_subjects),
            len(val_ids), len(val_subjects),
            len(test_ids), len(test_subjects),
        )
        return DataSplit(
            dataset_name=dataset_name,
            train_ids=train_ids,
            val_ids=val_ids,
            test_ids=test_ids,
            split_strategy=SplitStrategy.SUBJECT_DISJOINT,
            random_seed=self._seed,
        )

    def _stratified(
        self, records: list[ImageRecord], dataset_name: str
    ) -> DataSplit:
        """
        Stratified image-level split preserving per-class proportions.

        Subjects may appear in all three splits.  Only valid for open-set
        or verification benchmarks where identity leakage is acceptable.
        """
        by_subject: dict[SubjectID, list[str]] = defaultdict(list)
        for rec in records:
            by_subject[rec.subject_id].append(rec.image_id)

        train_ids, val_ids, test_ids = [], [], []
        rng = random.Random(self._seed)

        for subject, ids in sorted(by_subject.items()):
            shuffled = list(ids)
            rng.shuffle(shuffled)
            n = len(shuffled)
            n_test = max(1, round(n * self._test_frac)) if n > 2 else 0
            n_val = max(1, round(n * self._val_frac)) if n > 2 else 0
            test_ids.extend(shuffled[:n_test])
            val_ids.extend(shuffled[n_test: n_test + n_val])
            train_ids.extend(shuffled[n_test + n_val:])

        logger.info(
            "[%s] Stratified split — train: %d, val: %d, test: %d",
            dataset_name, len(train_ids), len(val_ids), len(test_ids),
        )
        return DataSplit(
            dataset_name=dataset_name,
            train_ids=train_ids,
            val_ids=val_ids,
            test_ids=test_ids,
            split_strategy=SplitStrategy.STRATIFIED,
            random_seed=self._seed,
        )

    def _random(
        self, records: list[ImageRecord], dataset_name: str
    ) -> DataSplit:
        """Simple random shuffle, no stratification."""
        ids = [r.image_id for r in records]
        rng = random.Random(self._seed)
        rng.shuffle(ids)

        n = len(ids)
        n_test = max(1, round(n * self._test_frac))
        n_val = max(1, round(n * self._val_frac))

        test_ids = ids[:n_test]
        val_ids = ids[n_test: n_test + n_val]
        train_ids = ids[n_test + n_val:]

        logger.info(
            "[%s] Random split — train: %d, val: %d, test: %d",
            dataset_name, len(train_ids), len(val_ids), len(test_ids),
        )
        return DataSplit(
            dataset_name=dataset_name,
            train_ids=train_ids,
            val_ids=val_ids,
            test_ids=test_ids,
            split_strategy=SplitStrategy.RANDOM,
            random_seed=self._seed,
        )

    def _k_fold(
        self,
        records: list[ImageRecord],
        dataset_name: str,
        k: int,
        fold_index: int,
    ) -> DataSplit:
        """
        Subject-disjoint k-fold cross-validation.

        The validation set is a slice of the training subjects equal to
        ``1/k`` of them, taken in a rotating window.
        """
        if k < 2:
            raise SplitError(f"k_folds must be >= 2, got {k}")
        if not 0 <= fold_index < k:
            raise SplitError(
                f"fold_index {fold_index} out of range for k={k}"
            )

        subjects = self._sorted_subjects(records)
        rng = random.Random(self._seed)
        rng.shuffle(subjects)

        n = len(subjects)
        if n < k:
            raise SplitError(
                f"Dataset '{dataset_name}' has only {n} subjects but k={k} folds "
                "were requested.  Reduce k."
            )

        fold_size = n // k
        test_start = fold_index * fold_size
        test_end = test_start + fold_size if fold_index < k - 1 else n

        test_subjects = set(subjects[test_start:test_end])
        remaining = subjects[:test_start] + subjects[test_end:]

        # Validation: first fold_size subjects from remaining
        n_val = max(1, fold_size)
        val_subjects = set(remaining[:n_val])
        train_subjects = set(remaining[n_val:])

        train_ids, val_ids, test_ids = [], [], []
        for rec in records:
            if rec.subject_id in train_subjects:
                train_ids.append(rec.image_id)
            elif rec.subject_id in val_subjects:
                val_ids.append(rec.image_id)
            else:
                test_ids.append(rec.image_id)

        logger.info(
            "[%s] K-fold %d/%d — train: %d, val: %d, test: %d",
            dataset_name, fold_index + 1, k,
            len(train_ids), len(val_ids), len(test_ids),
        )
        return DataSplit(
            dataset_name=dataset_name,
            train_ids=train_ids,
            val_ids=val_ids,
            test_ids=test_ids,
            split_strategy=SplitStrategy.K_FOLD,
            random_seed=self._seed,
            k_fold_index=fold_index,
            k_fold_total=k,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _sorted_subjects(records: list[ImageRecord]) -> list[SubjectID]:
        """Return a deterministically ordered list of unique subject IDs."""
        return sorted({r.subject_id for r in records})

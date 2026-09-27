from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import Any

import numpy as np

from faceeval.core.types import (
    FailureCase,
    ModelName,
    PerturbationCategory,
    PerturbationSpec,
)
from faceeval.failure.extractor import extract_image_features

logger = logging.getLogger(__name__)

# Feature names in the order they appear in the feature matrix
_FEATURE_NAMES: list[str] = [
    "mean_brightness",
    "brightness_std",
    "local_variance",
    "edge_density",
    "aspect_ratio",
    "perturbation_severity",
    "confidence_proxy",
] + [f"cat_{cat.value}" for cat in PerturbationCategory]


# ---------------------------------------------------------------------------
# Meta-model class
# ---------------------------------------------------------------------------

class FailureMetaModel:
    """
    Random Forest meta-model that predicts image-level failures.

    Parameters
    ----------
    model_name:
        The recognition model whose failures this meta-model predicts.
    n_estimators:
        Number of trees in the Random Forest.
    max_depth:
        Maximum tree depth (``None`` = fully grown).
    random_state:
        Reproducibility seed.
    """

    def __init__(
        self,
        model_name: ModelName,
        n_estimators: int = 100,
        max_depth: int | None = 8,
        random_state: int = 42,
    ) -> None:
        self._model_name    = model_name
        self._n_estimators  = n_estimators
        self._max_depth     = max_depth
        self._random_state  = random_state
        self._clf: Any      = None
        self._is_trained    = False

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def fit(
        self,
        success_cases_images: list[np.ndarray],
        success_cases_specs:  list[PerturbationSpec | None],
        failure_cases_images: list[np.ndarray],
        failure_cases_specs:  list[PerturbationSpec | None],
    ) -> dict[str, Any]:
        """
        Train the meta-model.

        Parameters
        ----------
        success_cases_images / success_cases_specs:
            Images and perturbation specs for correctly classified samples.
        failure_cases_images / failure_cases_specs:
            Images and perturbation specs for misclassified samples.

        Returns
        -------
        dict with training statistics (n_success, n_failure, class_balance).
        """
        X_pos, y_pos = self._build_features(failure_cases_images, failure_cases_specs, label=1)
        X_neg, y_neg = self._build_features(success_cases_images, success_cases_specs, label=0)

        if len(X_pos) == 0 or len(X_neg) == 0:
            logger.warning(
                "[%s] Meta-model training skipped: need both success and failure samples.",
                self._model_name,
            )
            return {"trained": False, "reason": "insufficient samples"}

        X = np.vstack([X_pos, X_neg])
        y = np.concatenate([y_pos, y_neg])

        logger.info(
            "[%s] Training failure meta-model: %d failures, %d successes.",
            self._model_name, len(X_pos), len(X_neg),
        )

        from sklearn.ensemble import RandomForestClassifier
        self._clf = RandomForestClassifier(
            n_estimators=self._n_estimators,
            max_depth=self._max_depth,
            class_weight="balanced",   # compensates for failure rarity
            random_state=self._random_state,
            n_jobs=-1,
        )
        self._clf.fit(X, y)
        self._is_trained = True

        return {
            "trained": True,
            "n_failure_samples": len(X_pos),
            "n_success_samples": len(X_neg),
            "class_balance": round(len(X_pos) / (len(X_pos) + len(X_neg)), 4),
            "n_features": X.shape[1],
        }

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def predict_failure_probability(
        self,
        images: list[np.ndarray],
        specs: list[PerturbationSpec | None],
    ) -> list[float]:
        """
        Predict probability of failure for each (image, spec) pair.

        Returns
        -------
        list[float]
            Failure probabilities in [0, 1], one per input.
        """
        self._require_trained()
        X, _ = self._build_features(images, specs, label=0)
        probs = self._clf.predict_proba(X)
        # Column 1 = probability of class 1 = failure
        return [float(p[1]) for p in probs]

    def predict(
        self,
        images: list[np.ndarray],
        specs: list[PerturbationSpec | None],
        threshold: float = 0.5,
    ) -> list[bool]:
        """
        Binary failure prediction (True = predicted failure).
        """
        probs = self.predict_failure_probability(images, specs)
        return [p >= threshold for p in probs]

    # ------------------------------------------------------------------
    # Interpretability
    # ------------------------------------------------------------------

    def feature_importances(self) -> dict[str, float]:
        """
        Return per-feature importance scores from the Random Forest.

        Feature importances sum to 1.0.  High-importance features explain
        which image and perturbation properties drive failure.
        """
        self._require_trained()
        importances = self._clf.feature_importances_
        n = min(len(importances), len(_FEATURE_NAMES))
        return {
            _FEATURE_NAMES[i]: float(importances[i])
            for i in range(n)
        }

    def top_failure_predictors(self, top_k: int = 5) -> list[tuple[str, float]]:
        """
        Return the ``top_k`` most important features for predicting failure,
        sorted by descending importance.
        """
        importances = self.feature_importances()
        return sorted(importances.items(), key=lambda x: x[1], reverse=True)[:top_k]

    def cross_validate(
        self,
        images: list[np.ndarray],
        specs: list[PerturbationSpec | None],
        labels: list[int],
        cv_folds: int = 5,
    ) -> dict[str, float]:
        """
        Cross-validate the meta-model and return accuracy, precision, recall, F1.
        """
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.model_selection import cross_validate as sk_cv

        X, _ = self._build_features(images, specs, label=0)
        y = np.array(labels)

        clf = RandomForestClassifier(
            n_estimators=self._n_estimators,
            max_depth=self._max_depth,
            class_weight="balanced",
            random_state=self._random_state,
        )
        cv_results = sk_cv(
            clf, X, y,
            cv=min(cv_folds, len(y)),
            scoring=["accuracy", "precision", "recall", "f1"],
            return_train_score=False,
        )
        return {
            "accuracy":  float(np.mean(cv_results["test_accuracy"])),
            "precision": float(np.mean(cv_results["test_precision"])),
            "recall":    float(np.mean(cv_results["test_recall"])),
            "f1":        float(np.mean(cv_results["test_f1"])),
            "cv_folds":  cv_folds,
        }

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str | Path) -> None:
        self._require_trained()
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({
                "clf": self._clf,
                "model_name": self._model_name,
                "hyperparameters": {
                    "n_estimators": self._n_estimators,
                    "max_depth": self._max_depth,
                    "random_state": self._random_state,
                },
            }, f, protocol=pickle.HIGHEST_PROTOCOL)
        logger.info("[%s] Meta-model saved to '%s'.", self._model_name, path)

    def load(self, path: str | Path) -> None:
        with open(path, "rb") as f:
            state = pickle.load(f)
        self._clf = state["clf"]
        self._model_name = state["model_name"]
        self._is_trained = True
        logger.info("[%s] Meta-model loaded from '%s'.", self._model_name, path)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _build_features(
        self,
        images: list[np.ndarray],
        specs: list[PerturbationSpec | None],
        label: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Build feature matrix and label vector."""
        rows: list[list[float]] = []
        categories = list(PerturbationCategory)

        for img, spec in zip(images, specs):
            img_feats = extract_image_features(img)
            severity = spec.severity if spec else 0.0
            conf_proxy = 0.5  # neutral when not available from prior run

            # Category one-hot
            cat = spec.category if spec else None
            cat_oh = [1.0 if c == cat else 0.0 for c in categories]

            row = [
                img_feats["mean_brightness"],
                img_feats["brightness_std"],
                img_feats["local_variance"],
                img_feats["edge_density"],
                img_feats["aspect_ratio"],
                severity,
                conf_proxy,
            ] + cat_oh
            rows.append(row)

        X = np.array(rows, dtype=np.float32)
        y = np.full(len(rows), label, dtype=int)
        return X, y

    def _require_trained(self) -> None:
        if not self._is_trained:
            raise RuntimeError(
                f"FailureMetaModel for '{self._model_name}' has not been trained. "
                "Call fit() first."
            )

    @property
    def is_trained(self) -> bool:
        return self._is_trained

    @property
    def model_name(self) -> ModelName:
        return self._model_name


# ---------------------------------------------------------------------------
# Batch training helper
# ---------------------------------------------------------------------------

def train_meta_models(
    model_names: list[ModelName],
    failure_cases: list[FailureCase],
    images_by_id: dict[str, np.ndarray],
    n_success_samples_per_model: int = 200,
    random_state: int = 42,
) -> dict[ModelName, FailureMetaModel]:
    """
    Train one ``FailureMetaModel`` per model.

    Parameters
    ----------
    model_names:
        Models to train meta-models for.
    failure_cases:
        All extracted failure cases.
    images_by_id:
        Dict mapping image_id → preprocessed image array.
        Used to build features for both failure and success samples.
    n_success_samples_per_model:
        Number of random success samples to draw for class balance.

    Returns
    -------
    dict: model_name → trained FailureMetaModel
    """
    rng = np.random.default_rng(random_state)
    all_image_ids = list(images_by_id.keys())
    meta_models: dict[ModelName, FailureMetaModel] = {}

    for model_name in model_names:
        model_failures = [c for c in failure_cases if c.model_name == model_name]
        if not model_failures:
            logger.warning("[%s] No failure cases — skipping meta-model training.", model_name)
            continue

        # Failure images
        fail_imgs = [
            images_by_id[c.image_id]
            for c in model_failures
            if c.image_id in images_by_id
        ]
        fail_specs = [
            c.perturbation_spec
            for c in model_failures
            if c.image_id in images_by_id
        ]

        # Success images (random sample from non-failure images)
        failure_ids = {c.image_id for c in model_failures}
        success_ids = [i for i in all_image_ids if i not in failure_ids]
        if len(success_ids) > n_success_samples_per_model:
            chosen = rng.choice(success_ids, n_success_samples_per_model, replace=False)
            success_ids = list(chosen)

        succ_imgs  = [images_by_id[i] for i in success_ids]
        succ_specs = [None] * len(succ_imgs)

        meta = FailureMetaModel(model_name=model_name, random_state=random_state)
        result = meta.fit(succ_imgs, succ_specs, fail_imgs, fail_specs)
        if result.get("trained"):
            meta_models[model_name] = meta

    return meta_models

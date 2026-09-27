"""
faceeval.models.traditional.eigenfaces
========================================
Eigenfaces (PCA), Fisherfaces (LDA), and PCA+SVM recognizers.

Eigenfaces (Turk & Pentland 1991)
----------------------------------
Project face images onto the top-k principal components of the training
set covariance matrix.  Classify in PCA space using a nearest-neighbour
or linear SVM classifier.

Fisherfaces (Belhumeur et al. 1997)
-------------------------------------
Reduce with PCA first (to avoid singular scatter matrices) then apply
LDA to maximise between-class scatter relative to within-class scatter.
More illumination-robust than Eigenfaces.

PCA + SVM
----------
Same PCA projection as Eigenfaces but uses a kernel SVM classifier
instead of nearest-neighbour, typically improving generalisation.
"""

from __future__ import annotations

import logging
from typing import Any

from faceeval.core.registry import register_model
from faceeval.core.types import (
    DistanceMetric,
    EmbeddingVector,
    ModelParadigm,
    TraditionalModelType,
)
from faceeval.models.traditional.base import TraditionalRecognizerMixin

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Eigenfaces
# ---------------------------------------------------------------------------

class EigenfacesRecognizer(TraditionalRecognizerMixin):
    """
    PCA-based Eigenfaces with nearest-centroid or SVM classification.

    Parameters
    ----------
    n_components:
        Number of principal components to retain.  ``None`` uses
        ``min(n_samples, n_features) - 1``.
    classifier:
        ``"svm"`` (linear SVM, default) or ``"knn"`` (k-nearest neighbour).
    svm_C:
        Regularisation parameter for the SVM classifier.
    knn_k:
        k for the KNN classifier (used when ``classifier="knn"``).
    whiten:
        If True, divide projected coordinates by their standard deviation
        (unit-variance PCA space).
    """

    MODEL_NAME      = TraditionalModelType.EIGENFACES.value
    MODEL_TYPE      = TraditionalModelType.EIGENFACES.value
    PARADIGM        = ModelParadigm.TRADITIONAL
    VERSION         = "1.0.0"
    EMBEDDING_DIM   = None          # set dynamically after fit
    DISTANCE_METRIC = DistanceMetric.EUCLIDEAN

    def __init__(
        self,
        n_components: int | None = 150,
        classifier: str = "svm",
        svm_C: float = 1.0,
        knn_k: int = 5,
        whiten: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            n_components=n_components,
            classifier=classifier,
            svm_C=svm_C,
            knn_k=knn_k,
            whiten=whiten,
            **kwargs,
        )
        self._n_components = n_components
        self._classifier_type = classifier
        self._svm_C = svm_C
        self._knn_k = knn_k
        self._whiten = whiten

    def _build_extractor(self) -> Any:
        from sklearn.decomposition import PCA
        return PCA(n_components=self._n_components, whiten=self._whiten)

    def _build_classifier(self) -> Any:
        if self._classifier_type == "knn":
            from sklearn.neighbors import KNeighborsClassifier
            return KNeighborsClassifier(n_neighbors=self._knn_k, metric="euclidean")
        from sklearn.svm import LinearSVC
        from sklearn.calibration import CalibratedClassifierCV
        base = LinearSVC(C=self._svm_C, max_iter=2000)
        return CalibratedClassifierCV(base, cv=3)

    def extract_features(self, X):  # type: ignore[override]
        self._require_trained()
        X_flat = self._flatten_batch(X)
        projected = self._extractor.transform(X_flat)
        self.EMBEDDING_DIM = projected.shape[1]
        return [row.tolist() for row in projected]


# ---------------------------------------------------------------------------
# Fisherfaces
# ---------------------------------------------------------------------------

class FisherfacesRecognizer(TraditionalRecognizerMixin):
    """
    LDA-based Fisherfaces (PCA whitening → LDA → SVM).

    Parameters
    ----------
    n_pca_components:
        PCA step keeps this many components.  Must be > n_classes - 1.
        Default ``None`` uses min(n_samples, n_features) - 1.
    n_lda_components:
        LDA components to retain.  Bounded by n_classes - 1.
        ``None`` uses all available.
    svm_C:
        SVM regularisation.
    """

    MODEL_NAME      = TraditionalModelType.FISHERFACES.value
    MODEL_TYPE      = TraditionalModelType.FISHERFACES.value
    PARADIGM        = ModelParadigm.TRADITIONAL
    VERSION         = "1.0.0"
    EMBEDDING_DIM   = None
    DISTANCE_METRIC = DistanceMetric.EUCLIDEAN

    def __init__(
        self,
        n_pca_components: int | None = 200,
        n_lda_components: int | None = None,
        svm_C: float = 1.0,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            n_pca_components=n_pca_components,
            n_lda_components=n_lda_components,
            svm_C=svm_C,
            **kwargs,
        )
        self._n_pca = n_pca_components
        self._n_lda = n_lda_components
        self._svm_C = svm_C
        self._lda: Any = None

    def fit(self, X_train, y_train):  # type: ignore[override]
        from sklearn.decomposition import PCA
        from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
        from sklearn.svm import LinearSVC
        from sklearn.calibration import CalibratedClassifierCV

        self._class_labels = sorted(set(y_train))
        n_classes = len(self._class_labels)
        X_flat = self._flatten_batch(X_train)

        # PCA step
        n_pca = min(
            self._n_pca or X_flat.shape[0] - 1,
            X_flat.shape[0] - 1,
            X_flat.shape[1],
        )
        self._extractor = PCA(n_components=n_pca, whiten=True)
        X_pca = self._extractor.fit_transform(X_flat)

        # LDA step
        n_lda = min(
            self._n_lda or n_classes - 1,
            n_classes - 1,
            X_pca.shape[1],
        )
        self._lda = LinearDiscriminantAnalysis(n_components=n_lda)
        X_lda = self._lda.fit_transform(X_pca, y_train)
        self.EMBEDDING_DIM = X_lda.shape[1]

        # Classifier
        base = LinearSVC(C=self._svm_C, max_iter=2000)
        self._clf = CalibratedClassifierCV(base, cv=min(3, n_classes))
        self._clf.fit(X_lda, y_train)
        self._is_trained = True
        logger.info("[%s] Trained — PCA:%d → LDA:%d → SVM, %d subjects",
                    self.MODEL_NAME, n_pca, n_lda, n_classes)

    def _transform(self, X_flat):  # type: ignore[override]
        X_pca = self._extractor.transform(X_flat)
        return self._lda.transform(X_pca)

    def extract_features(self, X):  # type: ignore[override]
        self._require_trained()
        X_flat = self._flatten_batch(X)
        X_lda = self._transform(X_flat)
        return [row.tolist() for row in X_lda]

    def _build_classifier(self) -> Any:   # not used (overriding fit directly)
        return None

    def save(self, path) -> None:
        self._require_trained()
        import pickle; from pathlib import Path
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({"extractor": self._extractor, "lda": self._lda,
                         "clf": self._clf, "class_labels": self._class_labels,
                         "hyperparameters": self._hyperparameters,
                         "embedding_dim": self.EMBEDDING_DIM}, f,
                        protocol=pickle.HIGHEST_PROTOCOL)

    def load(self, path) -> None:
        import pickle
        with open(path, "rb") as f:
            state = pickle.load(f)
        self._extractor = state["extractor"]; self._lda = state["lda"]
        self._clf = state["clf"]; self._class_labels = state["class_labels"]
        self._hyperparameters = state["hyperparameters"]
        self.EMBEDDING_DIM = state.get("embedding_dim"); self._is_trained = True


# ---------------------------------------------------------------------------
# PCA + SVM (explicit, separate from Eigenfaces for hyper clarity)
# ---------------------------------------------------------------------------

class PCASVMRecognizer(TraditionalRecognizerMixin):
    """
    PCA dimensionality reduction + RBF-kernel SVM classifier.

    Using an RBF SVM instead of a linear one gives substantially better
    results on face recognition when the PCA subspace is well-configured,
    at the cost of longer training.

    Parameters
    ----------
    n_components:
        PCA components.
    svm_C:
        SVM regularisation strength.
    svm_gamma:
        RBF kernel width.  ``"scale"`` uses 1/(n_features * X.var()).
    whiten:
        Unit-variance PCA space.
    """

    MODEL_NAME      = TraditionalModelType.PCA_SVM.value
    MODEL_TYPE      = TraditionalModelType.PCA_SVM.value
    PARADIGM        = ModelParadigm.TRADITIONAL
    VERSION         = "1.0.0"
    EMBEDDING_DIM   = None
    DISTANCE_METRIC = DistanceMetric.EUCLIDEAN

    def __init__(
        self,
        n_components: int | None = 150,
        svm_C: float = 10.0,
        svm_gamma: str | float = "scale",
        whiten: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            n_components=n_components,
            svm_C=svm_C,
            svm_gamma=svm_gamma,
            whiten=whiten,
            **kwargs,
        )
        self._n_components = n_components
        self._svm_C = svm_C
        self._svm_gamma = svm_gamma
        self._whiten = whiten

    def _build_extractor(self) -> Any:
        from sklearn.decomposition import PCA
        return PCA(n_components=self._n_components, whiten=self._whiten)

    def _build_classifier(self) -> Any:
        from sklearn.svm import SVC
        return SVC(
            C=self._svm_C,
            kernel="rbf",
            gamma=self._svm_gamma,
            probability=True,
            max_iter=5000,
        )

    def extract_features(self, X):  # type: ignore[override]
        self._require_trained()
        X_flat = self._flatten_batch(X)
        projected = self._extractor.transform(X_flat)
        return [row.tolist() for row in projected]


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

@register_model(
    key=TraditionalModelType.EIGENFACES.value,
    paradigm=ModelParadigm.TRADITIONAL,
    model_type=TraditionalModelType.EIGENFACES.value,
    description="PCA-based Eigenfaces with configurable classifier",
)
def _eigenfaces_factory(**kwargs: Any) -> EigenfacesRecognizer:
    return EigenfacesRecognizer(**kwargs)


@register_model(
    key=TraditionalModelType.FISHERFACES.value,
    paradigm=ModelParadigm.TRADITIONAL,
    model_type=TraditionalModelType.FISHERFACES.value,
    description="LDA-based Fisherfaces (PCA→LDA→SVM)",
)
def _fisherfaces_factory(**kwargs: Any) -> FisherfacesRecognizer:
    return FisherfacesRecognizer(**kwargs)


@register_model(
    key=TraditionalModelType.PCA_SVM.value,
    paradigm=ModelParadigm.TRADITIONAL,
    model_type=TraditionalModelType.PCA_SVM.value,
    description="PCA dimensionality reduction + RBF-SVM classifier",
)
def _pca_svm_factory(**kwargs: Any) -> PCASVMRecognizer:
    return PCASVMRecognizer(**kwargs)

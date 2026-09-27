from __future__ import annotations

import logging
from typing import Any, Callable, TypeVar

from faceeval.core.exceptions import RegistrationError, UnknownPluginError
from faceeval.core.types import ModelParadigm, ModelName

logger = logging.getLogger(__name__)

T = TypeVar("T")
FactoryFn = Callable[..., Any]


# ---------------------------------------------------------------------------
# Base registry
# ---------------------------------------------------------------------------

class _Registry:
    """Generic key → factory registry."""

    def __init__(self, name: str) -> None:
        self._name = name
        self._factories: dict[str, FactoryFn] = {}
        self._metadata: dict[str, dict[str, Any]] = {}

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register(
        self,
        key: str,
        factory: FactoryFn,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """
        Register a factory function under ``key``.

        Parameters
        ----------
        key:
            Unique string identifier (e.g. "eigenfaces", "gaussian_blur").
        factory:
            Zero-or-kwargs callable that produces an instance of the plugin.
        metadata:
            Optional dict of descriptive info (paradigm, category, etc.)
            stored alongside the factory for introspection.

        Raises
        ------
        RegistrationError
            If ``key`` is already registered.
        """
        if key in self._factories:
            raise RegistrationError(
                f"[{self._name}] Key '{key}' is already registered. "
                "Use force=True to override."
            )
        self._factories[key] = factory
        self._metadata[key] = metadata or {}
        logger.debug("[%s] Registered '%s'", self._name, key)

    def register_or_replace(
        self,
        key: str,
        factory: FactoryFn,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Register a factory, silently replacing any existing registration."""
        if key in self._factories:
            logger.warning(
                "[%s] Replacing existing registration for '%s'", self._name, key
            )
        self._factories[key] = factory
        self._metadata[key] = metadata or {}

    # ------------------------------------------------------------------
    # Lookup
    # ------------------------------------------------------------------

    def get(self, key: str, **kwargs: Any) -> Any:
        """
        Instantiate the plugin registered under ``key``.

        Parameters
        ----------
        key:
            Registered key string.
        **kwargs:
            Passed directly to the factory function.

        Returns
        -------
        Any
            An instance of the plugin.

        Raises
        ------
        UnknownPluginError
            If ``key`` has not been registered.
        """
        if key not in self._factories:
            raise UnknownPluginError(
                f"[{self._name}] No plugin registered for key '{key}'. "
                f"Registered keys: {sorted(self._factories)}"
            )
        return self._factories[key](**kwargs)

    def get_factory(self, key: str) -> FactoryFn:
        """Return the raw factory without instantiating it."""
        if key not in self._factories:
            raise UnknownPluginError(
                f"[{self._name}] No factory for '{key}'. "
                f"Registered: {sorted(self._factories)}"
            )
        return self._factories[key]

    def get_metadata(self, key: str) -> dict[str, Any]:
        """Return the metadata dict for a registered key."""
        if key not in self._metadata:
            raise UnknownPluginError(
                f"[{self._name}] No metadata for '{key}'."
            )
        return self._metadata[key]

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def list_registered(self) -> list[str]:
        """Return all registered keys sorted alphabetically."""
        return sorted(self._factories)

    def is_registered(self, key: str) -> bool:
        return key in self._factories

    def __len__(self) -> int:
        return len(self._factories)

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} name={self._name!r} n={len(self)}>"


# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------

class ModelRegistry(_Registry):
    """
    Registry for face recognition model factories.

    Each factory must accept keyword arguments corresponding to the model's
    hyperparameters and return an object conforming to
    ``faceeval.models.base.BaseRecognizer``.

    Metadata keys
    -------------
    paradigm : ModelParadigm
    model_type : str
    description : str
    """

    def __init__(self) -> None:
        super().__init__("ModelRegistry")

    def register_model(
        self,
        key: ModelName,
        paradigm: ModelParadigm,
        model_type: str,
        description: str = "",
    ) -> Callable[[FactoryFn], FactoryFn]:
        """
        Decorator factory for registering a model.

        Usage::

            @model_registry.register_model(
                key="eigenfaces",
                paradigm=ModelParadigm.TRADITIONAL,
                model_type=TraditionalModelType.EIGENFACES.value,
                description="PCA-based Eigenfaces recognizer",
            )
            def _make_eigenfaces(**kwargs):
                return EigenfacesRecognizer(**kwargs)
        """
        def decorator(fn: FactoryFn) -> FactoryFn:
            self.register(
                key=key,
                factory=fn,
                metadata={
                    "paradigm": paradigm,
                    "model_type": model_type,
                    "description": description,
                },
            )
            return fn
        return decorator

    def list_traditional(self) -> list[str]:
        return [
            k for k, m in self._metadata.items()
            if m.get("paradigm") == ModelParadigm.TRADITIONAL
        ]

    def list_deep(self) -> list[str]:
        return [
            k for k, m in self._metadata.items()
            if m.get("paradigm") == ModelParadigm.DEEP_LEARNING
        ]


# ---------------------------------------------------------------------------
# Perturbation registry
# ---------------------------------------------------------------------------

class PerturbationRegistry(_Registry):
    """
    Registry for image perturbation factories.

    Each factory must accept keyword arguments and return an object conforming
    to ``faceeval.perturbation.base.BasePerturbation``.

    Metadata keys
    -------------
    category : PerturbationCategory
    description : str
    severity_param : str  — name of the raw parameter that tracks severity
    """

    def __init__(self) -> None:
        super().__init__("PerturbationRegistry")

    def register_perturbation(
        self,
        key: str,
        category: str,
        description: str = "",
        severity_param: str = "",
    ) -> Callable[[FactoryFn], FactoryFn]:
        """
        Decorator factory for registering a perturbation.

        Usage::

            @perturbation_registry.register_perturbation(
                key="gaussian_blur",
                category=PerturbationCategory.BLUR.value,
                description="Gaussian blur with configurable sigma",
                severity_param="sigma",
            )
            def _make_gaussian_blur(**kwargs):
                return GaussianBlur(**kwargs)
        """
        def decorator(fn: FactoryFn) -> FactoryFn:
            self.register(
                key=key,
                factory=fn,
                metadata={
                    "category": category,
                    "description": description,
                    "severity_param": severity_param,
                },
            )
            return fn
        return decorator

    def list_by_category(self, category: str) -> list[str]:
        return [
            k for k, m in self._metadata.items()
            if m.get("category") == category
        ]


# ---------------------------------------------------------------------------
# Dataset registry
# ---------------------------------------------------------------------------

class DatasetRegistry(_Registry):
    """
    Registry for dataset loader factories.

    Metadata keys
    -------------
    description : str
    num_subjects_approx : int
    license : str
    """

    def __init__(self) -> None:
        super().__init__("DatasetRegistry")

    def register_dataset(
        self,
        key: str,
        description: str = "",
        num_subjects_approx: int = 0,
        license_str: str = "",
    ) -> Callable[[FactoryFn], FactoryFn]:
        """Decorator factory for registering a dataset loader."""
        def decorator(fn: FactoryFn) -> FactoryFn:
            self.register(
                key=key,
                factory=fn,
                metadata={
                    "description": description,
                    "num_subjects_approx": num_subjects_approx,
                    "license": license_str,
                },
            )
            return fn
        return decorator


# ---------------------------------------------------------------------------
# Singleton instances (module-level)
# ---------------------------------------------------------------------------

model_registry = ModelRegistry()
perturbation_registry = PerturbationRegistry()
dataset_registry = DatasetRegistry()


# ---------------------------------------------------------------------------
# Convenience decorator aliases (top-level imports for plugin authors)
# ---------------------------------------------------------------------------

def register_model(
    key: ModelName,
    paradigm: ModelParadigm,
    model_type: str,
    description: str = "",
) -> Callable[[FactoryFn], FactoryFn]:
    """Module-level alias for ``model_registry.register_model(...)``."""
    return model_registry.register_model(
        key=key,
        paradigm=paradigm,
        model_type=model_type,
        description=description,
    )


def register_perturbation(
    key: str,
    category: str,
    description: str = "",
    severity_param: str = "",
) -> Callable[[FactoryFn], FactoryFn]:
    """Module-level alias for ``perturbation_registry.register_perturbation(...)``."""
    return perturbation_registry.register_perturbation(
        key=key,
        category=category,
        description=description,
        severity_param=severity_param,
    )


def register_dataset(
    key: str,
    description: str = "",
    num_subjects_approx: int = 0,
    license_str: str = "",
) -> Callable[[FactoryFn], FactoryFn]:
    """Module-level alias for ``dataset_registry.register_dataset(...)``."""
    return dataset_registry.register_dataset(
        key=key,
        description=description,
        num_subjects_approx=num_subjects_approx,
        license_str=license_str,
    )

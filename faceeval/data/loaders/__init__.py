"""
faceeval.data.loaders
=====================
Dataset loader plugin package.

Importing this package triggers all @register_dataset decorators,
registering every loader with dataset_registry.

To add a new dataset:
1. Create faceeval/data/loaders/my_dataset.py
2. Subclass BaseDatasetLoader, implement get_info() and load_records()
3. Decorate the factory with @register_dataset(key="my_dataset", ...)
4. Add an import below.
"""

from faceeval.data.loaders.base import BaseDatasetLoader
from faceeval.data.loaders.lfw import LFWLoader
from faceeval.data.loaders.celeba import CelebALoader, VGGFace2Loader, FolderDatasetLoader

__all__ = [
    "BaseDatasetLoader",
    "LFWLoader",
    "CelebALoader",
    "VGGFace2Loader",
    "FolderDatasetLoader",
]

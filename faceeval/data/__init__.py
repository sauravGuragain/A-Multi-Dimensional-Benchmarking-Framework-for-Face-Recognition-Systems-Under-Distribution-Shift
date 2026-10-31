"""
faceeval.data
===============
Dataset management, splitting, and caching for face recognition benchmarks.

Submodules:
- splitter: Train/test splitting strategies (STRATIFIED, SUBJECT_DISJOINT, etc.)
- loaders: Dataset-specific loaders (LFW, synthetic, etc.)
"""

from faceeval.data.splitter import DatasetSplitter

__all__ = ["DatasetSplitter"]

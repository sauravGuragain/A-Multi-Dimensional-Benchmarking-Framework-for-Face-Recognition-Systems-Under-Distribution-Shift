"""
Shared pytest fixtures for the FaceEval-X test suite.

Run with: pytest tests/ -v
"""
import sys
from pathlib import Path

import pytest

# Ensure the project root is importable when running pytest from anywhere
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


@pytest.fixture
def tmp_data_dir(tmp_path):
    """A temporary directory for synthetic dataset fixtures."""
    d = tmp_path / "data"
    d.mkdir()
    return d


@pytest.fixture
def random_seed():
    return 42

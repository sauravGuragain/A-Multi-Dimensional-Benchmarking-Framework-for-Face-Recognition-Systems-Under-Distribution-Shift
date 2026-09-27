from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

from faceeval.storage.result_store import ResultStore


class Settings:
    """API runtime settings, read from environment variables."""

    def __init__(self) -> None:
        self.db_path = os.environ.get("FACEEVAL_DB_PATH", "experiments/results.db")
        self.figures_dir = os.environ.get("FACEEVAL_FIGURES_DIR", "reports/figures")
        self.reports_dir = os.environ.get("FACEEVAL_REPORTS_DIR", "reports")
        self.cors_origins = os.environ.get(
            "FACEEVAL_CORS_ORIGINS", "http://localhost:3000,http://localhost:5173"
        ).split(",")


@lru_cache
def get_settings() -> Settings:
    return Settings()


@lru_cache
def get_result_store() -> ResultStore:
    settings = get_settings()
    return ResultStore(db_path=settings.db_path)

from __future__ import annotations

import os
from functools import lru_cache

from api.readonly_store import ReadOnlyResultStore

_DEFAULT_CORS = (
    "http://localhost:5173,http://127.0.0.1:5173,"
    "http://localhost:3000,http://127.0.0.1:3000"
)


class Settings:
    """API runtime settings, read from environment variables."""

    def __init__(self) -> None:
        self.db_path = os.environ.get("FACEEVAL_DB_PATH", "experiments/results.db")
        self.figures_dir = os.environ.get("FACEEVAL_FIGURES_DIR", "reports/figures")
        self.reports_dir = os.environ.get("FACEEVAL_REPORTS_DIR", "reports")
        self.cors_origins = [
            o.strip()
            for o in os.environ.get("FACEEVAL_CORS_ORIGINS", _DEFAULT_CORS).split(",")
            if o.strip()
        ]


@lru_cache
def get_settings() -> Settings:
    return Settings()


def get_result_store() -> ReadOnlyResultStore:
    """Read-only view of the results database (never creates or writes it)."""
    return ReadOnlyResultStore(get_settings().db_path)

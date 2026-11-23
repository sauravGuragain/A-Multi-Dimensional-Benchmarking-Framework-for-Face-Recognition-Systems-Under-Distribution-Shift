"""
Read-only access to the FaceEval-X results database for the API.

The dashboard must never change research evidence.  ``ResultStore`` opens the
database read-write, creates it (and its parent directory) if it is missing,
and runs ``CREATE TABLE`` on every open, so it is not used by the API.  This
reader opens SQLite with ``mode=ro`` and ``PRAGMA query_only=ON``: a wrong
``FACEEVAL_DB_PATH`` produces a clear "database not found" error instead of
a new empty database, and no code path here can write.

A short-lived connection is opened per call, so runs written by the
experiment runner while the API is up appear without a restart.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

from faceeval.core.exceptions import RunNotFoundError


class DatabaseUnavailableError(Exception):
    """The database file is missing, locked, or not a FaceEval-X results DB."""


class StoredDataError(Exception):
    """A stored run exists but its JSON payload cannot be read."""


class ReadOnlyResultStore:
    def __init__(self, db_path: str | Path) -> None:
        self.db_path = Path(db_path)

    # -- connection ---------------------------------------------------------

    def _connect(self) -> sqlite3.Connection:
        if not self.db_path.is_file():
            raise DatabaseUnavailableError(
                f"Results database not found at '{self.db_path}'. "
                "Set FACEEVAL_DB_PATH or run an experiment first."
            )
        uri = self.db_path.resolve().as_uri() + "?mode=ro"
        try:
            conn = sqlite3.connect(uri, uri=True, timeout=5.0)
            conn.execute("PRAGMA query_only = ON")
            return conn
        except sqlite3.Error as exc:
            raise DatabaseUnavailableError(
                f"Cannot open results database '{self.db_path}': {exc}"
            ) from exc

    def _query(self, sql: str, params: tuple = ()) -> list[tuple]:
        conn = self._connect()
        try:
            return conn.execute(sql, params).fetchall()
        except sqlite3.Error as exc:
            raise DatabaseUnavailableError(
                f"Cannot read results database '{self.db_path}': {exc}"
            ) from exc
        finally:
            conn.close()

    # -- queries ------------------------------------------------------------

    def get(self, run_id: str) -> dict[str, Any]:
        rows = self._query("SELECT data FROM runs WHERE run_id = ?", (run_id,))
        if not rows:
            raise RunNotFoundError(run_id)
        try:
            data = json.loads(rows[0][0])
        except (TypeError, ValueError) as exc:
            raise StoredDataError(f"Stored data for run '{run_id}' is not valid JSON.") from exc
        if not isinstance(data, dict):
            raise StoredDataError(f"Stored data for run '{run_id}' has an unexpected shape.")
        return data

    def list_runs(
        self,
        experiment_name: str | None = None,
        status: str | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        where, params = _filters(experiment_name, status)
        rows = self._query(
            "SELECT run_id, experiment_name, status, started_at, completed_at "
            f"FROM runs{where} ORDER BY started_at DESC, run_id LIMIT ? OFFSET ?",
            (*params, limit, offset),
        )
        keys = ("run_id", "experiment_name", "status", "started_at", "completed_at")
        return [dict(zip(keys, r)) for r in rows]

    def count_runs(self, experiment_name: str | None = None, status: str | None = None) -> int:
        where, params = _filters(experiment_name, status)
        return int(self._query(f"SELECT COUNT(*) FROM runs{where}", tuple(params))[0][0])


def _filters(experiment_name: str | None, status: str | None) -> tuple[str, list[Any]]:
    clauses, params = [], []
    if experiment_name:
        clauses.append("experiment_name = ?")
        params.append(experiment_name)
    if status:
        clauses.append("status = ?")
        params.append(status)
    return (" WHERE " + " AND ".join(clauses)) if clauses else "", params

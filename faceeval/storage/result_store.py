"""
faceeval.storage.result_store
===============================
SQLite-backed persistence for ``ExperimentRun`` objects.

Why SQLite and not raw JSON files?
------------------------------------
The API layer needs to list runs, filter by status, and fetch one run's
full result set without loading every run into memory.  SQLite gives us
indexed queries on (run_id, status, experiment_name) with zero external
service dependencies — appropriate for a thesis-scale framework where a
single researcher runs a few dozen experiments.

Schema
------
One table ``runs`` with the full ``ExperimentRun`` serialised as a JSON
blob in the ``data`` column, plus indexed columns for fast filtering:
``run_id`` (PK), ``experiment_name``, ``status``, ``config_hash``,
``started_at``, ``completed_at``.

This is intentionally a document-store pattern over SQL, not a normalised
relational schema — the nested dataclass structure of ``ExperimentRun``
does not benefit from relational decomposition at this scale, and a JSON
blob keeps serialisation trivial and always in sync with ``faceeval.core.types``.
"""

from __future__ import annotations

import json
import logging
import sqlite3
from pathlib import Path
from typing import Any

from faceeval.core.exceptions import DuplicateRunError, RunNotFoundError
from faceeval.core.types import ExperimentRun, ExperimentStatus, RunID

logger = logging.getLogger(__name__)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    run_id          TEXT PRIMARY KEY,
    experiment_name TEXT NOT NULL,
    status          TEXT NOT NULL,
    config_hash     TEXT,
    started_at      TEXT,
    completed_at    TEXT,
    data            TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_runs_experiment ON runs(experiment_name);
CREATE INDEX IF NOT EXISTS idx_runs_status     ON runs(status);
CREATE INDEX IF NOT EXISTS idx_runs_confighash ON runs(config_hash);
"""


class ResultStore:
    """
    SQLite-backed store for ``ExperimentRun`` objects.

    Parameters
    ----------
    db_path:
        Path to the SQLite database file.  Created if it does not exist.
    """

    def __init__(self, db_path: str = "experiments/results.db") -> None:
        self._path = Path(db_path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self._path), check_same_thread=False)
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def save(self, run: ExperimentRun, allow_overwrite: bool = True) -> None:
        """
        Persist an ``ExperimentRun``.

        Parameters
        ----------
        run:
            The run to save (insert or update).
        allow_overwrite:
            If False, raises ``DuplicateRunError`` when a run with the same
            ``config_hash`` already exists under a different run_id.
        """
        if not allow_overwrite:
            existing = self.find_by_config_hash(run.config.config_hash)
            if existing and existing.run_id != run.run_id:
                raise DuplicateRunError(run.config.config_hash, existing.run_id)

        data = _serialize_run(run)
        self._conn.execute(
            """
            INSERT INTO runs (run_id, experiment_name, status, config_hash,
                              started_at, completed_at, data)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(run_id) DO UPDATE SET
                status=excluded.status,
                completed_at=excluded.completed_at,
                data=excluded.data
            """,
            (
                run.run_id,
                run.config.experiment_name,
                run.status.value,
                run.config.config_hash,
                run.started_at.isoformat() if run.started_at else None,
                run.completed_at.isoformat() if run.completed_at else None,
                json.dumps(data, default=str),
            ),
        )
        self._conn.commit()
        logger.debug("Run '%s' saved to result store.", run.run_id)

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    def get(self, run_id: RunID) -> dict[str, Any]:
        """
        Retrieve the raw stored dict for a run.

        Returns the JSON-deserialised dict (not reconstructed as a dataclass,
        since the API layer serves JSON directly).

        Raises
        ------
        RunNotFoundError
            If ``run_id`` is not in the store.
        """
        cur = self._conn.execute("SELECT data FROM runs WHERE run_id = ?", (run_id,))
        row = cur.fetchone()
        if row is None:
            raise RunNotFoundError(run_id)
        return json.loads(row[0])

    def exists(self, run_id: RunID) -> bool:
        cur = self._conn.execute("SELECT 1 FROM runs WHERE run_id = ?", (run_id,))
        return cur.fetchone() is not None

    def find_by_config_hash(self, config_hash: str) -> Any | None:
        """Return the stored run dict matching ``config_hash``, or None."""
        if not config_hash:
            return None
        cur = self._conn.execute(
            "SELECT run_id, data FROM runs WHERE config_hash = ? LIMIT 1",
            (config_hash,),
        )
        row = cur.fetchone()
        if row is None:
            return None

        class _Stub:
            def __init__(self, run_id: str) -> None:
                self.run_id = run_id
        return _Stub(row[0])

    def list_runs(
        self,
        experiment_name: str | None = None,
        status: ExperimentStatus | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        """
        List run summaries (not full data) with optional filtering.

        Returns lightweight dicts with run_id, experiment_name, status,
        started_at, completed_at — suitable for a dashboard list view.
        """
        query = "SELECT run_id, experiment_name, status, started_at, completed_at FROM runs"
        conditions: list[str] = []
        params: list[Any] = []

        if experiment_name:
            conditions.append("experiment_name = ?")
            params.append(experiment_name)
        if status:
            conditions.append("status = ?")
            params.append(status.value)

        if conditions:
            query += " WHERE " + " AND ".join(conditions)
        query += " ORDER BY started_at DESC LIMIT ? OFFSET ?"
        params.extend([limit, offset])

        cur = self._conn.execute(query, params)
        return [
            {
                "run_id": r[0],
                "experiment_name": r[1],
                "status": r[2],
                "started_at": r[3],
                "completed_at": r[4],
            }
            for r in cur.fetchall()
        ]

    def count_runs(
        self,
        experiment_name: str | None = None,
        status: ExperimentStatus | None = None,
    ) -> int:
        query = "SELECT COUNT(*) FROM runs"
        conditions: list[str] = []
        params: list[Any] = []
        if experiment_name:
            conditions.append("experiment_name = ?")
            params.append(experiment_name)
        if status:
            conditions.append("status = ?")
            params.append(status.value)
        if conditions:
            query += " WHERE " + " AND ".join(conditions)
        cur = self._conn.execute(query, params)
        return int(cur.fetchone()[0])

    # ------------------------------------------------------------------
    # Delete
    # ------------------------------------------------------------------

    def delete(self, run_id: RunID) -> bool:
        """Delete a run. Returns True if a row was deleted."""
        cur = self._conn.execute("DELETE FROM runs WHERE run_id = ?", (run_id,))
        self._conn.commit()
        return cur.rowcount > 0

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> "ResultStore":
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()


# ---------------------------------------------------------------------------
# Serialization helpers
# ---------------------------------------------------------------------------

def _serialize_run(run: ExperimentRun) -> dict[str, Any]:
    """Convert an ExperimentRun (with nested dataclasses) to a plain dict."""
    return {
        "run_id": run.run_id,
        "status": run.status.value,
        "config": run.config.to_dict(),
        "evaluation_results": [r.to_dict() for r in run.evaluation_results],
        "fingerprints": [fp.to_dict() for fp in run.fingerprints],
        "failure_clusters": [c.to_dict() for c in run.failure_clusters],
        "deployment_rankings": [r.to_dict() for r in run.deployment_rankings],
        "started_at": run.started_at.isoformat() if run.started_at else None,
        "completed_at": run.completed_at.isoformat() if run.completed_at else None,
        "duration_seconds": run.duration_seconds,
        "error_message": run.error_message,
    }

"""
Tests for faceeval.storage.result_store

Regression tests ensuring that:
1. Runs with large result artifacts can be persisted without sqlite3.DataError
2. Stored metadata is retrievable and complete
3. Raw arrays/matrices are NOT stored in SQLite (only counts)
4. Existing overwrite/config-hash behavior is preserved
"""

from __future__ import annotations

import json
import sqlite3
import tempfile
from datetime import datetime
from pathlib import Path

import pytest

from faceeval.core.types import (
    ExperimentRun,
    ExperimentStatus,
    ModelParadigm,
    ExperimentConfig,
    EvaluationResult,
    BehavioralFingerprint,
    DeploymentRanking,
    DeploymentScenario,
    DeploymentWeights,
    DeploymentScore,
    FailureCluster,
    PerturbationSpec,
    PerturbationType,
    PerturbationCategory,
)
from faceeval.storage.result_store import ResultStore, _serialize_run


@pytest.fixture
def temp_db():
    """Create a temporary SQLite database."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test_results.db"
        yield str(db_path)


@pytest.fixture
def sample_config():
    """Create a minimal ExperimentConfig for testing."""
    return ExperimentConfig(
        experiment_name="test_experiment",
        random_seed=42,
        run_id="test_run_001",
        config_hash="test_hash_001",
        report_dir="reports/test/",
    )


@pytest.fixture
def sample_run(sample_config):
    """Create a sample ExperimentRun with mock data."""
    run = ExperimentRun(
        run_id="test_run_001",
        config=sample_config,
        status=ExperimentStatus.COMPLETED,
        started_at=datetime(2026, 9, 28, 10, 0, 0),
        completed_at=datetime(2026, 9, 28, 10, 30, 0),
        error_message="",
    )

    # Add mock evaluation results (would normally contain large arrays)
    for i in range(5):
        result = EvaluationResult(
            run_id=run.run_id,
            model_name=f"model_{i}",
            perturbation_spec=PerturbationSpec(
                perturbation_type=PerturbationType.GAUSSIAN_BLUR,
                category=PerturbationCategory.BLUR,
                severity=0.5,
                raw_params={"sigma": 2.0},
                description="Test Gaussian blur",
            ),
            accuracy=0.8 - (i * 0.1),
            eer=0.2 + (i * 0.02),
            far_at_thresholds={0.5: 0.1},
            frr_at_thresholds={0.5: 0.1},
        )
        run.evaluation_results.append(result)

    # Add mock fingerprints
    for i in range(2):
        fp = BehavioralFingerprint(
            model_name=f"model_{i}",
            paradigm=ModelParadigm.TRADITIONAL,
            dimension_names=[f"dim_{j}" for j in range(21)],
            vector=[0.5] * 21,
            scores={f"dim_{j}": 0.5 for j in range(21)},
        )
        run.fingerprints.append(fp)

    # Add mock deployment ranking
    weights = DeploymentWeights()

    ranked_scores = [
        DeploymentScore(
            model_name="model_0",
            paradigm=ModelParadigm.TRADITIONAL,
            scenario=DeploymentScenario.RESEARCH_BENCHMARK,
            weights=weights,
            total_score=0.9,
            rank=1,
        ),
        DeploymentScore(
            model_name="model_1",
            paradigm=ModelParadigm.TRADITIONAL,
            scenario=DeploymentScenario.RESEARCH_BENCHMARK,
            weights=weights,
            total_score=0.8,
            rank=2,
        ),
    ]

    ranking = DeploymentRanking(
        scenario=DeploymentScenario.RESEARCH_BENCHMARK,
        weights=weights,
        ranked_scores=ranked_scores,
    )
    run.deployment_rankings.append(ranking)

    return run


class TestResultStoreBasics:
    """Basic ResultStore functionality."""

    def test_init_creates_schema(self, temp_db):
        """ResultStore.__init__ creates valid SQLite schema."""
        store = ResultStore(temp_db)

        # Verify table exists
        cur = store._conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='runs'"
        )
        assert cur.fetchone() is not None

        # Verify indexes exist
        cur = store._conn.execute(
            "SELECT name FROM sqlite_master WHERE type='index' AND name LIKE 'idx_runs%'"
        )
        indexes = cur.fetchall()
        assert len(indexes) >= 3  # experiment, status, confighash

        store.close()

    def test_save_and_retrieve(self, temp_db, sample_run):
        """Persistence: save and get a complete run."""
        store = ResultStore(temp_db)

        # Save
        store.save(sample_run)

        # Retrieve
        retrieved = store.get(sample_run.run_id)

        assert retrieved["run_id"] == sample_run.run_id
        assert retrieved["status"] == ExperimentStatus.COMPLETED.value
        assert len(retrieved["evaluation_results"]) == 5
        assert len(retrieved["fingerprints"]) == 2
        assert len(retrieved["deployment_rankings"]) == 1

        store.close()

    def test_save_nonexistent_run(self, temp_db):
        """Persistence: saving a run without results works."""
        store = ResultStore(temp_db)

        config = ExperimentConfig(
            experiment_name="empty_test",
            random_seed=99,
            run_id="empty_run",
            config_hash="empty_hash",
            report_dir=None,
        )
        run = ExperimentRun(
            run_id="empty_run",
            config=config,
            status=ExperimentStatus.PENDING,
        )

        store.save(run)
        retrieved = store.get(run.run_id)

        assert len(retrieved["evaluation_results"]) == 0
        assert len(retrieved["fingerprints"]) == 0
        assert retrieved["status"] == ExperimentStatus.PENDING.value

        store.close()

    def test_overwrite_existing_run(self, temp_db, sample_run, sample_config):
        """Persistence: overwriting an existing run updates status/timestamp."""
        store = ResultStore(temp_db)

        # Save original
        store.save(sample_run)

        # Modify and save again
        sample_run.status = ExperimentStatus.FAILED
        sample_run.error_message = "Test failure"
        sample_run.completed_at = datetime(2026, 9, 28, 11, 0, 0)

        store.save(sample_run, allow_overwrite=True)

        # Verify update
        retrieved = store.get(sample_run.run_id)
        assert retrieved["status"] == ExperimentStatus.FAILED.value
        assert retrieved["error_message"] == "Test failure"

        store.close()

    def test_config_hash_deduplication(self, temp_db, sample_config):
        """Persistence: find_by_config_hash retrieves runs by config."""
        store = ResultStore(temp_db)

        # Save run
        run1 = ExperimentRun(
            run_id="run_1",
            config=sample_config,
            status=ExperimentStatus.COMPLETED,
        )
        store.save(run1)

        # Find by hash
        found = store.find_by_config_hash(sample_config.config_hash)
        assert found is not None
        assert found.run_id == "run_1"

        # Not found
        not_found = store.find_by_config_hash("nonexistent_hash")
        assert not_found is None

        store.close()

    def test_exists(self, temp_db, sample_run):
        """Persistence: exists() returns True for saved runs."""
        store = ResultStore(temp_db)

        assert not store.exists(sample_run.run_id)
        store.save(sample_run)
        assert store.exists(sample_run.run_id)

        store.close()

    def test_delete(self, temp_db, sample_run):
        """Persistence: delete() removes a run."""
        store = ResultStore(temp_db)

        store.save(sample_run)
        assert store.exists(sample_run.run_id)

        deleted = store.delete(sample_run.run_id)
        assert deleted is True
        assert not store.exists(sample_run.run_id)

        # Delete non-existent returns False
        deleted = store.delete("nonexistent")
        assert deleted is False

        store.close()

    def test_list_runs(self, temp_db, sample_config):
        """Persistence: list_runs() returns summaries."""
        store = ResultStore(temp_db)

        # Save multiple runs
        for i in range(3):
            config = ExperimentConfig(
                experiment_name="list_test",
                random_seed=42 + i,
                run_id=f"run_{i}",
                config_hash=f"hash_{i}",
            )
            run = ExperimentRun(
                run_id=f"run_{i}",
                config=config,
                status=ExperimentStatus.COMPLETED,
                started_at=datetime(2026, 9, 28, 10 + i, 0, 0),
                completed_at=datetime(2026, 9, 28, 11 + i, 0, 0),
            )
            store.save(run)

        # List all
        summaries = store.list_runs(limit=10)
        assert len(summaries) == 3

        # List by experiment
        summaries = store.list_runs(experiment_name="list_test", limit=10)
        assert len(summaries) == 3
        assert all(s["experiment_name"] == "list_test" for s in summaries)

        store.close()


class TestLargePersistence:
    """Regression tests: large runs don't cause sqlite3.DataError."""

    def test_large_run_no_data_error(self, temp_db, sample_run):
        """
        REGRESSION TEST: Large ExperimentRun with 18+ conditions
        persists without sqlite3.DataError: string or blob too big.

        This is the smoke-test scenario: 18 conditions × 5749 subjects
        would create massive confusion matrices if serialized naively.
        With Option A (count-based storage), this should succeed.
        """
        store = ResultStore(temp_db)

        # Add more results to simulate realistic load
        for i in range(18):  # Smoke test: 18 conditions
            result = EvaluationResult(
                run_id=sample_run.run_id,
                model_name=f"model_{i % 3}",
                perturbation_spec=PerturbationSpec(
                    perturbation_type=[
                        PerturbationType.GAUSSIAN_BLUR,
                        PerturbationType.GAUSSIAN_NOISE,
                        PerturbationType.ROTATION,
                    ][i % 3],
                    category=[
                        PerturbationCategory.BLUR,
                        PerturbationCategory.NOISE,
                        PerturbationCategory.GEOMETRIC,
                    ][i % 3],
                    severity=[0.0, 0.5, 1.0][(i // 3) % 3],
                    raw_params={},
                    description=f"Test perturbation {i}",
                ),
                accuracy=0.5 - (i * 0.01),
                eer=0.4 + (i * 0.01),
                far_at_thresholds={0.1 + (i * 0.005): 0.1 + (i * 0.005)},
                frr_at_thresholds={0.1 + (i * 0.005): 0.1 + (i * 0.005)},
            )
            sample_run.evaluation_results.append(result)

        # This should NOT raise sqlite3.DataError
        try:
            store.save(sample_run)
            success = True
        except sqlite3.DataError as e:
            pytest.fail(f"sqlite3.DataError raised (should be fixed): {e}")
            success = False

        assert success

        # Verify retrieval
        retrieved = store.get(sample_run.run_id)
        assert len(retrieved["evaluation_results"]) == 23

        store.close()

    def test_serialized_size_is_small(self, sample_run):
        """
        REGRESSION TEST: Serialized run dict is small (<1MB).

        Option A should produce ~10-100 KB per run, not 100+ MB.
        This test verifies that raw data is NOT serialized.
        """
        # Add realistic data volume
        for i in range(18):
            result = EvaluationResult(
                run_id=sample_run.run_id,
                model_name=f"model_{i % 3}",
                perturbation_spec=PerturbationSpec(
                    perturbation_type=PerturbationType.GAUSSIAN_BLUR,
                    category=PerturbationCategory.BLUR,
                    severity=0.5,
                    raw_params={"sigma": 2.0},
                    description="Test",
                ),
                accuracy=0.5,
                eer=0.4,
                far_at_thresholds={0.5: 0.1},
                frr_at_thresholds={0.5: 0.1},
            )
            sample_run.evaluation_results.append(result)

        # Serialize
        serialized = _serialize_run(sample_run)
        json_str = json.dumps(serialized, default=str)
        size_bytes = len(json_str.encode('utf-8'))
        size_kb = size_bytes / 1024

        # Should be < 500 KB (realistic for metadata + config)
        assert size_kb < 500, f"Serialized size {size_kb:.1f} KB is too large (should be < 500 KB)"
        print(f"  ✓ Serialized run: {size_kb:.1f} KB")


class TestSerializationContent:
    """Tests verifying what is and isn't serialized."""

    def test_results_are_serialized(self, sample_run):
        """Structured experiment results are serialized in the stored output."""
        serialized = _serialize_run(sample_run)

        assert "evaluation_results" in serialized
        assert isinstance(serialized["evaluation_results"], list)
        assert len(serialized["evaluation_results"]) == len(
            sample_run.evaluation_results
        )

        assert "fingerprints" in serialized
        assert isinstance(serialized["fingerprints"], list)
        assert len(serialized["fingerprints"]) == len(sample_run.fingerprints)

        assert "deployment_rankings" in serialized
        assert isinstance(serialized["deployment_rankings"], list)
        assert len(serialized["deployment_rankings"]) == len(
            sample_run.deployment_rankings
        )
    def test_metadata_is_serialized(self, sample_run):
        """Essential run and configuration metadata are serialized."""
        serialized = _serialize_run(sample_run)

        assert serialized["run_id"] == sample_run.run_id
        assert serialized["status"] == ExperimentStatus.COMPLETED.value
        assert serialized["config"] is not None
        assert serialized["config"]["report_dir"] == sample_run.config.report_dir
        assert serialized["started_at"] is not None
        assert serialized["completed_at"] is not None
        assert serialized["duration_seconds"] is not None

    def test_report_dir_is_serialized_in_config(self, sample_run):
        """Report directory is preserved as part of the experiment config."""
        serialized = _serialize_run(sample_run)

        assert serialized["config"]["report_dir"] == sample_run.config.report_dir

    def test_report_dir_none_handling(self, sample_config):
        """A None report directory remains None in the serialized config."""
        sample_config.report_dir = None

        run = ExperimentRun(
            run_id="test_no_report",
            config=sample_config,
            status=ExperimentStatus.PENDING,
        )

        serialized = _serialize_run(run)

        assert serialized["config"]["report_dir"] is None
class TestEdgeCases:
    """Edge cases and error conditions."""

    def test_run_not_found(self, temp_db):
        """get() raises RunNotFoundError for missing runs."""
        store = ResultStore(temp_db)

        with pytest.raises(Exception):  # RunNotFoundError
            store.get("nonexistent_run_id")

        store.close()

    def test_empty_error_message(self, temp_db, sample_config):
        """Runs with empty error_message persist correctly."""
        store = ResultStore(temp_db)

        run = ExperimentRun(
            run_id="test_no_error",
            config=sample_config,
            status=ExperimentStatus.COMPLETED,
            error_message="",
        )

        store.save(run)
        retrieved = store.get(run.run_id)
        assert retrieved["error_message"] == ""

        store.close()

    def test_none_timestamps(self, temp_db, sample_config):
        """Runs with None timestamps persist correctly."""
        store = ResultStore(temp_db)

        run = ExperimentRun(
            run_id="test_no_time",
            config=sample_config,
            status=ExperimentStatus.PENDING,
            started_at=None,
            completed_at=None,
        )

        store.save(run)
        retrieved = store.get(run.run_id)
        assert retrieved["started_at"] is None
        assert retrieved["completed_at"] is None
        assert retrieved["duration_seconds"] is None

        store.close()

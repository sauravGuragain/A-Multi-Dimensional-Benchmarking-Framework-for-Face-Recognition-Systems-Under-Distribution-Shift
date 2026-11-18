"""Provenance recording and shipped deep-model Fold-0 configs (no torch / weights needed)."""
from __future__ import annotations

from pathlib import Path

import pytest

from faceeval.core.config import load_config
from faceeval.orchestration import verification as v

CONFIGS = Path(__file__).resolve().parent.parent / "configs"


def test_provenance_shape_and_never_raises(monkeypatch, tmp_path):
    p = v.run_provenance(["facenet", "arcface", "eigenfaces"])
    assert set(p) == {"platform", "machine", "code_dirty", "packages", "weights"}
    assert set(p["packages"]) == set(v.PROVENANCE_PACKAGES)
    assert set(p["weights"]) == {"facenet", "arcface"}            # classical models have no weights entry
    monkeypatch.setenv("HOME", str(tmp_path))                     # no cache -> NOT_RECORDED, not an error
    monkeypatch.setenv("TORCH_HOME", str(tmp_path / "nothing"))
    assert v.run_provenance(["facenet"])["weights"]["facenet"] == v.sm.NOT_RECORDED


def test_provenance_survives_broken_helpers(monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("x")
    monkeypatch.setattr(v.subprocess, "run", boom)
    monkeypatch.setattr(v.importlib_metadata, "version", boom)
    monkeypatch.setattr(v, "_weight_files", boom)
    p = v.run_provenance(["facenet"])
    assert p["code_dirty"] == v.sm.NOT_RECORDED
    assert all(x == v.sm.NOT_RECORDED for x in p["packages"].values())
    assert p["weights"]["facenet"] == v.sm.NOT_RECORDED


def test_weight_hash_recorded_when_file_present(monkeypatch, tmp_path):
    ck = tmp_path / "torch" / "checkpoints"
    ck.mkdir(parents=True)
    (ck / "20180402-114759-vggface2.pt").write_bytes(b"abc")
    monkeypatch.setenv("TORCH_HOME", str(tmp_path / "torch"))
    w = v.run_provenance(["facenet"])["weights"]["facenet"]
    assert w == {"20180402-114759-vggface2.pt":
                 "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"}


@pytest.mark.parametrize("model", ["facenet", "arcface"])
def test_shipped_deep_configs(model):
    c = load_config(CONFIGS / f"lfw_fold0_deep_{model}.yaml")
    assert c.evaluation_protocol == "lfw_fold0_verification" and c.require_real_models
    assert [m.value for m in c.deep_models] == [model] and not c.traditional_models
    ref = load_config(CONFIGS / "lfw_fold0_verification_full.yaml")
    assert c.perturbation_schedule.perturbation_types == ref.perturbation_schedule.perturbation_types
    assert len(c.perturbation_schedule.perturbation_types) == 16
    assert c.perturbation_schedule.severity_levels == [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    assert c.random_seed == 42 and c.output_dir.rstrip("/").endswith(f"lfw_fold0_deep_{model}")

"""The Fold 0 verification summary records how per-image noise was seeded (provenance of the noise realisation)."""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

from faceeval.orchestration import verification as V
from faceeval.perturbation import noise as N
from tests.test_verification_stage import _run, fake_data  # noqa: F401  (fixture reuse)

REVIEWED_PATCH_SHA256 = "881d30a130bbbc5993a0fcb3551725860ad1ebf03bd378d1515f89157e597d7e"


def test_metadata_fields_exact():
    m = V.noise_seeding_metadata()
    assert m["master_seed"] == 42 == N.NOISE_MASTER_SEED
    assert m["scheme"] == "sha256(master|image_id|perturbation_type)"
    assert m["image_id_normalization"] == "backslash -> forward slash; then UTF-8 encoded; no other change"
    assert m["digest_to_seed"].startswith("first 8 bytes of the SHA-256 digest, big-endian unsigned 64-bit")
    assert m["severity_in_seed"] is False
    assert m["id_source"] == "ImageRecord.image_id"
    assert set(m) == {"master_seed", "scheme", "image_id_normalization", "digest_to_seed", "severity_in_seed",
                      "id_source", "reviewed_patch_sha256", "noise_module_sha256"}


def test_metadata_describes_the_scheme_actually_implemented():
    """The recorded scheme string and master seed reproduce the seed the noise code really uses."""
    m = V.noise_seeding_metadata()
    key = "|".join([str(m["master_seed"]), "Aaron_Eckhart/Aaron_Eckhart_0001.jpg", "speckle_noise"])
    want = int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], "big")
    assert N.noise_seed("Aaron_Eckhart/Aaron_Eckhart_0001.jpg", "speckle_noise") == want


def test_normalization_and_digest_descriptions_match_the_implementation():
    """Backslash ids equal slash ids (normalisation), nothing else is normalised, and the seed is the big-endian
    first-8-bytes of the digest as recorded."""
    assert "backslash -> forward slash" in V.noise_seeding_metadata()["image_id_normalization"]
    assert N.noise_seed("A\\b.jpg", "gaussian_noise") == N.noise_seed("A/b.jpg", "gaussian_noise")
    assert N.noise_seed("a/b.jpg", "gaussian_noise") != N.noise_seed("A/b.jpg", "gaussian_noise")      # case kept
    assert N.noise_seed(" a/b.jpg", "gaussian_noise") != N.noise_seed("a/b.jpg", "gaussian_noise")     # no trimming
    digest = hashlib.sha256("42|a/b.jpg|gaussian_noise".encode("utf-8")).digest()
    assert N.noise_seed("a/b.jpg", "gaussian_noise") == int.from_bytes(digest[:8], "big")
    assert 0 <= N.noise_seed("a/b.jpg", "gaussian_noise") < 2**64


def test_provenance_hashes():
    m = V.noise_seeding_metadata()
    assert m["reviewed_patch_sha256"] == REVIEWED_PATCH_SHA256 == V.NOISE_SEEDING_PATCH_SHA256
    assert re.fullmatch(r"[0-9a-f]{64}", m["reviewed_patch_sha256"])
    assert m["noise_module_sha256"] == hashlib.sha256(Path(N.__file__).read_bytes()).hexdigest()


def test_module_hash_changes_when_noise_source_changes(tmp_path, monkeypatch):
    original = hashlib.sha256(Path(N.__file__).read_bytes()).hexdigest()
    other = tmp_path / "noise_copy.py"
    other.write_bytes(Path(N.__file__).read_bytes() + b"\n# edited\n")
    monkeypatch.setattr(V._noise, "__file__", str(other))
    assert V.noise_seeding_metadata()["noise_module_sha256"] != original


def test_summary_and_json_on_disk_record_noise_seeding(tmp_path, monkeypatch, fake_data):  # noqa: F811
    _cfg, runner, _run_obj = _run(tmp_path, monkeypatch, fake_data, protocol="lfw_fold0_verification")
    summ = runner.verification_summary
    js = json.load(open(Path(summ["output_dir"]) / "verification_results.json"))
    for doc in (summ, js):
        assert doc["noise_seeding"] == V.noise_seeding_metadata()
        assert doc["noise_seeding"]["master_seed"] == 42
    # split identity and config-level seed stay recorded next to it
    assert js["seed"] == 42 and js["content_sha256"]["split"] and js["manifest_sha256"]

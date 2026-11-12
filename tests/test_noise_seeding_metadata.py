"""The Fold 0 verification summary records how per-image noise was seeded (provenance of the noise realisation)."""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pytest

from faceeval.orchestration import verification as V
from faceeval.perturbation import noise as N
from tests.test_verification_stage import _run, fake_data  # noqa: F401  (fixture reuse)

def test_metadata_fields_exact():
    m = V.noise_seeding_metadata()
    assert m["protocol_version"] == N.NOISE_PROTOCOL_VERSION == "per_image_v2"
    assert m["master_seed"] == 42 == N.NOISE_MASTER_SEED
    assert m["scheme"] == "sha256(master|image_id|perturbation_type)"
    assert m["image_id_normalization"] == "Unicode NFC; then backslash -> forward slash; then UTF-8 encoded; no other change"
    assert m["digest_to_seed"].startswith("first 8 bytes of the SHA-256 digest, big-endian unsigned 64-bit")
    assert m["severity_in_seed"] is False
    assert m["id_source"] == "ImageRecord.image_id"
    assert m["perturbation_types"] == ["gaussian_noise", "salt_pepper_noise", "speckle_noise"]
    assert m["numpy_version"] == np.__version__
    assert m["stream_fingerprint"] == N.noise_stream_fingerprint()
    assert "polarity independent of severity" in m["salt_pepper"]
    assert set(m) == {"protocol_version", "master_seed", "scheme", "image_id_normalization", "digest_to_seed",
                      "severity_in_seed", "id_source", "perturbation_types", "salt_pepper", "numpy_version",
                      "stream_fingerprint", "noise_module_sha256", "base_module_sha256"}


def test_metadata_describes_the_scheme_actually_implemented():
    """The recorded scheme string and master seed reproduce the seed the noise code really uses."""
    m = V.noise_seeding_metadata()
    key = "|".join([str(m["master_seed"]), "Aaron_Eckhart/Aaron_Eckhart_0001.jpg", "speckle_noise"])
    want = int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], "big")
    assert N.noise_seed("Aaron_Eckhart/Aaron_Eckhart_0001.jpg", "speckle_noise") == want


def test_normalization_and_digest_descriptions_match_the_implementation():
    """Backslash ids equal slash ids (normalisation), nothing else is normalised, and the seed is the big-endian
    first-8-bytes of the digest as recorded."""
    assert "NFC" in V.noise_seeding_metadata()["image_id_normalization"]
    assert "backslash -> forward slash" in V.noise_seeding_metadata()["image_id_normalization"]
    assert N.noise_seed("A\\b.jpg", "gaussian_noise") == N.noise_seed("A/b.jpg", "gaussian_noise")
    assert N.noise_seed("a/b.jpg", "gaussian_noise") != N.noise_seed("A/b.jpg", "gaussian_noise")      # case kept
    assert N.noise_seed(" a/b.jpg", "gaussian_noise") != N.noise_seed("a/b.jpg", "gaussian_noise")     # no trimming
    digest = hashlib.sha256("42|a/b.jpg|gaussian_noise".encode("utf-8")).digest()
    assert N.noise_seed("a/b.jpg", "gaussian_noise") == int.from_bytes(digest[:8], "big")
    assert 0 <= N.noise_seed("a/b.jpg", "gaussian_noise") < 2**64


def test_provenance_hashes_are_the_live_module_fingerprints():
    from faceeval.perturbation import base as B
    m = V.noise_seeding_metadata()
    for key in ("noise_module_sha256", "base_module_sha256"):
        assert re.fullmatch(r"[0-9a-f]{64}", m[key])
    assert m["noise_module_sha256"] == hashlib.sha256(Path(N.__file__).read_bytes()).hexdigest()
    assert m["base_module_sha256"] == hashlib.sha256(Path(B.__file__).read_bytes()).hexdigest()
    assert m["noise_module_sha256"] != m["base_module_sha256"]


@pytest.mark.parametrize("module_attr,key,other_key", [
    ("_noise", "noise_module_sha256", "base_module_sha256"),
    ("_base", "base_module_sha256", "noise_module_sha256"),
])
def test_each_fingerprint_changes_when_its_source_changes(tmp_path, monkeypatch, module_attr, key, other_key):
    """Changing base.py (where image-id normalisation lives) must change a recorded hash -- it cannot go undetected."""
    before = V.noise_seeding_metadata()
    module = getattr(V, module_attr)
    edited = tmp_path / "edited.py"
    edited.write_bytes(Path(module.__file__).read_bytes() + b"\n# edited\n")
    monkeypatch.setattr(module, "__file__", str(edited))
    after = V.noise_seeding_metadata()
    assert after[key] != before[key] and after[other_key] == before[other_key]


def test_normalize_image_id_lives_in_the_fingerprinted_base_module():
    from faceeval.perturbation import base as B
    assert B.normalize_image_id.__module__ == "faceeval.perturbation.base"
    assert Path(B.__file__).read_bytes().count(b"def normalize_image_id") == 1


def test_stream_fingerprint_is_deterministic_interpretable_and_matches_reference():
    """General-purpose: holds for whatever approval policy is configured (base reference or a version-specific
    approval). Base-reference and version-specific behaviour are pinned by the dedicated tests below."""
    fp = N.noise_stream_fingerprint()
    assert fp == N.noise_stream_fingerprint()
    rng = np.random.default_rng(42)
    want = hashlib.sha256(rng.random(8).astype("<f8").tobytes() + rng.normal(size=8).astype("<f8").tobytes()).hexdigest()
    assert fp["sha256"] == want
    assert fp["bit_generator"] == type(np.random.default_rng(42).bit_generator).__name__      # the generator actually used
    assert fp["numpy_version"] == np.__version__
    assert "random(8) then normal(size=8)" in fp["probe"]
    approved = N.approved_stream_reference()
    assert fp["approved_reference"] == approved
    assert fp["matches_approved_reference"] is (fp["sha256"] == approved["sha256"])


def test_base_reference_policy_when_no_version_is_approved(monkeypatch):
    monkeypatch.setattr(N, "NOISE_STREAM_APPROVED_BY_VERSION", {})
    fp = N.noise_stream_fingerprint()
    assert fp["approved_reference"] == {**N.NOISE_STREAM_REFERENCE, "source": "base_reference"}
    assert fp["approved_reference"]["bit_generator"] == "PCG64"
    assert fp["matches_approved_reference"] is (fp["sha256"] == N.NOISE_STREAM_REFERENCE["sha256"])


def test_stream_fingerprint_flags_a_changed_stream(monkeypatch):
    monkeypatch.setattr(N, "NOISE_STREAM_APPROVED_BY_VERSION", {})       # isolate from any real approval
    monkeypatch.setitem(N.NOISE_STREAM_REFERENCE, "sha256", "0" * 64)
    fp = N.noise_stream_fingerprint()
    assert fp["matches_approved_reference"] is False and fp["sha256"] != "0" * 64      # recorded, never hidden
    assert fp["approved_reference"]["source"] == "base_reference"


def _drifted(monkeypatch):
    """Simulate a NumPy whose stream differs from the base reference; return the stream's real digest."""
    digest = N.noise_stream_fingerprint()["sha256"]
    monkeypatch.setattr(N, "NOISE_STREAM_APPROVED_BY_VERSION", {})       # isolate from any real approval
    monkeypatch.setitem(N.NOISE_STREAM_REFERENCE, "sha256", "0" * 64)
    return digest


def test_an_explicitly_approved_version_matches_its_own_approved_stream(monkeypatch):
    """The reported inconsistency: a version approved in NOISE_STREAM_APPROVED_BY_VERSION must be judged against its
    approval (not the base reference), in the fingerprint itself AND in the run metadata."""
    digest = _drifted(monkeypatch)
    assert N.noise_stream_fingerprint()["matches_approved_reference"] is False          # unapproved drift is flagged
    monkeypatch.setitem(N.NOISE_STREAM_APPROVED_BY_VERSION, np.__version__, digest)
    for fp in (N.noise_stream_fingerprint(), V.noise_seeding_metadata()["stream_fingerprint"]):
        assert fp["matches_approved_reference"] is True
        assert fp["approved_reference"] == {"numpy_version": np.__version__, "sha256": digest,
                                            "source": "version_specific_approval"}
        assert "bit_generator" not in fp["approved_reference"]          # an approval makes no generator claim
        assert fp["bit_generator"] == type(np.random.default_rng(42).bit_generator).__name__   # generator actually used
        assert fp["sha256"] == digest


def test_an_approval_never_accepts_an_unknown_stream(monkeypatch):
    digest = _drifted(monkeypatch)
    # approval recorded for this version but for a DIFFERENT stream -> still flagged
    monkeypatch.setitem(N.NOISE_STREAM_APPROVED_BY_VERSION, np.__version__, "1" * 64)
    fp = N.noise_stream_fingerprint()
    assert fp["matches_approved_reference"] is False and fp["sha256"] == digest
    assert fp["approved_reference"]["sha256"] == "1" * 64 and fp["approved_reference"]["source"] == "version_specific_approval"
    # approval for another version does not apply to this one
    monkeypatch.delitem(N.NOISE_STREAM_APPROVED_BY_VERSION, np.__version__)
    monkeypatch.setitem(N.NOISE_STREAM_APPROVED_BY_VERSION, "0.0.0", digest)
    fp = N.noise_stream_fingerprint()
    assert fp["matches_approved_reference"] is False and fp["approved_reference"]["source"] == "base_reference"


def test_approved_stream_reference_lookup_by_version(monkeypatch):
    base = N.approved_stream_reference("9.9.9")
    assert base["source"] == "base_reference" and base["sha256"] == N.NOISE_STREAM_REFERENCE["sha256"]
    monkeypatch.setitem(N.NOISE_STREAM_APPROVED_BY_VERSION, "9.9.9", "2" * 64)
    assert N.approved_stream_reference("9.9.9") == {"numpy_version": "9.9.9", "sha256": "2" * 64,
                                                    "source": "version_specific_approval"}
    assert N.approved_stream_reference("9.9.8")["source"] == "base_reference"


def test_id_seeded_types_are_derived_from_the_registry_and_match_perturb_batch_behaviour():
    from faceeval.core.registry import perturbation_registry
    from faceeval.core.types import PerturbationType as PT
    from faceeval.orchestration.perturbed_inputs import perturb_batch
    from faceeval.perturbation.composer import PerturbationComposer
    derived = V._id_seeded_perturbation_types()
    flagged = sorted(k for k in perturbation_registry.list_registered() if perturbation_registry.get(k).USES_IMAGE_ID)
    assert derived == flagged == ["gaussian_noise", "salt_pepper_noise", "speckle_noise"]
    c = PerturbationComposer(severity_levels=[0.0, 0.5])
    x = [np.full((16, 16, 3), 100, np.uint8)]
    for pt in PT:                                      # exactly the derived types demand ids in the benchmark path
        spec = c.specs_for_type(pt)[1]
        if pt.value in derived:
            with pytest.raises(ValueError):
                perturb_batch(c, spec, None, x, None)
        else:
            perturb_batch(c, spec, None, x, None)


def test_summary_and_json_on_disk_record_noise_seeding(tmp_path, monkeypatch, fake_data):  # noqa: F811
    _cfg, runner, _run_obj = _run(tmp_path, monkeypatch, fake_data, protocol="lfw_fold0_verification")
    summ = runner.verification_summary
    js = json.load(open(Path(summ["output_dir"]) / "verification_results.json"))
    for doc in (summ, js):
        assert doc["noise_seeding"] == V.noise_seeding_metadata()
        assert doc["noise_seeding"]["master_seed"] == 42
    # split identity and config-level seed stay recorded next to it
    assert js["seed"] == 42 and js["content_sha256"]["split"] and js["manifest_sha256"]

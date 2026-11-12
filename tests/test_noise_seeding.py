"""Per-image deterministic noise seeding: seed = SHA-256(master | image_id | perturbation type), severity-free."""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import unicodedata
from pathlib import Path

import numpy as np
import pytest

from faceeval.core.types import PerturbationType as PT
from faceeval.perturbation import noise as N
from faceeval.perturbation.composer import PerturbationComposer
from faceeval.core.exceptions import PerturbationError
from faceeval.orchestration.perturbed_inputs import perturb_batch
from faceeval.perturbation.base import validate_image_ids
from tests.test_perturbation_value_range import _face_u8

# Approved reference random stream, produced with NumPy 2.5.3 (PCG64). It is the stream behind every noise value.
GOLDEN_REFERENCE_NUMPY = "2.5.3"
GOLDEN = {
    "uniform": [0.07812690025788827, 0.9619213243373694, 0.2530243832282263],     # default_rng(noise_seed("a.jpg","gaussian_noise")).random(3)
    "normal": [0.33435510138103786, 1.1909245966617579, 0.2540350258999167],      # ... then .normal(size=3)
}
# Derived-stream golden values for a NumPy version whose stream differs and has been explicitly approved:
# {"<numpy version>": {"uniform": [...], "normal": [...]}}. It must list exactly the versions approved in
# faceeval.perturbation.noise.NOISE_STREAM_APPROVED_BY_VERSION (checked below), whose sha256 is the single approved
# fingerprint used by both the run metadata and this test. Empty on purpose: add an entry ONLY after a reviewed
# decision that the new stream is acceptable (and NOISE_PROTOCOL_VERSION has been bumped); never edit GOLDEN or add
# an entry just to make a test pass.
APPROVED_VERSION_GOLDENS: dict[str, dict] = {}
NOISE = [PT.GAUSSIAN_NOISE, PT.SALT_PEPPER_NOISE, PT.SPECKLE_NOISE]
SEVS = [0.2, 0.4, 0.6, 0.8, 1.0]


@pytest.fixture(scope="module")
def composer():
    return PerturbationComposer(severity_levels=[0.0] + SEVS)


@pytest.fixture(scope="module")
def img():
    return _face_u8()


def run(composer, pt, sev, image, image_id):
    spec = next(s for s in composer.specs_for_type(pt) if abs(s.severity - sev) < 1e-9)
    return composer._get_perturbation(pt).apply(image, spec.severity, image_id=image_id)


def test_seed_matches_independent_sha256():
    key = "42|Aaron_Eckhart/Aaron_Eckhart_0001.jpg|gaussian_noise"
    want = int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], "big")
    assert N.noise_seed("Aaron_Eckhart/Aaron_Eckhart_0001.jpg", "gaussian_noise") == want
    assert N.noise_seed("a\\b.jpg", "x") == N.noise_seed("a/b.jpg", "x")     # path separator neutral


def test_seed_ignores_severity_and_depends_on_type_and_id():
    assert N.noise_seed("a.jpg", "gaussian_noise") != N.noise_seed("b.jpg", "gaussian_noise")
    assert N.noise_seed("a.jpg", "gaussian_noise") != N.noise_seed("a.jpg", "speckle_noise")


@pytest.mark.parametrize("pt", NOISE)
def test_deterministic_reproducibility(composer, img, pt):
    a = run(composer, pt, 0.6, img, "p/x.jpg")
    b = run(composer, pt, 0.6, img.copy(), "p/x.jpg")
    assert np.array_equal(a, b)


@pytest.mark.parametrize("pt", NOISE)
def test_different_ids_give_different_noise(composer, img, pt):
    a = run(composer, pt, 0.8, img, "p/x.jpg")
    b = run(composer, pt, 0.8, img, "q/y.jpg")
    assert not np.array_equal(a, b)


@pytest.mark.parametrize("pt", NOISE)
def test_pair_images_get_independent_noise(composer, img, pt):
    """Genuine pair (same person, two files) and impostor pair (two people), even for pixel-identical inputs."""
    for ids in (("Ann/Ann_0001.jpg", "Ann/Ann_0002.jpg"), ("Ann/Ann_0001.jpg", "Bob/Bob_0001.jpg")):
        n1 = run(composer, pt, 1.0, img, ids[0]).astype(np.float64) - img
        n2 = run(composer, pt, 1.0, img, ids[1]).astype(np.float64) - img
        assert not np.array_equal(n1, n2)
        if pt is PT.GAUSSIAN_NOISE:
            assert abs(np.corrcoef(n1.ravel(), n2.ravel())[0, 1]) < 0.05


def test_gaussian_pattern_consistent_across_severities(composer, img):
    base = img.astype(np.float64)
    d = {s: (run(composer, PT.GAUSSIAN_NOISE, s, img, "p/x.jpg").astype(np.float64) - base).ravel() for s in (0.2, 0.4)}
    assert np.corrcoef(d[0.2], d[0.4])[0, 1] > 0.9


def test_speckle_pattern_consistent_across_severities(composer, img):
    base = img.astype(np.float64)
    d = {s: (run(composer, PT.SPECKLE_NOISE, s, img, "p/x.jpg").astype(np.float64) - base).ravel() for s in (0.2, 0.8)}
    assert np.corrcoef(d[0.2], d[0.8])[0, 1] > 0.9


def test_salt_pepper_corrupted_set_nested_across_severities(composer):
    flat = np.full((112, 112, 3), 128, np.uint8)
    prev = None
    for s in SEVS:
        cur = run(composer, PT.SALT_PEPPER_NOISE, s, flat, "p/x.jpg")[..., 0] != 128
        if prev is not None:
            assert (prev & ~cur).sum() == 0
        prev = cur
    assert prev.sum() > 0


@pytest.mark.parametrize("pt", NOISE)
def test_same_noise_for_uint8_and_float_with_same_id(composer, img, pt):
    a = run(composer, pt, 0.6, img, "p/x.jpg").astype(np.float64) / 255
    b = run(composer, pt, 0.6, img.astype(np.float32) / 255, "p/x.jpg").astype(np.float64)
    assert np.abs(a - b).max() < 0.01


@pytest.mark.parametrize("pt", NOISE)
def test_no_id_fallback_is_deterministic_and_content_dependent(composer, img, pt):
    a = run(composer, pt, 0.6, img, None)
    assert np.array_equal(a, run(composer, pt, 0.6, img.copy(), None))
    other = np.clip(img.astype(int) + 1, 0, 255).astype(np.uint8)
    assert not np.array_equal(a - img, run(composer, pt, 0.6, other, None) - other)


def test_apply_spec_requires_matching_ids(composer, img):
    spec = composer.specs_for_type(PT.GAUSSIAN_NOISE)[1]
    with pytest.raises(ValueError):
        composer.apply_spec([img, img], ["s", "s"], ["a"], spec)


def test_apply_spec_uses_ids(composer, img):
    spec = composer.specs_for_type(PT.GAUSSIAN_NOISE)[2]
    o = composer.apply_spec([img, img], ["s", "s"], ["a.jpg", "b.jpg"], spec).image_tensors
    assert not np.array_equal(o[0], o[1])


def test_other_perturbations_unaffected_by_image_id(composer, img):
    for pt in PT:
        if pt in NOISE:
            continue
        for s in (0.4, 1.0):
            spec = next(x for x in composer.specs_for_type(pt) if abs(x.severity - s) < 1e-9)
            p = composer._get_perturbation(pt)
            assert np.array_equal(p.apply(img, s), p.apply(img, s, image_id="z/z.jpg")), pt


def test_verification_path_passes_unique_record_ids(tmp_path, monkeypatch):
    import faceeval.orchestration.verification as V
    from faceeval.core.registry import model_registry  # noqa: F401
    from tests.test_perturbation_value_range import test_verification_path_feeds_models_sane_perturbed_inputs as base
    seen = []
    real = V.perturb_batch

    def spy(composer, spec, normalizer, X, chips, image_ids=None):
        seen.append((len(X), image_ids))
        return real(composer, spec, normalizer, X, chips, image_ids)
    monkeypatch.setattr(V, "perturb_batch", spy)
    base(tmp_path, monkeypatch)
    assert seen
    for n, ids in seen:
        assert ids is not None and len(ids) == n and len(set(ids)) == n


# ---------------------------------------------------------------------------------------------------------------
# Severity scaling on float inputs (no clipping / quantisation, so the scaling is exact)
# ---------------------------------------------------------------------------------------------------------------

@pytest.fixture(scope="module")
def fimg():
    return (np.random.default_rng(1).random((112, 112, 3)) * 0.6 + 0.2).astype(np.float32)


def _scaled_ratio(composer, pt, lo, hi, image):
    """Per-pixel residual ratio hi/lo on float input, restricted to pixels that are neither clipped nor
    dominated by float32 rounding, so any deviation from the expected scaling is a real one."""
    o_lo, o_hi = (run(composer, pt, s, image, "P/x.jpg").astype(np.float64) for s in (lo, hi))
    a, b = (o_lo - image).ravel(), (o_hi - image).ravel()
    ok = (o_lo.ravel() > 0) & (o_lo.ravel() < 1) & (o_hi.ravel() > 0) & (o_hi.ravel() < 1) & (np.abs(a) > 1e-2)
    assert ok.sum() > 10_000
    return b[ok] / a[ok]


def test_gaussian_severity_scaling_is_exact_on_float_input(composer, fimg):
    ratio = _scaled_ratio(composer, PT.GAUSSIAN_NOISE, 0.2, 0.8, fimg)
    assert abs(ratio.mean() - 4.0) < 0.005 and ratio.std() < 0.01         # std is linear in severity: 0.8/0.2 = 4


def test_speckle_severity_scaling_follows_sqrt_variance(composer, fimg):
    ratio = _scaled_ratio(composer, PT.SPECKLE_NOISE, 0.2, 0.8, fimg)
    assert abs(ratio.mean() - 2.0) < 0.005 and ratio.std() < 0.01         # variance linear -> std ~ sqrt(sev): sqrt(4) = 2


# ---------------------------------------------------------------------------------------------------------------
# Salt-and-pepper: nested locations AND stable polarity (approved protocol change)
# ---------------------------------------------------------------------------------------------------------------

def test_salt_pepper_polarity_is_stable_across_severity(composer):
    flat = np.full((112, 112, 3), 128, np.uint8)
    outs = {s: run(composer, PT.SALT_PEPPER_NOISE, s, flat, "p/x.jpg")[..., 0] for s in SEVS}
    for lo, hi in zip(SEVS, SEVS[1:]):
        hit = outs[lo] != 128
        assert hit.any()
        assert np.array_equal(outs[hi][hit], outs[lo][hit]), f"polarity changed between severity {lo} and {hi}"
        assert (outs[hi] != 128).sum() > hit.sum()                          # and density really grows


def test_salt_pepper_marginals_density_and_fifty_fifty(composer):
    flat = np.full((112, 112, 3), 128, np.uint8)
    n = 112 * 112
    for sev in (0.4, 1.0):
        out = run(composer, PT.SALT_PEPPER_NOISE, sev, flat, "p/x.jpg")[..., 0]
        density = sev * 0.20
        salt, pepper = (out == 255).sum(), (out == 0).sum()
        assert abs((salt + pepper) / n - density) < 0.02
        assert abs(salt / (salt + pepper) - 0.5) < 0.05


def test_salt_pepper_float_input_uses_zero_and_one(composer):
    flat = np.full((64, 64), 0.5, np.float32)
    out = run(composer, PT.SALT_PEPPER_NOISE, 1.0, flat, "p/x.jpg")
    assert set(np.unique(out)) == {0.0, 0.5, 1.0}


# ---------------------------------------------------------------------------------------------------------------
# Repeated runs, cross-process, golden stream values
# ---------------------------------------------------------------------------------------------------------------

_CHILD = """
import hashlib, logging, sys
logging.disable(50)
import numpy as np
import faceeval.perturbation
from faceeval.core.types import PerturbationType as PT
from faceeval.perturbation.composer import PerturbationComposer
from faceeval.perturbation.noise import noise_seed
c = PerturbationComposer(severity_levels=[0.0, 0.5, 1.0])
x = (np.random.default_rng(3).random((64, 64, 3)) * 200 + 20).astype(np.uint8)
h = hashlib.sha256()
for t in (PT.GAUSSIAN_NOISE, PT.SALT_PEPPER_NOISE, PT.SPECKLE_NOISE):
    for s in (0.5, 1.0):
        h.update(c._get_perturbation(t).apply(x, s, image_id="Aaron_Eckhart/Aaron_Eckhart_0001.jpg").tobytes())
    h.update(str(noise_seed("Zo\u00eb/a.jpg", t.value)).encode())
print(h.hexdigest())
"""


def _digest_here():
    ns: dict = {}
    exec(compile(_CHILD.replace("print(h.hexdigest())", "RESULT = h.hexdigest()"), "child", "exec"), ns)
    return ns["RESULT"]


def test_cross_process_determinism_and_repeated_runs():
    here = _digest_here()
    assert here == _digest_here()                                           # repeated run in-process
    root = str(Path(__file__).resolve().parents[1])
    outs = []
    for hashseed in ("1", "12345"):                                         # different hash randomisation per process
        env = {**os.environ, "PYTHONHASHSEED": hashseed, "PYTHONPATH": root, "PYTHONDONTWRITEBYTECODE": "1"}
        r = subprocess.run([sys.executable, "-c", _CHILD], capture_output=True, text=True, env=env, cwd=root, timeout=300)
        assert r.returncode == 0, r.stderr[-800:]
        outs.append(r.stdout.strip())
    assert outs[0] == outs[1] == here


def test_golden_and_approval_tables_are_consistent_with_the_noise_module():
    assert GOLDEN_REFERENCE_NUMPY == N.NOISE_STREAM_REFERENCE["numpy_version"]
    assert set(APPROVED_VERSION_GOLDENS) == set(N.NOISE_STREAM_APPROVED_BY_VERSION), (
        "a NumPy version must be approved in BOTH faceeval.perturbation.noise.NOISE_STREAM_APPROVED_BY_VERSION "
        "(fingerprint) and tests APPROVED_VERSION_GOLDENS (derived stream values), or in neither")


def test_golden_stream_values_detect_numpy_drift():
    """Sentinel for the exact random stream behind the noise. The criterion is the stream itself, not the version
    string: a different NumPy whose stream is identical passes; a changed stream fails with an explanation. The
    approved fingerprint comes from the same policy the run metadata uses (``approved_stream_reference``)."""
    approved = N.approved_stream_reference()
    if approved["source"] == "version_specific_approval":
        assert np.__version__ in APPROVED_VERSION_GOLDENS, (
            f"NumPy {np.__version__} is approved in NOISE_STREAM_APPROVED_BY_VERSION but tests/test_noise_seeding.py "
            f"APPROVED_VERSION_GOLDENS has no derived-stream values for it")
        ref = APPROVED_VERSION_GOLDENS[np.__version__]
        source = f"the approved entry for NumPy {np.__version__}"
    else:
        ref = GOLDEN
        source = f"the reference generated with NumPy {GOLDEN_REFERENCE_NUMPY}"
    why = (f"NumPy {np.__version__} draws a different random stream than {source}. Every previously recorded noise "
           f"realisation would change and results would no longer be reproducible across environments. Do NOT edit "
           f"the golden values to make this pass: review the change, bump NOISE_PROTOCOL_VERSION, and approve this "
           f"NumPy version explicitly in NOISE_STREAM_APPROVED_BY_VERSION (noise module) and APPROVED_VERSION_GOLDENS "
           f"(this test file).")
    assert N.noise_seed("a.jpg", "gaussian_noise") == int.from_bytes(
        hashlib.sha256(b"42|a.jpg|gaussian_noise").digest()[:8], "big")
    rng = np.random.default_rng(N.noise_seed("a.jpg", "gaussian_noise"))
    assert np.allclose(rng.random(3), ref["uniform"], atol=0, rtol=1e-12), why
    assert np.allclose(rng.normal(size=3), ref["normal"], atol=0, rtol=1e-12), why
    fp = N.noise_stream_fingerprint()
    assert fp["sha256"] == approved["sha256"], why
    assert fp["matches_approved_reference"] is True, why


# ---------------------------------------------------------------------------------------------------------------
# Unicode / path-separator normalisation
# ---------------------------------------------------------------------------------------------------------------

def test_canonically_equivalent_unicode_ids_get_identical_seed_and_noise(composer, img):
    nfc = unicodedata.normalize("NFC", "Zo\u00eb_Saldana/Zo\u00eb_Saldana_0001.jpg")
    nfd = unicodedata.normalize("NFD", nfc)
    assert nfc != nfd and len(nfd) > len(nfc)
    for pt in NOISE:
        assert N.noise_seed(nfc, pt.value) == N.noise_seed(nfd, pt.value)
        assert np.array_equal(run(composer, pt, 0.8, img, nfc), run(composer, pt, 0.8, img, nfd))
    assert N.noise_seed("Zoe_Saldana/x.jpg", "gaussian_noise") != N.noise_seed(nfc.replace("\u00eb", "e"), "speckle_noise")
    assert N.noise_seed("Zoe/x.jpg", "gaussian_noise") != N.noise_seed(nfc, "gaussian_noise")
    assert N.noise_seed("Zo\u00eb\\x.jpg", "speckle_noise") == N.noise_seed("Zo\u00eb/x.jpg", "speckle_noise")   # \ and /


def test_normalisation_changes_nothing_else():
    assert N.noise_seed("A/b.jpg", "speckle_noise") != N.noise_seed("a/b.jpg", "speckle_noise")        # case kept
    assert N.noise_seed(" a/b.jpg", "speckle_noise") != N.noise_seed("a/b.jpg", "speckle_noise")       # no trimming


# ---------------------------------------------------------------------------------------------------------------
# Invalid / duplicate ids are rejected (benchmark batches) -- legacy ad-hoc API still works
# ---------------------------------------------------------------------------------------------------------------

def _spec(composer, pt=PT.GAUSSIAN_NOISE, i=3):
    return composer.specs_for_type(pt)[i]


@pytest.mark.parametrize("bad,exc", [
    ([None, "b"], TypeError), ([5, "b"], TypeError), (["", "b"], ValueError), (["   ", "b"], ValueError),
    (["a", "a"], ValueError), (["a\\b", "a/b"], ValueError), (["Zo\u00eb", "Zoe\u0308"], ValueError),
    (["a"], ValueError), (["a", "b", "c"], ValueError), (None, ValueError),
])
def test_invalid_batch_ids_are_rejected(bad, exc):
    with pytest.raises(exc):
        validate_image_ids(bad, 2)


@pytest.mark.parametrize("pt", NOISE)
def test_perturb_batch_rejects_missing_and_bad_ids_for_noise(composer, img, pt):
    spec = _spec(composer, pt)
    for bad in (None, ["a"], ["a", "a"], ["", "b"], [None, "b"], [1, 2]):
        with pytest.raises((ValueError, TypeError)):
            perturb_batch(composer, spec, None, [img, img], None, bad)
    good = perturb_batch(composer, spec, None, [img, img], None, ["a.jpg", "b.jpg"])
    assert not np.array_equal(good[0], good[1])


def test_perturb_batch_still_accepts_no_ids_for_deterministic_perturbations(composer, img):
    out = perturb_batch(composer, _spec(composer, PT.GAUSSIAN_BLUR), None, [img, img], None)
    assert np.array_equal(out[0], out[1])
    with pytest.raises(ValueError):                                         # but supplied ids are always validated
        perturb_batch(composer, _spec(composer, PT.GAUSSIAN_BLUR), None, [img, img], None, ["a", "a"])


@pytest.mark.parametrize("bad,exc", [("", ValueError), ("  ", ValueError), (7, TypeError)])
def test_apply_rejects_a_supplied_bad_id_instead_of_falling_back(composer, img, bad, exc):
    with pytest.raises(exc):
        run(composer, PT.GAUSSIAN_NOISE, 0.6, img, bad)


def test_legacy_two_argument_api_still_works(composer, img):
    for pt in PT:
        p = composer._get_perturbation(pt)
        out = p.apply(img, 0.6)
        assert out.shape == img.shape and out.dtype == img.dtype
        assert np.array_equal(out, p.apply(img.copy(), 0.6))
    for pt in NOISE:                                    # severity 0 returns an unchanged COPY (new array, same values)
        out0 = composer._get_perturbation(pt).apply(img, 0.0)
        assert np.array_equal(out0, img) and out0 is not img


# ---------------------------------------------------------------------------------------------------------------
# Composer: strict failure handling, subject alignment, compose_specs
# ---------------------------------------------------------------------------------------------------------------

def test_apply_spec_is_strict_by_default_and_never_returns_a_silent_clean_image(composer, img, monkeypatch):
    p = composer._get_perturbation(PT.GAUSSIAN_NOISE)
    def boom(image, severity):
        raise RuntimeError("boom")
    monkeypatch.setattr(p, "_apply", boom)
    spec = _spec(composer)
    with pytest.raises(PerturbationError, match="a.jpg"):
        composer.apply_spec([img], ["s"], ["a.jpg"], spec)
    with pytest.raises(PerturbationError):
        composer.apply_all_specs([img], ["s"], ["a.jpg"], [spec])
    lenient = composer.apply_spec([img], ["s"], ["a.jpg"], spec, strict=False)    # explicit legacy mode only
    assert np.array_equal(lenient.image_tensors[0], img)


def test_apply_spec_validates_subject_alignment_and_ids(composer, img):
    spec = _spec(composer)
    with pytest.raises(ValueError):
        composer.apply_spec([img, img], ["s"], ["a", "b"], spec)
    with pytest.raises(ValueError):
        composer.apply_spec([img, img], ["s", "s"], ["a", "a"], spec)


def test_compose_specs_preserves_ids_and_is_reproducible(composer, img):
    chain = [_spec(composer, PT.BRIGHTNESS, 2), _spec(composer, PT.GAUSSIAN_NOISE, 3), _spec(composer, PT.SPECKLE_NOISE, 2)]
    a = composer.compose_specs([img, img], ["s1", "s2"], ["a.jpg", "b.jpg"], chain)
    b = composer.compose_specs([img, img], ["s1", "s2"], ["a.jpg", "b.jpg"], chain)
    assert a.original_ids == ["a.jpg", "b.jpg"] == b.original_ids
    assert all(np.array_equal(x, y) for x, y in zip(a.image_tensors, b.image_tensors))
    assert not np.array_equal(a.image_tensors[0], a.image_tensors[1])       # the id reached every stage
    swapped = composer.compose_specs([img, img], ["s1", "s2"], ["b.jpg", "a.jpg"], chain)
    assert np.array_equal(swapped.image_tensors[0], a.image_tensors[1])     # noise follows the id, not the position
    with pytest.raises(ValueError):
        composer.compose_specs([img, img], ["s1", "s2"], ["a.jpg", "a.jpg"], chain)


def test_runner_uses_record_ids_for_noise(composer, img):
    from types import SimpleNamespace
    from faceeval.orchestration.runner import ExperimentRunner
    from faceeval.preprocessing.normalizer import normalizer_for_model
    norm = normalizer_for_model("arcface")
    recs = [SimpleNamespace(image_id="Ann/Ann_0001.jpg"), SimpleNamespace(image_id="Bob/Bob_0001.jpg")]
    clean = [norm.normalize(img)] * 2
    out = ExperimentRunner._apply_perturbation(composer, _spec(composer), recs, clean, norm, [img, img])
    assert not np.array_equal(out[0], out[1])
    again = ExperimentRunner._apply_perturbation(composer, _spec(composer), recs, clean, norm, [img, img])
    assert all(np.array_equal(x, y) for x, y in zip(out, again))
    with pytest.raises(ValueError):
        ExperimentRunner._apply_perturbation(composer, _spec(composer), [recs[0], recs[0]], clean, norm, [img, img])


# ---------------------------------------------------------------------------------------------------------------
# End to end: repeated benchmark runs with all three noise types are identical
# ---------------------------------------------------------------------------------------------------------------

def test_end_to_end_noise_benchmark_is_deterministic_across_runs(tmp_path, monkeypatch):
    from faceeval.orchestration.runner import ExperimentRunner
    from tests.test_verification_stage import _config, _make_fake_lfw
    root = tmp_path / "data"
    _make_fake_lfw(root)
    summaries = []
    for tag in ("run_a", "run_b"):
        wd = tmp_path / tag
        wd.mkdir()
        monkeypatch.chdir(wd)
        cfg = _config(wd, root, "lfw_fold0_verification", deep=())
        cfg.perturbation_schedule.perturbation_types = list(NOISE)
        cfg.perturbation_schedule.severity_levels = [0.0, 0.5, 1.0]
        runner = ExperimentRunner(cfg)
        runner.run()
        assert runner.verification_summary["status"] == "COMPLETE"
        summaries.append(runner.verification_summary)
    ra, rb = (sm["results"] for sm in summaries)                            # no timing columns exist in the rows
    assert len(ra) == 2 * 3 * 3                                              # 2 models x 3 noise types x 3 severities
    assert {(r["model"], r["perturbation"], r["severity"]) for r in ra} == {
        (m, t, s_) for m in ("eigenfaces", "lbph") for t in ("gaussian_noise", "salt_pepper_noise", "speckle_noise")
        for s_ in (0.0, 0.5, 1.0)}
    dump = lambda rows: json.dumps(rows, sort_keys=True, default=str)       # NaN-safe (nan != nan under ==)
    assert dump(ra) == dump(rb)                                              # every metric, threshold and count identical
    assert summaries[0]["noise_seeding"] == summaries[1]["noise_seeding"]
    assert summaries[0]["noise_seeding"]["master_seed"] == 42
    # the noise really acted: for some model and noise type, a metric differs between clean (0.0) and severe (1.0)
    by = {(r["model"], r["perturbation"], r["severity"]): r for r in ra}
    assert any(by[(m, t, 0.0)]["auc"] != by[(m, t, 1.0)]["auc"] or by[(m, t, 0.0)]["eer"] != by[(m, t, 1.0)]["eer"]
               for (m, t, _s) in by)


# ---------------------------------------------------------------------------------------------------------------
# v3: ids are validated at every severity (including 0.0), stricter id checks, statistical independence,
# and the verification path hands noise the id that belongs to each image
# ---------------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("pt", NOISE + [PT.GAUSSIAN_BLUR, PT.BRIGHTNESS])
@pytest.mark.parametrize("bad,exc", [("", ValueError), ("   ", ValueError), (7, TypeError), (b"a", TypeError),
                                     ("a\x00b", ValueError), ("a\u200bb", ValueError)])
@pytest.mark.parametrize("sev", [0.0, 0.5])
def test_apply_rejects_invalid_supplied_id_at_every_severity(composer, img, pt, bad, exc, sev):
    with pytest.raises(exc):
        composer._get_perturbation(pt).apply(img, sev, image_id=bad)


@pytest.mark.parametrize("pt", NOISE)
def test_valid_id_at_severity_zero_returns_an_unchanged_copy(composer, img, pt):
    p = composer._get_perturbation(pt)
    for image_id in ("Aaron_Eckhart/Aaron_Eckhart_0001.jpg", None):
        out = p.apply(img, 0.0, image_id=image_id)
        assert np.array_equal(out, img) and out is not img and out.dtype == img.dtype


def test_invalid_severity_still_raises_first(composer, img):
    from faceeval.core.exceptions import InvalidSeverityError
    with pytest.raises(InvalidSeverityError):
        composer._get_perturbation(PT.GAUSSIAN_NOISE).apply(img, 1.5, image_id="a.jpg")


@pytest.mark.parametrize("bare", ["ab", "a.jpg", b"ab", bytearray(b"ab")])
def test_a_bare_string_is_rejected_as_the_id_collection(bare):
    with pytest.raises(TypeError, match="not a single"):
        validate_image_ids(bare, 2)


@pytest.mark.parametrize("bad", [
    "a\x00b", "a\nb", "a\tb", "a\x7fb", "a\x85b",                          # control characters (Cc)
    "a\u200bb", "a\u200cb", "a\u200db", "\ufeffa", "a\u202eb", "a\u2060b",  # zero-width / format (Cf)
    "a\ud800b",                                                          # lone surrogate (Cs)
])
def test_control_and_zero_width_characters_are_rejected_in_ids(bad):
    with pytest.raises(ValueError, match="character U\\+"):
        validate_image_ids([bad, "ok.jpg"], 2)


@pytest.mark.parametrize("good", [
    "Zo\u00eb_Saldana/Zo\u00eb_Saldana_0001.jpg", "Jos\u00e9_Mourinho/Jos\u00e9_Mourinho_0002.jpg",
    unicodedata.normalize("NFD", "Zo\u00eb/x.jpg"), "\u738b\u5c0f\u660e/\u738b_0001.jpg", "\u00d6zil/\u00d6zil_1.jpg",
    "Ren\u00e9e O'Brien-Smith Jr./a.b.jpg", "emoji_\U0001f642/x.jpg", "a b/c d.jpg", " leading/space.jpg",
])
def test_legitimate_unicode_ids_are_accepted_and_still_normalised(good, composer, img):
    validate_image_ids([good, "other/x.jpg"], 2)
    out = run(composer, PT.GAUSSIAN_NOISE, 0.6, img, good)
    nfc = unicodedata.normalize("NFC", good)
    assert np.array_equal(out, run(composer, PT.GAUSSIAN_NOISE, 0.6, img, nfc))          # NFC: same noise
    assert N.noise_seed(good, "speckle_noise") == N.noise_seed(good.replace("/", "\\"), "speckle_noise")   # slash


def _residual(composer, pt, sev, image, image_id):
    return (run(composer, pt, sev, image, image_id).astype(np.float64) - image).ravel()


def test_salt_pepper_independence_across_ids_statistical(composer):
    """Two different ids must corrupt (about) independent pixel sets with (about) independent polarity."""
    flat = np.full((112, 112, 3), 128, np.uint8)
    genuine = [(f"P{k:03d}/P{k:03d}_0001.jpg", f"P{k:03d}/P{k:03d}_0002.jpg") for k in range(12)]
    impostor = [(f"P{k:03d}/P{k:03d}_0001.jpg", f"P{k + 50:03d}/P{k + 50:03d}_0001.jpg") for k in range(12)]
    overlaps, agree, both = [], 0, 0
    for a_id, b_id in genuine + impostor:
        a = run(composer, PT.SALT_PEPPER_NOISE, 1.0, flat, a_id)[..., 0]
        b = run(composer, PT.SALT_PEPPER_NOISE, 1.0, flat, b_id)[..., 0]
        ca, cb = a != 128, b != 128
        overlaps.append((ca & cb).mean())
        agree += int(((a == b) & ca & cb).sum())
        both += int((ca & cb).sum())
    assert abs(np.mean(overlaps) - 0.2 * 0.2) < 0.005          # independent sets: overlap = density^2 = 0.04
    assert max(overlaps) < 0.06                                  # a shared noise array would give 0.20
    assert abs(agree / both - 0.5) < 0.05                        # independent polarity: half the shared pixels agree


def test_speckle_independence_across_ids_statistical(composer, fimg):
    pairs = [(f"P{k:03d}/P{k:03d}_0001.jpg", f"P{k:03d}/P{k:03d}_0002.jpg") for k in range(10)] \
        + [(f"P{k:03d}/P{k:03d}_0001.jpg", f"Q{k:03d}/Q{k:03d}_0001.jpg") for k in range(10)]
    corr = [abs(np.corrcoef(_residual(composer, PT.SPECKLE_NOISE, 1.0, fimg, a),
                            _residual(composer, PT.SPECKLE_NOISE, 1.0, fimg, b))[0, 1]) for a, b in pairs]
    assert max(corr) < 0.03 and np.mean(corr) < 0.012            # n=37632 values: sigma ~0.005; shared noise would be ~1.0
    same = abs(np.corrcoef(_residual(composer, PT.SPECKLE_NOISE, 1.0, fimg, "P000/x.jpg"),
                           _residual(composer, PT.SPECKLE_NOISE, 1.0, fimg, "P000/x.jpg"))[0, 1])
    assert same > 0.999                                          # control: the statistic does detect a shared stream


def test_verification_path_ids_belong_to_the_images_they_noise(tmp_path, monkeypatch):
    """Through the real Fold 0 verification run, for both the plain-array path (Eigenfaces/LBPH) and the uint8-chip
    path (ArcFace): every position k of the batch handed to perturb_batch holds the preprocessing of the record whose
    image_id is image_ids[k] -- checked by re-preprocessing each record on its own."""
    from faceeval.orchestration import verification as V
    from faceeval.orchestration.runner import ExperimentRunner
    from tests.test_verification_stage import _config, _make_fake_lfw
    real_pre, real_chips = ExperimentRunner._preprocess_records, ExperimentRunner._aligned_chips
    produced = []                                   # (output list object, pipeline, records, raw_images)

    def rec_pre(pipeline, records, raw_images):
        out = real_pre(pipeline, records, raw_images)
        produced.append((out, pipeline, list(records), raw_images, real_pre))
        return out

    def rec_chips(pipeline, records, raw_images):
        out = real_chips(pipeline, records, raw_images)
        produced.append((out, pipeline, list(records), raw_images, real_chips))
        return out
    monkeypatch.setattr(ExperimentRunner, "_preprocess_records", staticmethod(rec_pre))
    monkeypatch.setattr(ExperimentRunner, "_aligned_chips", staticmethod(rec_chips))

    checked = []
    real_pb = V.perturb_batch

    def spy(composer, spec, normalizer, X, chips, image_ids=None):
        source = X if chips is None else chips
        hit = next(p for p in produced if p[0] is source)           # the very list the batch was built from
        out, pipeline, records, raw_images, fn = hit
        assert [r.image_id for r in records] == list(image_ids)
        for k, rec in enumerate(records):                           # independent per-record re-preprocessing
            assert np.array_equal(np.asarray(source[k]), np.asarray(fn(pipeline, [rec], raw_images)[0])), rec.image_id
        assert len({np.asarray(a).tobytes() for a in source}) == len(source)      # not vacuous: images are distinct
        checked.append("chips" if chips is not None else "array")
        return real_pb(composer, spec, normalizer, X, chips, image_ids)
    monkeypatch.setattr(V, "perturb_batch", spy)

    root = tmp_path / "data"
    _make_fake_lfw(root)
    cfg = _config(tmp_path, root, "lfw_fold0_verification", deep=("arcface",))
    cfg.perturbation_schedule.perturbation_types = [PT.GAUSSIAN_NOISE]
    cfg.perturbation_schedule.severity_levels = [0.0, 1.0]
    monkeypatch.chdir(tmp_path)
    runner = ExperimentRunner(cfg)
    runner.run()
    assert runner.verification_summary["status"] == "COMPLETE"
    assert "array" in checked and "chips" in checked

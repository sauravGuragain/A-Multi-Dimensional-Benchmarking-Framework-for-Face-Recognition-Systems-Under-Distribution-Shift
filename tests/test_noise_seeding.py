"""Per-image deterministic noise seeding: seed = SHA-256(master | image_id | perturbation type), severity-free."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from faceeval.core.types import PerturbationType as PT
from faceeval.perturbation import noise as N
from faceeval.perturbation.composer import PerturbationComposer
from tests.test_perturbation_value_range import _face_u8

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

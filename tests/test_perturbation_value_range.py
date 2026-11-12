"""Regression tests for the perturbation value-range defect.

Defect (fixed): 8-bit inputs went through ``_to_float32`` ([0, 1]) but came back through ``_restore_dtype``
(expects [0, 255]), so Gaussian/speckle noise, brightness, contrast and gamma returned black images; JPEG on
float [0, 1] images returned values 0-255.  These tests use an independent, non-constant face-like image and
check absolute properties (not output == output): exact severity-0 copy, dtype, value range, black/constant
detection, distortion growth, uint8 <-> float32 agreement and closed-form expected values.
"""
from __future__ import annotations

import numpy as np
import pytest

from faceeval.core.types import PerturbationType
from faceeval.orchestration.perturbed_inputs import needs_chips, perturb_batch
from faceeval.perturbation.composer import PerturbationComposer
from faceeval.preprocessing.normalizer import normalizer_for_model

ALL_TYPES = list(PerturbationType)
GRID = [0.2, 0.4, 0.6, 0.8, 1.0]
# Severity-neutral centres: brightness shift = (s-0.5)*0.8, gamma = 4**(2s-1), contrast scale = 0.3*10**s.
# At the centre the perturbation is (nearly) the identity BY DESIGN, so distortion is V-shaped around it.
V_SHAPED = {PerturbationType.BRIGHTNESS: 0.5, PerturbationType.GAMMA: 0.5, PerturbationType.CONTRAST: 0.5233}


def _face_u8(size: int = 112) -> np.ndarray:
    """Deterministic non-constant BGR face-like image (skin ellipse, eyes, mouth, shading, texture), 40-215."""
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32) / size
    img = np.empty((size, size, 3), np.float32)
    bg = 95 + 50 * xx + 20 * yy
    skin = ((xx - 0.5) / 0.30) ** 2 + ((yy - 0.52) / 0.40) ** 2 < 1
    for c, tone in enumerate((135, 165, 195)):           # B, G, R
        img[..., c] = np.where(skin, tone - 25 * (xx - 0.5) ** 2 * 4, bg)
    for cx in (0.38, 0.62):                              # eyes
        img[(((xx - cx) / 0.06) ** 2 + ((yy - 0.42) / 0.035) ** 2) < 1] = 55
    img[(((xx - 0.5) / 0.12) ** 2 + ((yy - 0.72) / 0.03) ** 2) < 1] = (70, 60, 130)   # mouth
    img += np.random.default_rng(7).normal(0, 6, img.shape)
    return np.clip(img, 40, 215).astype(np.uint8)


@pytest.fixture(scope="module")
def composer():
    return PerturbationComposer(severity_levels=[0.0] + GRID)


@pytest.fixture(scope="module")
def images():
    u8 = _face_u8()
    return {"uint8": u8, "float32": u8.astype(np.float32) / 255.0}


def _unit(a: np.ndarray) -> np.ndarray:
    """Pixels on a common 0-1 scale (uint8 / 255, float unchanged)."""
    return a.astype(np.float64) / 255.0 if a.dtype == np.uint8 else a.astype(np.float64)


def _distortion(out, clean) -> float:
    return float(np.abs(_unit(out) - _unit(clean)).mean())


def _perturb(composer, ptype, img, severity):
    # one fixed image id: the same source image gets the same (per-image) noise in every representation
    return composer._get_perturbation(ptype).apply(img, severity, image_id="fixture/image_0001.jpg")


def test_all_sixteen_types_are_covered():
    assert len(ALL_TYPES) == 16


@pytest.mark.parametrize("rep", ["uint8", "float32"])
def test_test_image_is_realistic(images, rep):
    u = _unit(images[rep])
    assert u.std() > 0.1 and 0.25 < u.mean() < 0.75 and len(np.unique(images[rep])) > 100


@pytest.mark.parametrize("rep", ["uint8", "float32"])
@pytest.mark.parametrize("ptype", ALL_TYPES, ids=lambda p: p.value)
def test_severity_zero_is_exact_copy(composer, images, ptype, rep):
    img = images[rep]
    out = _perturb(composer, ptype, img, 0.0)
    assert np.array_equal(out, img) and out.dtype == img.dtype
    assert out is not img and not np.shares_memory(out, img)


@pytest.mark.parametrize("rep", ["uint8", "float32"])
@pytest.mark.parametrize("sev", GRID)
@pytest.mark.parametrize("ptype", ALL_TYPES, ids=lambda p: p.value)
def test_output_dtype_range_and_not_degenerate(composer, images, ptype, sev, rep):
    img = images[rep]
    out = _perturb(composer, ptype, img, sev)
    assert out.dtype == img.dtype and out.shape == img.shape
    u = _unit(out)
    assert np.isfinite(u).all() and u.min() >= 0.0 and u.max() <= 1.0 + 1e-6      # input's own range
    # Black / constant detection (the defect produced 2 distinct values with mean ~0)
    assert len(np.unique(out)) >= 16, f"{ptype.value}@{sev}: only {len(np.unique(out))} distinct values"
    assert u.std() >= 0.02, f"{ptype.value}@{sev}: (near-)constant output, std={u.std():.4f}"
    assert u.mean() >= 0.03, f"{ptype.value}@{sev}: black image, mean={u.mean():.4f}"
    # An 8-bit image must stay on the 0-255 scale (not collapse to 0/1); float must not leave [0, 1]
    if out.dtype == np.uint8:
        assert out.max() > 100


@pytest.mark.parametrize("rep", ["uint8", "float32"])
@pytest.mark.parametrize("ptype", ALL_TYPES, ids=lambda p: p.value)
def test_nonzero_severity_changes_the_image(composer, images, ptype, rep):
    img = images[rep]
    sev = 0.8 if ptype in V_SHAPED else 0.6          # the neutral centre of V-shaped types is the identity
    assert _distortion(_perturb(composer, ptype, img, sev), img) > 0.002


@pytest.mark.parametrize("rep", ["uint8", "float32"])
@pytest.mark.parametrize("ptype", [p for p in ALL_TYPES if p not in V_SHAPED], ids=lambda p: p.value)
def test_distortion_grows_with_severity(composer, images, ptype, rep):
    """Monotone types: distortion from the clean image never drops by more than 15% of the previous step
    (JPEG quality steps, rotation and fixed-seed crops are slightly uneven) and ends clearly above where it starts."""
    img = images[rep]
    d = [_distortion(_perturb(composer, ptype, img, s), img) for s in GRID]
    for prev, cur in zip(d, d[1:]):
        assert cur >= prev * 0.85, f"{ptype.value}: distortion {prev:.4f} -> {cur:.4f}"
    assert d[-1] > d[0] * 1.2, f"{ptype.value}: {d}"


@pytest.mark.parametrize("rep", ["uint8", "float32"])
@pytest.mark.parametrize("ptype", list(V_SHAPED), ids=lambda p: p.value)
def test_v_shaped_types_distort_more_away_from_neutral_centre(composer, images, ptype, rep):
    """Brightness/contrast/gamma are centred (severity ~0.5 = no change), so distortion grows in BOTH directions
    away from the centre; severity 0.0 is the identity only because apply() returns a copy there."""
    img = images[rep]
    up = [_distortion(_perturb(composer, ptype, img, s), img) for s in (0.6, 0.8, 1.0)]
    down = [_distortion(_perturb(composer, ptype, img, s), img) for s in (0.4, 0.2, 0.05)]
    assert up[0] < up[1] < up[2] and down[0] < down[1] < down[2], (up, down)
    assert _distortion(_perturb(composer, ptype, img, V_SHAPED[ptype]), img) < 0.01     # neutral centre


@pytest.mark.parametrize("ptype", ALL_TYPES, ids=lambda p: p.value)
@pytest.mark.parametrize("sev", [0.2, 0.6, 1.0])
def test_uint8_and_float32_inputs_agree(composer, images, ptype, sev):
    """Same image, two representations -> the same result on the common 0-1 scale (to rounding)."""
    a = _unit(_perturb(composer, ptype, images["uint8"], sev))
    b = _unit(_perturb(composer, ptype, images["float32"], sev))
    if ptype is PerturbationType.SCALING:
        # By design (unchanged here) the empty border is filled with 128 for uint8 but with the image mean for
        # float images, so compare only the shrunken face region.
        n = int(112 * composer._get_perturbation(ptype)._severity_to_params(sev)["scale_factor"])
        pad = (112 - n) // 2
        a, b = a[pad:pad + n, pad:pad + n], b[pad:pad + n, pad:pad + n]
    assert np.abs(a - b).mean() < 0.01, f"{ptype.value}@{sev}: mean |u8-f32| = {np.abs(a - b).mean():.4f}"


# ---- closed-form expectations (independent of the implementation's own helpers) -------------------------

@pytest.mark.parametrize("rep", ["uint8", "float32"])
def test_brightness_matches_closed_form(composer, images, rep):
    f = images["float32"].astype(np.float64)
    for sev in (0.2, 0.8, 1.0):
        want = np.clip(f + (sev - 0.5) * 2 * 0.40, 0, 1)
        got = _unit(_perturb(composer, PerturbationType.BRIGHTNESS, images[rep], sev))
        assert np.abs(got - want).max() <= 1.5 / 255


@pytest.mark.parametrize("rep", ["uint8", "float32"])
def test_gamma_matches_closed_form(composer, images, rep):
    f = images["float32"].astype(np.float64)
    for sev in (0.2, 0.8, 1.0):
        want = f ** (0.25 * 16 ** sev)
        got = _unit(_perturb(composer, PerturbationType.GAMMA, images[rep], sev))
        assert np.abs(got - want).max() <= 1.5 / 255


@pytest.mark.parametrize("rep", ["uint8", "float32"])
def test_contrast_matches_closed_form(composer, images, rep):
    f = images["float32"].astype(np.float64)
    for sev in (0.2, 0.8, 1.0):
        scale = 0.3 * 10 ** sev
        want = np.clip(f.mean() + scale * (f - f.mean()), 0, 1)
        got = _unit(_perturb(composer, PerturbationType.CONTRAST, images[rep], sev))
        assert np.abs(got - want).max() <= 2.5 / 255


@pytest.mark.parametrize("rep", ["uint8", "float32"])
@pytest.mark.parametrize("ptype,std_at_1", [(PerturbationType.GAUSSIAN_NOISE, 0.20)], ids=["gaussian_noise"])
def test_gaussian_noise_has_requested_strength_and_keeps_mean(composer, images, ptype, std_at_1, rep):
    """Added noise must be about std_at_1*severity on the 0-1 scale and must not darken the image."""
    img = images[rep]
    clean_mean = _unit(img).mean()
    for sev in (0.2, 0.6):
        out = _unit(_perturb(composer, ptype, img, sev))
        resid = out - _unit(img)
        assert 0.6 * std_at_1 * sev < resid.std() < 1.3 * std_at_1 * sev          # clipping trims it a little
        assert abs(out.mean() - clean_mean) < 0.03


@pytest.mark.parametrize("rep", ["uint8", "float32"])
def test_speckle_noise_keeps_brightness_and_adds_variation(composer, images, rep):
    img = images[rep]
    out = _unit(_perturb(composer, PerturbationType.SPECKLE_NOISE, img, 0.6))
    assert abs(out.mean() - _unit(img).mean()) < 0.03
    assert (out - _unit(img)).std() > 0.02


@pytest.mark.parametrize("rep", ["uint8", "float32"])
def test_jpeg_output_stays_in_input_range_and_close_to_input(composer, images, rep):
    """Float [0, 1] input used to come back as 0-255 floats."""
    img = images[rep]
    for sev in GRID:
        out = _perturb(composer, PerturbationType.JPEG_COMPRESSION, img, sev)
        assert out.dtype == img.dtype
        assert _unit(out).max() <= 1.0 + 1e-6, f"JPEG@{sev} left the input range (max={out.max()})"
        assert _distortion(out, img) < 0.12                                       # a codec artefact, not a different image


def test_jpeg_on_2d_float_image_stays_in_range(composer, images):
    """The traditional models feed (H, W) float [0, 1] grayscale arrays."""
    gray = images["float32"].mean(axis=2).astype(np.float32)
    out = _perturb(composer, PerturbationType.JPEG_COMPRESSION, gray, 0.6)
    assert out.shape == gray.shape and out.dtype == np.float32 and 0.0 <= out.min() and out.max() <= 1.0
    assert 0.0005 < np.abs(out - gray).mean() < 0.05


# ---- existing identification / verification paths ---------------------------------------------------------

@pytest.mark.parametrize("model", ["arcface", "facenet", "dlib_fr", "eigenfaces"])
@pytest.mark.parametrize("ptype", ALL_TYPES, ids=lambda p: p.value)
def test_perturb_batch_is_sane_for_every_normalisation_mode(composer, images, model, ptype):
    """perturb_batch (used by BOTH the identification and verification paths) must hand every model a plausible,
    non-degenerate perturbed input, whether it perturbs an 8-bit chip (arcface/facenet) or a [0, 1] array."""
    norm = normalizer_for_model(model)
    chip = images["uint8"]
    if model == "eigenfaces":
        chip = chip.mean(axis=2).astype(np.uint8)
    clean = norm.normalize(chip)
    sev = 0.8 if ptype in V_SHAPED else 0.6
    spec = next(s for s in composer.specs_for_type(ptype) if abs(s.severity - sev) < 1e-9)
    out = perturb_batch(composer, spec, norm, [clean], [chip] if needs_chips(norm) else None,
                        ["fixture/image_0001.jpg"])[0]      # noise perturbations require image ids
    assert out.shape == clean.shape and np.isfinite(out).all() and len(np.unique(out)) >= 16
    assert out.std() > 0.05 * clean.std()                                         # not flattened to a constant / black frame
    if model in ("dlib_fr", "eigenfaces"):                                       # [0, 1] float domain
        assert 0.0 <= out.min() and out.max() <= 1.0
    if model == "arcface":                                                       # (x - 127.5) / 128 domain
        assert -1.0 <= out.min() and out.max() <= 1.0
        assert (out * 128 + 127.5).mean() > 8                                     # i.e. a visible image, not black
    if model == "facenet":                                                       # per-image standardised RGB
        assert abs(out.mean()) < 1e-3 and abs(out.std() - 1) < 1e-2
    assert np.abs(out - clean).mean() > 1e-3                                      # and it really changed


def test_runner_apply_perturbation_uses_the_same_safe_path(composer, images):
    """Identification path: ExperimentRunner._apply_perturbation delegates to perturb_batch."""
    from faceeval.orchestration.runner import ExperimentRunner
    norm = normalizer_for_model("arcface")
    chip = images["uint8"]
    spec = next(s for s in composer.specs_for_type(PerturbationType.GAUSSIAN_NOISE) if s.severity == 0.6)
    from types import SimpleNamespace
    recs = [SimpleNamespace(image_id="fixture/image_0001.jpg")]
    out = ExperimentRunner._apply_perturbation(composer, spec, recs, [norm.normalize(chip)], norm, [chip])
    assert (out[0] * 128 + 127.5).mean() > 100 and len(np.unique(out[0])) > 200      # noisy face, not black


def test_verification_path_feeds_models_sane_perturbed_inputs(tmp_path, monkeypatch):
    """End to end through run_fold0_verification on a tiny synthetic LFW: a recording stand-in for a chip-mode
    deep model sees non-black, non-constant perturbed arrays for every previously broken perturbation."""
    from faceeval.core.registry import model_registry
    from faceeval.orchestration.runner import ExperimentRunner
    from tests.test_verification_stage import _config, _make_fake_lfw

    seen: dict[str, list[tuple[float, float, int]]] = {}
    current = {"label": None}

    class Recorder:
        def fit_embedding(self, X, y): pass
        def is_stub_backbone(self): return False
        def assert_real_backbone(self): pass
        def extract_features(self, X):
            lab = current["label"]
            for a in X:
                px = a * 128 + 127.5                     # back to the 0-255 scale of the ARCFACE normaliser
                seen.setdefault(lab, []).append((float(px.mean()), float(px.std()), len(np.unique(a))))
            return [list(np.resize(np.asarray(a, np.float32).reshape(-1)[::997], 8)) for a in X]

    real_get = model_registry.get
    monkeypatch.setattr(model_registry, "get", lambda name, **k: Recorder() if name == "arcface" else real_get(name, **k))
    import faceeval.orchestration.verification as V
    real_pb = V.perturb_batch
    def spy(composer, spec, normalizer, X, chips, image_ids=None):
        current["label"] = (spec.perturbation_type.value, round(spec.severity, 2))
        return real_pb(composer, spec, normalizer, X, chips, image_ids)
    monkeypatch.setattr(V, "perturb_batch", spy)

    root = tmp_path / "data"; _make_fake_lfw(root)
    cfg = _config(tmp_path, root, "lfw_fold0_verification", deep=("arcface",))
    cfg.traditional_models = []
    from faceeval.core.types import PerturbationType as PT
    cfg.perturbation_schedule.perturbation_types = [PT.GAUSSIAN_NOISE, PT.SPECKLE_NOISE, PT.BRIGHTNESS,
                                                    PT.CONTRAST, PT.GAMMA, PT.JPEG_COMPRESSION]
    cfg.perturbation_schedule.severity_levels = [0.0, 1.0]
    monkeypatch.chdir(tmp_path)
    runner = ExperimentRunner(cfg); runner.run()
    assert runner.verification_summary["status"] == "COMPLETE"
    for ptype in ("gaussian_noise", "speckle_noise", "brightness", "contrast", "gamma", "jpeg_compression"):
        stats = seen[(ptype, 1.0)]
        assert len(stats) > 50
        assert min(m for m, _, _ in stats) > 8, f"{ptype}: black perturbed input reached the model"
        assert min(s for _, s, _ in stats) > 3, f"{ptype}: constant perturbed input reached the model"

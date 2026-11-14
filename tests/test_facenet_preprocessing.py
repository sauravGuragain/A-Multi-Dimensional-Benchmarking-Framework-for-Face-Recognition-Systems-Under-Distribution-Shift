"""FaceNet-only preprocessing: fixed standardisation + centre crop 0.8; every other model is unchanged.

Context: a validation-only ablation (Fold 0, 684 genuine + 684 impostor pairs) showed that the old FaceNet pipeline
(per-image standardisation on a loosely framed chip) gave 0.794 accuracy / 0.859 AUC, while fixed standardisation
gave 0.958 / 0.994 and an additional 0.8 centre crop 0.991 / 0.9995. These tests pin the new FaceNet behaviour,
keep the old normalisation available as an explicit opt-in, and prove that no other model's preprocessing changed.
"""
import csv
import json
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from faceeval.core.types import PerturbationType
from faceeval.orchestration.perturbed_inputs import CHIP_MODES, needs_chips, perturb_batch
from faceeval.orchestration.runner import ExperimentRunner
from faceeval.perturbation.composer import PerturbationComposer
from faceeval.preprocessing import (
    CentreCropAligner, FaceAligner, NormalizationMode, Normalizer, describe_preprocessing, normalizer_for_model,
    pipeline_for_model,
)
from faceeval.preprocessing.normalizer import _MODEL_TO_MODE, _MODEL_TO_SIZE
from faceeval.preprocessing.pipeline import FACENET_CENTRE_CROP_FRACTION

OTHER_MODELS = ["arcface", "insightface", "dlib_fr", "eigenfaces", "fisherfaces", "lbph", "pca_svm", "hog_svm", "knn"]

# The model -> normalisation mapping before this change (everything except facenet must still match).
ORIGINAL_MODES = {
    "arcface": NormalizationMode.ARCFACE, "insightface": NormalizationMode.ARCFACE, "dlib_fr": NormalizationMode.DLIB,
    "eigenfaces": NormalizationMode.GRAYSCALE_EQ, "fisherfaces": NormalizationMode.GRAYSCALE_EQ,
    "lbph": NormalizationMode.GRAYSCALE, "pca_svm": NormalizationMode.GRAYSCALE_EQ,
    "hog_svm": NormalizationMode.GRAYSCALE, "knn": NormalizationMode.GRAYSCALE,
}
ORIGINAL_SIZES = {"facenet": 160, "arcface": 112, "insightface": 112, "dlib_fr": 150, "eigenfaces": 100,
                  "fisherfaces": 100, "lbph": 100, "pca_svm": 100, "hog_svm": 128, "knn": 100}


def _image(h=250, w=250, seed=0):
    return np.random.default_rng(seed).integers(0, 256, (h, w, 3), dtype=np.uint8)


def _ref_centre_crop(img, fraction, size):
    """Independent reference of the framing validated in the ablation (variant C_centre_crop_0.80)."""
    h, w = img.shape[:2]
    side = int(round(fraction * min(h, w)))
    y0, x0 = (h - side) // 2, (w - side) // 2
    return cv2.resize(img[y0:y0 + side, x0:x0 + side], (size, size), interpolation=cv2.INTER_AREA)


# ---------------------------------------------------------------- normalisation
def test_fixed_standardisation_values_channel_order_and_dtype():
    chip = np.zeros((160, 160, 3), np.uint8)
    chip[..., 0], chip[..., 1], chip[..., 2] = 10, 120, 250                  # B, G, R
    out = Normalizer(NormalizationMode.FACENET_FIXED, target_size=160).normalize(chip)
    assert out.dtype == np.float32 and out.shape == (160, 160, 3)
    assert np.allclose(out[0, 0], [(250 - 127.5) / 128, (120 - 127.5) / 128, (10 - 127.5) / 128])   # RGB order
    img = _image(160, 160, 1)
    assert np.allclose(Normalizer("facenet_fixed", 160).normalize(img), (img[:, :, ::-1].astype(np.float32) - 127.5) / 128.0)


def test_fixed_standardisation_keeps_brightness_information():
    img = _image(160, 160, 2).clip(60, 190)
    base = Normalizer("facenet_fixed", 160).normalize(img)
    shifted = Normalizer("facenet_fixed", 160).normalize((img + 20).astype(np.uint8))
    assert np.allclose(shifted - base, 20 / 128, atol=1e-6)                   # a global shift survives
    legacy = Normalizer("facenet", 160)
    assert np.allclose(legacy.normalize(img), legacy.normalize((img + 20).astype(np.uint8)), atol=1e-4)  # old mode cancelled it


def test_legacy_per_image_standardisation_is_still_available_as_explicit_opt_in():
    img = _image(160, 160, 3)
    out = Normalizer(NormalizationMode.FACENET, 160).normalize(img)
    assert abs(float(out.mean())) < 1e-4 and abs(float(out.std()) - 1.0) < 1e-3
    pipe = pipeline_for_model("facenet", normalization="facenet")
    assert pipe.normalizer.mode is NormalizationMode.FACENET
    assert isinstance(pipe.aligner, CentreCropAligner)                         # opt-in changes normalisation only
    assert describe_preprocessing(pipe)["normalization_mode"] == "facenet"


def test_default_mapping_facenet_new_everything_else_unchanged():
    assert _MODEL_TO_MODE["facenet"] is NormalizationMode.FACENET_FIXED
    assert normalizer_for_model("facenet").mode is NormalizationMode.FACENET_FIXED
    for m, mode in ORIGINAL_MODES.items():
        assert _MODEL_TO_MODE[m] is mode and normalizer_for_model(m).mode is mode
    assert _MODEL_TO_SIZE == ORIGINAL_SIZES
    assert set(_MODEL_TO_MODE) == set(ORIGINAL_MODES) | {"facenet"}


def test_fixed_mode_is_perturbed_on_the_uint8_chip_not_on_normalised_values():
    assert NormalizationMode.FACENET_FIXED in CHIP_MODES and NormalizationMode.FACENET in CHIP_MODES
    assert needs_chips(normalizer_for_model("facenet")) and needs_chips(normalizer_for_model("arcface"))
    assert not needs_chips(normalizer_for_model("eigenfaces"))
    pipe = pipeline_for_model("facenet")
    composer = PerturbationComposer(severity_levels=[0.0, 1.0])
    spec = [s for s in composer.specs_for_type(PerturbationType.BRIGHTNESS) if s.severity == 1.0][0]
    chips = [_ref_centre_crop(_image(250, 250, i), 0.8, 160) for i in range(3)]
    clean = [pipe.normalizer.normalize(c) for c in chips]
    out = perturb_batch(composer, spec, pipe.normalizer, clean, chips, [f"id{i}" for i in range(3)])
    brightened = [composer._get_perturbation(spec.perturbation_type).apply(c, 1.0) for c in chips]
    for o, b in zip(out, brightened):
        assert np.array_equal(o, pipe.normalizer.normalize(b))                # perturb uint8 chip, then normalise
        assert o.min() >= -127.5 / 128 - 1e-6 and o.max() <= 127.5 / 128 + 1e-6      # pixel range 0..255 -> [-0.996, 0.996]


# ---------------------------------------------------------------- centre-crop aligner
@pytest.mark.parametrize("shape", [(250, 250), (250, 200), (180, 300), (100, 100)])
def test_centre_crop_aligner_geometry_matches_reference(shape):
    img = _image(*shape, seed=4)
    chip = CentreCropAligner(160, 0.8).align(img, None).image
    assert chip.shape == (160, 160, 3) and chip.dtype == np.uint8
    assert np.array_equal(chip, _ref_centre_crop(img, 0.8, 160))


def test_centre_crop_aligner_ignores_detection_and_reports_method():
    det = SimpleNamespace(image_path="x.jpg", num_faces=0, primary_box=None)
    face = CentreCropAligner(160, 0.8).align(_image(), det)
    assert face is not None and face.alignment_method == "centre_crop" and face.source_path == "x.jpg"
    assert len(CentreCropAligner(160, 0.8).align_all(_image(), det)) == 1


@pytest.mark.parametrize("bad", [0.0, -0.1, 1.01, float("nan")])
def test_centre_crop_fraction_validated(bad):
    with pytest.raises(ValueError):
        CentreCropAligner(160, bad)


def test_centre_crop_fraction_one_is_whole_image_resize():
    img = _image(250, 250, 5)
    assert np.array_equal(CentreCropAligner(160, 1.0).align(img, None).image,
                          cv2.resize(img, (160, 160), interpolation=cv2.INTER_AREA))


# ---------------------------------------------------------------- pipeline wiring
def test_facenet_pipeline_uses_the_new_framing_and_normalisation():
    assert FACENET_CENTRE_CROP_FRACTION == 0.8
    pipe = pipeline_for_model("facenet")
    assert type(pipe.aligner) is CentreCropAligner and pipe.aligner.crop_fraction == 0.8
    assert pipe.normalizer.mode is NormalizationMode.FACENET_FIXED
    assert describe_preprocessing(pipe) == {"normalization_mode": "facenet_fixed", "alignment": "centre_crop", "crop_fraction": 0.8}
    assert describe_preprocessing(pipeline_for_model("facenet", crop_fraction=0.6))["crop_fraction"] == 0.6


def test_runner_chip_path_gives_facenet_the_centre_crop(tmp_path):
    img = _image(250, 250, 6)
    path = tmp_path / "a.jpg"
    cv2.imwrite(str(path), img)
    loaded = cv2.imread(str(path))
    rec = SimpleNamespace(image_path=str(path), image_id="a/a_0001.jpg")
    pipe = pipeline_for_model("facenet")
    (chip,) = ExperimentRunner._aligned_chips(pipe, [rec], None)               # exactly what the verification stage calls
    assert np.array_equal(chip, _ref_centre_crop(loaded, 0.8, 160))
    (arr,) = ExperimentRunner._preprocess_records(pipe, [rec], None)
    assert np.array_equal(arr, (chip[:, :, ::-1].astype(np.float32) - 127.5) / 128.0)


@pytest.mark.parametrize("model", OTHER_MODELS)
def test_other_models_keep_their_original_pipeline(model, tmp_path):
    pipe = pipeline_for_model(model)
    assert type(pipe.aligner) is FaceAligner                                   # not the centre-crop subclass
    assert pipe.normalizer.mode is ORIGINAL_MODES[model]
    assert describe_preprocessing(pipe)["crop_fraction"] is None
    assert describe_preprocessing(pipe)["alignment"] == "affine_fixed_landmarks"
    # chips equal those of a directly constructed original aligner (same code, same machine)
    img = _image(250, 250, 7)
    path = tmp_path / "b.jpg"
    cv2.imwrite(str(path), img)
    rec = SimpleNamespace(image_path=str(path), image_id="b/b_0001.jpg")
    (chip,) = ExperimentRunner._aligned_chips(pipe, [rec], None)
    loaded = cv2.imread(str(path))
    det = pipe._whole_image_detection(loaded, str(path))
    expected = FaceAligner(output_size=ORIGINAL_SIZES[model]).align(loaded, det).image
    assert np.array_equal(chip, expected)
    assert np.array_equal(ExperimentRunner._preprocess_records(pipe, [rec], None)[0],
                          normalizer_for_model(model).normalize(expected))


@pytest.mark.parametrize("model", OTHER_MODELS)
def test_crop_fraction_is_rejected_for_other_models(model):
    with pytest.raises(ValueError):
        pipeline_for_model(model, crop_fraction=0.8)


# ---------------------------------------------------------------- experiment outputs record the preprocessing
@pytest.fixture(scope="module")
def fake_data(tmp_path_factory):
    from tests.test_verification_stage import _make_fake_lfw
    root = tmp_path_factory.mktemp("fn_data")
    _make_fake_lfw(root)
    return root


def _verification_outputs(tmp_path, monkeypatch, data_root, deep):
    from tests.test_verification_stage import _run
    _, runner, _ = _run(tmp_path, monkeypatch, data_root, protocol="lfw_fold0_verification", deep=deep)
    out = runner.verification_summary["output_dir"]
    with open(f"{out}/verification_results.csv", newline="") as fh:
        reader = csv.DictReader(fh)
        rows = list(reader)
        header = reader.fieldnames
    return rows, header, json.load(open(f"{out}/verification_results.json"))


def test_outputs_record_exact_normalisation_and_crop_fraction(tmp_path, monkeypatch, fake_data):
    rows, header, data = _verification_outputs(tmp_path, monkeypatch, fake_data, ("facenet", "arcface"))
    assert {"normalization_mode", "alignment", "crop_fraction"} <= set(header)
    fn = [r for r in rows if r["model"] == "facenet"]
    ar = [r for r in rows if r["model"] == "arcface"]
    cl = [r for r in rows if r["model"] in ("eigenfaces", "lbph")]
    assert fn and ar and cl
    assert all((r["normalization_mode"], r["alignment"], float(r["crop_fraction"])) == ("facenet_fixed", "centre_crop", 0.8) for r in fn)
    assert all((r["normalization_mode"], r["alignment"], r["crop_fraction"]) == ("arcface", "affine_fixed_landmarks", "") for r in ar)
    assert all(r["normalization_mode"] == "" and r["alignment"] == "" and r["crop_fraction"] == "" for r in cl)
    prep = data["preprocessing"]
    assert prep["facenet"] == {"normalization_mode": "facenet_fixed", "alignment": "centre_crop", "crop_fraction": 0.8}
    assert prep["arcface"] == {"normalization_mode": "arcface", "alignment": "affine_fixed_landmarks", "crop_fraction": None}
    assert prep["eigenfaces"]["normalization_mode"] == "grayscale_eq" and prep["lbph"]["normalization_mode"] == "grayscale"
    assert all("normalization_mode" not in r for r in data["results"] if r["model"] in ("eigenfaces", "lbph"))
    assert all(r["normalization_mode"] == "facenet_fixed" and r["crop_fraction"] == 0.8 for r in data["results"] if r["model"] == "facenet")


def test_classical_only_run_keeps_the_original_csv_columns(tmp_path, monkeypatch, fake_data):
    rows, header, data = _verification_outputs(tmp_path, monkeypatch, fake_data, ())
    assert rows and not ({"normalization_mode", "alignment", "crop_fraction"} & set(header))
    assert set(data["preprocessing"]) == {"eigenfaces", "lbph"}

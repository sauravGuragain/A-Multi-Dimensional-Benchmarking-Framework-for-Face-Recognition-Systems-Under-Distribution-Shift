"""Regression tests for the independent review of f6082e4 (findings D1-D4, R1, R6, R7).

SCOPE: hermetic. dlib, its weights and every dataset are faked; nothing is downloaded and no real model runs. The one
real-dlib test (tests/test_dlib_alignment.py) remains environment-dependent and is skipped when dlib or the verified
weights are absent."""
from __future__ import annotations

import csv
import glob
import json
import logging
import os
import sys
import types
from pathlib import Path

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent))
import test_verification_stage as T  # noqa: E402
from faceeval.analysis import efficiency as E  # noqa: E402
from faceeval.analysis import evidence_index as EI  # noqa: E402
from faceeval.analysis import figures as F  # noqa: E402
from faceeval.analysis import multi_split as MS  # noqa: E402
from faceeval.core.config import ConfigLoader  # noqa: E402
from faceeval.core.exceptions import EmbeddingError, OrchestratorError  # noqa: E402
from faceeval.models.deep import dlib_weights as W  # noqa: E402
from faceeval.orchestration import verification as V  # noqa: E402
from faceeval.orchestration.runner import ExperimentRunner  # noqa: E402
from faceeval.preprocessing import dlib_align as A  # noqa: E402
from faceeval.preprocessing.aligner import FatalPreprocessingError  # noqa: E402
from faceeval.preprocessing.pipeline import describe_preprocessing, pipeline_for_model  # noqa: E402


# ------------------------------------------------------------------------------------------------ fake dlib
class Rect:
    def __init__(s, l, t, r, b): s.l, s.t, s.r, s.b = l, t, r, b
    def left(s): return s.l
    def top(s): return s.t
    def width(s): return s.r - s.l
    def height(s): return s.b - s.t


def fake_dlib(detect: bool, bad_predictor: bool = False):
    m = types.ModuleType("dlib")
    m.__version__ = "0.0-fake"
    class Det:
        def run(self, rgb, up, thr):
            h, w = rgb.shape[:2]
            return ([Rect(w // 4, h // 4, 3 * w // 4, 3 * h // 4)], [1.0], [0]) if detect else ([], [], [])
    m.get_frontal_face_detector = lambda: Det()
    def shape_predictor(path):
        if bad_predictor:
            raise RuntimeError("Error deserializing object (corrupt file)")
        return lambda rgb, rect: ("shape", rect)
    m.shape_predictor = shape_predictor
    m.rectangle = lambda l, t, r, b: Rect(l, t, r, b)
    m.get_face_chip = lambda rgb, shape, size, padding: (np.arange(size * size * 3).reshape(size, size, 3) % 251).astype(np.uint8)
    class Rec:
        def compute_face_descriptor(self, img, num_jitters=0):
            return np.random.default_rng(int(np.asarray(img).sum()) % 2**31).normal(size=128)
    m.face_recognition_model_v1 = lambda path: Rec()
    return m


@pytest.fixture
def with_dlib(monkeypatch, tmp_path):
    def make(detect=True, bad_predictor=False, resolve_error=None):
        monkeypatch.setitem(sys.modules, "dlib", fake_dlib(detect, bad_predictor))
        f = tmp_path / "w.dat"
        f.write_bytes(b"x")
        def resolve(spec, explicit_path=None, allow_download=True, **k):
            if resolve_error:
                raise W.WeightError(resolve_error)
            return f
        monkeypatch.setattr(W, "resolve", resolve)
    return make


# ------------------------------------------------------------------------------------------------ D1 / R1
def test_unloadable_landmark_weights_are_fatal_not_a_per_image_failure(with_dlib):
    with_dlib(resolve_error="SHA-256 mismatch for shape_predictor_5_face_landmarks.dat")
    al = A.DlibChipAligner()
    with pytest.raises(FatalPreprocessingError, match="cannot start.*SHA-256 mismatch"):
        al.ensure_ready()
    with pytest.raises(FatalPreprocessingError):
        al.align(np.zeros((50, 50, 3), np.uint8))


def test_corrupt_predictor_file_is_fatal(with_dlib):
    with_dlib(bad_predictor=True)
    with pytest.raises(FatalPreprocessingError, match="corrupt"):
        A.DlibChipAligner().ensure_ready()


def test_pipeline_process_reraises_the_fatal_error_but_still_records_ordinary_failures(with_dlib, tmp_path):
    with_dlib(resolve_error="weights missing")
    good = tmp_path / "a.jpg"
    cv2.imwrite(str(good), np.full((64, 64, 3), 90, np.uint8))
    p = pipeline_for_model("dlib_fr", alignment="dlib_chip")
    with pytest.raises(FatalPreprocessingError):
        p.process_path(str(good))
    r = pipeline_for_model("dlib_fr", alignment="dlib_chip").process_path(str(tmp_path / "missing.jpg"))   # per-image problem
    assert r.success is False and isinstance(r.error, str) and "cv2.imread" in r.error


def test_dlib_missing_fails_closed_only_when_real_models_are_required(monkeypatch):
    monkeypatch.setitem(sys.modules, "dlib", None)
    al = A.DlibChipAligner()
    assert al.ensure_ready(strict=False) is False
    with pytest.raises(FatalPreprocessingError, match="require_real_models"):
        al.ensure_ready(strict=True)


def test_R1_dlib_missing_valid_detection_result_runs_the_pipeline_with_a_truthful_label_and_one_warning(monkeypatch, tmp_path, caplog):
    monkeypatch.setitem(sys.modules, "dlib", None)
    for i in range(3):
        cv2.imwrite(str(tmp_path / f"{i}.jpg"), (np.random.default_rng(i).random((250, 250, 3)) * 255).astype(np.uint8))
    p = pipeline_for_model("dlib_fr", alignment="dlib_chip")
    with caplog.at_level(logging.WARNING, logger="faceeval.preprocessing.dlib_align"):
        rs = [p.process_path(str(tmp_path / f"{i}.jpg"), image_id=str(i)) for i in range(3)]
    assert all(r.success and r.array.shape == (150, 150, 3) for r in rs)           # valid DetectionResult went through super().align
    assert {r.aligned.alignment_method for r in rs} == {A.LEGACY_FALLBACK_METHOD}
    d = describe_preprocessing(p)
    assert d["alignment"] == A.LEGACY_FALLBACK_METHOD and d["requested_alignment"] == "dlib_chip"
    assert sum("dlib is not importable" in r.getMessage() for r in caplog.records) == 1       # once, not once per image
    assert p.aligner.stats()["n_generic_aligner_dlib_missing"] == 3 and p.aligner.stats()["n_centre_box_fallback"] == 0


def test_alignment_fallback_guard_records_stats_and_refuses_when_most_images_missed():
    sink = {}
    V._check_alignment_fallback("dlib_fr", {"centre_box_fallback_fraction": 0.2, "max_allowed_fraction": 0.5, "n_images": 10}, sink)
    assert sink["dlib_fr"]["n_images"] == 10
    with pytest.raises(EmbeddingError, match="centre-box fallback"):
        V._check_alignment_fallback("dlib_fr", {"centre_box_fallback_fraction": 0.51, "max_allowed_fraction": 0.5}, sink)
    assert "dlib_fr" in sink


def test_failed_embedding_guard():
    V._check_failed_fraction("m", np.array([True] * 6 + [False] * 4), "clean validation")          # 40% failed: allowed
    with pytest.raises(EmbeddingError, match="mostly failed"):
        V._check_failed_fraction("m", np.array([True] * 4 + [False] * 6), "clean validation")
    V._check_failed_fraction("m", np.array([], dtype=bool), "clean validation")


# --- end-to-end through the real runner / verification code with fake dlib
def _cfg(root, require_real=False):
    return f"""
experiment_name: dlibfix
random_seed: 42
dataset_names: [lfw]
data_root: {root}
split_strategy: stratified
evaluation_protocol: lfw_fold0_verification
require_real_models: {str(require_real).lower()}
traditional_models: []
deep_models: [dlib_fr]
perturbation_schedule: {{perturbation_types: [gaussian_blur], severity_levels: [0.0, 1.0], compose_perturbations: false}}
compute_calibration: false
compute_fairness: false
compute_fingerprints: false
compute_failure_analysis: false
compute_deployment_scores: false
deployment_scenarios: [research_benchmark]
profile_resources: false
use_feature_cache: false
feature_cache_dir: {root}/cache/
output_dir: {root}/out/
report_dir: {root}/rep/
export_png: false
mlflow_tracking_uri: sqlite:///{root}/mlflow.db
mlflow_experiment_name: dlibfix
device: cpu
"""


def _run(tmp_path, require_real=False):
    root = tmp_path / "d"
    T._make_fake_lfw(root)
    old = os.getcwd()
    os.chdir(root)
    err = None
    try:
        ExperimentRunner(ConfigLoader().loads(_cfg(root, require_real))).run()
    except Exception as exc:      # a refused/incomplete run raises; its partial outputs stay on disk
        err = exc
    finally:
        os.chdir(old)
    files = glob.glob(str(root / "out" / "verification" / "*" / "verification_results.json"))
    return (json.loads(Path(files[0]).read_text()) if files else None), err


def test_run_with_detected_faces_completes_and_records_alignment_counters(with_dlib, tmp_path):
    with_dlib(detect=True)
    s, err = _run(tmp_path)
    assert err is None and s["status"] == "COMPLETE"
    st = s["provenance"]["classical_rng"]["dlib_alignment_stats"]["dlib_fr"]
    assert st["n_centre_box_fallback"] == 0 and st["n_detected_by_dlib"] == st["n_images"] > 0
    assert s["preprocessing"]["dlib_fr"]["alignment"] == A.METHOD


def test_run_where_dlib_finds_no_face_is_refused_and_the_failure_is_recorded_at_run_level(with_dlib, tmp_path):
    with_dlib(detect=False)
    s, err = _run(tmp_path)
    assert s is not None and s["status"] == "INCOMPLETE" and err is not None
    assert any(f["model"] == "dlib_fr" and "centre-box fallback" in f["error"] for f in s["failures"])
    assert s["provenance"]["classical_rng"]["dlib_alignment_stats"]["dlib_fr"]["centre_box_fallback_fraction"] == 1.0
    assert s["n_rows"] == 0


def test_run_with_unloadable_landmark_weights_fails_visibly_before_any_row(with_dlib, tmp_path):
    with_dlib(resolve_error="landmark file corrupt")
    s, err = _run(tmp_path)
    assert err is not None and s["status"] == "INCOMPLETE" and s["n_rows"] == 0
    assert any("dlib_chip alignment cannot start" in f["error"] and "landmark file corrupt" in f["error"] for f in s["failures"])


def test_require_real_models_with_dlib_missing_aborts_the_run(monkeypatch, tmp_path):
    monkeypatch.setitem(sys.modules, "dlib", None)
    s, err = _run(tmp_path, require_real=True)
    assert isinstance(err, OrchestratorError)
    assert s is None or s["status"] == "INCOMPLETE"


# ------------------------------------------------------------------------------------------------ D2
HEX_A, HEX_B, HEX_C = "a" * 64, "b" * 64, "c" * 64          # the runner writes 64-hex SHA-256 digests


def _run_dir(tmp_path, k, seed, fold, models=("facenet",), duplicate_row=False, **over):
    import hashlib
    d = tmp_path / f"r{k}"
    d.mkdir()
    ids = [f"p{i}" for i in range(100)]
    test = ids[fold * 10:(fold + 1) * 10]
    raw = ("k=v\n" + "".join(f"train|{i}|{i}_1\n" for i in ids if i not in test) + "".join(f"test|{i}|{i}_1\n" for i in test)).encode()
    (d / "m.tsv").write_bytes(raw)
    j = {"status": "COMPLETE", "seed": seed, "fold": fold, "k_folds": 10, "protocol": "lfw_fold0_verification",
         "threshold_protocol": "t", "full_lfw": True, "dataset": {"fingerprint_sha256": "f", "scope": "official_lfw"},
         "pair_params": {"validation": {"cap": 1, "rho": 1}, "test": {"cap": 1, "rho": 1}},
         "manifest_path": str(d / "m.tsv"), "manifest_sha256": hashlib.sha256(raw).hexdigest(), "content_sha256": {"s": str(k)},
         "stub_backbones": [], "preprocessing": {"facenet": {"alignment": "centre_crop", "normalization_mode": "x", "crop_fraction": 0.8}},
         "provenance": {"classical_rng": {"protocol": "global_seed"}, "weights": {"facenet": {"w.pt": HEX_A}}}}
    for path, value in over.items():
        cur = j
        keys = path.split("__")
        for kk in keys[:-1]:
            cur = cur.setdefault(kk, {})
        cur[keys[-1]] = value
    (d / "verification_results.json").write_text(json.dumps(j))
    with open(d / "verification_results.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["model", "perturbation", "severity", "accuracy", "tar", "far", "auc", "eer"])
        w.writeheader()
        for model in models:
            row = {"model": model, "perturbation": "clean", "severity": 0.0, "accuracy": 0.9, "tar": 0.9, "far": 0.1, "auc": 0.95, "eer": 0.1}
            w.writerow(row)
            if duplicate_row:
                w.writerow(row)
    return d


def _drop(d, *path):
    """Remove a (possibly nested) key from a run's verification_results.json."""
    j = json.loads((d / "verification_results.json").read_text())
    cur = j
    for k in path[:-1]:
        cur = cur[k]
    cur.pop(path[-1])
    (d / "verification_results.json").write_text(json.dumps(j))


def _agg(tmp_path, dirs, name="o"):
    return MS.aggregate(dirs, tmp_path / name)


def test_comparable_runs_differing_only_in_seed_and_fold_are_accepted_and_the_checks_recorded(tmp_path):
    dirs = [_run_dir(tmp_path, 0, 7, 0), _run_dir(tmp_path, 1, 7, 1), _run_dir(tmp_path, 2, 9, 2)]
    prov = _agg(tmp_path, dirs)
    assert "classical RNG protocol" in prov["comparability"]["checked"] and prov["comparability"]["weights_unverified"] == []


@pytest.mark.parametrize("over,match", [
    ({"stub_backbones": ["facenet"]}, "STUB backbone"),
    ({"provenance__classical_rng__protocol": "isolated_seed"}, "classical RNG protocol differs"),
    ({"preprocessing__facenet__alignment": "affine"}, "preprocessing/alignment differs"),
    ({"dataset__scope": "custom_or_subset"}, "dataset scope differs"),
    ({"protocol": "other"}, "protocol differs"),
    ({"threshold_protocol": "other"}, "threshold_protocol differs"),
    ({"provenance__weights__facenet": {"w.pt": HEX_B}}, "weight file hashes differ"),
    ({"provenance__weights__facenet": "NOT_RECORDED"}, "recorded in some runs but not others"),
])
def test_incomparable_run_is_refused_with_an_actionable_message(tmp_path, over, match):
    dirs = [_run_dir(tmp_path, 0, 7, 0), _run_dir(tmp_path, 1, 7, 1, **over)]
    with pytest.raises(MS.MultiSplitError, match=match):
        _agg(tmp_path, dirs)


def test_a_stub_run_is_refused_even_when_it_is_the_first_run(tmp_path):
    with pytest.raises(MS.MultiSplitError, match="STUB"):
        _agg(tmp_path, [_run_dir(tmp_path, 0, 7, 0, stub_backbones=["facenet"]), _run_dir(tmp_path, 1, 7, 1)])


def test_partially_recorded_weights_are_accepted_only_with_the_explicit_flag_and_listed(tmp_path):
    a, b = _run_dir(tmp_path, 0, 7, 0), _run_dir(tmp_path, 1, 7, 1, provenance__weights__facenet="NOT_RECORDED")
    with pytest.raises(MS.MultiSplitError, match="--allow-unverified-weights"):
        MS.aggregate([a, b], tmp_path / "o1")
    prov = MS.aggregate([a, b], tmp_path / "o2", allow_unverified_weights=True)
    assert prov["comparability"]["weights_unverified"] == ["facenet"]
    c = _run_dir(tmp_path, 2, 7, 2, provenance__weights__facenet={"w.pt": HEX_C})
    with pytest.raises(MS.MultiSplitError, match="hashes differ"):                    # differing recorded hashes are never allowed
        MS.aggregate([a, c], tmp_path / "o3", allow_unverified_weights=True)
    assert MS.main(["--runs", str(a), str(b), "--out", str(tmp_path / "o4"), "--allow-unverified-weights"]) == 0
    assert MS.main(["--runs", str(a), str(b), "--out", str(tmp_path / "o5")]) == 2


def test_runs_that_predate_provenance_are_comparable_and_weights_are_listed_as_unverified(tmp_path):
    a, b = _run_dir(tmp_path, 0, 7, 0), _run_dir(tmp_path, 1, 7, 1)
    for d in (a, b):
        j = json.loads((d / "verification_results.json").read_text())
        j.pop("provenance")
        (d / "verification_results.json").write_text(json.dumps(j))
    prov = _agg(tmp_path, [a, b])
    assert prov["comparability"]["weights_unverified"] == ["facenet"]


# ---------------------------------------------------------------------------------------- review follow-up: model sets
def _msg(exc):
    return str(exc.value)


def test_a_run_missing_a_model_of_the_first_run_is_refused_and_names_it(tmp_path):
    a = _run_dir(tmp_path, 0, 7, 0, models=("facenet", "eigenfaces"))
    b = _run_dir(tmp_path, 1, 7, 1, models=("facenet",))
    with pytest.raises(MS.MultiSplitError) as e:
        _agg(tmp_path, [a, b])
    assert "model set differs" in _msg(e) and "missing ['eigenfaces']" in _msg(e) and "extra []" in _msg(e) and str(b) in _msg(e)


def test_a_run_with_an_extra_model_is_refused_and_names_it(tmp_path):
    """The first run alone used to define the model set, so an extra model in a later run was only caught indirectly."""
    a = _run_dir(tmp_path, 0, 7, 0, models=("facenet",))
    b = _run_dir(tmp_path, 1, 7, 1, models=("facenet", "arcface"))
    with pytest.raises(MS.MultiSplitError) as e:
        _agg(tmp_path, [a, b])
    assert "extra ['arcface']" in _msg(e) and "missing []" in _msg(e) and str(b) in _msg(e)


def test_model_set_check_also_protects_direct_callers_of_check_comparable(tmp_path):
    """check_comparable(...) is a public function: an extra pretrained model in a later run must not slip past its
    weight comparison just because the first run does not contain that model."""
    a = _run_dir(tmp_path, 0, 7, 0, models=("facenet",))
    b = _run_dir(tmp_path, 1, 7, 1, models=("facenet", "arcface"), provenance__weights__arcface={"w.onnx": HEX_B})
    with pytest.raises(MS.MultiSplitError, match="extra \\['arcface'\\]"):
        MS.check_comparable([a, b], [MS._load(a), MS._load(b)])


def test_same_models_in_a_different_order_are_accepted_and_the_cli_reports_the_refusal(tmp_path, capsys):
    a = _run_dir(tmp_path, 0, 7, 0, models=("facenet", "eigenfaces"))
    b = _run_dir(tmp_path, 1, 7, 1, models=("eigenfaces", "facenet"))
    assert _agg(tmp_path, [a, b])["n_conditions"] == 2
    c = _run_dir(tmp_path, 2, 7, 2, models=("facenet",))
    assert MS.main(["--runs", str(a), str(c), "--out", str(tmp_path / "cli")]) == 2
    assert "REFUSED" in capsys.readouterr().out


def test_condition_set_difference_names_the_missing_and_extra_conditions(tmp_path):
    a, b = _run_dir(tmp_path, 0, 7, 0), _run_dir(tmp_path, 1, 7, 1)
    rows = list(csv.DictReader(open(b / "verification_results.csv")))
    rows[0]["perturbation"] = "blur"
    with open(b / "verification_results.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    with pytest.raises(MS.MultiSplitError) as e:
        _agg(tmp_path, [a, b])
    assert "1 missing, 1 extra" in _msg(e) and "blur" in _msg(e) and "clean" in _msg(e)


# ----------------------------------------------------------------------------------- review follow-up: weight provenance
MALFORMED_WEIGHTS = [
    ({}, "empty object"),
    ("garbage", "only accepted string"),
    ("not_recorded", "only accepted string"),                  # spelling/case variants are not the explicit marker
    ("", "only accepted string"),
    (None, "NoneType"),
    (["a" * 64], "list"),
    (7, "int"),
    ({"w.pt": "abc"}, "64-hex"),
    ({"w.pt": 5}, "64-hex"),
    ({"w.pt": "z" * 64}, "64-hex"),
    ({"": "a" * 64}, "64-hex"),
    ({"w.pt": "NOT_RECORDED"}, "64-hex"),
]


@pytest.mark.parametrize("value,match", MALFORMED_WEIGHTS)
@pytest.mark.parametrize("position", [0, 1])
def test_malformed_weight_provenance_is_refused_wherever_it_occurs(tmp_path, value, match, position):
    runs = [_run_dir(tmp_path, 0, 7, 0), _run_dir(tmp_path, 1, 7, 1)]
    runs[position] = _run_dir(tmp_path, 2 + position, 7, position, provenance__weights__facenet=value)
    with pytest.raises(MS.MultiSplitError, match=match):
        MS.aggregate(runs, tmp_path / "o", allow_unverified_weights=True)         # the override must not excuse malformed data


@pytest.mark.parametrize("value,match", MALFORMED_WEIGHTS)
def test_identical_malformed_weight_provenance_in_every_run_is_still_refused(tmp_path, value, match):
    """Two runs that are malformed in the same way used to look 'the same' to the comparison."""
    runs = [_run_dir(tmp_path, k, 7, k, provenance__weights__facenet=value) for k in (0, 1)]
    with pytest.raises(MS.MultiSplitError, match=match):
        MS.aggregate(runs, tmp_path / "o")


@pytest.mark.parametrize("over,match", [
    ({"provenance__weights": []}, "provenance.weights must be an object"),
    ({"provenance__weights": "x"}, "provenance.weights must be an object"),
    ({"provenance": "x"}, "'provenance' must be an object"),
    ({"provenance": []}, "'provenance' must be an object"),
])
def test_malformed_provenance_containers_are_refused(tmp_path, over, match):
    runs = [_run_dir(tmp_path, 0, 7, 0), _run_dir(tmp_path, 1, 7, 1, **over)]
    with pytest.raises(MS.MultiSplitError, match=match):
        MS.aggregate(runs, tmp_path / "o", allow_unverified_weights=True)


def test_explicit_not_recorded_in_every_run_marks_pretrained_weights_unverified_without_a_flag(tmp_path):
    runs = [_run_dir(tmp_path, k, 7, k, models=("facenet", "eigenfaces"),
                     provenance__weights={"facenet": "NOT_RECORDED", "eigenfaces": "NOT_RECORDED"}) for k in (0, 1)]
    prov = MS.aggregate(runs, tmp_path / "o")
    assert prov["comparability"]["weights_unverified"] == ["facenet"]          # eigenfaces is trained here: nothing to verify


def test_empty_weights_container_and_absent_entries_are_legitimate_and_unverified(tmp_path):
    """The runner writes ``"weights": {}`` for a run without pretrained models; older runs have no provenance at all."""
    runs = [_run_dir(tmp_path, 0, 7, 0, provenance__weights={}), _run_dir(tmp_path, 1, 7, 1, provenance__weights={})]
    assert MS.aggregate(runs, tmp_path / "o1")["comparability"]["weights_unverified"] == ["facenet"]
    old = [_run_dir(tmp_path, 2, 7, 2), _run_dir(tmp_path, 3, 7, 3)]
    for d in old:
        _drop(d, "provenance", "weights")
    assert MS.aggregate(old, tmp_path / "o2")["comparability"]["weights_unverified"] == ["facenet"]


def test_absent_entry_in_one_run_is_partial_recording_and_needs_the_flag(tmp_path):
    a, b = _run_dir(tmp_path, 0, 7, 0), _run_dir(tmp_path, 1, 7, 1, provenance__weights={})
    with pytest.raises(MS.MultiSplitError, match="recorded in some runs but not others"):
        MS.aggregate([a, b], tmp_path / "o1")
    assert MS.aggregate([a, b], tmp_path / "o2", allow_unverified_weights=True)["comparability"]["weights_unverified"] == ["facenet"]


def test_mixed_recorded_unrecorded_and_differing_hashes_is_refused_even_with_the_override(tmp_path):
    a = _run_dir(tmp_path, 0, 7, 0)
    b = _run_dir(tmp_path, 1, 7, 1, provenance__weights__facenet="NOT_RECORDED")
    c = _run_dir(tmp_path, 2, 7, 2, provenance__weights__facenet={"w.pt": HEX_B})
    with pytest.raises(MS.MultiSplitError, match="weight file hashes differ"):
        MS.aggregate([a, b, c], tmp_path / "o", allow_unverified_weights=True)


def test_a_different_set_of_weight_files_is_a_difference_and_is_named(tmp_path):
    a = _run_dir(tmp_path, 0, 7, 0)
    b = _run_dir(tmp_path, 1, 7, 1, provenance__weights__facenet={"w.pt": HEX_A, "extra.onnx": HEX_B})
    with pytest.raises(MS.MultiSplitError) as e:
        MS.aggregate([a, b], tmp_path / "o", allow_unverified_weights=True)
    assert "weight file hashes differ" in _msg(e) and "extra.onnx" in _msg(e)


def test_digest_case_does_not_create_a_false_difference(tmp_path):
    a = _run_dir(tmp_path, 0, 7, 0, provenance__weights__facenet={"w.pt": "ab" * 32})
    b = _run_dir(tmp_path, 1, 7, 1, provenance__weights__facenet={"w.pt": ("ab" * 32).upper()})
    assert MS.aggregate([a, b], tmp_path / "o")["comparability"]["weights_unverified"] == []


# --------------------------------------------------------------------------- review follow-up: other silent acceptances
@pytest.mark.parametrize("value", ["nonsense", None, 3, []])
def test_unknown_classical_rng_protocol_is_refused_even_when_every_run_has_it(tmp_path, value):
    runs = [_run_dir(tmp_path, k, 7, k, provenance__classical_rng__protocol=value) for k in (0, 1)]
    with pytest.raises(MS.MultiSplitError, match="classical_rng.protocol"):
        MS.aggregate(runs, tmp_path / "o")


def test_classical_rng_that_is_not_an_object_is_refused_and_both_known_protocols_are_accepted(tmp_path):
    with pytest.raises(MS.MultiSplitError, match="classical_rng must be an object"):
        MS.aggregate([_run_dir(tmp_path, 0, 7, 0, provenance__classical_rng="x"), _run_dir(tmp_path, 1, 7, 1)], tmp_path / "o1")
    iso = [_run_dir(tmp_path, k, 7, k, provenance__classical_rng__protocol="isolated_seed") for k in (2, 3)]
    assert MS.aggregate(iso, tmp_path / "o2")["n_splits"] == 2


@pytest.mark.parametrize("value", ["facenet", {}, 0, None, ""])
def test_stub_backbones_that_is_not_a_list_is_refused(tmp_path, value):
    runs = [_run_dir(tmp_path, k, 7, k, stub_backbones=value) for k in (0, 1)]
    with pytest.raises(MS.MultiSplitError, match="stub_backbones must be a list"):
        MS.aggregate(runs, tmp_path / "o")


def test_fields_that_no_run_records_are_accepted_but_listed(tmp_path):
    """Older result files may lack comparison fields; they stay usable, but nothing is silently assumed."""
    runs = [_run_dir(tmp_path, k, 7, k) for k in (0, 1)]
    for d in runs:
        _drop(d, "stub_backbones")
        _drop(d, "threshold_protocol")
        _drop(d, "preprocessing")
    nr = MS.aggregate(runs, tmp_path / "o")["comparability"]["fields_not_recorded"]
    assert set(nr) == {"stub_backbones", "threshold_protocol", "preprocessing/alignment"} and all(len(v) == 2 for v in nr.values())
    full = MS.aggregate([_run_dir(tmp_path, 5, 7, 5), _run_dir(tmp_path, 6, 7, 6)], tmp_path / "o2")
    assert full["comparability"]["fields_not_recorded"] == {}


def test_a_field_recorded_in_one_run_but_not_another_is_still_refused(tmp_path):
    a, b = _run_dir(tmp_path, 0, 7, 0), _run_dir(tmp_path, 1, 7, 1)
    _drop(b, "threshold_protocol")
    with pytest.raises(MS.MultiSplitError, match="threshold_protocol differs"):
        MS.aggregate([a, b], tmp_path / "o")


def test_a_duplicated_condition_row_inside_one_run_is_refused_not_counted_as_a_split(tmp_path):
    a = _run_dir(tmp_path, 0, 7, 0)
    b = _run_dir(tmp_path, 1, 7, 1, duplicate_row=True)
    with pytest.raises(MS.MultiSplitError) as e:
        MS.aggregate([a, b], tmp_path / "o")
    assert "appears more than once" in _msg(e) and str(b) in _msg(e) and "lines 2 and 3" in _msg(e)


def test_unusable_result_rows_are_refused_with_an_actionable_message(tmp_path):
    def rewrite(d, fn):
        rows = list(csv.DictReader(open(d / "verification_results.csv")))
        fn(rows)
        with open(d / "verification_results.csv", "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=["model", "perturbation", "severity", "accuracy", "tar", "far", "auc", "eer"])
            w.writeheader()
            w.writerows(rows)

    a = _run_dir(tmp_path, 0, 7, 0)
    for k, (fn, match) in enumerate([(lambda rows: rows.clear(), "no result rows"),
                                     (lambda rows: rows[0].update(accuracy=""), "'accuracy' is missing or not numeric"),
                                     (lambda rows: rows[0].update(severity="high"), "severity 'high' is not a number"),
                                     (lambda rows: rows[0].update(model=""), "no 'model' value")]):
        b = _run_dir(tmp_path, 10 + k, 7, 1)
        rewrite(b, fn)
        with pytest.raises(MS.MultiSplitError, match=match):
            MS.aggregate([a, b], tmp_path / f"o{k}")



# ------------------------------------------------------------------------------------------------ D3
def _eff_dir(tmp_path, synthetic, comparable=True, name="e"):
    d = tmp_path / name
    d.mkdir()
    row = {"model": "facenet", "synthetic": synthetic, "n_images": 3}
    (d / "efficiency_results.csv").write_text("model,synthetic,n_images\nfacenet,%s,3\n" % synthetic)
    (d / "efficiency_results.json").write_text(json.dumps({"results": [row], "comparability": {
        "all_same_conditions": comparable, "differing_fields": [] if comparable else ["device_kind"]}}))
    return d


def test_synthetic_efficiency_output_is_invalid_not_present(tmp_path):
    idx = EI.build({"efficiency": _eff_dir(tmp_path, True)}, tmp_path / "out")
    e = next(p for p in idx["products"] if p["kind"] == "efficiency")
    assert e["status"] == "INVALID" and "NOT evidence" in e["problems"][0]
    assert idx["summary"]["INVALID"] == 1 and idx["summary"]["PRESENT"] == 0


def test_efficiency_csv_alone_marking_synthetic_is_enough_to_invalidate(tmp_path):
    d = _eff_dir(tmp_path, False)
    (d / "efficiency_results.csv").write_text("model,synthetic,n_images\nsynthetic,True,3\n")
    assert EI.inspect("efficiency", d)["status"] == "INVALID"


def test_real_efficiency_output_is_present_and_surfaces_comparability(tmp_path):
    ok = EI.inspect("efficiency", _eff_dir(tmp_path, False, True, "ok"))
    assert ok["status"] == "PRESENT" and ok["comparability"]["all_same_conditions"] is True and ok["caveats"] == []
    bad = EI.inspect("efficiency", _eff_dir(tmp_path, False, False, "bad"))
    assert bad["status"] == "PRESENT" and any("device_kind" in c and "do not compare" in c for c in bad["caveats"])
    md = tmp_path / "o"
    EI.build({"efficiency": tmp_path / "bad"}, md)
    assert "do not compare" in (md / "evidence_index.md").read_text()


def test_efficiency_json_that_is_not_harness_output_is_incomplete(tmp_path):
    d = _eff_dir(tmp_path, False)
    (d / "efficiency_results.json").write_text("{\"x\": 1}")
    assert EI.inspect("efficiency", d)["status"] == "INCOMPLETE"


# ------------------------------------------------------------------------------------------------ D4
def test_batch_larger_than_the_model_internal_batch_is_flagged_and_labelled():
    r = E.time_batches(lambda b: None, list(range(64)), 64, 0, internal_batch=32)
    assert r["requested_batch_size"] == 64 and r["model_internal_batch"] == 32 and r["effective_device_batch"] == 32
    assert r["limited_by_model"] is True and "sends 32 at a time" in r["label"]
    ok = E.time_batches(lambda b: None, list(range(64)), 8, 0, internal_batch=32)
    assert ok["limited_by_model"] is False and ok["effective_device_batch"] == 8
    one = E.time_batches(lambda b: None, list(range(8)), 4, 0, internal_batch=1)           # ONNX/dlib: one image at a time
    assert one["effective_device_batch"] == 1 and one["limited_by_model"] is True
    unl = E.time_batches(lambda b: None, list(range(8)), 4, 0)
    assert unl["model_internal_batch"] is None and unl["effective_device_batch"] == 4 and unl["limited_by_model"] is False


def test_internal_batch_per_model():
    assert E._internal_batch("facenet", types.SimpleNamespace(_batch_size=32)) == 32
    assert E._internal_batch("facenet", types.SimpleNamespace()) is None
    assert all(E._internal_batch(m, object()) == 1 for m in ("arcface", "insightface", "dlib_fr"))
    assert E._internal_batch("synthetic", object()) is None


def test_worker_and_csv_carry_the_batch_accounting(tmp_path):
    r = E.run_worker(E.SYNTHETIC, None, n_images=16, warmup=0, allow_stub=False, batch_sizes=(4,))
    assert r["model_internal_batch"] is None and r["batched"][0]["effective_device_batch"] == 4
    out = tmp_path / "o"
    assert E.main(["--models", E.SYNTHETIC, "--n-images", "16", "--warmup", "0", "--batch-sizes", "4", "--out", str(out)]) == 0
    header = (out / "efficiency_results.csv").read_text().splitlines()[0].split(",")
    assert {"batch4_effective_device_batch", "batch4_limited_by_model", "model_internal_batch"} <= set(header)


# ------------------------------------------------------------------------------------------------ R6
def _write(path, rows):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def test_calibration_figure_draws_points_not_lines_between_unrelated_conditions(tmp_path):
    rows = [{"model": "facenet", "perturbation": p, "severity": s, "label": f"{p}@{s}", "n_pairs": 10, "ece_before": 0.2, "ece_after": 0.05}
            for p, s in (("clean", 0.0), ("blur", 1.0), ("noise", 0.5))]
    _write(tmp_path / "calibration_results.csv", rows)
    fig, data = F.fig_calibration(rows)
    ax = fig.axes[0]
    assert len(data) == 3 and all(ln.get_linestyle() in ("None", "", " ") for ln in ax.lines)        # markers only
    assert [t.get_text() for t in ax.get_xticklabels()] == ["clean@0.0", "blur@1.0", "noise@0.5"]    # condition names, not indices
    F.plt.close(fig)


def test_valid_pairs_figure_with_all_nan_auc_raises_a_domain_error_and_writes_nothing(tmp_path):
    rows = [{"model": "m", "perturbation": "clean", "severity": 0.0, "label": "c", "valid_pair_rate_balanced": 0.9,
             "e2e_auc": "nan", "valid_auc": "nan"}]
    _write(tmp_path / "valid_pairs_by_condition.csv", rows)
    with pytest.raises(F.FigureError, match="no finite AUC"):
        F.make_figure("valid_pairs", tmp_path, tmp_path / "o")
    assert not (tmp_path / "o").exists()
    assert F.main(["valid_pairs", "--input", str(tmp_path), "--out", str(tmp_path / "o2")]) == 2


# ------------------------------------------------------------------------------------------------ R7
@pytest.mark.parametrize("module", ["multi_split", "figures", "env_pins", "valid_pairs", "efficiency", "fairness", "evidence_index", "fold0_results"])
def test_every_analysis_cli_starts_with_help(module, capsys):
    import importlib
    mod = importlib.import_module(f"faceeval.analysis.{module}")
    with pytest.raises(SystemExit) as ei:
        mod.main(["--help"])
    assert ei.value.code == 0 and "usage" in capsys.readouterr().out.lower()


def test_ci_workflow_smoke_checks_cover_the_new_clis():
    text = (Path(__file__).resolve().parents[1] / ".github" / "workflows" / "ci.yml").read_text()
    for m in ("multi_split", "figures", "env_pins", "evidence_index", "fairness"):
        assert f"python -m faceeval.analysis.{m} --help" in text, m

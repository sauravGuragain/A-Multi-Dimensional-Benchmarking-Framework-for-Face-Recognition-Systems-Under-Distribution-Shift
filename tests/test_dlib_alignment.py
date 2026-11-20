"""dlib_fr alignment, pinned weights and the InsightFace pin tool. Hermetic: dlib and the weight files are faked; tests that
need the real library skip when it (or the verified weights) is absent. Nothing here claims comparability of dlib_fr."""
from __future__ import annotations

import bz2
import hashlib
import json
import sys
import types
from pathlib import Path

import numpy as np
import pytest

from faceeval.analysis import env_pins
from faceeval.models.deep import dlib_weights as W
from faceeval.preprocessing import dlib_align as A
from faceeval.preprocessing.pipeline import alignment_for_config, describe_preprocessing, pipeline_for_model


class R:      # dlib.rectangle stand-in
    def __init__(s, l, t, r, b): s.l, s.t, s.r, s.b = l, t, r, b
    def left(s): return s.l
    def top(s): return s.t
    def width(s): return s.r - s.l
    def height(s): return s.b - s.t


def _fake_dlib(rects, scores):
    m = types.ModuleType("dlib")
    m.__version__ = "0.0-fake"
    class Det:
        def run(self, rgb, up, thr): return rects, scores, [0] * len(rects)
    m.get_frontal_face_detector = lambda: Det()
    m.shape_predictor = lambda path: (lambda rgb, rect: ("shape", rect))
    m.rectangle = lambda l, t, r, b: R(l, t, r, b)
    m.get_face_chip = lambda rgb, shape, size, padding: np.full((size, size, 3), 7, np.uint8)
    return m


@pytest.fixture
def fake(monkeypatch, tmp_path):
    def make(rects, scores):
        monkeypatch.setitem(sys.modules, "dlib", _fake_dlib(rects, scores))
        monkeypatch.setattr(A.W, "resolve", lambda spec, p=None, allow=True, **k: tmp_path / "x.dat")
    return make


def test_select_face_is_deterministic_largest_then_score_then_left():
    rs = [R(0, 0, 10, 10), R(5, 5, 25, 25), R(30, 0, 50, 20)]
    assert A.select_face(rs, [0.9, 0.1, 0.5]) == 2           # equal areas 400 vs 400: higher score wins
    assert A.select_face(rs, [0.9, 0.9, 0.9]) == 1           # tie on score: smaller left edge
    assert A.select_face([R(0, 0, 5, 5), R(0, 0, 9, 9)], [1.0, 0.0]) == 1
    with pytest.raises(ValueError):
        A.select_face([], [])


def test_aligner_returns_150px_bgr_chip_and_records_method(fake):
    fake([R(10, 10, 60, 60)], [1.2])
    f = A.DlibChipAligner().align(np.zeros((100, 100, 3), np.uint8), types.SimpleNamespace(image_path="p.jpg"))
    assert f.image.shape == (150, 150, 3) and f.image.dtype == np.uint8
    assert f.alignment_method == A.METHOD and (f.bounding_box.x, f.bounding_box.y) == (10, 10)
    a, b = A.DlibChipAligner().align(np.zeros((100, 100, 3), np.uint8)), A.DlibChipAligner().align(np.zeros((100, 100, 3), np.uint8))
    assert np.array_equal(a.image, b.image)                  # deterministic


def test_no_face_uses_the_counted_centre_box_fallback_never_none(fake):
    """Policy change (corrective patch): the experiment runner aborts a model when an aligner returns None, so a dlib miss
    goes through a labelled, counted centre-box fallback instead (see dlib_align module docstring)."""
    fake([], [])
    al = A.DlibChipAligner()
    f = al.align(np.zeros((100, 100, 3), np.uint8))
    assert f is not None and f.image.shape == (150, 150, 3) and f.alignment_method == A.CENTRE_BOX_METHOD
    assert (f.bounding_box.x, f.bounding_box.y, f.bounding_box.w) == (25, 25, 50)
    assert al.stats()["n_centre_box_fallback"] == 1 and al.stats()["centre_box_fallback_fraction"] == 1.0
    assert len(al.align_all(np.zeros((9, 9, 3), np.uint8))) == 1


def test_wrong_chip_size_is_refused():
    with pytest.raises(ValueError):
        A.DlibChipAligner(output_size=112)


def test_without_dlib_falls_back_visibly(monkeypatch):
    monkeypatch.setitem(sys.modules, "dlib", None)           # import raises ImportError
    al = A.DlibChipAligner()
    assert al.method == A.LEGACY_FALLBACK_METHOD and al.describe()["alignment"] == A.LEGACY_FALLBACK_METHOD
    assert al.describe()["dlib_version"] == "NOT_RECORDED" and al.align(np.zeros((9, 9, 3), np.uint8)) is None  # no DetectionResult given


def test_describe_records_method_detector_weights_and_pins(fake):
    fake([R(0, 0, 5, 5)], [1.0])
    d = A.DlibChipAligner().describe()
    assert d["alignment"] == A.METHOD and d["chip_size"] == 150 and d["padding"] == 0.25 and d["dlib_version"] == "0.0-fake"
    assert d["landmark_model_sha256_pinned"] == W.SHAPE_PREDICTOR_5.sha256 and W.SOURCE_COMMIT in d["weights_source"]
    json.dumps(d)


# ------------------------------------------------------------------ pipeline / config wiring
def test_default_pipeline_for_other_models_and_legacy_unchanged(fake):
    fake([R(0, 0, 5, 5)], [1.0])
    assert pipeline_for_model("dlib_fr").aligner.__class__.__name__ == "FaceAligner"
    assert pipeline_for_model("dlib_fr", alignment=None).aligner.__class__.__name__ == "FaceAligner"
    p = pipeline_for_model("dlib_fr", alignment="dlib_chip")
    assert isinstance(p.aligner, A.DlibChipAligner)
    assert "dlib_5pt" in describe_preprocessing(p)["alignment"]
    for m in ("arcface", "facenet", "eigenfaces"):
        with pytest.raises(ValueError):
            pipeline_for_model(m, alignment="dlib_chip")
    with pytest.raises(ValueError):
        pipeline_for_model("dlib_fr", alignment="bogus")


def test_alignment_for_config():
    C = types.SimpleNamespace
    assert alignment_for_config("dlib_fr", C(dlib_alignment="dlib_chip")) == "dlib_chip"
    assert alignment_for_config("dlib_fr", C(dlib_alignment="legacy")) is None
    assert alignment_for_config("dlib_fr", None) == "dlib_chip"
    assert alignment_for_config("arcface", C(dlib_alignment="dlib_chip")) is None


def test_dlib_alignment_field_serialisation(tmp_path):
    import yaml
    from faceeval.core.config import load_config
    p = tmp_path / "c.yaml"
    src = yaml.safe_load(open(Path(__file__).resolve().parents[1] / "configs" / "lfw_fold0_deep_dlib_fr.yaml"))
    p.write_text(yaml.safe_dump(src))
    default = load_config(str(p))
    assert "dlib_alignment" not in default.to_dict() and default.dlib_alignment == "dlib_chip"
    src["dlib_alignment"] = "legacy"
    p.write_text(yaml.safe_dump(src))
    legacy = load_config(str(p))
    assert legacy.to_dict()["dlib_alignment"] == "legacy" and legacy.config_hash != default.config_hash


# ------------------------------------------------------------------ weights: HTTPS + checksum
def _spec(payload=b"weights" * 50, url="https://example.test/w.bz2"):
    raw = bz2.compress(payload)
    return W.WeightSpec("w.dat", hashlib.sha256(payload).hexdigest(), hashlib.sha256(raw).hexdigest(), url, len(payload)), raw, payload


class Resp:
    def __init__(s, b): s.b = b
    def __enter__(s): return s
    def __exit__(s, *a): return False
    def read(s): return s.b


def test_pinned_specs_are_https_and_hashed():
    for s in (W.RECOGNITION, W.SHAPE_PREDICTOR_5):
        assert s.url.startswith("https://") and len(s.sha256) == 64 and len(s.archive_sha256) == 64 and W.SOURCE_COMMIT in s.url
    assert not any(s.url.startswith("http://") for s in (W.RECOGNITION, W.SHAPE_PREDICTOR_5))


def test_download_verifies_then_writes_atomically(tmp_path, monkeypatch):
    monkeypatch.setattr(W, "cache_dir", lambda: tmp_path)
    spec, raw, payload = _spec()
    p = W.resolve(spec, opener=lambda url, timeout: Resp(raw))
    assert p.read_bytes() == payload and list(tmp_path.glob("*.tmp")) == []
    assert W.resolve(spec, opener=lambda *a, **k: (_ for _ in ()).throw(AssertionError("cached file must not re-download"))) == p


def test_tampered_download_is_refused_and_nothing_written(tmp_path, monkeypatch):
    monkeypatch.setattr(W, "cache_dir", lambda: tmp_path)
    spec, raw, _ = _spec()
    with pytest.raises(W.WeightError, match="wrong SHA-256"):
        W.resolve(spec, opener=lambda url, timeout: Resp(bz2.compress(b"evil" * 50)))
    assert list(tmp_path.iterdir()) == []


def test_non_https_refused(tmp_path, monkeypatch):
    monkeypatch.setattr(W, "cache_dir", lambda: tmp_path)
    spec, raw, _ = _spec(url="http://dlib.net/w.bz2")
    with pytest.raises(W.WeightError, match="non-HTTPS"):
        W.resolve(spec, opener=lambda *a, **k: Resp(raw))


def test_corrupt_cache_and_explicit_path_are_refused(tmp_path, monkeypatch):
    monkeypatch.setattr(W, "cache_dir", lambda: tmp_path)
    spec, raw, payload = _spec()
    (tmp_path / spec.name).write_bytes(b"corrupt")
    with pytest.raises(W.WeightError, match="refusing to use"):
        W.resolve(spec, opener=lambda *a, **k: Resp(raw))
    good = tmp_path / "other.dat"
    good.write_bytes(payload)
    assert W.resolve(spec, explicit_path=str(good)) == good
    with pytest.raises(W.WeightError):
        W.resolve(spec, explicit_path=str(tmp_path / "missing.dat"))


def test_download_can_be_disabled(tmp_path, monkeypatch):
    monkeypatch.setattr(W, "cache_dir", lambda: tmp_path)
    spec, raw, _ = _spec()
    with pytest.raises(W.WeightError, match="downloading is disabled"):
        W.resolve(spec, allow_download=False, opener=lambda *a, **k: Resp(raw))
    monkeypatch.setenv(W.NO_DOWNLOAD_ENV, "1")
    with pytest.raises(W.WeightError, match="downloading is disabled"):
        W.resolve(spec, opener=lambda *a, **k: Resp(raw))


# ------------------------------------------------------------------ real dlib (skips when absent)
def test_real_dlib_chip_when_available():
    pytest.importorskip("dlib")
    try:
        W.resolve(W.SHAPE_PREDICTOR_5, allow_download=False)
    except W.WeightError:
        pytest.skip("verified shape predictor not cached")
    al = A.DlibChipAligner(allow_download=False)
    f = al.align(np.zeros((120, 120, 3), np.uint8))                    # blank image: dlib finds no face -> counted centre-box fallback
    assert f is not None and f.alignment_method == A.CENTRE_BOX_METHOD and f.image.shape == (150, 150, 3)
    assert al.stats()["n_centre_box_fallback"] == 1


# ------------------------------------------------------------------ env_pins
def _run(tmp_path, prov):
    (tmp_path / "verification_results.json").write_text(json.dumps({"provenance": prov}))
    return tmp_path


def test_env_pins_reads_recorded_versions_and_skips_not_recorded(tmp_path, capsys):
    r = _run(tmp_path, {"packages": {"insightface": "0.7.3", "onnxruntime": "1.18.0", "dlib": "NOT_RECORDED"}})
    assert env_pins.main(["--run", str(r), "--print"]) == 0
    out = capsys.readouterr().out
    assert "insightface==0.7.3" in out and "onnxruntime==1.18.0" in out and "dlib" not in out


def test_env_pins_refuses_missing_provenance_without_guessing(tmp_path, capsys):
    assert env_pins.main(["--run", str(_run(tmp_path, {"seed": 1})), "--print"]) == 2
    assert "Nothing is guessed" in capsys.readouterr().out
    assert env_pins.main(["--run", str(tmp_path / "nope"), "--print"]) == 2


def test_env_pins_write_never_overwrites_and_check_detects_mismatch(tmp_path, capsys):
    r = _run(tmp_path, {"packages": {"numpy": "0.0.1"}})
    out = tmp_path / "c.txt"
    assert env_pins.main(["--run", str(r), "--write", str(out)]) == 0 and out.read_text() == "numpy==0.0.1\n"
    assert env_pins.main(["--run", str(r), "--write", str(out)]) == 2
    assert env_pins.main(["--run", str(r), "--check"]) == 1
    assert "DIFFERENT" in capsys.readouterr().out

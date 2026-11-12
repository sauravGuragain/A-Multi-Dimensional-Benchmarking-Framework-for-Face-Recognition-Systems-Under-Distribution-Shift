"""Regression tests for the Audit 2 corrections (F1-F10)."""
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

import faceeval.evaluation.metrics as M
from faceeval.core.exceptions import OrchestratorError
from faceeval.models.deep.arcface import _InsightFaceWrapper
from faceeval.models.deep.insightface import _InsightFaceEmbedder, _flatten_nested_pack
from faceeval.orchestration import verification as V
from faceeval.orchestration.perturbed_inputs import needs_chips, perturb_batch
from faceeval.orchestration.runner import ExperimentRunner
from faceeval.perturbation.composer import PerturbationComposer
from faceeval.preprocessing.normalizer import Normalizer
from tests.test_verification_stage import _config, _make_fake_lfw


def _facenet_real_available():
    try:
        import facenet_pytorch, torch  # noqa: F401
        return True
    except Exception:
        return False


@pytest.fixture(scope="module")
def fake_data(tmp_path_factory):
    import tests.test_verification_stage as T
    old = T.N_SUBJECTS
    T.N_SUBJECTS = 60
    try:
        root = tmp_path_factory.mktemp("data60")
        _make_fake_lfw(root)
    finally:
        T.N_SUBJECTS = old
    return root


def _run(tmp_path, monkeypatch, data_root, allow_partial=False, **kw):
    monkeypatch.chdir(tmp_path)
    cfg = _config(tmp_path, data_root, **kw)
    cfg.allow_partial_results = allow_partial
    runner = ExperimentRunner(cfg)
    run = runner.run()
    return cfg, runner, run


# ---------------- F2: perturbation domain ----------------

def _chip(seed=0):
    return np.random.default_rng(seed).integers(0, 256, (112, 112, 3), dtype=np.uint8)


def _spec(composer, ptype_name, severity):
    from faceeval.core.types import PerturbationType
    specs = composer.specs_for_type(PerturbationType(ptype_name))
    return min(specs, key=lambda sp: abs(sp.severity - severity))


@pytest.mark.parametrize("mode,expect", [("arcface", True), ("facenet", True), ("imagenet", True),
                                         ("dlib", False), ("unit_range", False),
                                         ("grayscale", False), ("grayscale_eq", False)])
def test_needs_chips_only_for_non_unit_range_modes(mode, expect):
    assert needs_chips(Normalizer(mode=mode)) is expect


@pytest.mark.parametrize("mode", ["arcface", "facenet"])
def test_chip_perturbation_is_normalised_after_perturbing(mode):
    norm = Normalizer(mode=mode)
    composer = PerturbationComposer(severity_levels=[0.0, 1.0])
    chip = _chip()
    clean = [norm.normalize(chip)]
    for ptype in ("jpeg_compression", "gamma", "gaussian_noise", "brightness"):
        spec = _spec(composer, ptype, 1.0)
        out = perturb_batch(composer, spec, norm, clean, [chip], ["fixture/image_0001.jpg"])[0]   # noise needs ids
        pert = composer._get_perturbation(spec.perturbation_type).apply(chip, 1.0, image_id="fixture/image_0001.jpg")
        np.testing.assert_allclose(out, norm.normalize(pert))
        assert np.isfinite(out).all()
        if mode == "arcface":
            assert out.min() >= -1.001 and out.max() <= 1.001, ptype      # old path gave JPEG values up to ~170
    # severity 0 reproduces the clean normalised input exactly
    z = perturb_batch(composer, _spec(composer, "jpeg_compression", 0.0), norm, clean, [chip])[0]
    np.testing.assert_array_equal(z, clean[0])


def test_unit_range_models_keep_original_behaviour():
    norm = Normalizer(mode="grayscale_eq")
    composer = PerturbationComposer(severity_levels=[0.0, 1.0])
    arr = norm.normalize(_chip())
    spec = _spec(composer, "gaussian_blur", 1.0)
    got = perturb_batch(composer, spec, norm, [arr], None)[0]
    want = composer._get_perturbation(spec.perturbation_type).apply(arr, 1.0)
    np.testing.assert_array_equal(got, want)


def test_preprocess_records_equals_normalised_chips(tmp_path):
    from faceeval.preprocessing.pipeline import pipeline_for_model
    p = tmp_path / "a.jpg"
    cv2.imwrite(str(p), np.random.default_rng(1).integers(0, 255, (80, 80, 3), dtype=np.uint8))
    rec = SimpleNamespace(image_path=str(p), image_id="a.jpg")
    pipe = pipeline_for_model("arcface", skip_detection_for_pre_cropped=True)
    chips = ExperimentRunner._aligned_chips(pipe, [rec], None)
    assert chips[0].dtype == np.uint8
    np.testing.assert_array_equal(ExperimentRunner._preprocess_records(pipe, [rec], None)[0],
                                  pipe.normalizer.normalize(chips[0]))


# ---------------- F3: dataset validation / fingerprint ----------------

def _recs(root):
    return [SimpleNamespace(subject_id=f.parent.name, image_id=f"{f.parent.name}/{f.name}", image_path=str(f))
            for f in sorted((root / "lfw").rglob("*.jpg"))]


def test_dataset_fingerprint_stable_and_content_sensitive(fake_data, tmp_path):
    recs = _recs(fake_data)
    a = V.dataset_fingerprint(recs)
    assert a == V.dataset_fingerprint(list(reversed(recs)))          # order independent
    import shutil
    cp = tmp_path / "copy"; shutil.copytree(fake_data, cp)
    r2 = _recs(cp)
    assert r2[0].image_path != recs[0].image_path
    assert V.dataset_fingerprint(r2) == a                              # same ids + bytes, other paths
    with open(r2[0].image_path, "ab") as fh:
        fh.write(b"x")
    assert V.dataset_fingerprint(r2) != a


def test_dataset_fingerprint_missing_file_fails(fake_data, tmp_path):
    recs = _recs(fake_data)[:3]
    recs[1] = SimpleNamespace(**{**vars(recs[1]), "image_path": str(tmp_path / "nope.jpg")})
    with pytest.raises(OrchestratorError, match="missing"):
        V.dataset_fingerprint(recs)


def test_summary_records_actual_dataset(tmp_path, monkeypatch, fake_data):
    _, runner, _ = _run(tmp_path, monkeypatch, fake_data, protocol="lfw_fold0_verification", deep=())
    ds = runner.verification_summary["dataset"]
    assert ds["scope"] == "custom_or_subset" and ds["n_identities"] == 60
    assert len(ds["fingerprint_sha256"]) == 64
    assert runner.verification_summary["status"] == "COMPLETE"


# ---------------- F4: failures are not silently successful ----------------

def test_verification_failure_fails_run_but_keeps_outputs(tmp_path, monkeypatch, fake_data):
    from faceeval.models.traditional.lbph import LBPHRecognizer

    def boom(self, X, y):
        raise RuntimeError("boom")
    monkeypatch.setattr(LBPHRecognizer, "fit_embedding", boom)
    with pytest.raises(OrchestratorError, match="lbph"):
        _run(tmp_path, monkeypatch, fake_data, protocol="lfw_fold0_verification", deep=())
    out = next((tmp_path / "out" / "verification").iterdir())
    assert (out / "verification_results.csv").exists()               # eigenfaces rows kept
    assert json.loads((out / "verification_results.json").read_text())["status"] == "INCOMPLETE"


def test_allow_partial_results_keeps_previous_behaviour(tmp_path, monkeypatch, fake_data):
    from faceeval.models.traditional.lbph import LBPHRecognizer
    monkeypatch.setattr(LBPHRecognizer, "fit_embedding",
                        lambda self, X, y: (_ for _ in ()).throw(RuntimeError("boom")))
    _, runner, run = _run(tmp_path, monkeypatch, fake_data, allow_partial=True,
                          protocol="lfw_fold0_verification", deep=())
    assert runner.verification_summary["status"] == "INCOMPLETE" and runner.failures


def test_identification_failure_fails_run(tmp_path, monkeypatch, fake_data):
    from faceeval.models.traditional.lbph import LBPHRecognizer
    monkeypatch.setattr(LBPHRecognizer, "fit",
                        lambda self, X, y: (_ for _ in ()).throw(RuntimeError("boom")))
    with pytest.raises(OrchestratorError, match="lbph"):
        _run(tmp_path, monkeypatch, fake_data, protocol="identification", deep=())


# ---------------- F5: preflight + InsightFace pack repair ----------------

@pytest.mark.skipif(_facenet_real_available(), reason="real facenet available; stub path not reachable")
def test_preflight_fails_before_any_work(tmp_path, monkeypatch, fake_data):
    with pytest.raises(OrchestratorError, match="Preflight"):
        _run(tmp_path, monkeypatch, fake_data, protocol="lfw_fold0_verification", require_real=True)
    assert not (tmp_path / "out" / "verification").exists()           # nothing was computed


def test_flatten_nested_pack(tmp_path):
    root = tmp_path / "models"
    (root / "pk" / "pk").mkdir(parents=True)
    (root / "pk" / "pk" / "det.onnx").write_bytes(b"1")
    assert _flatten_nested_pack("pk", str(root)) is True
    assert (root / "pk" / "det.onnx").exists()
    assert _flatten_nested_pack("pk", str(root)) is False             # nothing left to do
    assert _flatten_nested_pack("absent", str(root)) is False


# ---------------- F6/F7: one embedding path, misses reported ----------------

class _FakeApp:
    def __init__(self, faces):
        self._faces = faces
        self.seen = []
        self.models = {"recognition": SimpleNamespace(input_size=(112, 112), get_feat=self._get_feat)}

    def get(self, img):
        self.seen.append(img)
        return self._faces

    def _get_feat(self, imgs):
        assert isinstance(imgs, list) and imgs[0].dtype == np.uint8 and imgs[0].shape == (112, 112, 3)
        return np.ones((len(imgs), 512), dtype=np.float32)


def _norm_face(seed=0):
    return Normalizer(mode="arcface").normalize(_chip(seed))


def test_arcface_detector_miss_is_zero_embedding_by_default():
    app = _FakeApp([])
    w = _InsightFaceWrapper(app)
    out = w.embed([_norm_face(), _norm_face(1)])
    assert not out.any() and (w.n_total, w.n_detector_miss, w.n_failed) == (2, 2, 2)
    np.testing.assert_array_equal(app.seen[0], _chip(0))              # lossless float->uint8 round trip


def test_arcface_crop_fallback_is_opt_in_and_uses_get_feat():
    w = _InsightFaceWrapper(_FakeApp([]), crop_fallback=True)
    out = w.embed([_norm_face()])
    assert np.isclose(np.linalg.norm(out[0]), 1.0) and w.n_detector_miss == 1 and w.n_failed == 0


def test_arcface_detected_face_uses_detector_embedding():
    face = SimpleNamespace(embedding=np.arange(1, 513, dtype=np.float32))
    w = _InsightFaceWrapper(_FakeApp([face]))
    assert w.embed([_norm_face()]).any() and w.n_detector_miss == 0


def test_insightface_embedder_counts_misses_and_rounds():
    app = _FakeApp([])
    e = _InsightFaceEmbedder(app, 512)
    out = e.embed([_norm_face()])
    assert not out.any() and (e.n_total, e.n_detector_miss) == (1, 1)
    np.testing.assert_array_equal(app.seen[0], _chip(0))


# ---------------- F1 / F10 ----------------

def test_adc_auc_has_no_removed_numpy_api(monkeypatch):
    from faceeval.core.types import ADCPoint, PerturbationType
    from faceeval.fingerprint.adc import _compute_auc
    if hasattr(np, "trapz"):
        monkeypatch.delattr(np, "trapz")                              # simulate NumPy >= 2.4
    pt = list(PerturbationType)[0]
    pts = [ADCPoint(s, a, pt) for s, a in [(0.0, 0.9), (0.5, 0.7), (1.0, 0.4)]]
    assert _compute_auc(pts, 10) == pytest.approx((0.775 - 0.1) / 0.9 * 0 + (0.5 * (0.9 + 0.7) / 2 + 0.5 * (0.7 + 0.4) / 2 - 0.1) / 0.9)


@pytest.mark.parametrize("decimals", [None, 1])
def test_roc_thresholds_aligned_with_points(decimals):
    rng = np.random.default_rng(2)
    y = rng.random(500) < 0.5
    s = rng.normal(0.3, 0.2, 500) + 0.3 * y
    if decimals is not None:
        s = np.round(s, decimals)
    roc = M.compute_verification_metrics(s.tolist(), y.tolist())["roc_curve"]
    assert len(roc.fpr) == len(roc.tpr) == len(roc.thresholds)
    for f, t, th in zip(roc.fpr, roc.tpr, roc.thresholds):
        pred = s >= th
        assert pred[~y].mean() == pytest.approx(f) and pred[y].mean() == pytest.approx(t)


# ---------------- F9: environment-independent content hashes ----------------

def _manifest(py="3.12.1", npv="1.26.4", code="abc"):
    from faceeval.evaluation import split_manifest as sm
    parts = {"train": [(f"t{i}", f"t{i}/{j}.jpg") for i in range(4) for j in range(3)],
             "validation": [(f"v{i}", f"v{i}/{j}.jpg") for i in range(2) for j in range(3)],
             "test": [(f"x{i}", f"x{i}/{j}.jpg") for i in range(2) for j in range(3)]}
    meta = {"dataset_name": "d", "fold": "0", "seed": "42", "python_version": py, "numpy_version": npv,
            "code_version": code, "shuffled_order_sha256": "0" * 64}
    return sm.build_manifest(parts, meta, {"train": 4, "validation": 2, "test": 2},
                             {"train": 12, "validation": 6, "test": 6})


def test_split_content_hash_ignores_environment_fields():
    from faceeval.evaluation import split_manifest as sm
    a, b = _manifest(), _manifest(py="3.13.0", npv="2.5.3", code="zzz")
    assert sm.manifest_sha256(a) != sm.manifest_sha256(b)             # full hash is environment dependent
    assert V.split_content_sha256(a) == V.split_content_sha256(b)


def test_pair_content_hash_ignores_numpy_version():
    from faceeval.evaluation import pairs as P
    imgs = [(f"id{i}", f"id{i}/{k}.jpg") for i in range(30) for k in range(4)]
    ps = P.build_pairs(imgs, seed=42, partition_code=1)
    h1 = V.pair_content_sha256(ps)
    ps.params["numpy_version"] = "9.9.9"
    assert V.pair_content_sha256(ps) == h1
    ps.params["seed"] = 43
    assert V.pair_content_sha256(ps) != h1


# ---------------- Round 3: full-LFW identity, early failures, preflight reuse, safe repair ----------------

def test_official_counts_alone_do_not_make_full_lfw(tmp_path, monkeypatch, fake_data):
    monkeypatch.setattr(V, "FULL_LFW_IDENTITIES", 60)
    monkeypatch.setattr(V, "FULL_LFW_IMAGES", len(_recs(fake_data)))
    _, runner, _ = _run(tmp_path, monkeypatch, fake_data, protocol="lfw_fold0_verification", deep=())
    s = runner.verification_summary
    assert s["dataset"]["counts_match_official_lfw"] is True
    assert s["full_lfw"] is False and s["dataset"]["scope"] == "custom_or_subset"


def test_verified_fingerprint_activates_strict_full_lfw_path(tmp_path, monkeypatch, fake_data):
    """Official counts + registered fingerprint -> full_lfw, strict (V1-count, build_fold0_pairs) path."""
    recs = _recs(fake_data)
    monkeypatch.setattr(V, "FULL_LFW_IDENTITIES", 60)
    monkeypatch.setattr(V, "FULL_LFW_IMAGES", len(recs))
    monkeypatch.setattr(V, "OFFICIAL_LFW_FINGERPRINTS", frozenset({V.dataset_fingerprint(recs)}))
    seen = {"expected": [], "fold0_codes": []}
    real_manifest, real_pairs = V.sm.build_manifest, V.pairs_mod.build_pairs
    def spy_manifest(parts, meta, exp_ids=None, exp_imgs=None):
        seen["expected"].append((exp_ids, exp_imgs))        # (None, None) => strict official V1 acceptance counts
        # the tiny fixture cannot meet the official counts, so relax them for the call only
        return real_manifest(parts, meta, {p: len({i for i, _ in r}) for p, r in parts.items()},
                             {p: len(r) for p, r in parts.items()})
    def spy_fold0_pairs(rows, code):
        seen["fold0_codes"].append(code)
        return real_pairs(rows, partition_code=code, **V.pairs_mod.FOLD0_DEFAULTS)
    monkeypatch.setattr(V.sm, "build_manifest", spy_manifest)
    monkeypatch.setattr(V.pairs_mod, "build_fold0_pairs", spy_fold0_pairs)
    _, runner, _ = _run(tmp_path, monkeypatch, fake_data, protocol="lfw_fold0_verification", deep=())
    s = runner.verification_summary
    assert s["full_lfw"] is True
    assert s["dataset"]["scope"] == "official_lfw" and s["dataset"]["counts_match_official_lfw"] is True
    assert seen["expected"] == [(None, None)]                           # official manifest requested once
    assert seen["fold0_codes"] == [V.PARTITION_CODE["validation"], V.PARTITION_CODE["test"]] == [1, 2]


def test_fingerprint_reads_each_file_once_in_bounded_chunks(fake_data, monkeypatch):
    import builtins
    recs = _recs(fake_data)[:5]
    opened, real_open = [], builtins.open
    def counting_open(f, *a, **k):
        fh = real_open(f, *a, **k)
        if str(f).endswith(".jpg"):
            opened.append(str(f))
            orig = fh.read
            fh.read = lambda n=-1: (_ for _ in ()).throw(AssertionError("unbounded read")) if n in (-1, None) else orig(n)
        return fh
    monkeypatch.setattr(builtins, "open", counting_open)
    V.dataset_fingerprint(recs)
    assert sorted(opened) == sorted(r.image_path for r in recs)


def test_verification_runs_one_fingerprint_per_run(tmp_path, monkeypatch, fake_data):
    calls = []
    real = V.dataset_fingerprint
    monkeypatch.setattr(V, "dataset_fingerprint", lambda r: calls.append(1) or real(r))
    _run(tmp_path, monkeypatch, fake_data, protocol="lfw_fold0_verification", deep=())
    assert len(calls) == 1


class _FakeDeep:
    """Stands in for a deep model whose backbone is (or is not) real."""
    loads = 0
    def __init__(self, real=True, miss=0.0):
        type(self).loads += 1
        self.real = real
    def assert_real_backbone(self):
        from faceeval.core.exceptions import EmbeddingError
        if not self.real:
            raise EmbeddingError("stub")


def test_early_real_model_failure_writes_incomplete_summary_and_keeps_rows(tmp_path, monkeypatch, fake_data):
    """Eigenfaces row is computed, then a deep model fails with require_real_models: summary still written."""
    from faceeval.core.registry import model_registry
    from faceeval.models.deep.facenet import FaceNetRecognizer
    from faceeval.core.exceptions import EmbeddingError
    n = []
    def assert_real(self):                       # preflight passes, the in-run check fails
        n.append(1)
        if len(n) > 1:
            raise EmbeddingError("backbone went away")
    monkeypatch.setattr(FaceNetRecognizer, "assert_real_backbone", assert_real)
    monkeypatch.setattr(FaceNetRecognizer, "is_stub_backbone", lambda self: True)       # fail inside verification
    cfg = _config(tmp_path, fake_data, "lfw_fold0_verification", deep=("facenet",), require_real=True)
    cfg.traditional_models = [m for m in cfg.traditional_models if m.value == "eigenfaces"]
    monkeypatch.chdir(tmp_path)
    runner = ExperimentRunner(cfg)
    with pytest.raises(OrchestratorError):
        runner.run()
    out = next((tmp_path / "out" / "verification").iterdir())
    data = json.loads((out / "verification_results.json").read_text())
    assert data["status"] == "INCOMPLETE" and data["failures"][0]["model"] == "facenet"
    assert data["n_rows"] > 0 and (out / "verification_results.csv").exists()   # completed work kept


def test_failure_before_any_model_still_writes_incomplete_summary(tmp_path, fake_data):
    recs = _recs(fake_data)[:4]
    recs[0].image_path = str(tmp_path / "gone.jpg")
    cfg = SimpleNamespace(output_dir=str(tmp_path), run_id="r1", random_seed=42)
    with pytest.raises(OrchestratorError, match="missing"):
        V.run_fold0_verification(config=cfg, records=recs, raw_images=None, model_names=[], trad_names=set(),
                                 composer=None, specs=[], dataset_name="d", preprocess=None,
                                 adapt_hyperparameters=None)
    data = json.loads((tmp_path / "verification" / "r1" / "verification_results.json").read_text())
    assert data["status"] == "INCOMPLETE" and "missing" in data["error"]


def test_preflight_holds_one_model_and_reuses_first(monkeypatch, tmp_path):
    from faceeval.core.registry import model_registry
    made = []
    class M:
        def __init__(self): made.append(self)
        def assert_real_backbone(self): pass
    monkeypatch.setattr(model_registry, "get", lambda name, **k: M())
    r = ExperimentRunner.__new__(ExperimentRunner)
    r._preflight_kept = {}
    r._preflight_models(["a", "b", "c", "lbph"], {"lbph"})
    assert len(made) == 3 and list(r._preflight_kept) == ["a"]        # only the first evaluated model is kept
    first = r._get_model("a")
    assert first is made[-1] and len(made) == 3                       # reused, no 4th load
    r._get_model("b"); assert len(made) == 4                          # others load once, at use


def test_flatten_nested_pack_never_overwrites_or_moves_other_files(tmp_path):
    root = tmp_path / "m"; pack = root / "pk"; nested = pack / "pk"; nested.mkdir(parents=True)
    (nested / "a.onnx").write_bytes(b"A-new"); (nested / "b.onnx").write_bytes(b"B-nested")
    (nested / "readme.txt").write_text("keep"); (pack / "notes.txt").write_text("unrelated")
    (pack / "b.onnx").write_bytes(b"B-existing")             # real destination collision
    assert _flatten_nested_pack("pk", str(root)) is True     # a.onnx moved, b.onnx collides
    assert (pack / "a.onnx").read_bytes() == b"A-new" and not (nested / "a.onnx").exists()
    assert (pack / "b.onnx").read_bytes() == b"B-existing"
    assert (nested / "b.onnx").read_bytes() == b"B-nested"   # not moved over the destination
    assert (nested / "readme.txt").read_text() == "keep" and (pack / "notes.txt").read_text() == "unrelated"


def test_flatten_nested_pack_moves_free_files_only(tmp_path):
    root = tmp_path / "m"; pack = root / "pk"; nested = pack / "pk"; nested.mkdir(parents=True)
    (nested / "a.onnx").write_bytes(b"A"); (nested / "b.onnx").write_bytes(b"B-nested")
    (nested / "readme.txt").write_text("keep")
    (pack / "b.onnx.bak").write_text("x")
    assert _flatten_nested_pack("pk", str(root)) is True
    assert (pack / "a.onnx").read_bytes() == b"A" and not (nested / "a.onnx").exists()
    assert (pack / "b.onnx").read_bytes() == b"B-nested"     # no existing b.onnx here -> moved
    assert (nested / "readme.txt").exists() and (pack / "b.onnx.bak").read_text() == "x"


def test_flatten_nested_pack_does_not_follow_dangling_symlink(tmp_path):
    root = tmp_path / "m"; pack = root / "p2"; nested = pack / "p2"; nested.mkdir(parents=True)
    (nested / "z.onnx").write_bytes(b"1")
    (pack / "a.txt").write_text("x")                         # pack has no .onnx directly -> repair applies
    import os
    os.symlink(str(root / "nowhere"), str(pack / "z.onnx"))
    assert _flatten_nested_pack("p2", str(root)) is False
    assert os.path.islink(pack / "z.onnx") and not os.path.exists(pack / "z.onnx")   # still dangling, untouched
    assert (nested / "z.onnx").read_bytes() == b"1"


def test_flatten_ignores_non_nested_layouts(tmp_path):
    (tmp_path / "pk").mkdir(); (tmp_path / "pk" / "x.onnx").write_bytes(b"1")
    assert _flatten_nested_pack("pk", str(tmp_path)) is False
    (tmp_path / "q" / "other").mkdir(parents=True); (tmp_path / "q" / "other" / "a.onnx").write_bytes(b"1")
    assert _flatten_nested_pack("q", str(tmp_path)) is False          # nested dir name != pack name
    assert (tmp_path / "q" / "other" / "a.onnx").exists()


def _fake_insightface(monkeypatch, app):
    import sys, types
    mod = types.ModuleType("insightface"); mod.app = types.SimpleNamespace(FaceAnalysis=lambda **k: app)
    app.prepare = lambda ctx_id=0: None
    monkeypatch.setitem(sys.modules, "insightface", mod)


def test_insightface_recognizer_load_retries_after_prepare_failure(monkeypatch, tmp_path):
    from faceeval.models.deep.insightface import InsightFaceRecognizer
    calls = {"prepare": 0}
    app = _FakeApp([])
    _fake_insightface(monkeypatch, app)
    def prepare(ctx_id=0):
        calls["prepare"] += 1
        if calls["prepare"] == 1:
            raise AssertionError("no detection model")
    app.prepare = prepare
    monkeypatch.setattr("faceeval.models.deep.insightface._flatten_nested_pack", lambda pack: True)
    m = InsightFaceRecognizer()
    assert m.is_real_backbone() and calls["prepare"] == 2


def test_arcface_real_backbone_check_uses_actual_assignment(monkeypatch):
    import sys, types
    from faceeval.models.deep import arcface as A
    from faceeval.models.deep.arcface import ArcFaceRecognizer
    app = _FakeApp([])
    _fake_insightface(monkeypatch, app)
    zoo = types.ModuleType("insightface.model_zoo"); zoo.get_model = lambda *a, **k: None
    monkeypatch.setitem(sys.modules, "insightface.model_zoo", zoo)
    monkeypatch.setattr(A, "_check_insightface", lambda: True)
    m = ArcFaceRecognizer()
    assert m.is_real_backbone() and type(m._backbone).__name__ == "_InsightFaceWrapper"
    monkeypatch.setattr(A, "_check_insightface", lambda: False)
    assert not ArcFaceRecognizer().is_real_backbone()               # stub path rejected by allow-list


def test_verification_reporting_path_uses_backbone_counters_synthetic(tmp_path, monkeypatch, fake_data):
    """Reporting path only: run_fold0_verification turns a backbone's n_total/n_detector_miss counters
    into row fields. The counters are synthetic (a fake backbone on Eigenfaces); this does NOT exercise
    the real ArcFace/InsightFace wrappers (see the wrapper counter tests above)."""
    from faceeval.models.traditional.eigenfaces import EigenfacesRecognizer
    class Counting:
        n_total = 0; n_detector_miss = 0
    orig_extract, orig_fit = EigenfacesRecognizer.extract_features, EigenfacesRecognizer.fit_embedding
    def fit(self, X, y):                         # deep models have a loaded backbone before embedding
        self._backbone = Counting()
        return orig_fit(self, X, y)
    monkeypatch.setattr(EigenfacesRecognizer, "fit_embedding", fit)
    def extract(self, X):
        bb = self._backbone
        bb.n_total += len(X); bb.n_detector_miss += len(X) // 4
        return orig_extract(self, X)
    monkeypatch.setattr(EigenfacesRecognizer, "extract_features", extract)
    _, runner, _ = _run(tmp_path, monkeypatch, fake_data, protocol="lfw_fold0_verification", deep=())
    rows = [r for r in runner.verification_summary["results"] if r["model"] == "eigenfaces"]
    assert rows and all(0.2 < r["detector_miss_rate"] < 0.3 and 0.2 < r["val_detector_miss_rate"] < 0.3 for r in rows)


def test_get_model_reuses_retained_falsy_model_without_registry_call(monkeypatch):
    from faceeval.core.registry import model_registry
    class Falsy:
        def __bool__(self): return False
    calls = []
    monkeypatch.setattr(model_registry, "get", lambda name, **k: calls.append(name))
    r = ExperimentRunner.__new__(ExperimentRunner)
    kept = Falsy()
    r._preflight_kept = {"m": kept}
    assert r._get_model("m") is kept and calls == [] and "m" not in r._preflight_kept
    monkeypatch.setattr(model_registry, "get", lambda name, **k: ("fresh", name))
    assert r._get_model("m") == ("fresh", "m")               # consumed: next call goes to the registry

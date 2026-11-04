"""Tests for the Fold 0 verification stage integrated in ExperimentRunner (fake LFW on disk)."""
import csv
import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from faceeval.core.exceptions import OrchestratorError
from faceeval.orchestration import verification as V
from faceeval.orchestration.runner import ExperimentRunner

N_SUBJECTS = 120


def _make_fake_lfw(root: Path) -> None:
    rng = np.random.default_rng(0)
    for s in range(N_SUBJECTS):
        d = root / "lfw" / f"Person_{s:03d}"
        d.mkdir(parents=True)
        base = rng.integers(0, 255, (100, 100, 3)).astype(np.float32)
        for k in range(8 + (s % 3)):
            img = np.clip(base + rng.normal(0, 12, base.shape), 0, 255).astype(np.uint8)
            cv2.imwrite(str(d / f"Person_{s:03d}_{k + 1:04d}.jpg"), img)


@pytest.fixture(scope="module")
def fake_data(tmp_path_factory):
    root = tmp_path_factory.mktemp("data")
    _make_fake_lfw(root)
    return root


def _config(tmp_path, data_root, protocol, deep=("facenet",), require_real=False):
    from faceeval.core.config import ConfigLoader
    deep_yaml = "".join(f"  - {m}\n" for m in deep)
    deep_block = f"deep_models:\n{deep_yaml}" if deep else "deep_models: []\n"
    text = f"""
experiment_name: vtest_{protocol}
random_seed: 42
dataset_names: [lfw]
data_root: {data_root}
split_strategy: stratified
evaluation_protocol: {protocol}
require_real_models: {str(require_real).lower()}
traditional_models: [eigenfaces, lbph]
{deep_block}
perturbation_schedule:
  perturbation_types: [gaussian_blur, brightness]
  severity_levels: [0.0, 1.0]
  compose_perturbations: false
compute_calibration: false
compute_fairness: false
compute_fingerprints: false
compute_failure_analysis: false
compute_deployment_scores: false
deployment_scenarios: [research_benchmark]
profile_resources: false
use_feature_cache: false
feature_cache_dir: {tmp_path}/cache/
output_dir: {tmp_path}/out/
report_dir: {tmp_path}/rep/
export_png: false
mlflow_tracking_uri: sqlite:///{tmp_path}/mlflow.db
mlflow_experiment_name: vtest
device: cpu
"""
    return ConfigLoader().loads(text)


def _run(tmp_path, monkeypatch, data_root, **kw):
    monkeypatch.chdir(tmp_path)
    cfg = _config(tmp_path, data_root, **kw)
    runner = ExperimentRunner(cfg)
    try:
        run = runner.run()
    finally:
        pass
    return cfg, runner, run


def test_config_fields_roundtrip(tmp_path, fake_data):
    cfg = _config(tmp_path, fake_data, "lfw_fold0_verification", require_real=True)
    d = cfg.to_dict()
    assert d["evaluation_protocol"] == "lfw_fold0_verification"
    assert d["require_real_models"] is True


def test_config_rejects_bad_protocol(tmp_path, fake_data):
    with pytest.raises(Exception):
        _config(tmp_path, fake_data, "bogus")


def _brute_best_acc(s, y):
    """Reference: try every distinct threshold plus one above the max and one at the min."""
    cands = sorted(set(s.tolist()) | {np.nextafter(s.max(), np.inf)})
    return max(((s >= t) == y).mean() for t in cands)


def _acc(s, y, t):
    return float(((s >= t) == y).mean())


def test_threshold_mixed():
    s = np.array([0.9, 0.8, 0.7, 0.2, 0.1])
    y = np.array([True, True, False, False, False])
    t = V.best_accuracy_threshold(s, y)
    assert t == 0.8 and _acc(s, y, t) == 1.0


def test_threshold_all_impostor_rejects_everything():
    s = np.array([0.9, 0.5, 0.1])
    y = np.zeros(3, bool)
    t = V.best_accuracy_threshold(s, y)
    assert t > s.max() and _acc(s, y, t) == 1.0


def test_threshold_all_genuine_accepts_everything():
    s = np.array([0.9, 0.5, 0.1])
    y = np.ones(3, bool)
    t = V.best_accuracy_threshold(s, y)
    assert t == s.min() and _acc(s, y, t) == 1.0


def test_threshold_reject_all_beats_any_cut():
    # impostors score higher than genuines: best possible is to reject everything of the majority
    s = np.array([0.9, 0.8, 0.7, 0.2])
    y = np.array([False, False, False, True])
    t = V.best_accuracy_threshold(s, y)
    assert t > s.max() and _acc(s, y, t) == 0.75


def test_threshold_ties_move_together():
    s = np.array([0.5, 0.5, 0.5, 0.1])
    y = np.array([True, False, True, False])
    t = V.best_accuracy_threshold(s, y)
    assert t in (0.5, 0.1) and _acc(s, y, t) == 0.75 and t == 0.5   # never splits the 0.5 tie
    s2 = np.full(4, 0.3); y2 = np.array([True, True, True, False])
    t2 = V.best_accuracy_threshold(s2, y2)
    assert t2 == 0.3 and _acc(s2, y2, t2) == 0.75                     # accept-all


def test_threshold_matches_brute_force_random():
    rng = np.random.default_rng(3)
    for _ in range(200):
        n = int(rng.integers(1, 30))
        s = np.round(rng.random(n), 1)                                # many ties
        y = rng.random(n) < 0.5
        t = V.best_accuracy_threshold(s, y)
        assert _acc(s, y, t) == pytest.approx(_brute_best_acc(s, y))


@pytest.mark.parametrize("scores,labels", [
    ([], []),
    ([0.1, 0.2], [True]),
    ([[0.1, 0.2]], [[True, False]]),
    ([0.1, float("nan")], [True, False]),
    ([0.1, float("inf")], [True, False]),
    ([0.1, 0.2], [2, 0]),
])
def test_threshold_invalid_input(scores, labels):
    with pytest.raises(ValueError):
        V.best_accuracy_threshold(scores, labels)


def test_threshold_accepts_zero_one_labels():
    t = V.best_accuracy_threshold([0.9, 0.1], [1, 0])
    assert t == 0.9


def test_shipped_verification_config_flags_match_protocol():
    from faceeval.core.config import load_config
    c = load_config(Path(__file__).resolve().parent.parent / "configs" / "lfw_fold0_verification_full.yaml")
    assert c.evaluation_protocol == "lfw_fold0_verification" and c.require_real_models
    assert not (c.compute_calibration or c.compute_fairness or c.compute_fingerprints
                or c.compute_failure_analysis or c.compute_deployment_scores)


def test_verification_protocol_end_to_end(tmp_path, monkeypatch, fake_data):
    cfg, runner, run = _run(tmp_path, monkeypatch, fake_data, protocol="lfw_fold0_verification")
    summ = runner.verification_summary
    out = Path(summ["output_dir"])
    assert (out / "split_manifest_fold0.tsv").exists()
    assert (out / "split_manifest_fold0.tsv.sha256").exists()
    assert (out / "pairs_validation_fold0.tsv").exists() and (out / "pairs_test_fold0.tsv").exists()
    assert summ["full_lfw"] is False                       # fake subset -> relaxed counts, flagged
    rows = list(csv.DictReader(open(out / "verification_results.csv")))
    assert {r["model"] for r in rows} == {"eigenfaces", "lbph", "facenet"}
    assert len(rows) == 3 * 4                              # 3 models x (2 types x 2 severities)
    for r in rows:
        assert 0.0 <= float(r["auc"]) <= 1.0 and 0.0 <= float(r["accuracy"]) <= 1.0
    # identification untouched in verification-only mode
    assert run.evaluation_results == []
    js = json.load(open(out / "verification_results.json"))
    assert js["manifest_sha256"] == summ["manifest_sha256"]
    # stub flag recorded for facenet when the real backbone is missing
    fn = [r for r in rows if r["model"] == "facenet"]
    assert all(r["backbone"] in ("real", "STUB") for r in fn)


def test_manifest_and_pairs_deterministic(tmp_path, monkeypatch, fake_data):
    a = tmp_path / "a"; b = tmp_path / "b"; a.mkdir(); b.mkdir()
    _, r1, _ = _run(a, monkeypatch, fake_data, protocol="lfw_fold0_verification", deep=())
    _, r2, _ = _run(b, monkeypatch, fake_data, protocol="lfw_fold0_verification", deep=())
    s1, s2 = r1.verification_summary, r2.verification_summary
    assert s1["pair_manifest_sha256"] == s2["pair_manifest_sha256"]
    ra = lambda s: Path(s["manifest_path"]).read_bytes().split(b"code_version")[0]
    assert ra(s1) == ra(s2)


def test_threshold_uses_validation_only(tmp_path, monkeypatch, fake_data):
    _, runner, _ = _run(tmp_path, monkeypatch, fake_data, protocol="lfw_fold0_verification", deep=())
    rows = runner.verification_summary["results"]
    # one frozen threshold per model across all perturbation conditions
    for m in {r["model"] for r in rows}:
        assert len({r["threshold"] for r in rows if r["model"] == m}) == 1


def test_both_protocols(tmp_path, monkeypatch, fake_data):
    _, runner, run = _run(tmp_path, monkeypatch, fake_data, protocol="both", deep=())
    assert runner.verification_summary["n_rows"] > 0
    assert len(run.evaluation_results) > 0


def _facenet_real_available():
    try:
        import facenet_pytorch, torch  # noqa: F401
        return True
    except Exception:
        return False


@pytest.mark.skipif(_facenet_real_available(), reason="real facenet available; stub path not reachable")
def test_require_real_models_blocks_stub(tmp_path, monkeypatch, fake_data):
    with pytest.raises(OrchestratorError):
        _run(tmp_path, monkeypatch, fake_data, protocol="lfw_fold0_verification", require_real=True)
    with pytest.raises(OrchestratorError):
        _run(tmp_path / "x" if (tmp_path / "x").mkdir() is None else tmp_path, monkeypatch, fake_data,
             protocol="identification", require_real=True)


# ---- real-backbone detection (allow-list) ----

def _fake_backbone(name):
    return type(name, (), {})()


def test_facenet_real_backbone_detection():
    from faceeval.models.deep.facenet import FaceNetRecognizer
    m = FaceNetRecognizer(pretrained="vggface2")
    m._backbone = _fake_backbone("_FaceNetStub"); assert not m.is_real_backbone()
    m._backbone = _fake_backbone("SomethingElse"); assert not m.is_real_backbone()
    m._backbone = _fake_backbone("InceptionResnetV1"); assert m.is_real_backbone()
    with pytest.raises(Exception):
        m._backbone = _fake_backbone("_FaceNetStub"); m.assert_real_backbone()
    r = FaceNetRecognizer(pretrained=None)          # randomly initialised weights
    r._backbone = _fake_backbone("InceptionResnetV1")
    assert not r.is_real_backbone() and r.is_stub_backbone()
    with pytest.raises(Exception):
        r.assert_real_backbone()


def test_arcface_other_models_allow_list():
    from faceeval.models.deep.arcface import ArcFaceRecognizer
    m = ArcFaceRecognizer()
    m._backbone = _fake_backbone("_ArcFaceStub"); assert not m.is_real_backbone()
    m._backbone = _fake_backbone("_InsightFaceWrapper"); assert m.is_real_backbone()

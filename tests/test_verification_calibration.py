"""Calibration integrated in the Fold-0 verification runner.

SCOPE: unit tests use synthetic scores; the integration tests run the real runner on the fake on-disk LFW of
``test_verification_stage`` (random noise images, stub FaceNet backbone). They test the PROTOCOL and plumbing
(validation-only fit, additive artifacts, failure isolation); no number here says anything about real models.
"""
import csv
import json
import math
import os
import sys
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent))
import test_verification_stage as T  # noqa: E402
from faceeval.evaluation import verification_calibration as C  # noqa: E402
from faceeval.orchestration.runner import ExperimentRunner  # noqa: E402


# ------------------------------------------------------------------ unit
def _logistic_data(n=4000, a=6.0, b=-1.0, seed=0):
    rng = np.random.default_rng(seed)
    s = rng.uniform(-0.2, 0.9, n)
    p = 1 / (1 + np.exp(-(a * s + b)))
    return s, rng.random(n) < p


def test_fit_platt_recovers_known_parameters():
    s, y = _logistic_data()
    cal = C.fit_platt(s, y)
    assert cal.a == pytest.approx(6.0, abs=0.5) and cal.b == pytest.approx(-1.0, abs=0.3) and cal.n_fit == len(s)
    assert (cal.predict(np.array([-1.0, 0.0, 1.0])) > 0).all()


@pytest.mark.parametrize("scores,labels,msg", [
    ([0.1] * 30, [True] * 30, "at least"),
    ([0.1] * 15 + [0.2] * 5, [True] * 15 + [False] * 5, "at least"),
    ([float("nan")] * 20 + [0.1] * 20, [True] * 20 + [False] * 20, "finite"),
    ([], [], "non-empty"),
])
def test_fit_platt_refuses_bad_input(scores, labels, msg):
    with pytest.raises(ValueError, match=msg):
        C.fit_platt(np.array(scores), np.array(labels))


def test_metrics_known_values():
    y = np.array([1, 0] * 50, dtype=bool)
    m = C.calibration_metrics(np.full(100, 0.9), y)
    assert m["ece"] == pytest.approx(0.4) and m["mce"] == pytest.approx(0.4)
    assert m["brier"] == pytest.approx(0.5 * 0.01 + 0.5 * 0.81)
    assert m["nll"] == pytest.approx(-(0.5 * math.log(0.9) + 0.5 * math.log(0.1)))
    perfect = C.calibration_metrics(np.where(y, 1.0, 0.0), y)
    assert perfect["ece"] == 0.0 and perfect["brier"] == 0.0
    with pytest.raises(ValueError):
        C.calibration_metrics(np.array([1.2]), np.array([True]))
    with pytest.raises(ValueError):
        C.calibration_metrics(np.array([0.5, 0.5]), np.array([True]))


def test_calibration_improves_validation_and_held_out_when_scores_are_miscalibrated():
    sv, yv = _logistic_data(seed=1)
    st, yt = _logistic_data(seed=2)
    cal = C.fit_platt(sv, yv)
    before = C.calibration_metrics(C.naive_probability(st), yt)
    after = C.calibration_metrics(cal.predict(st), yt)
    assert after["ece"] < before["ece"] and after["brier"] < before["brier"] and after["nll"] < before["nll"]


def test_recorder_fits_on_validation_only_and_isolates_errors(tmp_path):
    sv, yv = _logistic_data(seed=1)
    st, yt = _logistic_data(n=500, seed=2)
    rec = C.CalibrationRecorder()
    assert rec.fit("m", sv, yv, 0.3) and rec.has("m")
    a0 = rec.models["m"]["platt_a"]
    rec.record("m", "blur", 0.5, "blur_s0.50", st, ~yt)            # flipping TEST labels cannot change the fit
    assert rec.models["m"]["platt_a"] == a0 and len(rec.rows) == 1
    rec.record("m", "blur", 1.0, "bad", np.array([np.nan] * 500), yt)   # error recorded, not raised
    assert len(rec.rows) == 1 and len(rec.errors) == 1
    assert not rec.fit("bad", np.zeros(5), np.zeros(5, bool), 0.0) and not rec.has("bad") and len(rec.errors) == 2
    info = rec.write(tmp_path / "x.calibration")
    assert info["status"] == "PARTIAL" and not list((tmp_path / "x.calibration").glob("*.tmp*"))
    z = np.load(tmp_path / "x.calibration" / "validation_scores__m.npz")
    assert set(z.files) == {"scores", "labels", "threshold"} and z["scores"].shape == sv.shape
    summ = json.loads((tmp_path / "x.calibration" / "calibration_summary.json").read_text())
    assert summ["method"] == C.METHOD and "test labels are never used" in summ["fit_data"]


# ------------------------------------------------------------------ integration
@contextmanager
def pushd(d):
    old = os.getcwd()
    os.chdir(d)
    try:
        yield
    finally:
        os.chdir(old)


def _verification(tmp_path, data, calibrate):
    tmp_path.mkdir(parents=True)
    cfg = T._config(tmp_path, data, protocol="lfw_fold0_verification")
    cfg.compute_calibration = calibrate
    import faceeval.models.deep.facenet as fn
    with pytest.MonkeyPatch.context() as mp, pushd(tmp_path):
        mp.setattr(fn, "_FACENET_AVAILABLE", False)            # hermetic: stub FaceNet even on a machine with real weights
        mp.setitem(sys.modules, "facenet_pytorch", None)
        runner = ExperimentRunner(cfg)
        runner.run()
    return runner.verification_summary


@pytest.fixture(scope="module")
def runs(tmp_path_factory):
    data = tmp_path_factory.mktemp("data")
    T._make_fake_lfw(data)
    base = tmp_path_factory.mktemp("runs")
    return _verification(base / "off", data, False), _verification(base / "on", data, True)


def _rows(summary):
    return list(csv.DictReader(open(Path(summary["output_dir"]) / "verification_results.csv")))


def test_calibration_off_by_default_flag_leaves_no_artifacts(runs):
    off, _ = runs
    assert "calibration" not in off
    assert not Path(off["output_dir"] + ".calibration").exists()


def test_calibration_does_not_change_verification_results_or_run_directory(runs):
    off, on = runs
    r_off, r_on = _rows(off), _rows(on)
    assert len(r_off) == len(r_on) == 12
    assert [list(r.keys()) for r in r_off] == [list(r.keys()) for r in r_on]
    for a, b in zip(r_off, r_on):
        assert a == b                                              # identical numbers: nothing leaked into the protocol
    assert off["manifest_sha256"] == on["manifest_sha256"] and off["content_sha256"] == on["content_sha256"]
    names = lambda s: sorted(p.name for p in Path(s["output_dir"]).iterdir())  # noqa: E731
    assert names(off) == names(on)                                   # the strictly checked run dir is unchanged


def test_calibration_artifacts_are_complete_and_sibling(runs):
    _, on = runs
    run_dir = Path(on["output_dir"])
    cal_dir = run_dir.with_name(run_dir.name + ".calibration")
    assert cal_dir.is_dir() and cal_dir.parent == run_dir.parent and cal_dir not in run_dir.parents
    assert on["calibration"] == {"status": "OK", "directory": cal_dir.name, "n_condition_rows": 12, "n_errors": 0}
    rows = list(csv.DictReader(open(cal_dir / "calibration_results.csv")))
    assert {r["model"] for r in rows} == {"eigenfaces", "lbph", "facenet"} and len(rows) == 12
    for r in rows:
        for k in ("ece_before", "ece_after", "brier_before", "brier_after", "nll_after"):
            assert float(r[k]) >= 0.0 and math.isfinite(float(r[k]))
        assert float(r["base_rate_genuine"]) == 0.5
    summ = json.loads((cal_dir / "calibration_summary.json").read_text())
    assert set(summ["models"]) == {"eigenfaces", "lbph", "facenet"} and summ["errors"] == []
    for m in summ["models"].values():
        assert math.isfinite(m["platt_a"]) and m["n_validation_pairs"] > 0
        assert m["validation_in_sample"]["after"]["nll"] <= m["validation_in_sample"]["before"]["nll"] + 1e-9
    for m in ("eigenfaces", "lbph", "facenet"):
        z = np.load(cal_dir / f"validation_scores__{m}.npz")
        assert z["scores"].shape == z["labels"].shape


def test_calibration_failures_never_fail_the_run(tmp_path, monkeypatch):
    data = tmp_path / "data"
    T._make_fake_lfw(data)

    def boom(*a, **k):
        raise RuntimeError("calibration exploded")
    monkeypatch.setattr(C, "fit_platt", boom)
    s = _verification(tmp_path / "a", data, True)
    assert s["status"] == "COMPLETE" and s["calibration"]["status"] == "PARTIAL" and s["calibration"]["n_condition_rows"] == 0
    monkeypatch.undo()
    monkeypatch.setattr(C.CalibrationRecorder, "write", boom)
    s = _verification(tmp_path / "b", data, True)
    assert s["status"] == "COMPLETE" and s["calibration"]["status"] == "FAILED" and len(_rows(s)) == 12

"""Tests for faceeval.analysis.efficiency (runtime / peak-memory harness for the deep models).

SCOPE: these tests check the harness logic only. The ``synthetic`` pseudo-model is a matrix product on random arrays,
NOT a face model; its timings and memory numbers mean nothing and are never reported. No real FaceNet/ArcFace
timing, memory or weights are exercised here: that measurement must be run on the maintainer's machine.
"""
import importlib.util
import json
import math
import sys
import os
from pathlib import Path

import numpy as np
import pytest

from faceeval.analysis import efficiency as E

REPO = Path(__file__).resolve().parent.parent


@pytest.fixture(autouse=True)
def _pythonpath(monkeypatch):
    monkeypatch.setenv("PYTHONPATH", str(REPO) + os.pathsep + os.environ.get("PYTHONPATH", ""))


def test_summarise_and_time_calls():
    s = E.summarise_ms([0.001, 0.002, 0.003, 0.004])
    assert s["mean"] == pytest.approx(2.5) and s["p50"] == pytest.approx(2.5) and s["p95"] == pytest.approx(3.85)
    calls = []
    t = E.time_calls(calls.append, [1, 2, 3], warmup=2)
    assert len(t) == 3 and all(x >= 0 for x in t) and calls == [1, 2, 1, 2, 3]   # warm-up untimed, cycles items
    with pytest.raises(E.EfficiencyError):
        E.time_calls(print, [], 0)


def test_rss_helpers_return_positive_numbers():
    assert E.peak_rss_mb() > 1.0
    cur = E.current_rss_mb()
    assert cur == E.NOT_RECORDED or cur > 1.0
    big = np.ones((40_000_000,), dtype=np.uint8); big[::4096] = 2           # touch ~40 MB
    assert E.peak_rss_mb() >= 40.0 and big[1] == 1


def test_hardware_info_has_provenance_keys():
    h = E.hardware_info()
    assert {"platform", "machine", "cpu_count", "python", "numpy", "torch", "torch_cuda_available"} <= set(h)


def test_list_images_deterministic_and_refuses_too_few(tmp_path):
    import cv2
    for n in ("b.jpg", "a.png", "c.txt"):
        (tmp_path / n).parent.mkdir(exist_ok=True)
    cv2.imwrite(str(tmp_path / "b.jpg"), np.zeros((8, 8, 3), np.uint8))
    cv2.imwrite(str(tmp_path / "a.png"), np.zeros((8, 8, 3), np.uint8))
    (tmp_path / "c.txt").write_text("x")
    assert [p.name for p in E.list_images(tmp_path, 2)] == ["a.png", "b.jpg"]
    with pytest.raises(E.EfficiencyError, match="fewer than"):
        E.list_images(tmp_path, 3)


def test_worker_synthetic_result_schema():
    r = E.run_worker(E.SYNTHETIC, None, n_images=12, warmup=2, allow_stub=False)
    assert r["synthetic"] is True and r["n_embedded"] == 12 and r["n_preprocess_failed"] == 0
    assert r["embed_ms_mean"] > 0 and r["images_per_s"] > 0 and r["rss_peak_mb"] > 1.0
    assert r["embed_ms_p95"] >= r["embed_ms_p50"] >= 0 and not math.isnan(r["load_s"])
    assert "hardware" in r


def test_run_all_uses_a_fresh_process_per_model_and_writes_new_dir_only(tmp_path):
    out = tmp_path / "out"
    res = E.run_all([E.SYNTHETIC], None, 8, 1, out)
    assert len(res) == 1 and sorted(p.name for p in out.iterdir()) == ["efficiency_results.csv", "efficiency_results.json"]
    j = json.loads((out / "efficiency_results.json").read_text())
    assert j["results"][0]["synthetic"] is True and "must not be reported" in j["note"]
    with pytest.raises(E.EfficiencyError, match="non-empty"):
        E.run_all([E.SYNTHETIC], None, 8, 1, out)
    with pytest.raises(E.EfficiencyError, match="duplicate"):
        E.run_all([E.SYNTHETIC, E.SYNTHETIC], None, 8, 1, tmp_path / "o2")


def test_unknown_model_and_missing_images_dir_are_refused_cleanly(tmp_path):
    with pytest.raises(E.EfficiencyError, match="worker failed"):
        E.run_all(["no_such_model"], tmp_path, 1, 0, tmp_path / "a")
    with pytest.raises(E.EfficiencyError, match="images-dir|worker failed"):
        E.run_all(["facenet"], None, 1, 0, tmp_path / "b")
    assert not (tmp_path / "a").exists() and not (tmp_path / "b").exists()      # nothing written on refusal


@pytest.mark.skipif(importlib.util.find_spec("facenet_pytorch") is not None, reason="a real FaceNet backbone is installed")
def test_stub_backbone_is_refused_unless_allowed(tmp_path, monkeypatch):
    import cv2
    import faceeval.models.deep.facenet as fn
    monkeypatch.setattr(fn, "_FACENET_AVAILABLE", False)          # hermetic: a machine WITH real FaceNet weights too
    monkeypatch.setitem(sys.modules, "facenet_pytorch", None)
    for i in range(3):
        cv2.imwrite(str(tmp_path / f"f{i}.jpg"), (np.random.default_rng(i).random((160, 160, 3)) * 255).astype(np.uint8))
    with pytest.raises(E.EfficiencyError, match="stub"):
        E.run_all(["facenet"], tmp_path, 3, 0, tmp_path / "out")


def test_cli_exit_codes_and_documented_options(tmp_path, capsys):
    assert E.main(["--models", E.SYNTHETIC, "--n-images", "6", "--warmup", "1", "--out", str(tmp_path / "o")]) == 0
    assert "SYNTHETIC SELF-TEST" in capsys.readouterr().out
    assert E.main(["--models", E.SYNTHETIC, "--out", str(tmp_path / "o")]) == 2
    assert E.main(["--n-images", "0", "--models", "x", "--out", str(tmp_path / "z")]) == 2
    opts = {o for a in E.build_parser()._actions for o in a.option_strings}
    assert {"--models", "--images-dir", "--n-images", "--warmup", "--out", "--overwrite", "--allow-stub"} <= opts


def test_gpu_fields_are_not_recorded_when_cuda_is_absent(monkeypatch):
    import builtins
    real_import = builtins.__import__

    def no_torch(name, *a, **k):
        if name == "torch":
            raise ImportError("no torch here")
        return real_import(name, *a, **k)
    monkeypatch.setattr(builtins, "__import__", no_torch)
    sync, peak, reset = E.gpu_hooks()
    assert sync() is None and reset() is None and peak() == E.NOT_RECORDED
    h = E.hardware_info()
    assert h["gpu_name"] == E.NOT_RECORDED and h["cuda_version"] == E.NOT_RECORDED
    monkeypatch.setattr(builtins, "__import__", real_import)
    r = E.run_worker(E.SYNTHETIC, None, n_images=6, warmup=1, allow_stub=False)
    assert "gpu_peak_allocated_mb" in r


def test_repeated_synthetic_runs_have_identical_schema_and_counts():
    a = E.run_worker(E.SYNTHETIC, None, n_images=8, warmup=1, allow_stub=False)
    b = E.run_worker(E.SYNTHETIC, None, n_images=8, warmup=1, allow_stub=False)
    assert set(a) == set(b) and a["n_embedded"] == b["n_embedded"] == 8
    assert a["image_list_sha256"] == b["image_list_sha256"]


def test_setup_time_is_reported_separately_from_per_image_embedding():
    r = E.run_worker(E.SYNTHETIC, None, n_images=6, warmup=1, allow_stub=False)
    assert "load_s" in r and "embed_ms_mean" in r and r["load_s"] >= 0 and r["embed_ms_mean"] > 0   # distinct fields
    assert r["rss_after_load_mb"] != E.NOT_RECORDED


def test_device_label_and_gpu_accounting_only_for_torch_cuda():
    assert E.device_label("cuda:0") == "cuda:0" and E.device_label("cpu") == "cpu"
    for odd in ("auto", None):
        assert "NOT measured" in E.device_label(odd)
    r = E.run_worker(E.SYNTHETIC, None, n_images=6, warmup=1, allow_stub=False)
    assert r["device_used"] == "cpu (numpy)" and r["gpu_peak_allocated_mb"] == E.NOT_RECORDED


# ---------------------------------------------------------------- device selection (no hardware needed)
@pytest.mark.parametrize("requested,cuda,mps,expected", [
    ("auto", False, False, "cpu"), ("auto", True, False, "cuda"), ("auto", False, True, "cpu"),   # MPS is never implicit
    ("auto", True, True, "cuda"), ("cpu", True, True, "cpu"), ("cpu", False, False, "cpu"),
    ("cuda", True, False, "cuda"), ("cuda:1", True, False, "cuda:1"), ("mps", False, True, "mps"),
    ("best", True, True, "cuda"), ("best", False, True, "mps"), ("best", False, False, "cpu"),
    ("CUDA", True, False, "cuda"), (" Mps ", False, True, "mps"),
])
def test_select_device_matrix(requested, cuda, mps, expected):
    assert E.select_device(requested, cuda_available=cuda, mps_available=mps).name == expected


@pytest.mark.parametrize("requested,cuda,mps,msg", [
    ("cuda", False, True, "CUDA is not available"), ("cuda:0", False, False, "CUDA is not available"),
    ("mps", True, False, "MPS backend is not available"), ("tpu", True, True, "unknown --device"),
    ("cuda:x", True, False, "unknown --device"),
])
def test_select_device_refuses_unavailable_or_unknown_devices_without_fallback(requested, cuda, mps, msg):
    with pytest.raises(E.EfficiencyError, match=msg):
        E.select_device(requested, cuda_available=cuda, mps_available=mps)


class _FakeTorch:
    """Stand-in for torch: records calls; no hardware."""
    def __init__(self, cuda=False, mps=False, mps_built=True):
        self.calls = []
        outer = self

        class cuda_ns:
            @staticmethod
            def is_available(): return cuda
            @staticmethod
            def synchronize(): outer.calls.append("cuda.sync")
            @staticmethod
            def reset_peak_memory_stats(): outer.calls.append("cuda.reset")
            @staticmethod
            def max_memory_allocated(): return 5 * 1024 * 1024
        class mps_ns:
            @staticmethod
            def is_available(): return mps
            @staticmethod
            def is_built(): return mps_built
            @staticmethod
            def synchronize(): outer.calls.append("mps.sync")
            @staticmethod
            def current_allocated_memory(): return 7 * 1024 * 1024
        class backends:
            pass
        backends.mps = mps_ns
        self.cuda, self.mps, self.backends = cuda_ns, mps_ns, backends


def test_detect_accelerators_with_stand_in_torch_for_cpu_cuda_and_mps():
    assert E.detect_accelerators(_FakeTorch()) == (False, False)
    assert E.detect_accelerators(_FakeTorch(cuda=True)) == (True, False)
    assert E.detect_accelerators(_FakeTorch(mps=True)) == (False, True)
    assert E.detect_accelerators(_FakeTorch(mps=True, mps_built=False)) == (False, False)
    assert E.detect_accelerators(object()) == (False, False)               # no usable torch: never raises


def test_device_hooks_cuda_mps_and_cpu_report_only_what_is_measurable():
    t = _FakeTorch(cuda=True, mps=True)
    cu = E.device_hooks("cuda", t)
    cu.reset_peak(); cu.sync()
    assert t.calls == ["cuda.reset", "cuda.sync"] and cu.peak_mb() == 5.0 and cu.current_mb() == E.NOT_RECORDED
    mp = E.device_hooks("mps", t)
    mp.sync()
    assert t.calls[-1] == "mps.sync" and mp.current_mb() == 7.0 and mp.peak_mb() == E.NOT_RECORDED     # MPS: current, never a peak
    cpu = E.device_hooks("cpu", t)
    assert cpu.sync() is None and cpu.peak_mb() == E.NOT_RECORDED and cpu.current_mb() == E.NOT_RECORDED
    broken = E.device_hooks("cuda", object())                               # torch without the backend: degrade, do not raise
    assert broken.peak_mb() == E.NOT_RECORDED


def test_device_kind_and_label():
    assert [E.device_kind(x) for x in ("cuda:0", "mps", "cpu (numpy)", "onnxruntime (x)", None)] == ["cuda", "mps", "cpu", "other", "other"]
    assert E.device_label("mps") == "mps" and E.device_label("cuda:1") == "cuda:1"


# ---------------------------------------------------------------- protocol phases
def test_synchronise_runs_inside_every_timed_call_and_warmup():
    n = {"sync": 0, "calls": 0}
    def fn(x): n["calls"] += 1
    def sync(): n["sync"] += 1
    t = E.time_calls(fn, [1, 2, 3], warmup=2, after=sync)
    assert len(t) == 3 and n["calls"] == 5 and n["sync"] == 5


def test_time_batches_throughput_arithmetic_and_short_input():
    seen = []
    r = E.time_batches(lambda b: seen.append(len(b)), list(range(10)), batch_size=4, warmup=1)
    assert r["n_batches"] == 2 and seen == [4, 4, 4] and r["ms_per_image_mean"] * 4 == pytest.approx(r["ms_per_batch_mean"])
    assert r["images_per_s"] == pytest.approx(1000.0 * 4 / r["ms_per_batch_mean"])
    assert "skipped" in E.time_batches(lambda b: None, [1, 2], batch_size=4, warmup=0)
    with pytest.raises(E.EfficiencyError):
        E.time_batches(lambda b: None, [1], batch_size=0, warmup=0)


def test_worker_reports_every_phase_separately():
    r = E.run_worker(E.SYNTHETIC, None, n_images=40, warmup=2, allow_stub=False, batch_sizes=(4, 16, 64))
    for k in ("load_s", "first_preprocess_ms", "first_embed_ms", "preprocess_ms_mean", "embed_ms_mean", "embed_ms_p95",
              "rss_before_load_mb", "rss_after_load_mb", "rss_peak_after_load_mb", "rss_peak_mb", "device_requested",
              "device_kind", "device_detail"):
        assert k in r, k
    assert [b["batch_size"] for b in r["batched"]] == [4, 16, 64] and "skipped" in r["batched"][2]      # 40 images < 64
    assert r["batched"][0]["n_batches"] == 10 and r["batched"][1]["n_batches"] == 2
    assert r["device_kind"] == "cpu" and r["device_requested"] == "auto" and r["rss_peak_mb"] >= r["rss_peak_after_load_mb"]


def test_worker_refuses_accelerators_for_the_synthetic_model_and_bad_batch_sizes():
    with pytest.raises(E.EfficiencyError):
        E.run_worker(E.SYNTHETIC, None, 6, 1, False, device="cuda")
    with pytest.raises(E.EfficiencyError, match="batch sizes"):
        E.run_worker(E.SYNTHETIC, None, 6, 1, False, batch_sizes=(0,))


def test_non_torch_models_refuse_an_accelerator_request(tmp_path, monkeypatch):
    import cv2
    for i in range(3):
        cv2.imwrite(str(tmp_path / f"f{i}.jpg"), (np.random.default_rng(i).random((64, 64, 3)) * 255).astype(np.uint8))
    monkeypatch.setattr(E, "detect_accelerators", lambda *a, **k: (True, True))
    for model in ("arcface", "insightface", "dlib_fr"):
        with pytest.raises(E.EfficiencyError, match="not on a torch device"):
            E.run_worker(model, tmp_path, 3, 0, True, device="cuda")
    monkeypatch.setattr(E, "detect_accelerators", lambda *a, **k: (False, False))
    with pytest.raises(E.EfficiencyError, match="MPS backend is not available"):
        E.run_worker("facenet", tmp_path, 3, 0, True, device="mps")


def test_comparability_flags_rows_measured_under_different_conditions():
    hw = {"platform": "p", "machine": "m", "python": "3", "torch": "2", "onnxruntime": "1"}
    base = {"device_kind": "cpu", "n_images": 10, "image_list_sha256": "a", "warmup_calls": 5, "batch_sizes": [8], "hardware": hw}
    assert E.comparability([base, dict(base)])["all_same_conditions"] is True
    other = dict(base, device_kind="cuda", n_images=20, hardware=dict(hw, torch="3"))
    c = E.comparability([base, other])
    assert c["all_same_conditions"] is False and set(c["differing_fields"]) == {"device_kind", "n_images", "hardware.torch"}


def test_run_all_writes_comparability_batches_and_device_columns(tmp_path, capsys):
    out = tmp_path / "o"
    assert E.main(["--models", E.SYNTHETIC, "--n-images", "20", "--warmup", "1", "--batch-sizes", "4", "8", "--out", str(out)]) == 0
    j = json.loads((out / "efficiency_results.json").read_text())
    assert j["comparability"]["all_same_conditions"] is True and j["results"][0]["batched"][0]["batch_size"] == 4
    header = (out / "efficiency_results.csv").read_text().splitlines()[0].split(",")
    assert {"device_kind", "batch4_images_per_s", "batch8_ms_per_image", "first_embed_ms", "rss_peak_after_load_mb"} <= set(header)
    assert E.main(["--models", E.SYNTHETIC, "--device", "mps", "--out", str(tmp_path / "o2")]) == 2


# ---------------------------------------------------------------- current RSS (Linux / macOS / psutil / NOT_RECORDED)
import ctypes  # noqa: E402
import sys as _sys  # noqa: E402
import types as _types  # noqa: E402


class _FakeLibproc:
    """Stand-in for libproc: writes pti_resident_size into the struct the harness passes by reference."""
    def __init__(self, resident_bytes=0, returns=None, raises=None):
        self.calls, self._res, self._ret, self._raises = [], resident_bytes, returns, raises

        def proc_pidinfo(pid, flavor, arg, buf, size):
            self.calls.append((pid, flavor, arg, size))
            if self._raises:
                raise self._raises
            buf._obj.pti_resident_size = self._res
            return size if self._ret is None else self._ret
        self.proc_pidinfo = proc_pidinfo


def test_proc_taskinfo_layout_matches_the_apple_header():
    assert ctypes.sizeof(E._ProcTaskInfo) == 96                       # 6 x uint64 + 12 x int32
    assert E._ProcTaskInfo.pti_resident_size.offset == 8 and E._ProcTaskInfo.pti_virtual_size.offset == 0
    assert E._ProcTaskInfo.pti_policy.offset == 48 and E._ProcTaskInfo.pti_priority.offset == 92
    assert E.PROC_PIDTASKINFO == 4


def test_macos_current_rss_converts_bytes_to_mb_and_asks_for_the_right_flavor():
    lib = _FakeLibproc(resident_bytes=300 * 1024 * 1024)
    assert E._macos_rss_mb(lib) == 300.0
    pid, flavor, arg, size = lib.calls[0]
    assert pid == __import__("os").getpid() and flavor == 4 and arg == 0 and size == 96
    assert E._macos_rss_mb(_FakeLibproc(resident_bytes=1536 * 1024)) == 1.5


@pytest.mark.parametrize("lib", [_FakeLibproc(300 << 20, returns=-1), _FakeLibproc(300 << 20, returns=40),
                                 _FakeLibproc(0), _FakeLibproc(raises=OSError("no libproc")), object()])
def test_macos_native_failure_returns_none(lib):
    assert E._macos_rss_mb(lib) is None


def _no_linux(monkeypatch):
    monkeypatch.setattr(E, "_linux_rss_mb", lambda *a, **k: None)


def test_macos_api_failure_falls_back_to_optional_psutil(monkeypatch):
    _no_linux(monkeypatch)
    monkeypatch.setattr(E, "_macos_rss_mb", lambda *a, **k: None)
    ps = _types.SimpleNamespace(Process=lambda: _types.SimpleNamespace(memory_info=lambda: _types.SimpleNamespace(rss=200 * 1024 * 1024)))
    monkeypatch.setitem(_sys.modules, "psutil", ps)
    assert E.current_rss_mb("darwin") == 200.0


def test_macos_native_is_used_before_psutil_and_only_on_darwin(monkeypatch):
    _no_linux(monkeypatch)
    monkeypatch.setattr(E, "_macos_rss_mb", lambda *a, **k: 123.0)
    monkeypatch.setattr(E, "_psutil_rss_mb", lambda: 999.0)
    assert E.current_rss_mb("darwin") == 123.0
    assert E.current_rss_mb("linux") == 999.0                         # the macOS call is not attempted elsewhere
    monkeypatch.setattr(E, "_psutil_rss_mb", lambda: None)
    assert E.current_rss_mb("win32") == E.NOT_RECORDED


def test_linux_statm_behaviour(tmp_path, monkeypatch):
    f = tmp_path / "statm"
    f.write_text("5000 2560 100 10 0 400 0\n")
    monkeypatch.setattr(E.os, "sysconf", lambda name: 4096)
    assert E._linux_rss_mb(str(f)) == 2560 * 4096 / (1024.0 * 1024.0) == 10.0
    f.write_text("garbage")
    assert E._linux_rss_mb(str(f)) is None and E._linux_rss_mb(str(tmp_path / "missing")) is None
    f.write_text("1")
    assert E._linux_rss_mb(str(f)) is None
    f.write_text("5000 2560\n")                                       # current_rss_mb tries Linux first, on every platform
    orig = E._linux_rss_mb
    monkeypatch.setattr(E, "_linux_rss_mb", lambda *a, **k: orig(str(f)))
    monkeypatch.setattr(E, "_macos_rss_mb", lambda *a, **k: 1.0)
    assert E.current_rss_mb("darwin") == 10.0 and E.current_rss_mb("linux") == 10.0


def test_not_recorded_only_when_every_method_fails_and_peak_is_never_substituted(monkeypatch):
    _no_linux(monkeypatch)
    monkeypatch.setattr(E, "_macos_rss_mb", lambda *a, **k: None)
    monkeypatch.setattr(E, "_psutil_rss_mb", lambda: None)
    monkeypatch.setattr(E, "peak_rss_mb", lambda: 4242.0)
    assert E.current_rss_mb("darwin") == E.NOT_RECORDED != 4242.0
    r = E.run_worker(E.SYNTHETIC, None, n_images=6, warmup=0, allow_stub=False)
    assert r["rss_before_load_mb"] == r["rss_after_load_mb"] == E.NOT_RECORDED      # honest gap, run still succeeds
    assert r["rss_peak_mb"] == 4242.0 and r["rss_peak_after_load_mb"] == 4242.0     # peak fields stay peak fields


def test_peak_and_current_are_distinct_quantities_on_this_machine():
    cur = E.current_rss_mb()
    if cur == E.NOT_RECORDED:
        pytest.skip("no current-RSS method available on this platform")
    assert 0 < cur <= E.peak_rss_mb() + 1.0                      # a current size never exceeds the process peak


@pytest.mark.skipif(_sys.platform != "darwin", reason="exercises the real libproc; only meaningful on macOS (not run in the Linux cloud sandbox)")
def test_real_macos_libproc_returns_a_plausible_current_rss():
    v = E._macos_rss_mb()
    assert v is not None and 1.0 < v <= E.peak_rss_mb() + 1.0

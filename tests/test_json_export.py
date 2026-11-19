"""The runner writes a complete, valid JSON report and survives a failing JSON export."""
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))
import cv2  # noqa: E402
import numpy as np  # noqa: E402
from test_verification_stage import _run  # noqa: E402


@pytest.fixture(scope="module")
def small_data(tmp_path_factory):
    """40 people x 8 images of 64x64 noise: enough for the identification path, fast enough for CI."""
    root = tmp_path_factory.mktemp("small")
    rng = np.random.default_rng(1)
    for s in range(40):
        d = root / "lfw" / f"Person_{s:03d}"
        d.mkdir(parents=True)
        base = rng.integers(0, 255, (64, 64, 3)).astype(np.float32)
        for k in range(8):
            cv2.imwrite(str(d / f"Person_{s:03d}_{k + 1:04d}.jpg"),
                        np.clip(base + rng.normal(0, 10, base.shape), 0, 255).astype(np.uint8))
    return root


def _reports(tmp_path):
    return sorted((tmp_path / "rep").glob("experiment_report_*.json"))


def test_identification_run_writes_valid_json_report(tmp_path, monkeypatch, small_data):
    cfg, runner, run = _run(tmp_path, monkeypatch, small_data, protocol="identification", deep=())
    files = _reports(tmp_path)
    assert len(files) == 1 and not list((tmp_path / "rep").glob("*.tmp"))
    d = json.loads(files[0].read_text())
    assert d["schema_version"] == 2 and d["run_id"] == run.run_id
    assert len(d["evaluation_results"]) == d["num_evaluation_results"] == len(run.evaluation_results) > 0
    assert {"failure_clusters", "deployment_rankings", "fingerprints", "extra"} <= set(d)
    assert d["extra"] == {}                                      # identification only: no verification block


def test_a_failing_json_export_does_not_fail_the_run(tmp_path, monkeypatch, small_data):
    from faceeval.reporting import generator
    def boom(self, report, extra=None):
        raise OSError("disk full")
    monkeypatch.setattr(generator.ReportGenerator, "export_json", boom)
    cfg, runner, run = _run(tmp_path, monkeypatch, small_data, protocol="identification", deep=())
    assert run.evaluation_results and not _reports(tmp_path)
    assert list((tmp_path / "rep").glob("evaluation_results_*.csv"))   # the CSV exports still happened


def test_export_json_is_atomic_and_cleans_up_on_error(tmp_path, monkeypatch):
    import faceeval.reporting.generator as g
    gen = g.ReportGenerator(output_dir=str(tmp_path), run_id="r1")

    class Cfg:
        def to_dict(self): return {"x": 1}

    class R:
        run_id, experiment_name, generated_at, summary_text = "r1", "e", __import__("datetime").datetime(2026, 1, 1), "s"
        config = Cfg(); evaluation_results = []; fingerprints = []; failure_clusters = []; deployment_rankings = []; figures = []
    assert Path(gen.export_json(R)).is_file()
    monkeypatch.setattr(g.json, "dump", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("x")))
    (tmp_path / "experiment_report_r1.json").unlink()
    with pytest.raises(RuntimeError):
        gen.export_json(R)
    assert not list(tmp_path.glob("*.tmp")) and not (tmp_path / "experiment_report_r1.json").exists()

"""Figures are drawn only from existing result files; refusals write nothing; output is traceable and repeatable."""
from __future__ import annotations

import csv
import hashlib
import json

import pytest

from faceeval.analysis import figures as F


def _write(path, rows):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


@pytest.fixture
def inputs(tmp_path):
    d = tmp_path / "in"
    d.mkdir()
    cal = [{"model": m, "perturbation": p, "severity": s, "label": f"{p}@{s}", "n_pairs": 100, "ece_before": 0.2 + k * 0.01,
            "ece_after": 0.03 + k * 0.001} for m in ("facenet", "arcface") for k, (p, s) in enumerate([("clean", 0.0), ("blur", 1.0)])]
    _write(d / "calibration_results.csv", cal)
    ms = [{"model": m, "perturbation": "clean", "severity": 0.0, "n_splits": 6, "model_kind": kind, "interval_status": st,
           "auc_mean": 0.9 + i * 0.01, "auc_sd": 0.01, "accuracy_mean": 0.8, "accuracy_sd": 0.02}
          for i, (m, kind, st) in enumerate([("facenet", "pretrained", "spread_across_disjoint_test_identities"),
                                             ("eigenfaces", "trained_per_split", "corrected_interval_only")])]
    _write(d / "multi_split_by_condition.csv", ms)
    vp = [{"model": m, "perturbation": "clean", "severity": 0.0, "label": f"c{k}", "valid_pair_rate_balanced": 0.99 - 0.1 * k,
           "e2e_auc": 0.95 - 0.05 * k, "valid_auc": 0.96 - 0.04 * k} for m in ("arcface", "facenet") for k in range(3)]
    _write(d / "valid_pairs_by_condition.csv", vp)
    row = lambda m, dev, n=10: {"model": m, "synthetic": False, "n_images": n, "device_kind": dev, "load_s": 1.0, "embed_ms_mean": 5.0,
                                "images_per_s": 100.0, "rss_peak_mb": 500.0}
    (d / "efficiency_results.json").write_text(json.dumps({"results": [row("facenet", "cpu"), row("arcface", "cpu")],
                                                           "comparability": {"all_same_conditions": True, "differing_fields": []}}))
    return d


@pytest.mark.parametrize("kind", ["calibration", "multi_split", "valid_pairs", "efficiency"])
def test_each_figure_is_written_with_data_and_provenance(inputs, tmp_path, kind):
    out = F.make_figure(kind, inputs, tmp_path / "o")
    assert out["png"].read_bytes()[:8] == b"\x89PNG\r\n\x1a\n" and out["png"].stat().st_size > 2000
    prov = json.loads(out["provenance"].read_text())
    src = inputs / F.FILES[kind]
    assert prov["input_sha256"] == hashlib.sha256(src.read_bytes()).hexdigest() and prov["plotted_rows"] >= 2
    assert len(out["data"].read_text().splitlines()) == prov["plotted_rows"] + 1


def test_plotted_values_are_exactly_the_input_values(inputs, tmp_path):
    out = F.make_figure("calibration", inputs, tmp_path / "o")
    got = list(csv.DictReader(open(out["data"])))
    want = list(csv.DictReader(open(inputs / "calibration_results.csv")))
    assert [float(g["ece_after"]) for g in got] == [float(w["ece_after"]) for w in want]
    vp = list(csv.DictReader(open(F.make_figure("valid_pairs", inputs, tmp_path / "o2")["data"])))
    assert {r["model"] for r in vp} == {"arcface", "facenet"} and float(vp[0]["valid_pair_rate_balanced"]) == pytest.approx(0.99)


def test_regeneration_is_repeatable(inputs, tmp_path):
    a = F.make_figure("multi_split", inputs, tmp_path / "a")
    b = F.make_figure("multi_split", inputs, tmp_path / "b")
    assert a["data"].read_bytes() == b["data"].read_bytes() and a["png"].read_bytes() == b["png"].read_bytes()


def test_inputs_are_never_modified_and_outputs_not_overwritten(inputs, tmp_path):
    before = {p.name: p.read_bytes() for p in inputs.iterdir()}
    F.make_figure("calibration", inputs / "calibration_results.csv", tmp_path / "o")
    assert {p.name: p.read_bytes() for p in inputs.iterdir()} == before
    with pytest.raises(F.FigureError, match="non-empty"):
        F.make_figure("calibration", inputs, tmp_path / "o")
    F.make_figure("calibration", inputs, tmp_path / "o", overwrite=True)


def test_refusals_write_nothing(inputs, tmp_path):
    bad = tmp_path / "bad"
    bad.mkdir()
    (bad / "calibration_results.csv").write_text("model,x\nfacenet,1\n")
    with pytest.raises(F.FigureError, match="lacks column"):
        F.make_figure("calibration", bad, tmp_path / "o1")
    (bad / "calibration_results.csv").write_text("model,perturbation,severity,label,n_pairs,ece_before,ece_after\nm,clean,0,c,1,abc,0.1\n")
    with pytest.raises(F.FigureError, match="non-numeric"):
        F.make_figure("calibration", bad, tmp_path / "o2")
    with pytest.raises(F.FigureError, match="not found"):
        F.make_figure("valid_pairs", tmp_path / "nowhere", tmp_path / "o3")
    with pytest.raises(F.FigureError, match="no rows"):
        F.make_figure("multi_split", inputs, tmp_path / "o4", perturbation="blur")
    assert not (tmp_path / "o1").exists() and not (tmp_path / "o2").exists() and not (tmp_path / "o4").exists()


def test_efficiency_refuses_synthetic_and_flags_incomparable_rows(inputs, tmp_path):
    j = json.loads((inputs / "efficiency_results.json").read_text())
    j["results"][0]["synthetic"] = True
    (inputs / "efficiency_results.json").write_text(json.dumps(j))
    with pytest.raises(F.FigureError, match="synthetic"):
        F.make_figure("efficiency", inputs, tmp_path / "o")
    assert not (tmp_path / "o").exists()
    F.make_figure("efficiency", inputs, tmp_path / "o2", allow_synthetic=True)
    j["results"][0]["synthetic"] = False
    j["comparability"] = {"all_same_conditions": False, "differing_fields": ["device_kind"]}
    (inputs / "efficiency_results.json").write_text(json.dumps(j))
    assert F.make_figure("efficiency", inputs, tmp_path / "o3")["png"].is_file()


def test_cli(inputs, tmp_path, capsys):
    assert F.main(["multi_split", "--input", str(inputs), "--out", str(tmp_path / "c"), "--metric", "accuracy"]) == 0
    assert F.main(["multi_split", "--input", str(inputs), "--out", str(tmp_path / "c2"), "--metric", "nope"]) == 2
    assert "REFUSED" in capsys.readouterr().out

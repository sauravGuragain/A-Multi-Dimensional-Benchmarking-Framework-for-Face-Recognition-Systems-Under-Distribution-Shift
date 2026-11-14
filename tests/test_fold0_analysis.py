"""Tests for faceeval.analysis.fold0_results (post-hoc analysis of the archived LFW Fold 0 results).

SCOPE: the logic tests run on a SYNTHETIC archive built in ``tmp_path`` (the same file layout and columns as
``reports/experiment_results``). The ``real archive`` tests at the end run on the archived CSV/JSON files that are
committed in this repository and recompute every checked number independently with the ``csv`` module. No experiment
is run and no archived file is modified (a test hashes the archive before and after).
"""
import copy
import csv
import hashlib
import json
import math
from pathlib import Path

import pytest

from faceeval.analysis import fold0_results as A

REPO = Path(__file__).resolve().parent.parent
REAL_ROOT = REPO / "reports" / "experiment_results"

BASE_AUC = {"eigenfaces": 0.64, "fisherfaces": 0.69, "lbph": 0.58, "hog_svm": 0.62}
STRENGTH = {p: 0.001 * (i + 1) for i, p in enumerate(A.PERTURBATION_ORDER)}     # AUC lost per unit severity
STRENGTH["sunglasses"] = 0.1                                                      # the designed worst perturbation
COLUMNS = ["model", "paradigm", "backbone", "perturbation", "severity", "label", "embedding_dim", "threshold",
           "detector_miss_rate", "val_detector_miss_rate", "accuracy", "tar", "far", "n_genuine",
           "n_impostor_balanced", "auc", "eer", "frr_at_far_0.001", "frr_at_far_0.01", "frr_at_far_0.1",
           "n_impostor_full", "auc_full", "eer_full", "frr_at_far_0.001_full", "frr_at_far_0.01_full",
           "frr_at_far_0.1_full", "far_full"]


def synth_rows(models, shift=0.0):
    rows = []
    for m in models:
        for p in A.PERTURBATION_ORDER:
            for s in A.SEVERITIES:
                auc = BASE_AUC[m] - STRENGTH[p] * s + shift
                acc = 0.5 + 0.8 * (auc - 0.5)
                rows.append({
                    "model": m, "paradigm": "traditional", "backbone": "n/a", "perturbation": p, "severity": s,
                    "label": f"{p}_s{s:.2f}", "embedding_dim": 100, "threshold": 0.1 + 0.01 * len(m),
                    "detector_miss_rate": float("nan"), "val_detector_miss_rate": float("nan"), "accuracy": acc,
                    "tar": acc + 0.04, "far": 0.5 - (acc - 0.5) + 0.04, "n_genuine": 764, "n_impostor_balanced": 764,
                    "auc": auc, "eer": 1.0 - auc, "frr_at_far_0.001": 0.95, "frr_at_far_0.01": 0.9, "frr_at_far_0.1": 0.7,
                    "n_impostor_full": 7640, "auc_full": auc + 0.001, "eer_full": 1.0 - auc, "frr_at_far_0.001_full": 0.96,
                    "frr_at_far_0.01_full": 0.9, "frr_at_far_0.1_full": 0.7, "far_full": 0.3})
    return rows


def meta_for(rows, code="aaaa", manifest="m1", **over):
    meta = {"protocol": "lfw_fold0_verification", "full_lfw": True, "status": "COMPLETE", "failures": [],
            "n_rows": len(rows), "environment": {"code_version": code, "numpy_version": "1.26.4", "python_version": "3.12"},
            "manifest_sha256": manifest, "content_sha256": {"split": "s", "pairs": {"validation": "pv", "test": "pt"}},
            "dataset": {"fingerprint_sha256": "fp"},
            "noise_seeding": {"protocol_version": "per_image_v2", "master_seed": 42,
                              "stream_fingerprint": {"matches_approved_reference": True}},
            "results": rows}
    meta.update(over)
    return meta


def write_run(root, run, rows, meta):
    d = Path(root) / run
    d.mkdir(parents=True, exist_ok=True)
    with open(d / "verification_results.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=COLUMNS, lineterminator="\n")
        w.writeheader()
        for r in rows:
            w.writerow({c: ("nan" if isinstance(r[c], float) and math.isnan(r[c]) else r[c]) for c in COLUMNS})
    (d / "verification_results.json").write_text(json.dumps(meta))


@pytest.fixture
def archive(tmp_path):
    root = tmp_path / "experiment_results"
    classical = synth_rows(["fisherfaces", "lbph", "hog_svm"])
    final = synth_rows(["eigenfaces"])
    hist = synth_rows(["eigenfaces"], shift=0.001)
    write_run(root, "classical", classical, meta_for(classical))
    write_run(root, "eigenfaces_final", final, meta_for(final))
    write_run(root, "eigenfaces_per_image_noise", hist, meta_for(hist, code="old", manifest="m_old"))
    return root


def digest_tree(root):
    return {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(Path(root).rglob("*")) if p.is_file()}


# ---------------------------------------------------------------- coverage and structure
def test_primary_row_count_model_perturbation_and_severity_coverage(archive):
    rows, prov = A.load_primary(archive)
    assert len(rows) == 384
    assert {r["model"] for r in rows} == set(A.MODEL_ORDER)
    assert {r["perturbation"] for r in rows} == set(A.PERTURBATION_ORDER) and len(A.PERTURBATION_ORDER) == 16
    assert sorted({r["severity"] for r in rows}) == list(A.SEVERITIES)
    assert set(prov["runs"]) == {"classical", "eigenfaces_final"}                 # the historical run is not a primary run


def test_missing_condition_is_rejected(archive):
    rows = synth_rows(["fisherfaces", "lbph", "hog_svm"])[:-1]
    write_run(archive, "classical", rows, meta_for(rows))
    with pytest.raises(A.AnalysisError, match="condition grid differs|expected 384"):
        A.load_primary(archive)


def test_duplicate_condition_is_rejected(archive):
    rows = synth_rows(["fisherfaces", "lbph", "hog_svm"])
    rows[-1] = dict(rows[0])
    write_run(archive, "classical", rows, meta_for(rows))
    with pytest.raises(A.AnalysisError, match="duplicate"):
        A.load_primary(archive)


def test_unexpected_model_is_rejected(archive):
    rows = synth_rows(["eigenfaces", "lbph", "hog_svm"])          # classical run must hold fisherfaces/lbph/hog_svm
    write_run(archive, "classical", rows, meta_for(rows))
    with pytest.raises(A.AnalysisError, match="expected models"):
        A.load_primary(archive)


def test_primary_runs_must_share_manifest_and_code(archive):
    final = synth_rows(["eigenfaces"])
    write_run(archive, "eigenfaces_final", final, meta_for(final, manifest="other"))
    with pytest.raises(A.AnalysisError, match="manifest_sha256"):
        A.load_primary(archive)


def test_incomplete_run_is_rejected(archive):
    final = synth_rows(["eigenfaces"])
    write_run(archive, "eigenfaces_final", final, meta_for(final, status="PARTIAL"))
    with pytest.raises(A.AnalysisError, match="complete"):
        A.load_primary(archive)


def test_csv_and_json_disagreement_is_detected(archive):
    final = synth_rows(["eigenfaces"])
    bad = copy.deepcopy(final)
    bad[5]["auc"] += 0.01
    write_run(archive, "eigenfaces_final", final, meta_for(bad))
    with pytest.raises(A.AnalysisError, match="differs between CSV"):
        A.load_primary(archive)


def test_inconsistent_clean_baseline_is_rejected(archive):
    rows = synth_rows(["fisherfaces", "lbph", "hog_svm"])
    idx = next(i for i, r in enumerate(rows) if r["model"] == "lbph" and r["perturbation"] == "gamma" and r["severity"] == 0.0)
    rows[idx]["auc"] += 0.02
    write_run(archive, "classical", rows, meta_for(rows))
    with pytest.raises(A.AnalysisError, match="severity-0.0"):
        A.load_primary(archive)


# ---------------------------------------------------------------- calculations
def test_severity_grouping(archive):
    rows, _ = A.load_primary(archive)
    t = A.table_auc_by_severity(rows)
    assert len(t) == 4 * 6 and all(r["n_valid"] == 16 for r in t)
    r = next(x for x in t if x["model"] == "eigenfaces" and x["severity"] == 0.6)
    expected = [BASE_AUC["eigenfaces"] - STRENGTH[p] * 0.6 for p in A.PERTURBATION_ORDER]
    assert r["mean_auc"] == pytest.approx(sum(expected) / 16) and r["min_auc"] == pytest.approx(min(expected))
    assert r["max_auc"] == pytest.approx(max(expected))


def test_clean_baseline_values_and_ranks(archive):
    rows, _ = A.load_primary(archive)
    t = {r["model"]: r for r in A.table_clean_baseline(rows)}
    assert t["fisherfaces"]["auc"] == pytest.approx(0.69) and t["lbph"]["auc"] == pytest.approx(0.58)
    assert [t[m]["rank_auc"] for m in ("fisherfaces", "eigenfaces", "hog_svm", "lbph")] == [1, 2, 3, 4]
    assert [t[m]["rank_eer"] for m in ("fisherfaces", "eigenfaces", "hog_svm", "lbph")] == [1, 2, 3, 4]
    assert t["lbph"]["threshold"] == pytest.approx(0.1 + 0.01 * len("lbph"))


def test_harm_sign_convention():
    assert A.harm("auc", 0.60, 0.65) == pytest.approx(0.05)         # higher-is-better metric fell: harm > 0
    assert A.harm("accuracy", 0.70, 0.65) == pytest.approx(-0.05)   # improved: harm < 0
    assert A.harm("eer", 0.45, 0.40) == pytest.approx(0.05)         # lower-is-better metric rose: harm > 0
    assert A.harm("far", 0.30, 0.35) == pytest.approx(-0.05)
    assert A.harm("tar", 0.5, 0.5) == 0.0


def test_auc_degradation_calculations(archive):
    rows, _ = A.load_primary(archive)
    cells = A.table_cells_long(rows)
    assert len(cells) == 384 * len(A.METRICS)
    c = next(x for x in cells if x["model"] == "fisherfaces" and x["perturbation"] == "sunglasses" and x["severity"] == 0.8 and x["metric"] == "auc")
    assert c["delta"] == pytest.approx(-0.08) and c["harm"] == pytest.approx(0.08) and c["clean"] == pytest.approx(0.69)
    assert all(x["harm"] == 0.0 for x in cells if x["severity"] == 0.0)
    hm = {r["perturbation"]: r for r in A.table_heatmap(cells, "auc", 1.0)}
    assert hm["sunglasses"]["lbph"] == pytest.approx(0.1) and hm["gaussian_blur"]["eigenfaces"] == pytest.approx(0.001)
    mean = {r["perturbation"]: r for r in A.table_heatmap(cells, "auc", None)}
    assert mean["sunglasses"]["eigenfaces"] == pytest.approx(0.1 * (0.2 + 0.4 + 0.6 + 0.8 + 1.0) / 5)


def test_rankings_summary_and_cross_model_table(archive):
    rows, _ = A.load_primary(archive)
    cells = A.table_cells_long(rows)
    rk = A.table_rankings(cells, "auc")
    for m in A.MODEL_ORDER:
        mine = [r for r in rk if r["model"] == m]
        assert sorted(r["rank_mean"] for r in mine) == list(range(1, 17))
        assert sorted(r["rank_sev1"] for r in mine) == list(range(1, 17))
        assert mine[0]["perturbation"] == "sunglasses" and mine[0]["rank_mean"] == 1     # the designed worst perturbation
    s = {(r["model"], r["metric"]): r for r in A.table_degradation_summary(rows, cells)}
    e = s[("eigenfaces", "auc")]
    assert e["worst_perturbation"] == "sunglasses" and e["worst_severity"] == 1.0 and e["worst_harm"] == pytest.approx(0.1)
    assert e["n_conditions"] == 80 and e["n_valid"] == 80 and e["n_improved"] == 0
    exp_mean = sum(STRENGTH[p] * (0.2 + 0.4 + 0.6 + 0.8 + 1.0) / 5 for p in A.PERTURBATION_ORDER) / 16
    assert e["mean_harm_all"] == pytest.approx(exp_mean)
    x = A.table_perturbation_across_models(cells, "auc")
    assert x[0]["perturbation"] == "sunglasses" and x[0]["n_models_harmed_sev1"] == 4


def test_statistical_helpers_against_known_values():
    lo, hi = A.wilson_interval(0.5, 100)
    assert (lo, hi) == (pytest.approx(0.4038, abs=1e-3), pytest.approx(0.5962, abs=1e-3))
    assert all(math.isnan(v) for v in A.wilson_interval(float("nan"), 100))
    n = 764
    var_half = (0.25 + 2 * (n - 1) * (1 / 3 - 0.25)) / (n * n)             # Hanley-McNeil at AUC = 0.5
    assert A.hanley_mcneil_se(0.5, n, n) == pytest.approx(math.sqrt(var_half))
    assert 0.014 < A.hanley_mcneil_se(0.635, 764, 764) < 0.015
    assert math.isnan(A.hanley_mcneil_se(float("nan"), 764, 764))


def test_indicative_tables_are_labelled_and_use_the_clean_cell(archive):
    rows, _ = A.load_primary(archive)
    iv = {r["model"]: r for r in A.table_clean_intervals(rows)}
    assert iv["eigenfaces"]["auc_lo"] < iv["eigenfaces"]["auc"] < iv["eigenfaces"]["auc_hi"]
    assert "not computed" in iv["eigenfaces"]["eer_interval"]
    cells = A.table_auc_cells_indicative(rows)
    assert len(cells) == 4 * 16 * 5 and all(c["severity"] > 0 for c in cells)
    big = next(c for c in cells if c["model"] == "fisherfaces" and c["perturbation"] == "sunglasses" and c["severity"] == 1.0)
    assert big["abs_harm_exceeds_scale"] is True and big["indicative_scale_95"] > 0.03
    tiny = next(c for c in cells if c["model"] == "fisherfaces" and c["perturbation"] == "gaussian_blur" and c["severity"] == 0.2)
    assert tiny["abs_harm_exceeds_scale"] is False


# ---------------------------------------------------------------- missing / invalid metric handling
def _with_bad_cell(archive, column, value):
    rows = synth_rows(["fisherfaces", "lbph", "hog_svm"])
    idx = next(i for i, r in enumerate(rows) if r["model"] == "lbph" and r["perturbation"] == "gamma" and r["severity"] == 0.6)
    rows[idx][column] = value
    write_run(archive, "classical", rows, meta_for(rows))


@pytest.mark.parametrize("value", [float("nan"), 1.5, -0.1, float("inf"), "abc"])
def test_invalid_metrics_are_rejected_by_default(archive, value):
    _with_bad_cell(archive, "auc", value)
    with pytest.raises(A.InvalidMetricError):
        A.load_primary(archive)


def test_nan_is_kept_explicitly_and_excluded_from_aggregates_when_allowed(archive):
    _with_bad_cell(archive, "auc", float("nan"))
    rows, _ = A.load_primary(archive, allow_nan_metrics=True)
    cells = A.table_cells_long(rows)
    bad = next(c for c in cells if c["model"] == "lbph" and c["perturbation"] == "gamma" and c["severity"] == 0.6 and c["metric"] == "auc")
    assert math.isnan(bad["value"]) and math.isnan(bad["harm"])
    sev = next(r for r in A.table_auc_by_severity(rows) if r["model"] == "lbph" and r["severity"] == 0.6)
    assert sev["n_valid"] == 15 and not math.isnan(sev["mean_auc"])
    summ = next(r for r in A.table_degradation_summary(rows, cells) if r["model"] == "lbph" and r["metric"] == "auc")
    assert summ["n_valid"] == 79 and summ["n_conditions"] == 80 and not math.isnan(summ["mean_harm_all"])
    rank = next(r for r in A.table_rankings(cells, "auc") if r["model"] == "lbph" and r["perturbation"] == "gamma")
    assert rank["n_valid"] == 4
    ind = [c for c in A.table_auc_cells_indicative(rows) if c["model"] == "lbph" and c["perturbation"] == "gamma" and c["severity"] == 0.6]
    assert ind[0]["abs_harm_exceeds_scale"] == ""                      # undecidable, never silently False


def test_detector_miss_nan_is_expected_and_accepted(archive):
    rows, _ = A.load_primary(archive)
    assert all(math.isnan(r["detector_miss_rate"]) for r in rows)


def test_write_csv_renders_nan_empty_and_refuses_empty_tables(tmp_path):
    A.write_csv(tmp_path / "t.csv", [{"a": 1.0, "b": float("nan"), "c": True}])
    assert (tmp_path / "t.csv").read_text() == "a,b,c\n1,,true\n"
    with pytest.raises(A.AnalysisError):
        A.write_csv(tmp_path / "e.csv", [])


# ---------------------------------------------------------------- historical separation
def test_historical_run_is_never_part_of_the_primary_rows(archive):
    (archive / "eigenfaces_per_image_noise" / "verification_results.csv").write_text("garbage\n")
    rows, prov = A.load_primary(archive)            # must not even read the historical run
    assert len(rows) == 384 and "eigenfaces_per_image_noise" not in prov["runs"]


def test_historical_comparison_table(archive):
    t, prov = A.table_historical_vs_final(archive)
    assert len(t) == 96 and sum(r["is_noise_perturbation"] for r in t) == 18
    r = next(x for x in t if x["perturbation"] == "gamma" and x["severity"] == 0.4)
    assert r["auc_diff"] == pytest.approx(0.001) and r["auc_historical"] == pytest.approx(r["auc_final"] + 0.001)
    assert prov["eigenfaces_per_image_noise"]["code_version"] == "old"
    assert prov["eigenfaces_final"]["code_version"] == "aaaa"


# ---------------------------------------------------------------- pipeline: outputs, safety, consistency with sources
def test_run_analysis_outputs_match_source_csvs_and_do_not_touch_the_archive(archive, tmp_path):
    before = digest_tree(archive)
    out = tmp_path / "analysis"
    written = A.run_analysis(archive, out, figures=True)
    assert digest_tree(archive) == before                               # archived files are untouched
    assert {p.name for p in (out / "figures").glob("*.png")} == {f"fig{i}_{n}.png" for i, n in enumerate(
        ["clean_baseline", "auc_harm_heatmap_severity1", "auc_harm_heatmap_mean_severity", "auc_severity_mean",
         "auc_severity_by_perturbation", "degradation_rankings"], 1)}
    assert all(p.stat().st_size > 2000 for p in written.values() if p.suffix == ".png")
    # recompute from the raw CSVs, independently of the module
    raw = {}
    for run in ("classical", "eigenfaces_final"):
        for r in csv.DictReader(open(archive / run / "verification_results.csv")):
            raw[(r["model"], r["perturbation"], round(float(r["severity"]), 3))] = r
    long_rows = list(csv.DictReader(open(out / "tables" / "table10_all_cells_long.csv")))
    assert len(long_rows) == 384 * 5
    for r in long_rows:
        src = raw[(r["model"], r["perturbation"], round(float(r["severity"]), 3))]
        clean = raw[(r["model"], r["perturbation"], 0.0)]
        assert float(r["value"]) == pytest.approx(float(src[r["metric"]]), abs=1e-9)
        sign = 1 if r["metric"] in ("far", "eer") else -1
        assert float(r["harm"]) == pytest.approx(sign * (float(src[r["metric"]]) - float(clean[r["metric"]])), abs=1e-9)
    heat = list(csv.DictReader(open(out / "tables" / "table3_auc_harm_heatmap_severity1.csv")))
    assert len(heat) == 16 and list(heat[0].keys()) == ["perturbation", *A.MODEL_ORDER]
    for r in heat:
        for m in A.MODEL_ORDER:
            exp = float(raw[(m, r["perturbation"], 0.0)]["auc"]) - float(raw[(m, r["perturbation"], 1.0)]["auc"])
            assert float(r[m]) == pytest.approx(exp, abs=1e-9)
    prov = json.loads((out / "provenance.json").read_text())
    assert prov["n_primary_rows"] == 384 and "eigenfaces_per_image_noise" in prov["historical_run"]
    assert "eigenfaces_per_image_noise" not in prov["primary_runs"]
    hist = list(csv.DictReader(open(out / "tables" / "table11_historical_vs_final_eigenfaces.csv")))
    assert len(hist) == 96


def test_output_directory_cannot_be_inside_an_archived_run(archive):
    with pytest.raises(A.AnalysisError, match="refusing to write"):
        A.run_analysis(archive, archive / "classical" / "analysis", figures=False)
    with pytest.raises(A.AnalysisError, match="refusing to write"):
        A.run_analysis(archive, archive / "eigenfaces_per_image_noise", figures=False)


def test_pipeline_is_deterministic(archive, tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    A.run_analysis(archive, a, figures=False)
    A.run_analysis(archive, b, figures=False)
    assert digest_tree(a) == digest_tree(b)


# ---------------------------------------------------------------- the real archived results (committed in this repo)
needs_archive = pytest.mark.skipif(not (REAL_ROOT / "classical" / "verification_results.csv").is_file(),
                                   reason="archived results are not present in this checkout")


@needs_archive
def test_real_archive_primary_rows_and_provenance():
    rows, prov = A.load_primary(REAL_ROOT)
    assert len(rows) == 384 and {r["model"] for r in rows} == set(A.MODEL_ORDER)
    c, f = prov["runs"]["classical"], prov["runs"]["eigenfaces_final"]
    assert c["code_version"] == f["code_version"] and c["manifest_sha256"] == f["manifest_sha256"]
    assert c["split_content_sha256"] == f["split_content_sha256"]
    assert c["noise_protocol_version"] == "per_image_v2" and c["noise_master_seed"] == 42
    assert c["noise_stream_matches_approved"] is True and f["noise_stream_matches_approved"] is True


@needs_archive
def test_real_archive_tables_match_the_source_csvs(tmp_path):
    out = tmp_path / "analysis"
    A.run_analysis(REAL_ROOT, out, figures=False)
    src = {}
    for run in ("classical", "eigenfaces_final"):
        for r in csv.DictReader(open(REAL_ROOT / run / "verification_results.csv")):
            src[(r["model"], r["perturbation"], round(float(r["severity"]), 3))] = r
    t1 = {r["model"]: r for r in csv.DictReader(open(out / "tables" / "table1_clean_baseline.csv"))}
    for m in A.MODEL_ORDER:
        clean = src[(m, "gamma", 0.0)]
        for col in ("accuracy", "tar", "far", "auc", "eer", "threshold"):
            assert float(t1[m][col]) == pytest.approx(float(clean[col]), abs=1e-9)
    summ = {(r["model"], r["metric"]): r for r in csv.DictReader(open(out / "tables" / "table7_degradation_summary.csv"))}
    for m in A.MODEL_ORDER:
        harms = [float(src[(m, p, 0.0)]["auc"]) - float(src[(m, p, s)]["auc"]) for p in A.PERTURBATION_ORDER for s in A.SEVERITIES if s > 0]
        assert float(summ[(m, "auc")]["mean_harm_all"]) == pytest.approx(sum(harms) / 80, abs=1e-9)
        assert float(summ[(m, "auc")]["worst_harm"]) == pytest.approx(max(harms), abs=1e-9)


@needs_archive
def test_real_archive_historical_non_noise_rows_agree_on_counts_based_metrics():
    t, prov = A.table_historical_vs_final(REAL_ROOT)
    assert len(t) == 96
    for r in t:
        if not r["is_noise_perturbation"]:
            for metric in ("accuracy", "tar", "far"):
                assert r[f"{metric}_diff"] == 0.0, (r["perturbation"], r["severity"], metric)
    clean = [r for r in t if r["severity"] == 0.0]
    assert all(r[f"{m}_diff"] == 0.0 for r in clean for m in ("accuracy", "tar", "far"))
    assert prov["eigenfaces_per_image_noise"]["code_version"] != prov["eigenfaces_final"]["code_version"]

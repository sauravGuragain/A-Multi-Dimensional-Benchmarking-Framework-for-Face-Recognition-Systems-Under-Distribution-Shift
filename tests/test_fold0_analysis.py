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

FOUR = A.CLASSICAL_MODELS
SIX = A.MODEL_ORDER
BASE_AUC = {"eigenfaces": 0.64, "fisherfaces": 0.69, "lbph": 0.58, "hog_svm": 0.62, "facenet": 0.995, "arcface": 0.99}
STRENGTH = {p: 0.001 * (i + 1) for i, p in enumerate(A.PERTURBATION_ORDER)}     # AUC lost per unit severity
STRENGTH["sunglasses"] = 0.1                                                      # the designed worst perturbation
COLUMNS = ["model", "paradigm", "backbone", "perturbation", "severity", "label", "embedding_dim", "threshold",
           "detector_miss_rate", "val_detector_miss_rate", "accuracy", "tar", "far", "n_genuine",
           "n_impostor_balanced", "auc", "eer", "frr_at_far_0.001", "frr_at_far_0.01", "frr_at_far_0.1",
           "n_impostor_full", "auc_full", "eer_full", "frr_at_far_0.001_full", "frr_at_far_0.01_full",
           "frr_at_far_0.1_full", "far_full"]


DEEP_EXTRA = ["normalization_mode", "alignment", "crop_fraction", "val_balanced_accuracy", "val_n_failed_embeddings",
              "n_test_images", "n_failed_embeddings", "n_failed_pairs_balanced", "n_failed_pairs_full", "scores_file"]
N_IMG, N_BAL, N_FULL = 1338, 1528, 8404                      # test images, 764+764 balanced pairs, 764+7640 full-pool pairs
HEAVY_FAILURE = {"face_mask", "sunglasses"}                    # designed so that ArcFace loses every image at severity >= 0.8


def arcface_failure(p, s):
    """(failed images, failed balanced pairs, failed full-pool pairs) of the synthetic ArcFace run."""
    f = min(1.0, 1.25 * s) if p in HEAVY_FAILURE else (0.3 * s if p == "rotation" else 0.0)
    if f == 0.0:
        return 0, 0, 0
    if f >= 1.0:
        return N_IMG, N_BAL, N_FULL
    pair = 1.0 - (1.0 - f) ** 2
    return round(N_IMG * f), round(N_BAL * pair), round(N_FULL * pair)


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
                if m in A.DEEP_MODELS:
                    fi, fb, ff = arcface_failure(p, s) if m == "arcface" else (0, 0, 0)
                    rows[-1].update({"paradigm": "deep_learning", "backbone": "real", "embedding_dim": 512,
                                     "normalization_mode": "x", "alignment": "y", "crop_fraction": 0.8,
                                     "val_balanced_accuracy": 0.9, "val_n_failed_embeddings": 0 if m == "facenet" else 3,
                                     "n_test_images": N_IMG, "n_failed_embeddings": fi, "n_failed_pairs_balanced": fb,
                                     "n_failed_pairs_full": ff, "scores_file": f"{m}__{p}__sev{s:.2f}.npz"})
    return rows


def meta_for(rows, code="aaaa", manifest="m1", **over):
    meta = {"protocol": "lfw_fold0_verification", "full_lfw": True, "status": "COMPLETE", "failures": [],
            "n_rows": len(rows), "environment": {"code_version": code, "numpy_version": "1.26.4", "python_version": "3.12"},
            "manifest_sha256": manifest, "content_sha256": {"split": "s", "pairs": {"validation": "pv", "test": "pt"}},
            "dataset": {"fingerprint_sha256": "fp", "counts_match_official_lfw": True},
            "pair_params": {"test": {"n_images": 1338, "n_genuine": 764, "n_pool": 7640, "n_balanced": 764},
                            "validation": {"n_images": 1259, "n_genuine": 684, "n_pool": 6840, "n_balanced": 684}},
            "threshold_protocol": "max-accuracy on clean validation balanced pairs; frozen for test",
            "stub_backbones": [], "fold": 0, "k_folds": 10, "seed": 42, "n_rows_expected": len(rows),
            "noise_seeding": {"protocol_version": "per_image_v2", "master_seed": 42,
                              "stream_fingerprint": {"matches_approved_reference": True}},
            "results": rows}
    meta.update(over)
    return meta


def write_run(root, run, rows, meta):
    d = Path(root) / run
    d.mkdir(parents=True, exist_ok=True)
    cols = COLUMNS + [c for c in DEEP_EXTRA if c in rows[0]]
    with open(d / "verification_results.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, lineterminator="\n")
        w.writeheader()
        for r in rows:
            w.writerow({c: ("nan" if isinstance(r[c], float) and math.isnan(r[c]) else r[c]) for c in cols})
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
    assert {r["model"] for r in rows} == set(FOUR)
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


def test_distinct_code_and_manifest_between_classical_runs_are_accepted_but_recorded(archive):
    final = synth_rows(["eigenfaces"])
    write_run(archive, "eigenfaces_final", final, meta_for(final, code="zzzz", manifest="other"))
    rows, prov = A.load_primary(archive)
    assert len(rows) == 384
    assert prov["runs"]["eigenfaces_final"]["manifest_sha256"] == "other" and prov["runs"]["classical"]["manifest_sha256"] == "m1"
    assert prov["runs"]["eigenfaces_final"]["code_version"] == "zzzz" and prov["runs"]["classical"]["code_version"] == "aaaa"


def test_primary_runs_must_share_the_split(archive):
    final = synth_rows(["eigenfaces"])
    write_run(archive, "eigenfaces_final", final, meta_for(final, content_sha256={"split": "other", "pairs": {"validation": "pv", "test": "pt"}}))
    with pytest.raises(A.AnalysisError, match="split_content_sha256"):
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
    for m in FOUR:
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
    assert len(heat) == 16 and list(heat[0].keys()) == ["perturbation", *FOUR]
    for r in heat:
        for m in FOUR:
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
    assert len(rows) == 384 and {r["model"] for r in rows} == set(FOUR)
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
    for m in FOUR:
        clean = src[(m, "gamma", 0.0)]
        for col in ("accuracy", "tar", "far", "auc", "eer", "threshold"):
            assert float(t1[m][col]) == pytest.approx(float(clean[col]), abs=1e-9)
    summ = {(r["model"], r["metric"]): r for r in csv.DictReader(open(out / "tables" / "table7_degradation_summary.csv"))}
    for m in FOUR:
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


# =====================================================================================================================
# Six-model analysis (FaceNet and ArcFace runs live OUTSIDE the results root and are passed as explicit directories)
# =====================================================================================================================
@pytest.fixture
def six(archive, tmp_path):
    """The four-model archive plus FaceNet and ArcFace runs in separate, differently named run trees."""
    fn, ar = synth_rows(["facenet"]), synth_rows(["arcface"])
    fdir = tmp_path / "runs" / "fn_run" / "experiments" / "x" / "verification" / "b567"
    adir = tmp_path / "runs" / "ar_run" / "experiments" / "y" / "verification" / "4bdf"
    write_run(fdir.parent, fdir.name, fn, meta_for(fn, code="bbbb", manifest="m_fn"))
    write_run(adir.parent, adir.name, ar, meta_for(ar, code="cccc", manifest="m_ar"))
    return archive, {"facenet": fdir, "arcface": adir}


def _rewrite_deep(deep, name, rows, **meta_over):
    d = deep[name]
    write_run(d.parent, d.name, rows, meta_for(rows, **meta_over))


def test_six_models_give_576_rows_and_all_grid_cells(six):
    archive, deep = six
    rows, prov = A.load_primary(archive, deep_runs=deep)
    assert len(rows) == 576 and {r["model"] for r in rows} == set(SIX)
    assert {(r["model"], r["perturbation"], r["severity"]) for r in rows} == {
        (m, p, s) for m in SIX for p in A.PERTURBATION_ORDER for s in A.SEVERITIES}
    assert set(prov["runs"]) == {"classical", "eigenfaces_final", "facenet", "arcface"}
    assert "eigenfaces_per_image_noise" not in prov["runs"]
    for run, files in prov["input_files"].items():
        assert set(files) == {"verification_results.csv", "verification_results.json"}
        assert all(len(v["sha256"]) == 64 for v in files.values())


def test_default_without_deep_runs_is_still_the_four_model_analysis(archive):
    rows, prov = A.load_primary(archive)
    assert len(rows) == 384 and {r["model"] for r in rows} == set(FOUR)


def test_unknown_deep_run_name_is_rejected(six):
    archive, deep = six
    with pytest.raises(A.AnalysisError, match="unknown deep run"):
        A.load_primary(archive, deep_runs={"insightface": deep["facenet"]})


def test_deep_run_directory_without_files_is_rejected(six, tmp_path):
    archive, deep = six
    with pytest.raises(A.AnalysisError, match="missing archived file"):
        A.load_primary(archive, deep_runs={**deep, "facenet": tmp_path / "nowhere"})


def test_missing_condition_in_a_deep_run_is_rejected(six):
    archive, deep = six
    rows = synth_rows(["arcface"])[:-1]
    _rewrite_deep(deep, "arcface", rows)
    with pytest.raises(A.AnalysisError, match="arcface.*condition grid differs"):
        A.load_primary(archive, deep_runs=deep)


def test_duplicate_condition_in_a_deep_run_is_rejected(six):
    archive, deep = six
    rows = synth_rows(["facenet"])
    rows[-1] = dict(rows[0])
    _rewrite_deep(deep, "facenet", rows)
    with pytest.raises(A.AnalysisError, match="facenet.*duplicate"):
        A.load_primary(archive, deep_runs=deep)


def test_deep_run_holding_the_wrong_model_is_rejected(six):
    archive, deep = six
    rows = synth_rows(["facenet"])
    _rewrite_deep(deep, "arcface", rows)                              # the ArcFace directory holds FaceNet rows
    with pytest.raises(A.AnalysisError, match="expected models"):
        A.load_primary(archive, deep_runs=deep)


def test_each_run_must_have_exactly_96_rows_per_model(six):
    archive, deep = six
    fn = synth_rows(["facenet"])
    A.validate_run(fn, "facenet", ("facenet",))                                  # exactly 96 rows: accepted
    assert len(fn) == 96
    with pytest.raises(A.AnalysisError, match="condition grid differs"):         # 95 rows
        A.validate_run(fn[:-1], "facenet", ("facenet",))
    with pytest.raises(A.AnalysisError, match="duplicate"):                      # 97 rows
        A.validate_run(fn + [dict(fn[0])], "facenet", ("facenet",))
    with pytest.raises(A.AnalysisError, match="model coverage differs"):         # a second model smuggled in
        A.validate_run(fn + synth_rows(["arcface"]), "facenet", ("facenet",))
    _rewrite_deep(deep, "facenet", fn + [dict(fn[0])])
    with pytest.raises(A.AnalysisError, match="facenet.*duplicate"):
        A.load_primary(archive, deep_runs=deep)


def test_threshold_is_validated_per_model_independently(six):
    archive, deep = six
    rows, _ = A.load_primary(archive, deep_runs=deep)                  # six different thresholds across models are fine
    assert len({r["threshold"] for r in rows}) >= 3
    bad = synth_rows(["arcface"])
    bad[17]["threshold"] += 0.05                                       # one arcface cell differs from the other 95
    _rewrite_deep(deep, "arcface", bad)
    with pytest.raises(A.AnalysisError, match="arcface: the frozen threshold is not constant"):
        A.load_primary(archive, deep_runs=deep)


def test_deep_clean_baseline_must_be_consistent_across_perturbations(six):
    archive, deep = six
    rows = synth_rows(["facenet"])
    idx = next(i for i, r in enumerate(rows) if r["perturbation"] == "gamma" and r["severity"] == 0.0)
    rows[idx]["auc"] -= 0.01
    _rewrite_deep(deep, "facenet", rows)
    with pytest.raises(A.AnalysisError, match="facenet.*severity-0.0 'auc'"):
        A.load_primary(archive, deep_runs=deep)


@pytest.mark.parametrize("key,over", [
    ("split_content_sha256", {"content_sha256": {"split": "other", "pairs": {"validation": "pv", "test": "pt"}}}),
    ("pair_content_sha256", {"content_sha256": {"split": "s", "pairs": {"validation": "pv", "test": "OTHER"}}}),
    ("pair_content_sha256", {"content_sha256": {"split": "s", "pairs": {"validation": "OTHER", "test": "pt"}}}),
    ("dataset_fingerprint_sha256", {"dataset": {"fingerprint_sha256": "other"}}),
    ("noise_protocol_version", {"noise_seeding": {"protocol_version": "per_image_v1", "master_seed": 42,
                                                  "stream_fingerprint": {"matches_approved_reference": True}}}),
    ("noise_master_seed", {"noise_seeding": {"protocol_version": "per_image_v2", "master_seed": 7,
                                             "stream_fingerprint": {"matches_approved_reference": True}}}),
    ("seed", {"seed": 99}),
])
def test_deep_run_must_share_split_pairs_dataset_and_noise_protocol(six, key, over):
    archive, deep = six
    rows = synth_rows(["facenet"])
    _rewrite_deep(deep, "facenet", rows, code="bbbb", manifest="m_fn", **over)
    with pytest.raises(A.AnalysisError, match=key):
        A.load_primary(archive, deep_runs=deep)


def test_noise_stream_must_match_the_approved_reference(six):
    archive, deep = six
    rows = synth_rows(["facenet"])
    _rewrite_deep(deep, "facenet", rows, noise_seeding={"protocol_version": "per_image_v2", "master_seed": 42,
                                                        "stream_fingerprint": {"matches_approved_reference": False}})
    with pytest.raises(A.AnalysisError, match="approved reference"):
        A.load_primary(archive, deep_runs=deep)


@pytest.mark.parametrize("over", [{"status": "INCOMPLETE"}, {"full_lfw": False}, {"failures": [{"x": 1}]}])
def test_incomplete_or_failed_deep_runs_are_rejected(six, over):
    archive, deep = six
    rows = synth_rows(["arcface"])
    _rewrite_deep(deep, "arcface", rows, **over)
    with pytest.raises(A.AnalysisError, match="complete"):
        A.load_primary(archive, deep_runs=deep)


def test_distinct_code_versions_and_manifests_across_all_runs_are_accepted_and_preserved(six):
    archive, deep = six
    _, prov = A.load_primary(archive, deep_runs=deep)
    codes = {k: v["code_version"] for k, v in prov["runs"].items()}
    manifests = {k: v["manifest_sha256"] for k, v in prov["runs"].items()}
    assert codes == {"classical": "aaaa", "eigenfaces_final": "aaaa", "facenet": "bbbb", "arcface": "cccc"}
    assert manifests == {"classical": "m1", "eigenfaces_final": "m1", "facenet": "m_fn", "arcface": "m_ar"}
    for v in prov["runs"].values():
        assert v["numpy_version"] == "1.26.4" and v["python_version"] == "3.12" and v["split_content_sha256"] == "s"


def test_deep_csv_and_json_must_agree_on_metrics_and_failure_counters(six):
    archive, deep = six
    rows = synth_rows(["arcface"])
    bad = copy.deepcopy(rows)
    bad[40]["auc"] += 0.01
    _rewrite_deep(deep, "arcface", rows)
    d = deep["arcface"]
    (d / "verification_results.json").write_text(json.dumps(meta_for(bad, code="cccc", manifest="m_ar")))
    with pytest.raises(A.AnalysisError, match="auc differs between CSV"):
        A.load_primary(archive, deep_runs=deep)
    bad = copy.deepcopy(rows)
    k = next(i for i, r in enumerate(rows) if r["n_failed_embeddings"] > 0)
    bad[k]["n_failed_embeddings"] += 1
    (d / "verification_results.json").write_text(json.dumps(meta_for(bad, code="cccc", manifest="m_ar")))
    with pytest.raises(A.AnalysisError, match="n_failed_embeddings differs between CSV"):
        A.load_primary(archive, deep_runs=deep)
    short = copy.deepcopy(rows)[:-1]
    (d / "verification_results.json").write_text(json.dumps(meta_for(short, code="cccc", manifest="m_ar")))
    with pytest.raises(A.AnalysisError, match="CSV has 96 rows but JSON has 95"):
        A.load_primary(archive, deep_runs=deep)


@pytest.mark.parametrize("mutate,match", [
    (lambda r: r.update(n_failed_pairs_balanced=5, n_failed_pairs_full=5, n_failed_embeddings=0), "without any failed embedding"),
    (lambda r: r.update(n_failed_embeddings=N_IMG + 1), "exceed test images"),
    (lambda r: r.update(n_failed_pairs_balanced=N_BAL + 1, n_failed_embeddings=3), "inconsistent failed-pair"),
    (lambda r: r.update(n_failed_embeddings=N_IMG, n_failed_pairs_balanced=10, n_failed_pairs_full=20), "every image failed"),
    (lambda r: r.update(backbone="STUB"), "STUB"),
])
def test_inconsistent_failure_counters_and_stub_backbones_are_rejected(six, mutate, match):
    archive, deep = six
    rows = synth_rows(["arcface"])
    idx = next(i for i, r in enumerate(rows) if r["perturbation"] == "gaussian_blur" and r["severity"] == 0.4)
    mutate(rows[idx])
    _rewrite_deep(deep, "arcface", rows)
    with pytest.raises(A.AnalysisError, match=match):
        A.load_primary(archive, deep_runs=deep)


def test_deep_rows_without_failure_columns_are_rejected(six):
    archive, deep = six
    d = deep["facenet"]
    write_run(d.parent, d.name, synth_rows(["facenet"]), meta_for(synth_rows(["facenet"]), code="bbbb", manifest="m_fn"))
    text = (d / "verification_results.csv").read_text().splitlines()
    head = text[0].split(",")
    k = head.index("n_failed_pairs_full")
    stripped = "\n".join(",".join(c for i, c in enumerate(line.split(",")) if i != k) for line in text) + "\n"
    (d / "verification_results.csv").write_text(stripped)
    js = json.loads((d / "verification_results.json").read_text())
    for r in js["results"]:
        r.pop("n_failed_pairs_full")
    (d / "verification_results.json").write_text(json.dumps(js))
    with pytest.raises(A.AnalysisError, match="missing failure diagnostic column 'n_failed_pairs_full'"):
        A.load_primary(archive, deep_runs=deep)


@pytest.mark.parametrize("value", ["-1", "2.5", "nan", "abc"])
def test_failure_counters_must_be_non_negative_integers(value):
    raw = {"model": "arcface", "perturbation": "gamma", "severity": "0.2", "n_failed_embeddings": value}
    with pytest.raises(A.InvalidMetricError):
        A.parse_row(raw)


# ---------------------------------------------------------------- six-model tables
def test_all_tables_cover_six_models(six):
    archive, deep = six
    rows, _ = A.load_primary(archive, deep_runs=deep)
    cells = A.table_cells_long(rows)
    assert len(cells) == 576 * len(A.METRICS)
    t1 = A.table_clean_baseline(rows)
    assert [r["model"] for r in t1] == list(SIX) and sorted(r["rank_auc"] for r in t1) == [1, 2, 3, 4, 5, 6]
    assert next(r for r in t1 if r["model"] == "facenet")["rank_auc"] == 1
    assert [r["model"] for r in A.table_clean_intervals(rows)] == list(SIX)
    assert list(A.table_heatmap(cells, "auc", 1.0)[0]) == ["perturbation", *SIX]
    assert list(A.table_heatmap(cells, "auc", None)[0]) == ["perturbation", *SIX]
    assert len(A.table_auc_by_severity(rows)) == 6 * 6
    assert len(A.table_rankings(cells, "auc")) == 6 * 16
    assert len(A.table_degradation_summary(rows, cells)) == 6 * 3
    x = A.table_perturbation_across_models(cells, "auc")
    assert all(r["n_models"] == 6 for r in x) and max(r["n_models_harmed_sev1"] for r in x) <= 6
    assert len(A.table_auc_cells_indicative(rows)) == 6 * 16 * 5


# ---------------------------------------------------------------- failure diagnostics
def test_arcface_diagnostics_keep_zero_failure_and_total_failure_conditions(six):
    archive, deep = six
    rows, _ = A.load_primary(archive, deep_runs=deep)
    d = {(r["model"], r["perturbation"], r["severity"]): r for r in A.table_failure_diagnostics(rows)}
    assert len(d) == 576
    zero = d[("arcface", "gaussian_blur", 0.6)]
    assert zero["n_failed_embeddings"] == 0 and zero["n_failed_pairs_balanced"] == 0 and zero["n_failed_pairs_full"] == 0
    assert zero["image_failure_rate"] == 0.0 and zero["diagnostics_applicable"] is True
    clean = d[("arcface", "sunglasses", 0.0)]
    assert clean["n_failed_embeddings"] == 0 and clean["is_clean_condition"] is True
    worst = d[("arcface", "sunglasses", 1.0)]
    assert (worst["n_failed_embeddings"], worst["n_failed_pairs_balanced"], worst["n_failed_pairs_full"]) == (N_IMG, N_BAL, N_FULL)
    assert worst["image_failure_rate"] == 1.0 and worst["balanced_pair_failure_rate"] == 1.0 and worst["full_pair_failure_rate"] == 1.0
    mid = d[("arcface", "sunglasses", 0.2)]
    assert 0 < mid["n_failed_embeddings"] < N_IMG
    assert all(d[("facenet", p, s)]["n_failed_embeddings"] == 0 and d[("facenet", p, s)]["n_failed_pairs_full"] == 0
               for p in A.PERTURBATION_ORDER for s in A.SEVERITIES)
    # the archived end-to-end metrics are carried through unchanged, not recomputed
    src = {(r["model"], r["perturbation"], r["severity"]): r for r in rows}
    assert worst["auc_end_to_end"] == src[("arcface", "sunglasses", 1.0)]["auc"]


def test_failure_rates_use_the_image_and_the_pair_denominators(six):
    archive, deep = six
    rows, _ = A.load_primary(archive, deep_runs=deep)
    r = next(x for x in A.table_failure_diagnostics(rows) if x["model"] == "arcface" and x["perturbation"] == "rotation" and x["severity"] == 0.6)
    fi, fb, ff = arcface_failure("rotation", 0.6)
    assert (r["n_test_images"], r["n_balanced_pairs"], r["n_full_pairs"]) == (N_IMG, N_BAL, N_FULL)
    assert r["image_failure_rate"] == pytest.approx(fi / N_IMG)
    assert r["balanced_pair_failure_rate"] == pytest.approx(fb / N_BAL)
    assert r["full_pair_failure_rate"] == pytest.approx(ff / N_FULL)
    assert r["image_failure_rate"] != pytest.approx(r["balanced_pair_failure_rate"])        # an image rate is not a pair rate
    assert r["val_n_failed_embeddings"] == 3


def test_safe_rate_handles_zero_and_negative_denominators():
    assert A.safe_rate(3, 4) == 0.75 and math.isnan(A.safe_rate(0, 0)) and math.isnan(A.safe_rate(1, -1))


def test_classical_diagnostics_are_blank_not_fabricated(six):
    archive, deep = six
    rows, _ = A.load_primary(archive, deep_runs=deep)
    per = [r for r in A.table_failure_diagnostics(rows) if r["model"] in FOUR]
    assert len(per) == 4 * 96 and all(r["diagnostics_applicable"] is False for r in per)
    blanks = ("n_test_images", "n_failed_embeddings", "image_failure_rate", "n_balanced_pairs", "n_failed_pairs_balanced",
              "balanced_pair_failure_rate", "n_full_pairs", "n_failed_pairs_full", "full_pair_failure_rate", "val_n_failed_embeddings")
    assert all(r[c] == "" for r in per for c in blanks)
    summ = {r["model"]: r for r in A.table_failure_model_summary(rows)}
    for m in FOUR:
        assert summ[m] == {"model": m, "label": A.MODEL_LABELS[m], "diagnostics_applicable": False}
    four_only, _ = A.load_primary(archive)
    assert {r["diagnostics_applicable"] for r in A.table_failure_diagnostics(four_only)} == {False}


def test_model_level_failure_totals_use_matching_denominators_and_keep_condition_variation(six):
    archive, deep = six
    rows, _ = A.load_primary(archive, deep_runs=deep)
    s = {r["model"]: r for r in A.table_failure_model_summary(rows)}
    fn, ar = s["facenet"], s["arcface"]
    assert fn["perturbed_sum_failed_embeddings"] == 0 and fn["perturbed_n_conditions_with_any_failure"] == 0
    assert fn["perturbed_pooled_image_failure_rate"] == 0.0
    expected = sum(arcface_failure(p, sv)[0] for p in A.PERTURBATION_ORDER for sv in A.SEVERITIES if sv > 0)
    assert ar["perturbed_n_conditions"] == 80 and ar["clean_n_conditions"] == 16
    assert ar["perturbed_sum_failed_embeddings"] == expected and ar["perturbed_sum_image_denominator"] == 80 * N_IMG
    assert ar["perturbed_pooled_image_failure_rate"] == pytest.approx(expected / (80 * N_IMG))
    assert ar["perturbed_sum_balanced_pair_denominator"] == 80 * N_BAL and ar["perturbed_sum_full_pair_denominator"] == 80 * N_FULL
    assert ar["clean_sum_failed_embeddings"] == 0 and ar["all_rows_sum_failed_embeddings"] == expected
    assert ar["perturbed_min_condition_image_failure_rate"] == 0.0 and ar["perturbed_max_condition_image_failure_rate"] == 1.0
    n_total = sum(1 for p in HEAVY_FAILURE for sv in A.SEVERITIES if sv >= 0.8)
    assert ar["perturbed_n_conditions_all_images_failed"] == n_total == 4
    assert ar["perturbed_n_conditions_with_any_failure"] == 2 * 5 + 5                      # 2 heavy types x 5 + rotation x 5
    assert "not unique failed images" in ar["note"]


# ---------------------------------------------------------------- pipeline: separate output, safety, figures, CLI
def _dirs_digest(archive, deep):
    return {"archive": digest_tree(archive), **{k: digest_tree(v) for k, v in deep.items()}}


def test_six_model_run_writes_a_separate_directory_and_does_not_touch_any_input(six, tmp_path):
    archive, deep = six
    before = _dirs_digest(archive, deep)
    out = tmp_path / "analysis_six"
    written = A.run_analysis(archive, out, figures=True, deep_runs=deep)
    assert _dirs_digest(archive, deep) == before
    assert not (archive / "analysis_six").exists()
    names = {p.name for p in (out / "figures").glob("*.png")}
    assert names == {"fig1_clean_baseline.png", "fig2_auc_harm_heatmap_severity1.png", "fig3_auc_harm_heatmap_mean_severity.png",
                     "fig4_auc_severity_mean.png", "fig5_auc_severity_by_perturbation.png", "fig6_degradation_rankings.png",
                     "fig7_failure_rates.png"}
    assert all(p.stat().st_size > 2000 for p in written.values() if p.suffix == ".png")
    tables = {p.name for p in (out / "tables").glob("*.csv")}
    assert len(tables) == 13 and "table12_failure_diagnostics_by_condition.csv" in tables and "table13_failure_diagnostics_model_summary.csv" in tables
    long_rows = list(csv.DictReader(open(out / "tables" / "table10_all_cells_long.csv")))
    assert len(long_rows) == 576 * 5
    prov = json.loads((out / "provenance.json").read_text())
    assert prov["n_primary_rows"] == 576 and prov["models"] == list(SIX)
    assert set(prov["primary_runs"]) == {"classical", "eigenfaces_final", "facenet", "arcface"}
    assert prov["primary_runs"]["arcface"]["code_version"] == "cccc" and prov["primary_runs"]["arcface"]["manifest_sha256"] == "m_ar"
    assert prov["primary_runs"]["facenet"]["code_version"] == "bbbb"
    assert "code_version" in prov["cross_run_checks"]["recorded_not_compared"]
    assert "not valid-pairs-only" in prov["failure_semantics"]
    assert set(prov["deep_run_directories"]) == {"facenet", "arcface"} and set(prov["input_files"]) == set(prov["primary_runs"])


def test_historical_eigenfaces_comparison_stays_isolated_in_the_six_model_run(six, tmp_path):
    archive, deep = six
    out = tmp_path / "o"
    A.run_analysis(archive, out, figures=False, deep_runs=deep)
    hist = list(csv.DictReader(open(out / "tables" / "table11_historical_vs_final_eigenfaces.csv")))
    assert len(hist) == 96 and "model" not in hist[0]                       # Eigenfaces-only table, no model column
    for name in ("table1_clean_baseline", "table10_all_cells_long", "table12_failure_diagnostics_by_condition"):
        assert "eigenfaces_per_image_noise" not in (out / "tables" / f"{name}.csv").read_text()
    prov = json.loads((out / "provenance.json").read_text())
    assert "eigenfaces_per_image_noise" in prov["historical_run"] and "eigenfaces_per_image_noise" not in prov["primary_runs"]
    rows, _ = A.load_primary(archive, deep_runs=deep)
    assert sum(1 for r in rows if r["model"] == "eigenfaces") == 96


def test_six_model_tables_match_the_source_csvs(six, tmp_path):
    archive, deep = six
    out = tmp_path / "o"
    A.run_analysis(archive, out, figures=False, deep_runs=deep)
    raw = {}
    for d in (archive / "classical", archive / "eigenfaces_final", deep["facenet"], deep["arcface"]):
        for r in csv.DictReader(open(d / "verification_results.csv")):
            raw[(r["model"], r["perturbation"], round(float(r["severity"]), 3))] = r
    for r in csv.DictReader(open(out / "tables" / "table12_failure_diagnostics_by_condition.csv")):
        src = raw[(r["model"], r["perturbation"], round(float(r["severity"]), 3))]
        if r["model"] in FOUR:
            assert r["n_failed_embeddings"] == "" and "n_failed_embeddings" not in src
        else:
            assert int(r["n_failed_embeddings"]) == int(src["n_failed_embeddings"])
            assert int(r["n_failed_pairs_balanced"]) == int(src["n_failed_pairs_balanced"])
            assert int(r["n_failed_pairs_full"]) == int(src["n_failed_pairs_full"])
            assert float(r["auc_end_to_end"]) == pytest.approx(float(src["auc"]), abs=1e-9)
    heat = list(csv.DictReader(open(out / "tables" / "table3_auc_harm_heatmap_severity1.csv")))
    assert list(heat[0]) == ["perturbation", *SIX]
    for r in heat:
        for m in SIX:
            exp = float(raw[(m, r["perturbation"], 0.0)]["auc"]) - float(raw[(m, r["perturbation"], 1.0)]["auc"])
            assert float(r[m]) == pytest.approx(exp, abs=1e-9)


def test_output_directory_guards_for_six_model_runs(six, tmp_path):
    archive, deep = six
    with pytest.raises(A.AnalysisError, match="refusing to write inside"):
        A.run_analysis(archive, deep["arcface"] / "analysis", figures=False, deep_runs=deep)
    with pytest.raises(A.AnalysisError, match="refusing to write inside"):
        A.run_analysis(archive, deep["facenet"], figures=False, deep_runs=deep)
    with pytest.raises(A.AnalysisError, match="contains the archived run directory"):
        A.run_analysis(archive, tmp_path / "runs", figures=False, deep_runs=deep)
    out = tmp_path / "existing"
    A.run_analysis(archive, out, figures=False, deep_runs=deep)
    with pytest.raises(A.AnalysisError, match="non-empty directory"):                      # an earlier analysis is never replaced silently
        A.run_analysis(archive, out, figures=False, deep_runs=deep)
    A.run_analysis(archive, out, figures=False, deep_runs=deep, overwrite=True)
    empty = tmp_path / "empty"
    empty.mkdir()
    A.run_analysis(archive, empty, figures=False, deep_runs=deep)                          # an empty directory is fine


def test_six_model_pipeline_is_deterministic(six, tmp_path):
    archive, deep = six
    a, b = tmp_path / "a", tmp_path / "b"
    A.run_analysis(archive, a, figures=False, deep_runs=deep)
    A.run_analysis(archive, b, figures=False, deep_runs=deep)
    assert digest_tree(a) == digest_tree(b)


def test_figure_functions_work_for_four_and_six_models_and_follow_the_model_set(six, tmp_path):
    archive, deep = six
    for models_expected, kw in ((FOUR, {}), (SIX, {"deep_runs": deep})):
        rows, _ = A.load_primary(archive, **kw)
        cells = A.table_cells_long(rows)
        paths = A.make_figures(rows, cells, tmp_path / f"f{len(models_expected)}")
        assert len(paths) == (7 if kw else 6) and all(p.stat().st_size > 2000 for p in paths)
        assert A._models_in(rows) == tuple(models_expected)
    assert {m: A.MODEL_COLORS[m] for m in FOUR} == {"eigenfaces": "#0072B2", "fisherfaces": "#E69F00", "lbph": "#009E73", "hog_svm": "#CC79A7"}
    assert len(set(A.MODEL_COLORS.values())) == 6                                         # six distinct colours


def test_single_deep_model_runs_are_supported(six, tmp_path):
    archive, deep = six
    rows, _ = A.load_primary(archive, deep_runs={"facenet": deep["facenet"]})
    assert len(rows) == 480 and A._models_in(rows) == (*FOUR, "facenet")
    assert A.make_figures(rows, A.table_cells_long(rows), tmp_path / "f")


def test_cli_six_model_run_and_refusals(six, tmp_path, capsys):
    archive, deep = six
    out = tmp_path / "analysis_20261119_six_models"
    argv = ["--results-root", str(archive), "--facenet-run", str(deep["facenet"]), "--arcface-run", str(deep["arcface"]),
            "--out", str(out), "--no-figures"]
    assert A.main(argv) == 0
    assert (out / "tables" / "table12_failure_diagnostics_by_condition.csv").is_file()
    assert A.main(argv) == 2                                                               # existing non-empty output directory
    assert "ANALYSIS REFUSED" in capsys.readouterr().out
    assert A.main(argv + ["--overwrite"]) == 0
    with pytest.raises(SystemExit) as e:
        A.main(["--help"])
    assert e.value.code == 0
    help_text = capsys.readouterr().out
    assert "--facenet-run" in help_text and "--arcface-run" in help_text and "end-to-end" in help_text.lower()
    assert "valid-pairs-only" in help_text


def test_module_documentation_states_the_failure_semantics():
    doc = A.__doc__
    assert "576" in doc and "END-TO-END" in doc and "zero vector" in doc and "NOT valid-pairs-only" in doc
    assert "code_version" in doc and "manifest_sha256" in doc


# ---------------------------------------------------------------- the real committed classical archive, unchanged by the six-model code
@needs_archive
def test_real_archive_default_analysis_is_still_the_four_model_analysis(tmp_path):
    out = tmp_path / "a"
    A.run_analysis(REAL_ROOT, out, figures=False)
    assert len(list(csv.DictReader(open(out / "tables" / "table1_clean_baseline.csv")))) == 4
    diag = list(csv.DictReader(open(out / "tables" / "table12_failure_diagnostics_by_condition.csv")))
    assert len(diag) == 4 * 96 and {r["diagnostics_applicable"] for r in diag} == {"false"}


# =====================================================================================================================
# Final audit: CSV <-> JSON reconciliation of EVERY duplicated field, run metadata, diagnostics assumptions, provenance
# and output safety. (Synthetic fixtures only: they prove the logic, not the content of the real FaceNet/ArcFace archives.)
# =====================================================================================================================
DEEP_ROW_FIELDS = list(synth_rows(["arcface"])[0])


def _mutate_value(field, value):
    """A different, still well-formed value for ``field`` (so only the CSV/JSON disagreement can fail)."""
    if field == "model":
        return "facenet" if value == "arcface" else "arcface"
    if field == "perturbation":
        return "gamma" if value != "gamma" else "contrast"
    if field == "severity":
        return value + 0.2
    if isinstance(value, bool):
        return not value
    if isinstance(value, float):
        if math.isnan(value):
            return 0.1
        return value + 0.01 if value + 0.01 <= 1.0 else value - 0.01
    if isinstance(value, int):
        return value + 1
    return str(value) + "x"


def _deep_with_one_mutated_side(six, field, side, run="arcface"):
    archive, deep = six
    good = synth_rows([run])
    bad = copy.deepcopy(good)
    k = 40
    bad[k][field] = _mutate_value(field, bad[k][field])
    d = deep[run]
    csv_rows, json_rows = (bad, good) if side == "csv" else (good, bad)
    write_run(d.parent, d.name, csv_rows, meta_for(json_rows, code="cccc", manifest="m_ar"))
    return archive, deep


@pytest.mark.parametrize("side", ["json", "csv"])
@pytest.mark.parametrize("field", DEEP_ROW_FIELDS)
def test_every_duplicated_field_is_reconciled_between_csv_and_json(six, field, side):
    archive, deep = _deep_with_one_mutated_side(six, field, side)
    with pytest.raises(A.AnalysisError, match="differs between CSV|present in only one"):
        A.load_primary(archive, deep_runs=deep)


@pytest.mark.parametrize("field", ["n_impostor_full", "n_genuine", "threshold", "accuracy", "embedding_dim", "label"])
def test_classical_runs_are_reconciled_too(archive, field):
    rows = synth_rows(["fisherfaces", "lbph", "hog_svm"])
    bad = copy.deepcopy(rows)
    bad[7][field] = _mutate_value(field, bad[7][field])
    write_run(archive, "classical", rows, meta_for(bad))
    with pytest.raises(A.AnalysisError, match="differs between CSV"):
        A.load_primary(archive)


def test_numeric_tolerance_is_absolute_and_tight(six):
    archive, deep = six
    good = synth_rows(["arcface"])
    near, far = copy.deepcopy(good), copy.deepcopy(good)
    near[40]["auc"] += 5e-10
    far[40]["auc"] += 5e-9
    d = deep["arcface"]
    write_run(d.parent, d.name, good, meta_for(near, code="cccc", manifest="m_ar"))
    A.load_primary(archive, deep_runs=deep)                                         # inside 1e-9: accepted
    write_run(d.parent, d.name, good, meta_for(far, code="cccc", manifest="m_ar"))
    with pytest.raises(A.AnalysisError, match="auc differs between CSV"):
        A.load_primary(archive, deep_runs=deep)


def _arcface_json(six, edit):
    archive, deep = six
    d = deep["arcface"]
    rows = synth_rows(["arcface"])
    write_run(d.parent, d.name, rows, meta_for(rows, code="cccc", manifest="m_ar"))
    js = json.loads((d / "verification_results.json").read_text())
    edit(js)
    (d / "verification_results.json").write_text(json.dumps(js))
    return archive, deep


@pytest.mark.parametrize("edit,match", [
    (lambda js: js["results"][3].pop("n_failed_pairs_full"), "present in only one"),
    (lambda js: js["results"][3].pop("auc"), "present in only one"),
    (lambda js: js["results"][3].update(unexpected_extra=1), "present in only one"),
    (lambda js: js.pop("results"), "no 'results' list"),
    (lambda js: js.update(results={"a": 1}), "no 'results' list"),
    (lambda js: js["results"].__setitem__(3, [1, 2]), "not an object"),
    (lambda js: js["results"][3].update(auc="abc"), "not numeric in the JSON"),
    (lambda js: js["results"][3].update(threshold=True), "not numeric in the JSON"),
    (lambda js: js["results"][3].update(n_failed_embeddings=True), "differs between CSV"),
    (lambda js: js["results"][3].update(n_test_images=1338.5), "differs between CSV"),
    (lambda js: js["results"][3].update(n_test_images="1338"), "differs between CSV"),
    (lambda js: js["results"].reverse(), "differs between CSV"),
])
def test_missing_extra_and_malformed_json_fields_fail_cleanly(six, edit, match):
    archive, deep = _arcface_json(six, edit)
    with pytest.raises(A.AnalysisError, match=match):                               # never a KeyError/ValueError
        A.load_primary(archive, deep_runs=deep)


def test_integer_valued_json_floats_and_null_for_nan_are_accepted(six):
    def edit(js):
        js["results"][3]["n_failed_embeddings"] = float(js["results"][3]["n_failed_embeddings"])      # 0.0 == 0
        js["results"][4]["detector_miss_rate"] = None                                                   # NaN written as null
    archive, deep = _arcface_json(six, edit)
    rows, _ = A.load_primary(archive, deep_runs=deep)
    assert len(rows) == 576


def test_csv_column_missing_or_extra_is_detected(six):
    archive, deep = six
    d = deep["arcface"]
    lines = (d / "verification_results.csv").read_text().splitlines()
    k = lines[0].split(",").index("scores_file")
    (d / "verification_results.csv").write_text("\n".join(",".join(c for i, c in enumerate(l.split(",")) if i != k) for l in lines) + "\n")
    with pytest.raises(A.AnalysisError, match="column 'scores_file' is present in only one"):
        A.load_primary(archive, deep_runs=deep)


def test_nan_metrics_must_agree_when_nan_is_allowed(archive):
    rows = synth_rows(["fisherfaces", "lbph", "hog_svm"])
    nan_rows = copy.deepcopy(rows)
    nan_rows[10]["auc"] = float("nan")
    write_run(archive, "classical", nan_rows, meta_for(nan_rows))
    A.load_primary(archive, allow_nan_metrics=True)                                  # NaN in both files agrees
    write_run(archive, "classical", nan_rows, meta_for(rows))                        # NaN in the CSV only
    with pytest.raises(A.AnalysisError, match="auc differs between CSV"):
        A.load_primary(archive, allow_nan_metrics=True)


# ---------------------------------------------------------------- run-level metadata versus rows
@pytest.mark.parametrize("over,match", [
    ({"pair_params": {"test": {"n_images": 1338, "n_genuine": 765, "n_pool": 7640, "n_balanced": 764}}}, "n_genuine"),
    ({"pair_params": {"test": {"n_images": 1338, "n_genuine": 764, "n_pool": 7640, "n_balanced": 700}}}, "n_impostor_balanced"),
    ({"pair_params": {"test": {"n_images": 1338, "n_genuine": 764, "n_pool": 7000, "n_balanced": 764}}}, "n_impostor_full"),
    ({"pair_params": {"test": {"n_images": 1300, "n_genuine": 764, "n_pool": 7640, "n_balanced": 764}}}, "n_test_images"),
    ({"pair_params": {"test": {"n_images": 1338, "n_genuine": 764, "n_pool": 7640, "n_balanced": 764},
                      "validation": {"n_images": 2}}}, "validation images"),
    ({"n_rows_expected": 100}, "n_rows_expected"),
    ({"stub_backbones": ["arcface"]}, "stub backbones"),
    ({"dataset": {"fingerprint_sha256": "fp", "counts_match_official_lfw": False}}, "official LFW"),
    ({"threshold_protocol": "max-accuracy on the test pairs"}, "clean validation"),
])
def test_run_metadata_must_agree_with_the_rows(six, over, match):
    archive, deep = six
    rows = synth_rows(["arcface"])
    _rewrite_deep(deep, "arcface", rows, code="cccc", manifest="m_ar", **over)
    with pytest.raises(A.AnalysisError, match=match):
        A.load_primary(archive, deep_runs=deep)


def test_metadata_fields_an_older_archive_did_not_record_are_skipped(six):
    archive, deep = six
    rows = synth_rows(["facenet"])
    d = deep["facenet"]
    meta = meta_for(rows, code="bbbb", manifest="m_fn")
    for k in ("pair_params", "threshold_protocol", "n_rows_expected", "stub_backbones"):
        meta.pop(k)
    meta["dataset"] = {"fingerprint_sha256": "fp"}
    write_run(d.parent, d.name, rows, meta)
    assert len(A.load_primary(archive, deep_runs=deep)[0]) == 576


# ---------------------------------------------------------------- failure-diagnostic assumptions
def _arcface_cell(six, mutate):
    archive, deep = six
    rows = synth_rows(["arcface"])
    idx = next(i for i, r in enumerate(rows) if r["perturbation"] == "gaussian_blur" and r["severity"] == 0.4)
    mutate(rows[idx])
    _rewrite_deep(deep, "arcface", rows, code="cccc", manifest="m_ar")
    return archive, deep


def test_balanced_failed_pairs_cannot_exceed_full_pool_failed_pairs(six):
    # protocol: the balanced impostors are a sample of the full impostor pool and the genuine pairs are shared
    archive, deep = _arcface_cell(six, lambda r: r.update(n_failed_embeddings=10, n_failed_pairs_balanced=100, n_failed_pairs_full=50))
    with pytest.raises(A.AnalysisError, match="inconsistent failed-pair counts"):
        A.load_primary(archive, deep_runs=deep)


def test_a_failed_image_need_not_appear_in_any_balanced_pair(six):
    # NOT guaranteed by the protocol (balanced pairs are a sample), so zero balanced failures with failed images is accepted
    archive, deep = _arcface_cell(six, lambda r: r.update(n_failed_embeddings=3, n_failed_pairs_balanced=0, n_failed_pairs_full=9))
    rows, _ = A.load_primary(archive, deep_runs=deep)
    cell = next(r for r in A.table_failure_diagnostics(rows) if r["model"] == "arcface" and r["perturbation"] == "gaussian_blur" and r["severity"] == 0.4)
    assert cell["n_failed_embeddings"] == 3 and cell["n_failed_pairs_balanced"] == 0 and cell["balanced_pair_failure_rate"] == 0.0


def test_validation_failure_count_is_one_number_per_model(six):
    archive, deep = _arcface_cell(six, lambda r: r.update(val_n_failed_embeddings=4))
    with pytest.raises(A.AnalysisError, match="val_n_failed_embeddings is not constant"):
        A.load_primary(archive, deep_runs=deep)


def test_clean_condition_failure_counters_must_be_identical_across_perturbations(six):
    archive, deep = six
    rows = synth_rows(["arcface"])
    idx = next(i for i, r in enumerate(rows) if r["perturbation"] == "gamma" and r["severity"] == 0.0)
    rows[idx].update(n_failed_embeddings=2, n_failed_pairs_balanced=2, n_failed_pairs_full=9)
    _rewrite_deep(deep, "arcface", rows, code="cccc", manifest="m_ar")
    with pytest.raises(A.AnalysisError, match="severity-0.0 'n_failed_embeddings'"):
        A.load_primary(archive, deep_runs=deep)


def test_failed_images_are_kept_in_the_denominators_and_metrics(six):
    archive, deep = six
    rows, _ = A.load_primary(archive, deep_runs=deep)
    worst = next(r for r in rows if r["model"] == "arcface" and r["perturbation"] == "sunglasses" and r["severity"] == 1.0)
    assert worst["n_failed_embeddings"] == worst["n_test_images"] == N_IMG        # every image failed ...
    assert worst["n_genuine"] + worst["n_impostor_balanced"] == N_BAL              # ... yet all 1,528 pairs are still scored
    source = next(r for r in synth_rows(["arcface"]) if r["perturbation"] == "sunglasses" and r["severity"] == 1.0)
    assert worst["auc"] == source["auc"]                                           # the archived end-to-end value, not re-estimated


def test_model_failure_summary_is_recomputed_independently_from_the_source_csv(six, tmp_path):
    archive, deep = six
    out = tmp_path / "o"
    A.run_analysis(archive, out, figures=False, deep_runs=deep)
    summ = {r["model"]: r for r in csv.DictReader(open(out / "tables" / "table13_failure_diagnostics_model_summary.csv"))}
    src = list(csv.DictReader(open(deep["arcface"] / "verification_results.csv")))
    pert = [r for r in src if float(r["severity"]) > 0]
    clean = [r for r in src if float(r["severity"]) == 0]
    s = summ["arcface"]
    assert int(s["perturbed_sum_failed_embeddings"]) == sum(int(r["n_failed_embeddings"]) for r in pert)
    assert int(s["perturbed_sum_image_denominator"]) == sum(int(r["n_test_images"]) for r in pert) == 80 * N_IMG
    assert float(s["perturbed_pooled_image_failure_rate"]) == pytest.approx(
        sum(int(r["n_failed_embeddings"]) for r in pert) / sum(int(r["n_test_images"]) for r in pert))
    assert int(s["perturbed_sum_failed_pairs_full"]) == sum(int(r["n_failed_pairs_full"]) for r in pert)
    assert int(s["perturbed_sum_full_pair_denominator"]) == sum(int(r["n_genuine"]) + int(r["n_impostor_full"]) for r in pert)
    assert int(s["clean_n_conditions"]) == len(clean) == 16 and int(s["all_rows_sum_failed_embeddings"]) == sum(int(r["n_failed_embeddings"]) for r in src)
    for m in FOUR:
        assert summ[m]["diagnostics_applicable"] == "false" and summ[m]["perturbed_sum_failed_embeddings"] == ""


# ---------------------------------------------------------------- provenance
@pytest.mark.parametrize("key,over", [
    ("fold", {"fold": 1}), ("k_folds", {"k_folds": 5}), ("seed", {"seed": 7}),
])
def test_split_identity_fields_must_match_across_runs(six, key, over):
    archive, deep = six
    rows = synth_rows(["facenet"])
    _rewrite_deep(deep, "facenet", rows, code="bbbb", manifest="m_fn", **over)
    with pytest.raises(A.AnalysisError, match=key):
        A.load_primary(archive, deep_runs=deep)


def test_provenance_identifies_each_run_its_directory_and_file_hashes(six, tmp_path):
    archive, deep = six
    out = tmp_path / "o"
    A.run_analysis(archive, out, figures=False, deep_runs=deep)
    prov = json.loads((out / "provenance.json").read_text())
    dirs = prov["run_directories"]
    assert dirs["facenet"] == str(deep["facenet"].resolve()) and dirs["arcface"] == str(deep["arcface"].resolve())
    assert dirs["classical"] == str((archive / "classical").resolve()) and dirs["eigenfaces_final"] == str((archive / "eigenfaces_final").resolve())
    for run, d in dirs.items():
        for fname, rec in prov["input_files"][run].items():
            assert rec["path"] == str(Path(d) / fname)
            assert rec["sha256"] == hashlib.sha256((Path(d) / fname).read_bytes()).hexdigest()
    for run, p in prov["primary_runs"].items():
        assert p["status"] == "COMPLETE" and p["full_lfw"] is True and p["failures"] == 0 and p["n_rows_csv"] == p["n_rows_json"]
        assert (p["fold"], p["k_folds"], p["seed"]) == (0, 10, 42) and p["noise_stream_matches_approved"] is True
        assert p["n_rows_expected"] == p["n_rows_csv"] and p["counts_match_official_lfw"] is True
    assert prov["stale_files_not_rewritten"] == []


# ---------------------------------------------------------------- output safety
def test_a_rejected_run_creates_no_output_and_changes_no_input(six, tmp_path):
    archive, deep = six
    rows = synth_rows(["arcface"])[:-1]
    _rewrite_deep(deep, "arcface", rows, code="cccc", manifest="m_ar")
    before, out = _dirs_digest(archive, deep), tmp_path / "never_created"
    with pytest.raises(A.AnalysisError):
        A.run_analysis(archive, out, figures=True, deep_runs=deep)
    assert not out.exists() and _dirs_digest(archive, deep) == before
    assert A.main(["--results-root", str(archive), "--facenet-run", str(deep["facenet"]), "--arcface-run", str(deep["arcface"]),
                   "--out", str(out)]) == 2
    assert not out.exists()


def test_output_path_that_is_a_file_or_the_results_root_is_refused(six, tmp_path):
    archive, deep = six
    f = tmp_path / "afile"
    f.write_text("x")
    with pytest.raises(A.AnalysisError, match="not a directory"):
        A.run_analysis(archive, f, figures=False, deep_runs=deep)
    with pytest.raises(A.AnalysisError, match="contains the archived run directory"):
        A.run_analysis(archive, archive, figures=False, deep_runs=deep)
    with pytest.raises(A.AnalysisError, match="refusing to write inside"):
        A.run_analysis(archive, archive / "eigenfaces_final" / "x", figures=False, deep_runs=deep)


def test_overwrite_never_deletes_and_lists_files_it_did_not_rewrite(six, tmp_path):
    archive, deep = six
    out = tmp_path / "o"
    A.run_analysis(archive, out, figures=True, deep_runs=deep)
    keep = out / "tables" / "table99_from_an_older_analysis.csv"
    keep.write_text("keep me\n")
    A.run_analysis(archive, out, figures=False, deep_runs=deep, overwrite=True)      # no figures this time
    assert keep.read_text() == "keep me\n" and (out / "figures" / "fig7_failure_rates.png").is_file()   # nothing deleted
    stale = json.loads((out / "provenance.json").read_text())["stale_files_not_rewritten"]
    assert "tables/table99_from_an_older_analysis.csv" in stale and "figures/fig7_failure_rates.png" in stale
    assert "tables/table1_clean_baseline.csv" not in stale


def test_six_model_output_including_figures_is_deterministic(six, tmp_path):
    archive, deep = six
    a, b = tmp_path / "a", tmp_path / "b"
    A.run_analysis(archive, a, figures=True, deep_runs=deep)
    A.run_analysis(archive, b, figures=True, deep_runs=deep)
    da, db = digest_tree(a), digest_tree(b)
    assert da == db and len(da) == 13 + 7 + 1


def test_output_inventory_columns_and_row_counts(six, tmp_path):
    archive, deep = six
    out = tmp_path / "o"
    A.run_analysis(archive, out, figures=False, deep_runs=deep)
    expected = {"table1_clean_baseline": 6, "table2_clean_intervals_indicative": 6, "table3_auc_harm_heatmap_severity1": 16,
                "table4_auc_harm_heatmap_mean_severity": 16, "table5_auc_by_severity": 36, "table6_degradation_rankings_auc": 96,
                "table7_degradation_summary": 18, "table8_perturbation_across_models_auc": 16, "table9_auc_cells_indicative_scale": 480,
                "table10_all_cells_long": 2880, "table11_historical_vs_final_eigenfaces": 96,
                "table12_failure_diagnostics_by_condition": 576, "table13_failure_diagnostics_model_summary": 6}
    assert {p.stem for p in (out / "tables").glob("*.csv")} == set(expected)
    for name, n in expected.items():
        rows = list(csv.DictReader(open(out / "tables" / f"{name}.csv")))
        assert len(rows) == n, name
    t12 = list(csv.DictReader(open(out / "tables" / "table12_failure_diagnostics_by_condition.csv")))
    assert list(t12[0])[:6] == ["model", "label", "perturbation", "severity", "is_clean_condition", "diagnostics_applicable"]
    assert [r["model"] for r in t12][::96] == list(SIX)                                # canonical six-model order
    t1 = [r["model"] for r in csv.DictReader(open(out / "tables" / "table1_clean_baseline.csv"))]
    assert t1 == list(SIX)


def test_cli_default_is_the_four_model_analysis_and_single_deep_run_works(six, tmp_path):
    archive, deep = six
    out4 = tmp_path / "four"
    assert A.main(["--results-root", str(archive), "--out", str(out4)]) == 0
    assert len(list((out4 / "tables").glob("*.csv"))) == 13 and len(list((out4 / "figures").glob("*.png"))) == 6
    assert not (out4 / "figures" / "fig7_failure_rates.png").exists()
    assert json.loads((out4 / "provenance.json").read_text())["n_primary_rows"] == 384
    out5 = tmp_path / "five"
    assert A.main(["--results-root", str(archive), "--facenet-run", str(deep["facenet"]), "--out", str(out5), "--no-figures"]) == 0
    assert json.loads((out5 / "provenance.json").read_text())["models"] == [*FOUR, "facenet"]


def test_figures_render_for_all_zero_and_all_failure_cases(six, tmp_path):
    archive, deep = six
    rows, _ = A.load_primary(archive, deep_runs={"facenet": deep["facenet"]})        # FaceNet only: every failure rate is 0
    assert [p.name for p in A.make_figures(rows, A.table_cells_long(rows), tmp_path / "zero")][-1] == "fig7_failure_rates.png"
    all_fail = synth_rows(["arcface"])
    for r in all_fail:
        if r["severity"] > 0:
            r.update(n_failed_embeddings=N_IMG, n_failed_pairs_balanced=N_BAL, n_failed_pairs_full=N_FULL)
    _rewrite_deep(deep, "arcface", all_fail, code="cccc", manifest="m_ar")
    rows, _ = A.load_primary(archive, deep_runs={"arcface": deep["arcface"]})        # ArcFace only: every perturbed image failed
    paths = A.make_figures(rows, A.table_cells_long(rows), tmp_path / "all")
    assert len(paths) == 7 and all(p.stat().st_size > 2000 for p in paths)


# =====================================================================================================================
# One COMBINED deep run (FaceNet + ArcFace in a single verification run, e.g. benchmarks/lfw_fold0_2026-11-20/deep_core)
# =====================================================================================================================
COMBINED = A.COMBINED_DEEP_RUN
DEEP_CORE = REPO / "benchmarks" / "lfw_fold0_2026-11-20" / "deep_core"
needs_deep_core = pytest.mark.skipif(not (DEEP_CORE / "verification_results.csv").is_file() or not (REAL_ROOT / "classical").is_dir(),
                                     reason="the archived combined deep run / classical archive is not present")


def _write_combined(deep, rows, **meta_over):
    d = deep[COMBINED]
    write_run(d.parent, d.name, rows, meta_for(rows, code="dddd", manifest="m_deep", **meta_over))


@pytest.fixture
def combined(archive, tmp_path):
    """The four-model archive plus ONE run directory that holds both deep models (192 rows)."""
    d = tmp_path / "runs" / "deep_core_run" / "verification" / "5ee3"
    deep = {COMBINED: d}
    _write_combined(deep, synth_rows(["facenet", "arcface"]))
    return archive, deep


def _cond_key(r):
    return (r["model"], r["perturbation"], r["severity"])


def _pick(rows, model, perturbation="gaussian_blur", severity=0.4):
    return next(r for r in rows if r["model"] == model and r["perturbation"] == perturbation and r["severity"] == severity)


def test_combined_deep_run_gives_the_same_576_rows_as_two_separate_runs(six, combined):
    archive, separate = six
    _, comb = combined
    rows_sep, prov_sep = A.load_primary(archive, deep_runs=separate)
    rows_comb, prov_comb = A.load_primary(archive, deep_runs=comb)
    assert len(rows_comb) == 576 and {r["model"] for r in rows_comb} == set(SIX)
    assert (json.dumps(sorted(rows_comb, key=_cond_key), sort_keys=True)
            == json.dumps(sorted(rows_sep, key=_cond_key), sort_keys=True))
    assert set(prov_comb["runs"]) == {"classical", "eigenfaces_final", COMBINED}
    p = prov_comb["runs"][COMBINED]
    assert p["n_rows_csv"] == p["n_rows_json"] == 192 and p["code_version"] == "dddd" and p["manifest_sha256"] == "m_deep"
    assert set(prov_comb["input_files"][COMBINED]) == {"verification_results.csv", "verification_results.json"}
    assert prov_comb["run_directories"][COMBINED] == str(comb[COMBINED].resolve())


def test_combined_and_separate_runs_write_identical_tables_and_all_figures(six, combined, tmp_path):
    archive, separate = six
    _, comb = combined
    a, b = tmp_path / "out_sep", tmp_path / "out_comb"
    A.run_analysis(archive, a, figures=True, deep_runs=separate)
    prov_b = A.run_analysis(archive, b, figures=True, deep_runs=comb)
    names = sorted(p.name for p in (a / "tables").iterdir())
    assert names == sorted(p.name for p in (b / "tables").iterdir()) and len(names) == 13
    for n in names:
        assert (a / "tables" / n).read_bytes() == (b / "tables" / n).read_bytes(), n
    assert len(list((b / "figures").glob("fig*.png"))) == 7 and "provenance" in prov_b
    prov = json.loads((b / "provenance.json").read_text())
    assert prov["n_primary_rows"] == 576 and prov["models"] == list(SIX)
    assert set(prov["deep_run_directories"]) == {COMBINED} and set(prov["primary_runs"]) == {"classical", "eigenfaces_final", COMBINED}


def test_combined_run_does_not_touch_any_input(combined, tmp_path):
    archive, deep = combined
    before = (digest_tree(archive), digest_tree(deep[COMBINED]))
    A.run_analysis(archive, tmp_path / "out", figures=False, deep_runs=deep)
    assert (digest_tree(archive), digest_tree(deep[COMBINED])) == before


def test_combined_run_requires_exactly_the_two_deep_models_and_each_models_96_rows(combined):
    archive, deep = combined
    rows = synth_rows(["facenet", "arcface"])
    A.validate_run(rows, "deep", A.DEEP_MODELS)                                    # exactly 192 rows: accepted
    with pytest.raises(A.AnalysisError, match="condition grid differs"):             # 191 rows
        A.validate_run(rows[:-1], "deep", A.DEEP_MODELS)
    with pytest.raises(A.AnalysisError, match="duplicate"):                          # 193 rows
        A.validate_run(rows + [dict(rows[0])], "deep", A.DEEP_MODELS)
    with pytest.raises(A.AnalysisError, match="model coverage differs"):             # one deep model missing
        A.validate_run(synth_rows(["facenet"]), "deep", A.DEEP_MODELS)


def _only(model):
    return lambda rows: [r for r in rows if r["model"] == model]


def _drop_last(rows):
    return rows[:-1]


def _duplicate(rows):
    rows = list(rows)
    rows[5] = dict(rows[0])
    return rows


def _extra_classical_model(rows):
    extra = synth_rows(["eigenfaces"])
    for r in extra:                                         # same columns as the deep rows so that the CSV can be written
        r.update(normalization_mode="x", alignment="y", crop_fraction=0.8, val_balanced_accuracy=0.9, val_n_failed_embeddings=0,
                 n_test_images=N_IMG, n_failed_embeddings=0, n_failed_pairs_balanced=0, n_failed_pairs_full=0, scores_file="x")
    return rows + extra


def _threshold_varies(rows):
    _pick(rows, "arcface")["threshold"] += 0.05
    return rows


def _clean_baseline_differs(rows):
    _pick(rows, "facenet", "gaussian_blur", 0.0)["accuracy"] -= 0.02
    return rows


def _failure_counters_inconsistent(rows):
    _pick(rows, "arcface").update(n_failed_pairs_balanced=5, n_failed_pairs_full=5, n_failed_embeddings=0)
    return rows


def _stub_backbone(rows):
    _pick(rows, "facenet")["backbone"] = "STUB"
    return rows


@pytest.mark.parametrize("mutate,match", [
    (_only("facenet"), r"expected models \['arcface', 'facenet'\], found \['facenet'\]"),
    (_only("arcface"), r"expected models \['arcface', 'facenet'\], found \['arcface'\]"),
    (_extra_classical_model, r"expected models \['arcface', 'facenet'\], found \['arcface', 'eigenfaces', 'facenet'\]"),
    (_drop_last, "condition grid differs"),
    (_duplicate, "duplicate"),
    (_threshold_varies, "frozen threshold is not constant"),
    (_clean_baseline_differs, "severity-0.0 'accuracy' differs"),
    (_failure_counters_inconsistent, "without any failed embedding"),
    (_stub_backbone, "STUB"),
])
def test_invalid_or_incomplete_combined_runs_are_rejected(combined, mutate, match):
    archive, deep = combined
    _write_combined(deep, mutate(synth_rows(["facenet", "arcface"])))
    with pytest.raises(A.AnalysisError, match=match):
        A.load_primary(archive, deep_runs=deep)


@pytest.mark.parametrize("over,match", [
    ({"status": "INCOMPLETE"}, "not a complete"),
    ({"failures": ["boom"]}, "not a complete"),
    ({"full_lfw": False}, "not a complete"),
    ({"content_sha256": {"split": "other", "pairs": {"validation": "pv", "test": "pt"}}}, "disagree on 'split_content_sha256'"),
    ({"dataset": {"fingerprint_sha256": "other", "counts_match_official_lfw": True}}, "disagree on 'dataset_fingerprint_sha256'"),
    ({"seed": 7}, "disagree on 'seed'"),
    ({"noise_seeding": {"protocol_version": "per_image_v2", "master_seed": 42,
                        "stream_fingerprint": {"matches_approved_reference": False}}}, "noise stream"),
])
def test_combined_run_metadata_is_validated_like_any_other_run(combined, over, match):
    archive, deep = combined
    _write_combined(deep, synth_rows(["facenet", "arcface"]), **over)
    with pytest.raises(A.AnalysisError, match=match):
        A.load_primary(archive, deep_runs=deep)


def test_combined_run_csv_and_json_must_agree(combined):
    archive, deep = combined
    rows = synth_rows(["facenet", "arcface"])
    _write_combined(deep, rows)
    bad = copy.deepcopy(rows)
    _pick(bad, "facenet", "gamma", 0.6)["auc"] += 0.01
    (deep[COMBINED] / "verification_results.json").write_text(json.dumps(meta_for(bad, code="dddd", manifest="m_deep")))
    with pytest.raises(A.AnalysisError, match="auc differs between CSV"):
        A.load_primary(archive, deep_runs=deep)
    short = copy.deepcopy(rows)[:-1]
    (deep[COMBINED] / "verification_results.json").write_text(json.dumps(meta_for(short, code="dddd", manifest="m_deep")))
    with pytest.raises(A.AnalysisError, match="CSV has 192 rows but JSON has 191"):
        A.load_primary(archive, deep_runs=deep)


def test_combined_run_without_failure_columns_is_rejected(combined):
    archive, deep = combined
    d = deep[COMBINED]
    text = (d / "verification_results.csv").read_text().splitlines()
    k = text[0].split(",").index("n_failed_embeddings")
    (d / "verification_results.csv").write_text(
        "\n".join(",".join(c for i, c in enumerate(line.split(",")) if i != k) for line in text) + "\n")
    js = json.loads((d / "verification_results.json").read_text())
    for r in js["results"]:
        r.pop("n_failed_embeddings")
    (d / "verification_results.json").write_text(json.dumps(js))
    with pytest.raises(A.AnalysisError, match="missing failure diagnostic column 'n_failed_embeddings'"):
        A.load_primary(archive, deep_runs=deep)


@pytest.mark.parametrize("name", ["facenet", "arcface"])
def test_combined_run_cannot_be_mixed_with_a_per_model_run(six, combined, name):
    archive, separate = six
    _, comb = combined
    with pytest.raises(A.AnalysisError, match="cannot be used together"):
        A.load_primary(archive, deep_runs={COMBINED: comb[COMBINED], name: separate[name]})


def test_a_combined_run_given_as_two_per_model_runs_is_still_refused_with_a_hint(combined):
    """The old behaviour is kept (a per-model run must hold exactly its model); the message now says what to do."""
    archive, deep = combined
    d = deep[COMBINED]
    with pytest.raises(A.AnalysisError, match=r"facenet: expected models \['facenet'\], found \['arcface', 'facenet'\].*--deep-run"):
        A.load_primary(archive, deep_runs={"facenet": d, "arcface": d})
    with pytest.raises(A.AnalysisError, match=r"arcface: expected models \['arcface'\], found \['arcface', 'facenet'\].*--deep-run"):
        A.load_primary(archive, deep_runs={"arcface": d})


def test_a_per_model_run_holding_the_wrong_model_gets_no_combined_hint(six):
    archive, deep = six
    _rewrite_deep(deep, "arcface", synth_rows(["facenet"]))
    with pytest.raises(A.AnalysisError) as e:
        A.load_primary(archive, deep_runs=deep)
    assert "expected models ['arcface'], found ['facenet']" in str(e.value) and "--deep-run" not in str(e.value)


def test_combined_run_output_directory_guards(combined, tmp_path):
    archive, deep = combined
    with pytest.raises(A.AnalysisError, match="inside the archived run directory"):
        A.run_analysis(archive, deep[COMBINED] / "analysis", figures=False, deep_runs=deep)
    with pytest.raises(A.AnalysisError, match="contains the archived run directory"):
        A.run_analysis(archive, deep[COMBINED].parent, figures=False, deep_runs=deep)
    out = tmp_path / "out"
    A.run_analysis(archive, out, figures=False, deep_runs=deep)
    with pytest.raises(A.AnalysisError, match="non-empty directory"):
        A.run_analysis(archive, out, figures=False, deep_runs=deep)


def test_cli_deep_run_option(combined, six, tmp_path, capsys):
    archive, deep = combined
    _, separate = six
    out = tmp_path / "analysis_combined"
    argv = ["--results-root", str(archive), "--deep-run", str(deep[COMBINED]), "--out", str(out), "--no-figures"]
    assert A.main(argv) == 0
    assert len(list((out / "tables").glob("table*.csv"))) == 13
    assert A.main(argv) == 2 and "ANALYSIS REFUSED" in capsys.readouterr().out         # existing non-empty output directory
    assert A.main(["--results-root", str(archive), "--deep-run", str(deep[COMBINED]), "--facenet-run", str(separate["facenet"]),
                   "--out", str(tmp_path / "mixed"), "--no-figures"]) == 2
    assert "cannot be used together" in capsys.readouterr().out and not (tmp_path / "mixed").exists()
    with pytest.raises(SystemExit):
        A.main(["--help"])
    assert "--deep-run" in capsys.readouterr().out


# ----------------------------------------------------------------------------- the real archived combined run
@needs_deep_core
def test_real_combined_archive_gives_six_models_and_is_not_modified(tmp_path):
    before = (digest_tree(REAL_ROOT), digest_tree(DEEP_CORE))
    rows, prov = A.load_primary(REAL_ROOT, deep_runs={COMBINED: DEEP_CORE})
    assert len(rows) == 576 and {r["model"] for r in rows} == set(SIX)
    p = prov["runs"][COMBINED]
    assert p["status"] == "COMPLETE" and p["failures"] == 0 and p["n_rows_csv"] == p["n_rows_json"] == 192
    assert p["stub_backbones"] == [] and p["noise_stream_matches_approved"] is True
    A.run_analysis(REAL_ROOT, tmp_path / "out", figures=False, deep_runs={COMBINED: DEEP_CORE})
    assert (digest_tree(REAL_ROOT), digest_tree(DEEP_CORE)) == before


@needs_deep_core
def test_real_combined_archive_tables_match_the_source_csv(tmp_path):
    """Independent recomputation: the clean baseline of FaceNet and ArcFace and every ArcFace failure fraction are read
    straight from the archived CSV with the ``csv`` module."""
    out = tmp_path / "out"
    A.run_analysis(REAL_ROOT, out, figures=False, deep_runs={COMBINED: DEEP_CORE})
    src = {(r["model"], r["perturbation"], round(float(r["severity"]), 3)): r
           for r in csv.DictReader(open(DEEP_CORE / "verification_results.csv"))}
    assert len(src) == 192
    t1 = {r["model"]: r for r in csv.DictReader(open(out / "tables" / "table1_clean_baseline.csv"))}
    assert set(t1) == set(SIX)
    for m in A.DEEP_MODELS:
        clean = src[(m, "gamma", 0.0)]
        for col in ("accuracy", "tar", "far", "auc", "eer", "threshold"):
            assert float(t1[m][col]) == pytest.approx(float(clean[col]), abs=1e-9)
    t12 = list(csv.DictReader(open(out / "tables" / "table12_failure_diagnostics_by_condition.csv")))
    arc = {(r["perturbation"], round(float(r["severity"]), 3)): r for r in t12 if r["model"] == "arcface"}
    assert len(arc) == 96
    for (pert, sev), r in arc.items():
        s = src[("arcface", pert, sev)]
        assert int(float(r["n_failed_embeddings"])) == int(float(s["n_failed_embeddings"]))
        assert int(float(r["n_test_images"])) == int(float(s["n_test_images"]))

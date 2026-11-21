"""Tests for faceeval.analysis.valid_pairs (valid-pairs-only re-analysis of a saved deep-model Fold 0 run).

SCOPE: every test runs on a SYNTHETIC run directory built in ``tmp_path``. The fixture is written with the production
helpers of ``faceeval.orchestration.verification`` (``_scores``, ``_valid_rows``, ``_sanitize``, ``_accuracy_at``,
``_flat_metrics``, ``_pairs_with_failure``), so the archived columns are produced the same way a real run produces
them. The embeddings are random vectors, NOT real LFW images and NOT FaceNet/ArcFace output: these tests establish the
code's logic only and say nothing about real ArcFace or FaceNet performance. No archive is modified (a test hashes the
run directory before and after).
"""
import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pytest

from faceeval.analysis import valid_pairs as V
from faceeval.orchestration import verification as ver

N_ID, PER_ID = 40, 3
N_IMG = N_ID * PER_ID
THR = 0.30
SEVERITIES = (0.0, 0.5, 1.0)
PERTURBATIONS = ("gaussian_blur", "rotation", "sunglasses")


def _pairs(rng):
    ident = np.repeat(np.arange(N_ID), PER_ID)
    gen = [(i * PER_ID + a, i * PER_ID + b) for i in range(N_ID) for a, b in ((0, 1), (0, 2), (1, 2))]
    imp = set()
    while len(imp) < 600:
        a, b = sorted(rng.choice(N_IMG, 2, replace=False).tolist())
        if ident[a] != ident[b]:
            imp.add((a, b))
    imp = sorted(imp)
    return gen, imp[:len(gen)], imp


def _failed_images(sev, perturbation):
    """Deterministic nested failure sets: none at severity 0, more images at higher severity."""
    n = {0.0: 0, 0.5: 12, 1.0: 40}[sev] if perturbation == "sunglasses" else {0.0: 0, 0.5: 0, 1.0: 6}[sev]
    return np.arange(n) * 3          # image indices 0, 3, 6, ...


def build_run(root: Path, *, failures=True, model="arcface"):
    rng = np.random.default_rng(7)
    gen, bal_imp, pool = _pairs(rng)
    bal = gen + bal_imp
    full = gen + pool
    bal_lab = np.r_[np.ones(len(gen), bool), np.zeros(len(bal_imp), bool)]
    full_lab = np.r_[np.ones(len(gen), bool), np.zeros(len(pool), bool)]
    centroid = rng.normal(size=(N_ID, 16))
    base = np.repeat(centroid, PER_ID, axis=0)
    run = root / "run"
    (run / "scores").mkdir(parents=True)
    np.savez_compressed(run / "scores" / "test_pairs.npz", image_ids=np.array([f"img{i}" for i in range(N_IMG)]),
                        bal_pairs=np.asarray(bal, dtype=np.int32), bal_labels=bal_lab,
                        full_pairs=np.asarray(full, dtype=np.int32), full_labels=full_lab)
    rows = []
    for p in PERTURBATIONS:
        for s in SEVERITIES:
            noise = rng.normal(size=base.shape) * (0.6 + 0.8 * s)
            emb = base + noise
            emb /= np.linalg.norm(emb, axis=1, keepdims=True)
            if failures:
                emb[_failed_images(s, p)] = 0.0              # a detector miss: all-zero embedding
            valid = ver._valid_rows(emb)
            emb = ver._sanitize(emb, valid)
            s_bal, s_full = ver._scores(emb, bal), ver._scores(emb, full)
            fname = f"{model}__{p}__sev{s:.2f}.npz"
            np.savez_compressed(run / "scores" / fname, valid=valid, s_bal=s_bal, s_full=s_full,
                                threshold=np.float64(THR))
            row = {"model": model, "paradigm": "deep_learning", "backbone": model, "perturbation": p, "severity": s,
                   "label": f"{p}_s{s:.2f}", "threshold": THR, "n_test_images": int(valid.size),
                   "n_failed_embeddings": int((~valid).sum()),
                   "n_failed_pairs_balanced": ver._pairs_with_failure(valid, bal),
                   "n_failed_pairs_full": ver._pairs_with_failure(valid, full),
                   "n_genuine": int(bal_lab.sum()), "n_impostor_balanced": int((~bal_lab).sum()),
                   "n_impostor_full": int((~full_lab).sum()), "scores_file": fname}
            row.update(ver._accuracy_at(s_bal, bal_lab, THR))
            row.update(ver._flat_metrics(s_bal, bal_lab, ""))
            row.update(ver._flat_metrics(s_full, full_lab, "_full"))
            row["far_full"] = ver._accuracy_at(s_full, full_lab, THR)["far"]
            rows.append(row)
    with open(run / "verification_results.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    (run / "verification_results.json").write_text(json.dumps(
        {"protocol": "lfw_fold0_verification", "status": "COMPLETE", "n_rows": len(rows), "fold": 0, "seed": 42,
         "results": rows}, default=str))
    return run, dict(bal=bal, full=full, bal_lab=bal_lab, full_lab=full_lab)


def tree_hash(d: Path):
    return {str(p.relative_to(d)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(d.rglob("*")) if p.is_file()}


def by_label(rows):
    return {r["label"]: r for r in rows}


# ---------------------------------------------------------------- correctness
def test_reproduces_archive_and_matches_manual_valid_only_computation(tmp_path):
    run, d = build_run(tmp_path)
    rows, summary, prov = V.analyse_run(run)
    assert len(rows) == 9 and prov["archive_mismatches"] == [] and all(r["archive_reproduced"] for r in rows)
    r = by_label(rows)["sunglasses_s1.00"]
    z = np.load(run / "scores" / "arcface__sunglasses__sev1.00.npz")
    valid = z["valid"]
    keep = np.array([valid[a] and valid[b] for a, b in d["bal"]])
    sc, lab = z["s_bal"][keep], d["bal_lab"][keep]
    assert r["n_valid_pairs_balanced"] == int(keep.sum()) < len(d["bal"])
    assert r["n_valid_genuine"] == int(lab.sum()) and r["n_valid_impostor"] == int((~lab).sum())
    pred = sc >= THR
    assert r["valid_tar"] == pytest.approx(pred[lab].mean())
    assert r["valid_far"] == pytest.approx(pred[~lab].mean())
    assert r["valid_accuracy"] == pytest.approx((pred == lab).mean())
    m = V.compute_verification_metrics(sc.tolist(), lab.tolist())
    assert r["valid_auc"] == pytest.approx(m["auc"]) and r["valid_eer"] == pytest.approx(m["eer"])
    assert r["threshold"] == THR                                  # frozen threshold, never re-tuned


def test_failed_pairs_are_not_removed_from_the_archive_columns(tmp_path):
    run, d = build_run(tmp_path)
    rows, _, _ = V.analyse_run(run)
    r = by_label(rows)["sunglasses_s1.00"]
    csv_row = {x["label"]: x for x in csv.DictReader(open(run / "verification_results.csv"))}["sunglasses_s1.00"]
    # e2e_* is the archived end-to-end value over ALL 1,528-style pairs; failed pairs stay in it
    assert r["n_pairs_balanced"] == len(d["bal"])
    assert r["e2e_auc"] == pytest.approx(float(csv_row["auc"])) and r["e2e_accuracy"] == pytest.approx(float(csv_row["accuracy"]))
    assert r["n_valid_pairs_balanced"] == len(d["bal"]) - int(csv_row["n_failed_pairs_balanced"])


def test_without_failures_valid_only_equals_end_to_end(tmp_path):
    run, _ = build_run(tmp_path, failures=False, model="facenet")
    rows, summary, _ = V.analyse_run(run)
    for r in rows:
        assert r["valid_pair_rate_balanced"] == 1.0 and r["n_failed_embeddings"] == 0
        for k in ("accuracy", "tar", "far", "auc", "eer"):
            assert r[f"valid_{k}"] == pytest.approx(r[f"e2e_{k}"]) == pytest.approx(r[f"common_{k}"])
    assert summary[0]["n_conditions_with_failures"] == 0


def test_common_pair_set_is_identical_across_conditions_and_a_subset_of_each_valid_set(tmp_path):
    run, d = build_run(tmp_path)
    rows, summary, _ = V.analyse_run(run)
    assert len({r["n_common_valid_pairs"] for r in rows}) == 1
    n_common = rows[0]["n_common_valid_pairs"]
    assert 0 < n_common <= min(r["n_valid_pairs_balanced"] for r in rows)
    clean = by_label(rows)["sunglasses_s0.00"]
    assert clean["n_valid_pairs_balanced"] == len(d["bal"]) > n_common     # a failure elsewhere shrinks the common set
    assert summary[0]["n_common_valid_pairs"] == n_common
    assert summary[0]["failed_embeddings_summed_over_conditions_NOT_UNIQUE_IMAGES"] == sum(r["n_failed_embeddings"] for r in rows)


def test_too_few_valid_pairs_gives_nan_with_status_not_a_number(tmp_path):
    run, _ = build_run(tmp_path)
    # make every image invalid in one condition
    f = run / "scores" / "arcface__sunglasses__sev1.00.npz"
    z = dict(np.load(f))
    z["valid"] = np.zeros_like(z["valid"])
    np.savez_compressed(f, **z)
    with pytest.raises(V.ValidPairsError, match="n_failed_embeddings"):    # archived counts no longer agree: refused
        V.analyse_run(run)
    # consistent version: rebuild the whole condition archived as all-failed
    run2, d = build_run(tmp_path / "b")
    f = run2 / "scores" / "arcface__sunglasses__sev1.00.npz"
    z = dict(np.load(f))
    z["valid"] = np.zeros(N_IMG, bool)
    z["s_bal"] = np.zeros(len(d["bal"]))
    z["s_full"] = np.zeros(len(d["full"]))
    np.savez_compressed(f, **z)
    rows = list(csv.DictReader(open(run2 / "verification_results.csv")))
    for r in rows:
        if r["label"] == "sunglasses_s1.00":
            acc = ver._accuracy_at(z["s_bal"], d["bal_lab"], THR)
            r.update({k: repr(v) for k, v in acc.items()})
            r.update({k: repr(v) for k, v in ver._flat_metrics(z["s_bal"], d["bal_lab"], "").items() if k in ("auc", "eer")})
            r.update({k: repr(v) for k, v in ver._flat_metrics(z["s_full"], d["full_lab"], "_full").items() if k in ("auc_full", "eer_full")})
            r.update({"n_failed_embeddings": str(N_IMG), "n_failed_pairs_balanced": str(len(d["bal"])),
                      "n_failed_pairs_full": str(len(d["full"]))})
    with open(run2 / "verification_results.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    out, _, _ = V.analyse_run(run2)
    r = by_label(out)["sunglasses_s1.00"]
    assert r["valid_status"].startswith("too_few_valid_pairs") and math.isnan(r["valid_auc"]) and r["n_valid_pairs_balanced"] == 0
    assert not math.isnan(r["e2e_auc"])                                      # the archived end-to-end value is kept


# ---------------------------------------------------------------- refusals
def test_refuses_tampered_scores_and_inconsistent_inputs(tmp_path):
    run, _ = build_run(tmp_path)
    f = run / "scores" / "arcface__rotation__sev0.50.npz"
    z = dict(np.load(f))
    z["s_bal"] = z["s_bal"] + 0.01
    np.savez_compressed(f, **z)
    with pytest.raises(V.ValidPairsError, match="not reproducible"):
        V.analyse_run(run)
    rows, _, prov = V.analyse_run(run, allow_archive_mismatch=True)           # explicit inspection mode
    assert prov["archive_mismatches"] and not by_label(rows)["rotation_s0.50"]["archive_reproduced"]


def test_refuses_threshold_mismatch_non_complete_status_and_classical_run(tmp_path):
    run, _ = build_run(tmp_path / "a")
    f = run / "scores" / "arcface__rotation__sev0.50.npz"
    z = dict(np.load(f)); z["threshold"] = np.float64(0.31)
    np.savez_compressed(f, **z)
    with pytest.raises(V.ValidPairsError, match="threshold"):
        V.analyse_run(run)
    run, _ = build_run(tmp_path / "b")
    j = json.loads((run / "verification_results.json").read_text()); j["status"] = "INCOMPLETE"
    (run / "verification_results.json").write_text(json.dumps(j))
    with pytest.raises(V.ValidPairsError, match="not COMPLETE"):
        V.analyse_run(run)
    run, _ = build_run(tmp_path / "c")
    rows = list(csv.DictReader(open(run / "verification_results.csv")))
    for r in rows:
        r["scores_file"] = ""
    with open(run / "verification_results.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    with pytest.raises(V.ValidPairsError, match="not a deep-model run"):
        V.analyse_run(run)


def test_refuses_missing_score_file_and_pair_index_out_of_range(tmp_path):
    run, _ = build_run(tmp_path / "a")
    (run / "scores" / "arcface__rotation__sev0.50.npz").unlink()
    with pytest.raises(V.ValidPairsError, match="missing score file"):
        V.analyse_run(run)
    run, d = build_run(tmp_path / "b")
    tp = dict(np.load(run / "scores" / "test_pairs.npz"))
    tp["bal_pairs"] = tp["bal_pairs"].copy(); tp["bal_pairs"][0, 0] = N_IMG + 5
    np.savez_compressed(run / "scores" / "test_pairs.npz", **tp)
    with pytest.raises(V.ValidPairsError, match="indexes outside"):
        V.analyse_run(run)


# ---------------------------------------------------------------- output safety and CLI
def test_run_directory_is_never_modified_and_outputs_are_written_only_to_out(tmp_path):
    run, _ = build_run(tmp_path)
    before = tree_hash(run)
    out = tmp_path / "out"
    written = V.run_valid_pairs_analysis(run, out)
    assert tree_hash(run) == before
    assert sorted(p.name for p in out.iterdir()) == sorted([V.BY_CONDITION, V.MODEL_SUMMARY, V.PROVENANCE])
    prov = json.loads(written["provenance"].read_text())
    assert prov["semantics"] == V.SEMANTICS and prov["model"] == "arcface" and prov["is_full_grid"] is False
    assert set(prov["score_file_sha256"]) == {r["scores_file"] for r in csv.DictReader(open(run / "verification_results.csv"))}
    assert len(list(csv.DictReader(open(written["by_condition"])))) == 9


def test_output_directory_safety(tmp_path):
    run, _ = build_run(tmp_path)
    out = tmp_path / "out"; out.mkdir(); (out / "keep.txt").write_text("x")
    with pytest.raises(V.ValidPairsError, match="non-empty"):
        V.run_valid_pairs_analysis(run, out)
    V.run_valid_pairs_analysis(run, out, overwrite=True)
    assert (out / "keep.txt").read_text() == "x"                              # nothing is deleted
    with pytest.raises(V.ValidPairsError, match="inside the archived run"):
        V.run_valid_pairs_analysis(run, run / "analysis")
    with pytest.raises(V.ValidPairsError, match="contains the archived run"):
        V.run_valid_pairs_analysis(run, tmp_path)


def test_a_refused_run_writes_nothing(tmp_path):
    run, _ = build_run(tmp_path)
    (run / "scores" / "arcface__rotation__sev0.50.npz").unlink()
    out = tmp_path / "out"
    with pytest.raises(V.ValidPairsError):
        V.run_valid_pairs_analysis(run, out)
    assert not out.exists()


def test_cli_exit_codes_and_message(tmp_path, capsys):
    run, _ = build_run(tmp_path)
    assert V.main(["--run", str(run), "--out", str(tmp_path / "o")]) == 0
    assert "by_condition" in capsys.readouterr().out
    assert V.main(["--run", str(run), "--out", str(tmp_path / "o")]) == 2
    assert "VALID-PAIRS ANALYSIS REFUSED" in capsys.readouterr().out
    assert V.main(["--run", str(tmp_path / "nope"), "--out", str(tmp_path / "o2")]) == 2


def test_parser_options_are_the_documented_ones():
    opts = {o for a in V.build_parser()._actions for o in a.option_strings}
    assert {"--run", "--out", "--overwrite", "--allow-archive-mismatch"} <= opts


def test_semantics_text_states_the_limits():
    s = V.SEMANTICS.lower()
    assert "end-to-end" in s and "zero vectors" in s and "conditional" in s and "frozen" in s and "no confidence" in s


# ---------------------------------------------------------------------------------------------------------------------
# A COMBINED run (two models in one verification run, as in benchmarks/lfw_fold0_2026-11-20/deep_core) is analysed one
# model at a time with ``model=`` / ``--model``. Every other check is unchanged.
# ---------------------------------------------------------------------------------------------------------------------
def _combined(tmp_path):
    """A combined run built from two single-model runs (FaceNet without failures, ArcFace with detector misses)."""
    import shutil
    run_fn, _ = build_run(tmp_path / "fn", failures=False, model="facenet")
    run_ar, _ = build_run(tmp_path / "ar", failures=True, model="arcface")
    dest = tmp_path / "combined"
    shutil.copytree(run_fn / "scores", dest / "scores")
    for f in (run_ar / "scores").iterdir():
        if f.name == "test_pairs.npz":
            assert f.read_bytes() == (dest / "scores" / f.name).read_bytes()          # one shared test pair set
        else:
            shutil.copy(f, dest / "scores" / f.name)
    rows = []
    for run in (run_fn, run_ar):
        with open(run / "verification_results.csv", newline="") as fh:
            rows += list(csv.DictReader(fh))
    with open(dest / "verification_results.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    (dest / "verification_results.json").write_text(json.dumps(
        {"protocol": "lfw_fold0_verification", "status": "COMPLETE", "n_rows": len(rows), "fold": 0, "seed": 42,
         "results": rows}, default=str))
    return dest, run_fn, run_ar


def _without_run_dir(prov):
    return {k: v for k, v in prov.items()
            if k not in ("run_directory", "model_filter", "models_in_run", "n_rows_in_run", "run_json", "input_sha256")}


def test_multi_model_run_is_refused_without_model_and_names_the_option(tmp_path):
    dest, _, _ = _combined(tmp_path)
    with pytest.raises(V.ValidPairsError, match=r"exactly one model, found \['arcface', 'facenet'\].*--model"):
        V.analyse_run(dest)
    assert V.main(["--run", str(dest), "--out", str(tmp_path / "o")]) == 2
    assert not (tmp_path / "o").exists()                                                  # a refused run writes nothing


@pytest.mark.parametrize("model", ["facenet", "arcface"])
def test_model_selection_gives_exactly_the_single_model_run_result(tmp_path, model):
    dest, run_fn, run_ar = _combined(tmp_path)
    single = {"facenet": run_fn, "arcface": run_ar}[model]
    rows_c, summ_c, prov_c = V.analyse_run(dest, model=model)
    rows_s, summ_s, prov_s = V.analyse_run(single)
    assert rows_c == rows_s and summ_c == summ_s and len(rows_c) == 9 and {r["model"] for r in rows_c} == {model}
    assert _without_run_dir(prov_c) == _without_run_dir(prov_s)
    assert prov_c["model_filter"] == model and prov_c["models_in_run"] == ["arcface", "facenet"] and prov_c["n_rows_in_run"] == 18
    assert prov_s["model_filter"] is None and prov_s["n_rows_in_run"] == 9
    assert prov_c["input_sha256"]["scores/test_pairs.npz"] == prov_s["input_sha256"]["scores/test_pairs.npz"]
    assert set(prov_c["score_file_sha256"]) == {f"{model}__{r['perturbation']}__sev{r['severity']:.2f}.npz" for r in rows_c}


def test_models_of_a_combined_run_keep_their_own_failure_profile(tmp_path):
    dest, _, _ = _combined(tmp_path)
    fn, _, _ = V.analyse_run(dest, model="facenet")
    ar, _, _ = V.analyse_run(dest, model="arcface")
    assert all(r["n_failed_embeddings"] == 0 and r["valid_pair_rate_balanced"] == 1.0 for r in fn)
    assert any(r["n_failed_embeddings"] > 0 and r["valid_pair_rate_balanced"] < 1.0 for r in ar)


def test_unknown_model_is_refused_and_a_single_model_run_checks_the_name(tmp_path):
    dest, run_fn, _ = _combined(tmp_path)
    with pytest.raises(V.ValidPairsError, match=r"--model 'insightface' is not in this run \(models in the CSV: \['arcface', 'facenet'\]\)"):
        V.analyse_run(dest, model="insightface")
    with pytest.raises(V.ValidPairsError, match="--model 'arcface' is not in this run"):
        V.analyse_run(run_fn, model="arcface")                                          # a facenet-only run
    assert V.analyse_run(run_fn, model="facenet")[0] == V.analyse_run(run_fn)[0]        # naming the only model is harmless


def test_checks_apply_to_the_selected_model_and_do_not_leak_across_models(tmp_path):
    dest, _, _ = _combined(tmp_path)
    target = dest / "scores" / "arcface__sunglasses__sev1.00.npz"
    z = dict(np.load(target, allow_pickle=False))
    z["s_bal"] = z["s_bal"] + 0.01                                                      # tamper with ONE ArcFace score file
    np.savez_compressed(target, **z)
    with pytest.raises(V.ValidPairsError, match="not reproducible from the saved scores"):
        V.analyse_run(dest, model="arcface")
    assert len(V.analyse_run(dest, model="facenet")[0]) == 9                            # FaceNet's own files are intact
    with pytest.raises(V.ValidPairsError, match="exactly one model"):
        V.analyse_run(dest)


def test_duplicate_condition_and_missing_score_file_in_a_combined_run_are_refused(tmp_path):
    dest, _, _ = _combined(tmp_path)
    (dest / "scores" / "facenet__rotation__sev0.50.npz").unlink()
    with pytest.raises(V.ValidPairsError, match="missing score file facenet__rotation__sev0.50.npz"):
        V.analyse_run(dest, model="facenet")
    assert len(V.analyse_run(dest, model="arcface")[0]) == 9
    with open(dest / "verification_results.csv", newline="") as fh:
        rows = list(csv.DictReader(fh))
    rows.append(dict(rows[0]))
    with open(dest / "verification_results.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    with pytest.raises(V.ValidPairsError, match="duplicate"):
        V.analyse_run(dest, model="arcface")


def test_combined_run_directory_is_never_modified_and_cli_model_option_works(tmp_path, capsys):
    dest, _, _ = _combined(tmp_path)
    before = {str(p.relative_to(dest)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(dest.rglob("*")) if p.is_file()}
    for model in ("facenet", "arcface"):
        out = tmp_path / f"out_{model}"
        assert V.main(["--run", str(dest), "--model", model, "--out", str(out)]) == 0
        assert {p.name for p in out.iterdir()} == {V.BY_CONDITION, V.MODEL_SUMMARY, V.PROVENANCE}
        assert json.loads((out / V.PROVENANCE).read_text())["model_filter"] == model
    assert V.main(["--run", str(dest), "--model", "insightface", "--out", str(tmp_path / "bad")]) == 2
    assert "REFUSED" in capsys.readouterr().out and not (tmp_path / "bad").exists()
    after = {str(p.relative_to(dest)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(dest.rglob("*")) if p.is_file()}
    assert before == after
    assert "--model" in V.build_parser().format_help()


REAL_DEEP_CORE = Path(__file__).resolve().parent.parent / "benchmarks" / "lfw_fold0_2026-11-20" / "deep_core"


@pytest.mark.skipif(not (REAL_DEEP_CORE / "scores" / "test_pairs.npz").is_file(), reason="the archived deep_core run is not present")
@pytest.mark.parametrize("model", ["facenet", "arcface"])
def test_real_archived_deep_core_scores_reproduce_the_archived_values(tmp_path, model):
    """The archived end-to-end values of every one of the 96 conditions are reproduced from the saved scores (no
    mismatch allowed), and the run directory is not modified."""
    before = {p.name: p.stat().st_size for p in sorted((REAL_DEEP_CORE / "scores").iterdir())}
    rows, summary, prov = V.analyse_run(REAL_DEEP_CORE, model=model)
    assert len(rows) == 96 and prov["is_full_grid"] and prov["archive_mismatches"] == [] and all(r["archive_reproduced"] for r in rows)
    assert prov["n_rows_in_run"] == 192 and prov["models_in_run"] == ["arcface", "facenet"]
    assert before == {p.name: p.stat().st_size for p in sorted((REAL_DEEP_CORE / "scores").iterdir())}

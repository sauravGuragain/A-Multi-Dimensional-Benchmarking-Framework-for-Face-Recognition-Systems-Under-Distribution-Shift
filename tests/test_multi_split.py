"""Multi-split / multi-seed verification: configurable split seed and fold, Fold-0 backward compatibility, leakage
and pair-count invariants.

SCOPE: the integration tests run the real runner on the fake on-disk LFW of ``test_verification_stage`` (random noise
images; stub FaceNet). They test the protocol and its invariants, not any model's accuracy.
"""
import copy
import csv
import json
import math
import os
import sys
from contextlib import contextmanager
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))
import test_verification_stage as T  # noqa: E402
from faceeval.evaluation import pairs as P  # noqa: E402
from faceeval.evaluation.split_invariants import SplitInvariantError, check_split_and_pairs  # noqa: E402
from faceeval.orchestration.runner import ExperimentRunner  # noqa: E402


@contextmanager
def pushd(d):
    old = os.getcwd()
    os.chdir(d)
    try:
        yield
    finally:
        os.chdir(old)


def run_split(tmp_path, data, seed=42, fold=None):
    tmp_path.mkdir(parents=True)
    cfg = T._config(tmp_path, data, protocol="lfw_fold0_verification", deep=())
    cfg.random_seed = seed
    if fold is not None:
        cfg.verification_fold = fold
    with pushd(tmp_path):
        r = ExperimentRunner(cfg)
        r.run()
    return r.verification_summary


@pytest.fixture(scope="module")
def data(tmp_path_factory):
    d = tmp_path_factory.mktemp("data")
    T._make_fake_lfw(d)
    return d


@pytest.fixture(scope="module")
def splits(tmp_path_factory, data):
    base = tmp_path_factory.mktemp("runs")
    return {"ref_default": run_split(base / "a", data),
            "ref_explicit": run_split(base / "b", data, seed=42, fold=0),
            "s42f3": run_split(base / "c", data, seed=42, fold=3),
            "s7f0": run_split(base / "d", data, seed=7, fold=0),
            "s7f0_again": run_split(base / "e", data, seed=7, fold=0)}


class _M:
    def __init__(self, metadata, partitions):
        self.metadata, self.partitions = metadata, partitions


def manifest(summary):
    """Parse the canonical manifest text (head line + 'partition|identity|image' lines) without the official-count check."""
    import hashlib
    raw = Path(summary["manifest_path"]).read_bytes()
    assert hashlib.sha256(raw).hexdigest() == summary["manifest_sha256"]
    head, *lines = raw.decode("utf-8").splitlines()
    meta = dict(kv.split("=", 1) for kv in head.split("|"))
    parts = {"train": [], "validation": [], "test": []}
    for ln in lines:
        p, i, im = ln.split("|")
        parts[p].append((i, im))
    return _M(meta, parts)


def idents(summary, part):
    return {i for i, _ in manifest(summary).partitions[part]}


def test_reference_split_is_unchanged_and_carries_no_new_block(splits):
    a, b = splits["ref_default"], splits["ref_explicit"]
    assert "split" not in a and "split" not in b and a["fold"] == 0 and a["seed"] == 42
    assert a["content_sha256"] == b["content_sha256"] and a["pair_params"] == b["pair_params"]
    dump = lambda s: json.dumps(s["results"], sort_keys=True)            # noqa: E731  (NaN-safe: NaN != NaN as floats)
    assert dump(a) == dump(b)
    assert a["pair_params"]["validation"]["seed"] == 42          # the historical pair seed


def test_other_splits_complete_and_record_their_split(splits):
    for key, (seed, fold) in {"s42f3": (42, 3), "s7f0": (7, 0)}.items():
        s = splits[key]
        assert s["status"] == "COMPLETE" and s["seed"] == seed and s["fold"] == fold
        blk = s["split"]
        assert blk["reference_split"] is False and blk["invariants_checked"] and blk["pair_params"] == {"seed": seed, "cap": 10, "rho": 10}
        for part in ("validation", "test"):
            c = blk["counts"][part]
            assert c["n_balanced"] == c["n_genuine"] > 0 and c["n_pool"] == 10 * c["n_genuine"]
            assert s["pair_params"][part]["seed"] == seed and s["pair_params"][part]["cap"] == 10 and s["pair_params"][part]["rho"] == 10
        assert manifest(s).metadata["seed"] == str(seed) and manifest(s).metadata["fold"] == str(fold)


def test_partitions_are_identity_disjoint_in_every_split(splits):
    for s in splits.values():
        tr, va, te = idents(s, "train"), idents(s, "validation"), idents(s, "test")
        assert not (tr & va) and not (tr & te) and not (va & te)
        imgs = {p: {m for _, m in manifest(s).partitions[p]} for p in ("train", "validation", "test")}
        assert not (imgs["train"] & imgs["test"]) and not (imgs["validation"] & imgs["test"])


def test_splits_differ_by_seed_and_fold_but_are_deterministic(splits):
    c = lambda k: splits[k]["content_sha256"]["split"]  # noqa: E731
    assert c("s7f0") == c("s7f0_again")                                  # same (seed, fold) -> identical split
    assert splits["s7f0"]["content_sha256"] == splits["s7f0_again"]["content_sha256"]
    assert len({c("ref_default"), c("s42f3"), c("s7f0")}) == 3
    assert idents(splits["ref_default"], "test").isdisjoint(idents(splits["s42f3"], "test"))   # folds partition the identities
    assert [r["auc"] for r in splits["s7f0"]["results"]] == [r["auc"] for r in splits["s7f0_again"]["results"]]


def test_fold_out_of_range_is_rejected_at_config_level(tmp_path, data):
    from pydantic import ValidationError
    cfg_text = T._config(tmp_path, data, protocol="lfw_fold0_verification")
    assert cfg_text.verification_fold == 0
    from faceeval.core.config import ConfigLoader
    base = """
experiment_name: x
random_seed: 1
verification_fold: 10
dataset_names: [lfw]
data_root: %s
evaluation_protocol: lfw_fold0_verification
traditional_models: [eigenfaces]
deep_models: []
perturbation_schedule: {perturbation_types: [gaussian_blur], severity_levels: [0.0], compose_perturbations: false}
""" % data
    with pytest.raises((ValidationError, ValueError, Exception)) as e:
        ConfigLoader().loads(base)
    assert "verification_fold" in str(e.value) or "less than or equal" in str(e.value)


def test_runner_rejects_fold_outside_range(tmp_path, data):
    tmp_path.mkdir(exist_ok=True)
    cfg = T._config(tmp_path, data, protocol="lfw_fold0_verification", deep=())
    cfg.verification_fold = 10                                          # bypasses the schema: the runner must still refuse
    with pushd(tmp_path):
        with pytest.raises(Exception) as e:
            ExperimentRunner(cfg).run()
    assert "verification_fold" in str(e.value)


# ---------------------------------------------------------------- invariant checker
def _toy():
    parts = {"validation": [("a", "a1"), ("a", "a2"), ("b", "b1"), ("b", "b2"), ("c", "c1"), ("c", "c2")],
             "test": [("d", "d1"), ("d", "d2"), ("e", "e1"), ("e", "e2")],
             "train": [("f", "f1"), ("f", "f2")]}
    sets = {"validation": P.build_pairs(parts["validation"], seed=3, partition_code=1, cap=2, rho=2),
            "test": P.build_pairs(parts["test"], seed=3, partition_code=2, cap=2, rho=2)}
    return parts, sets


def test_invariants_accept_a_valid_split():
    parts, sets = _toy()
    out = check_split_and_pairs(parts, sets, rho=2)
    assert out["test"]["n_balanced"] == out["test"]["n_genuine"]


def test_invariants_detect_identity_and_image_leakage_and_pair_corruption():
    parts, sets = _toy()
    leaky = copy.deepcopy(parts); leaky["train"].append(("d", "d3"))
    with pytest.raises(SplitInvariantError, match="identity leakage between test and train|identity leakage"):
        check_split_and_pairs(leaky, sets, rho=2)
    dup = copy.deepcopy(parts); dup["train"].append(("z", "d1"))
    with pytest.raises(SplitInvariantError, match="image overlap"):
        check_split_and_pairs(dup, sets, rho=2)
    bad = copy.deepcopy(sets["test"]); bad.genuine[0] = (0, 2)            # joins two identities
    with pytest.raises(SplitInvariantError, match="genuine pair joins two different identities"):
        check_split_and_pairs(parts, {"test": bad}, rho=2)
    bad2 = copy.deepcopy(sets["test"]); bad2.balanced = bad2.balanced[:-1]
    with pytest.raises(SplitInvariantError, match="balanced impostors"):
        check_split_and_pairs(parts, {"test": bad2}, rho=2)
    with pytest.raises(SplitInvariantError, match="not the validation partition"):
        check_split_and_pairs(parts, {"validation": sets["test"]}, rho=2)


# ---------------------------------------------------------------- aggregation
def _run_dir(summary):
    return Path(summary["manifest_path"]).parent


def test_aggregate_mean_sd_ci_and_provenance(splits, tmp_path):
    import csv
    import math
    from faceeval.analysis import multi_split as MS
    runs = [_run_dir(splits[k]) for k in ("ref_default", "s42f3", "s7f0")]
    out = tmp_path / "agg"
    prov = MS.aggregate(runs, out)
    assert prov["n_splits"] == 3
    rows = list(csv.DictReader(open(out / MS.OUT_TABLE)))
    one = rows[0]
    vals = []
    for r in runs:
        for x in csv.DictReader(open(r / "verification_results.csv")):
            if (x["model"], x["perturbation"], float(x["severity"])) == (one["model"], one["perturbation"], float(one["severity"])):
                vals.append(float(x["accuracy"]))
    import numpy as np
    assert len(vals) == 3 and math.isclose(float(one["accuracy_mean"]), np.mean(vals))
    assert math.isclose(float(one["accuracy_sd"]), np.std(vals, ddof=1), abs_tol=1e-12)
    half = MS.t_crit(3) * np.std(vals, ddof=1) / math.sqrt(3)
    assert math.isclose(float(one["accuracy_ci_hi"]) - float(one["accuracy_mean"]), half, abs_tol=1e-9)
    assert "not independent" in prov["caveat"]
    runs_csv = list(csv.DictReader(open(out / MS.OUT_RUNS)))
    assert {(r["seed"], r["fold"]) for r in runs_csv} == {("42", "0"), ("42", "3"), ("7", "0")}


def test_aggregate_single_split_gives_nan_not_invented(splits, tmp_path):
    import csv
    from faceeval.analysis import multi_split as MS
    MS.aggregate([_run_dir(splits["s7f0"])], tmp_path / "one")
    r = next(csv.DictReader(open(tmp_path / "one" / MS.OUT_TABLE)))
    assert r["accuracy_sd"] == "nan" and r["accuracy_ci_lo"] == "nan"


def test_aggregate_refuses_duplicates_inside_run_and_nonempty_out(splits, tmp_path):
    from faceeval.analysis import multi_split as MS
    a, b = _run_dir(splits["s7f0"]), _run_dir(splits["s7f0_again"])
    with pytest.raises(MS.MultiSplitError, match="duplicate split"):
        MS.aggregate([a, b], tmp_path / "x")
    with pytest.raises(MS.MultiSplitError, match="inside a run directory"):
        MS.aggregate([a], a / "agg")
    (tmp_path / "full").mkdir()
    (tmp_path / "full" / "f").write_text("x")
    with pytest.raises(MS.MultiSplitError, match="not empty"):
        MS.aggregate([a], tmp_path / "full")
    assert MS.main(["--runs", str(a), "--out", str(tmp_path / "full")]) == 2


def test_aggregate_refuses_incomplete_and_different_dataset(splits, tmp_path):
    import shutil
    from faceeval.analysis import multi_split as MS
    a = _run_dir(splits["s7f0"])
    bad = tmp_path / "bad"
    shutil.copytree(a, bad)
    j = json.loads((bad / "verification_results.json").read_text())
    j["status"] = "INCOMPLETE"
    (bad / "verification_results.json").write_text(json.dumps(j))
    with pytest.raises(MS.MultiSplitError, match="not COMPLETE"):
        MS.aggregate([bad], tmp_path / "o1")
    other = tmp_path / "other"
    shutil.copytree(_run_dir(splits["s42f3"]), other)
    j = json.loads((other / "verification_results.json").read_text())
    j["dataset"]["fingerprint_sha256"] = "0" * 64
    (other / "verification_results.json").write_text(json.dumps(j))
    with pytest.raises(MS.MultiSplitError, match="fingerprint"):
        MS.aggregate([a, other], tmp_path / "o2")


def test_design_report_distinguishes_disjoint_folds_from_overlapping_seeds(splits, tmp_path):
    from faceeval.analysis import multi_split as MS
    folds = MS.aggregate([_run_dir(splits["ref_default"]), _run_dir(splits["s42f3"])], tmp_path / "f")["design"]
    assert folds["test_identity_sets_disjoint"] is True and folds["seeds"] == [42]
    assert "disjoint (folds of ONE seed)" in folds["interval_interpretation"]
    import hashlib

    def fake(seed, fold, test_ids):                    # a hand-built canonical manifest: head line + 'part|identity|image'
        raw = ("k=v\n" + "".join(f"test|{i}|{i}_1\n" for i in test_ids)).encode()
        path = tmp_path / f"m_{seed}_{fold}.tsv"
        path.write_bytes(raw)
        return {"seed": seed, "fold": fold, "manifest_path": str(path), "manifest_sha256": hashlib.sha256(raw).hexdigest()}, []
    overlap = MS.design_report([fake(1, 0, ["a", "b", "c"]), fake(2, 0, ["c", "d", "e"])])
    assert overlap["test_identity_sets_disjoint"] is False
    assert math.isclose(overlap["max_pairwise_test_identity_jaccard"], 1 / 5) and "OVERLAP" in overlap["interval_interpretation"]
    other_seeds = MS.design_report([fake(1, 0, ["a"]), fake(2, 0, ["b"])])
    assert other_seeds["test_identity_sets_disjoint"] is True and "different seeds" in other_seeds["interval_interpretation"]


def test_design_report_when_manifest_is_unavailable(splits, tmp_path):
    import shutil
    from faceeval.analysis import multi_split as MS
    a = tmp_path / "moved"
    shutil.copytree(_run_dir(splits["s7f0"]), a)
    j = json.loads((a / "verification_results.json").read_text())
    j["manifest_path"] = str(tmp_path / "gone.tsv")
    (a / "verification_results.json").write_text(json.dumps(j))
    d = MS.aggregate([a], tmp_path / "o")["design"]
    assert d["test_identity_sets_disjoint"] == "NOT_RECORDED" and "could not be checked" in d["interval_interpretation"]


# ---------------------------------------------------------------- interval reliability (synthetic runs, no pipeline)
def _synthetic_runs(tmp_path, specs, models=("facenet", "eigenfaces", "mystery"), acc_by_run=None):
    """specs: list of (seed, fold, train_ids, test_ids). Returns run directories with COMPLETE-looking results."""
    import hashlib
    dirs = []
    for k, (seed, fold, train, test) in enumerate(specs):
        d = tmp_path / f"r{k}"
        d.mkdir()
        raw = ("k=v\n" + "".join(f"train|{i}|{i}_1\n" for i in train) + "".join(f"test|{i}|{i}_1\n" for i in test)).encode()
        (d / "m.tsv").write_bytes(raw)
        j = {"status": "COMPLETE", "seed": seed, "fold": fold, "k_folds": 10, "dataset": {"fingerprint_sha256": "f"},
             "pair_params": {"validation": {"cap": 1, "rho": 1}, "test": {"cap": 1, "rho": 1}},
             "manifest_path": str(d / "m.tsv"), "manifest_sha256": hashlib.sha256(raw).hexdigest(), "content_sha256": {"split": str(k)}}
        (d / "verification_results.json").write_text(json.dumps(j))
        acc = 0.80 + 0.01 * (k % 3) if acc_by_run is None else acc_by_run[k]
        with open(d / "verification_results.csv", "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=["model", "perturbation", "severity", "accuracy", "tar", "far", "auc", "eer"])
            w.writeheader()
            for m in models:
                w.writerow({"model": m, "perturbation": "clean", "severity": 0.0, "accuracy": acc, "tar": acc, "far": 0.1, "auc": acc, "eer": 0.1})
        dirs.append(d)
    return dirs


def _rows(out):
    with open(out / "multi_split_by_condition.csv", newline="") as fh:
        return {r["model"]: r for r in csv.DictReader(fh)}


def _folds(n, n_ids=100):
    ids = [f"p{i}" for i in range(n_ids)]
    out = []
    for f in range(n):                       # one seed, disjoint test blocks of 10, training = everything else
        test = ids[f * 10:(f + 1) * 10]
        out.append((7, f, [i for i in ids if i not in test], test))
    return out


def test_pretrained_vs_trained_status_in_a_clean_one_seed_fold_design(tmp_path):
    from faceeval.analysis import multi_split as MS
    prov = MS.aggregate(_synthetic_runs(tmp_path, _folds(6)), tmp_path / "o")
    rows = _rows(tmp_path / "o")
    assert prov["design"]["design_kind"] == "folds_within_one_seed" and prov["design"]["test_identity_sets_disjoint"] is True
    assert prov["design"]["train_test_identity_ratio"] == pytest.approx(10 / 90)
    assert rows["facenet"]["model_kind"] == "pretrained" and rows["facenet"]["interval_status"] == MS.STATUS_PRETRAINED_OK
    assert rows["facenet"]["accuracy_corrected_ci_lo"] == "nan"
    assert rows["eigenfaces"]["model_kind"] == "trained_per_split" and rows["eigenfaces"]["interval_status"] == MS.STATUS_CORRECTED
    assert rows["mystery"]["model_kind"] == "unknown" and rows["mystery"]["interval_status"] == MS.STATUS_DESCRIPTIVE


def test_corrected_interval_is_wider_by_the_nadeau_bengio_factor(tmp_path):
    from faceeval.analysis import multi_split as MS
    accs = [0.80, 0.82, 0.79, 0.84, 0.81, 0.83]
    MS.aggregate(_synthetic_runs(tmp_path, _folds(6), acc_by_run=accs), tmp_path / "o")
    r = _rows(tmp_path / "o")["eigenfaces"]
    naive = float(r["accuracy_ci_hi"]) - float(r["accuracy_ci_lo"])
    corr = float(r["accuracy_corrected_ci_hi"]) - float(r["accuracy_corrected_ci_lo"])
    n, ratio = 6, 10 / 90
    assert corr / naive == pytest.approx(math.sqrt((1 / n + ratio) / (1 / n)))
    assert corr > naive


def test_too_few_splits_and_single_split_statuses(tmp_path):
    from faceeval.analysis import multi_split as MS
    (tmp_path / "a").mkdir()
    MS.aggregate(_synthetic_runs(tmp_path / "a", _folds(3)), tmp_path / "o3")
    assert {r["interval_status"] for r in _rows(tmp_path / "o3").values() if r["model_kind"] != "unknown"} == {MS.STATUS_FEW}
    (tmp_path / "b").mkdir()
    MS.aggregate(_synthetic_runs(tmp_path / "b", _folds(1)), tmp_path / "o1")
    assert {r["interval_status"] for r in _rows(tmp_path / "o1").values()} == {MS.STATUS_SINGLE}


def test_different_seeds_with_overlapping_test_people_are_descriptive_only_for_every_model(tmp_path):
    from faceeval.analysis import multi_split as MS
    ids = [f"p{i}" for i in range(100)]
    specs = [(s, 0, ids[20:], ids[:20] if s % 2 else ids[10:30]) for s in range(1, 7)]
    prov = MS.aggregate(_synthetic_runs(tmp_path, specs), tmp_path / "o")
    assert prov["design"]["design_kind"] == "different_seeds" and prov["design"]["test_identity_sets_disjoint"] is False
    assert {r["interval_status"] for r in _rows(tmp_path / "o").values()} == {MS.STATUS_DESCRIPTIVE}
    assert all(r["accuracy_corrected_ci_lo"] == "nan" for r in _rows(tmp_path / "o").values())
    assert prov["interval_status_counts"] == {MS.STATUS_DESCRIPTIVE: 3} and "population confidence interval" in prov["caveat"]


def test_trained_model_without_readable_manifest_falls_back_to_descriptive(tmp_path):
    from faceeval.analysis import multi_split as MS
    dirs = _synthetic_runs(tmp_path, _folds(6))
    for d in dirs:
        (d / "m.tsv").unlink()
    MS.aggregate(dirs, tmp_path / "o")
    assert {r["interval_status"] for r in _rows(tmp_path / "o").values()} == {MS.STATUS_DESCRIPTIVE}


def test_model_kind_covers_every_registered_model():
    from faceeval.analysis import multi_split as MS
    import faceeval.models  # noqa: F401
    from faceeval.core.registry import model_registry
    names = set(model_registry.list_models()) if hasattr(model_registry, "list_models") else set(MS.PRETRAINED_MODELS | MS.TRAINED_MODELS)
    assert names <= (MS.PRETRAINED_MODELS | MS.TRAINED_MODELS)

"""Subgroup fairness tool. SYNTHETIC labelled data only (hand-built score files): the tests prove the arithmetic,
coverage accounting and refusals. They prove nothing about any real model or dataset."""
import csv
import json
import math

import numpy as np
import pytest

from faceeval.analysis import fairness as F


def make_run(tmp_path, n_per=40, thr=0.5):
    """Two groups A (0..) and B. In A genuine pairs score high & impostors low (perfect); in B half the genuine pairs fall
    below the threshold (TAR 0.5)."""
    run = tmp_path / "run"
    (run / "scores").mkdir(parents=True)
    ids, pairs, labels, s = [], [], [], []
    def add_group(prefix, bad_tar):
        base = len(ids)
        for i in range(2 * n_per * 2):
            ids.append(f"{prefix}{i}")
        for k in range(n_per):                      # genuine pairs
            pairs.append((base + 2 * k, base + 2 * k + 1)); labels.append(True)
            s.append(0.2 if (bad_tar and k % 2 == 0) else 0.9)
        for k in range(n_per):                      # impostor pairs
            pairs.append((base + 2 * k, base + 2 * n_per + 2 * k)); labels.append(False); s.append(0.1)
    add_group("A", False); add_group("B", True)
    pairs.append((0, len(ids) - 1)); labels.append(False); s.append(0.1)       # one cross-group pair
    np.savez(run / "scores" / "test_pairs.npz", image_ids=np.array(ids), bal_pairs=np.array(pairs),
             bal_labels=np.array(labels), full_pairs=np.array(pairs), full_labels=np.array(labels))
    np.savez(run / "scores" / "m__clean__sev0.0.npz", valid=np.ones(len(ids), bool), s_bal=np.array(s),
             s_full=np.array(s), threshold=np.float64(thr))
    with open(run / "verification_results.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["model", "perturbation", "severity", "label", "scores_file"]); w.writeheader()
        w.writerow({"model": "m", "perturbation": "clean", "severity": 0.0, "label": "clean", "scores_file": "scores/m__clean__sev0.0.npz"})
    (run / "verification_results.json").write_text(json.dumps({"status": "COMPLETE"}))
    attr = tmp_path / "attr.csv"
    with open(attr, "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["image_id", "group"])
        for i in ids:
            w.writerow([i, i[0]])
    return run, attr, ids


def test_group_metrics_disparity_and_coverage(tmp_path):
    run, attr, ids = make_run(tmp_path)
    cov = F.run(run, attr, tmp_path / "out")
    assert cov["image_coverage"] == 1.0 and cov["n_cross_or_unknown_pairs"] == 1
    rows = {r["group"]: r for r in csv.DictReader(open(tmp_path / "out" / F.BY_GROUP))}
    assert float(rows["A"]["tar"]) == 1.0 and float(rows["A"]["far"]) == 0.0
    assert math.isclose(float(rows["B"]["tar"]), 0.5) and float(rows["B"]["far"]) == 0.0
    d = next(csv.DictReader(open(tmp_path / "out" / F.DISPARITY)))
    assert math.isclose(float(d["tar_max_minus_min"]), 0.5) and math.isclose(float(d["tar_max_over_min"]), 2.0)
    assert d["far_max_over_min"] == "nan"             # min FAR is 0 -> ratio undefined, never invented


def test_missing_labels_are_unknown_not_inferred(tmp_path):
    run, attr, ids = make_run(tmp_path)
    keep = [r for r in csv.reader(open(attr))]
    with open(attr, "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(keep[0]); [w.writerow(r) for r in keep[1:] if not r[0].startswith("B")]
    cov = F.run(run, attr, tmp_path / "out")
    assert cov["n_images_with_group"] == len(ids) // 2 and set(cov["groups"]) == {"A"}
    d = next(csv.DictReader(open(tmp_path / "out" / F.DISPARITY)))
    assert d["n_groups_used"] == "1" and d["tar_max_minus_min"] == "nan"      # one group -> no disparity


def test_small_group_is_insufficient_and_nan(tmp_path):
    run, attr, _ = make_run(tmp_path, n_per=10)
    F.run(run, attr, tmp_path / "out", min_pairs=30)
    r = next(csv.DictReader(open(tmp_path / "out" / F.BY_GROUP)))
    assert r["sufficient"] == "False" and r["tar"] == "nan"


def test_refusals(tmp_path):
    run, attr, _ = make_run(tmp_path)
    with pytest.raises(F.FairnessError, match="inside the run"):
        F.run(run, attr, run / "o")
    bad = tmp_path / "bad.csv"; bad.write_text("id,g\n1,a\n")
    with pytest.raises(F.FairnessError, match="needs columns"):
        F.load_attributes(bad)
    dup = tmp_path / "dup.csv"; dup.write_text("image_id,group\nx,a\nx,b\n")
    with pytest.raises(F.FairnessError, match="conflicting"):
        F.load_attributes(dup)
    (run / "verification_results.json").write_text(json.dumps({"status": "INCOMPLETE"}))
    with pytest.raises(F.FairnessError, match="not COMPLETE"):
        F.run(run, attr, tmp_path / "o2")
    assert F.main(["--run", str(run), "--attributes", str(attr), "--out", str(tmp_path / "o3")]) == 2
    assert not (tmp_path / "o3").exists()              # a refused run writes nothing


def test_no_matching_ids_is_a_clean_refusal_and_writes_nothing(tmp_path):
    run, _attr, _ids = make_run(tmp_path)
    other = tmp_path / "other.csv"
    other.write_text("image_id,group\nzzz1,a\nzzz2,b\n")
    with pytest.raises(F.FairnessError, match="no row of the attribute file matches"):
        F.run(run, other, tmp_path / "o")
    assert not (tmp_path / "o").exists()
    assert F.main(["--run", str(run), "--attributes", str(other), "--out", str(tmp_path / "o")]) == 2


def test_attribute_file_provenance_is_recorded(tmp_path):
    import hashlib
    run, attr, _ids = make_run(tmp_path)
    cov = F.run(run, attr, tmp_path / "out", source="synthetic unit-test labels")
    assert cov["attribute_file"]["sha256"] == hashlib.sha256(attr.read_bytes()).hexdigest()
    assert cov["attribute_file"]["source_note"] == "synthetic unit-test labels"
    cov2 = F.run(run, attr, tmp_path / "out2")
    assert cov2["attribute_file"]["source_note"] == "NOT_RECORDED"

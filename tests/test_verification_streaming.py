"""Incremental results, failed-embedding accounting and saved pair scores of the Fold 0 verification stage.

Why: results used to be written only after every model x condition finished (a killed run lost everything),
ArcFace detector misses became zero embeddings that were scored like any other pair with no per-condition count,
validation accuracy was only in the log, and no pair-level scores were kept. These tests pin the additions and
prove that classical rows keep their original columns.
"""
import csv
import json
import os
from pathlib import Path

import numpy as np
import pytest
from sklearn.metrics import roc_auc_score

from faceeval.core.exceptions import OrchestratorError
from faceeval.orchestration import verification as V

from tests.test_verification_stage import _run, fake_data  # noqa: F401  (fixture + helper reused)

NEW_DEEP_COLUMNS = ("val_balanced_accuracy", "val_n_failed_embeddings", "n_test_images", "n_failed_embeddings",
                    "n_failed_pairs_balanced", "n_failed_pairs_full", "scores_file")
DEEP = ("facenet", "arcface")
CLASSICAL = ("eigenfaces", "lbph")


def _force_arcface_stub(mp):
    """Deterministic on every machine: never load the real insightface detector (it misses all synthetic faces)."""
    from faceeval.models.deep import arcface as arc
    mp.setattr(arc, "_check_insightface", lambda: False)


def _outputs(runner):
    out = Path(runner.verification_summary["output_dir"])
    with open(out / "verification_results.csv", newline="") as fh:
        rows = list(csv.DictReader(fh))
    data = json.load(open(out / "verification_results.json"))
    stream = [json.loads(line) for line in open(out / "verification_rows.jsonl")]
    return out, rows, data, stream


@pytest.fixture(scope="module")
def clean_run(tmp_path_factory, fake_data):
    mp = pytest.MonkeyPatch()
    _force_arcface_stub(mp)
    try:
        _, runner, _ = _run(tmp_path_factory.mktemp("clean"), mp, fake_data,
                            protocol="lfw_fold0_verification", deep=DEEP)
        yield _outputs(runner)
    finally:
        mp.undo()


# ------------------------------------------------------------------ unit tests of the helpers
def test_valid_rows_flags_only_zero_vectors():
    emb = np.array([[1.0, 0.0], [0.0, 0.0], [0.6, 0.8]])
    assert _list(V._valid_rows(emb)) == [True, False, True]


def _list(a):
    return [bool(x) for x in a]


def test_pairs_with_failure_counts_pairs_touching_an_invalid_image():
    valid = np.array([True, False, True, True])
    pairs = [(0, 2), (0, 1), (1, 1), (2, 3)]
    assert V._pairs_with_failure(valid, pairs) == 2


def test_row_stream_is_readable_before_close_and_keeps_every_record(tmp_path):
    path = tmp_path / "s.jsonl"
    s = V._RowStream(path)
    s.write("plan", {"n_rows_expected": 2})
    s.write("row", {"model": "m", "auc": float("nan")})
    lines = [json.loads(x) for x in open(path)]                     # a second handle sees it: flushed to disk
    assert [x["type"] for x in lines] == ["plan", "row"] and lines[1]["model"] == "m"
    s.close()


# ------------------------------------------------------------------ a normal run
def test_stream_matches_final_results_and_ends_cleanly(clean_run):
    out, rows, data, stream = clean_run
    assert stream[0]["type"] == "plan" and stream[0]["n_rows_expected"] == len(rows) == data["n_rows_expected"]
    assert stream[-1] == {"type": "end", "status": "COMPLETE", "n_rows": len(rows), "n_rows_expected": len(rows)}
    streamed = [{k: v for k, v in r.items() if k != "type"} for r in stream if r["type"] == "row"]
    assert len(streamed) == len(rows) and not [r for r in stream if r["type"] == "failure"]
    dump = lambda x: json.dumps(x, sort_keys=True, default=str)       # NaN-safe comparison
    assert dump(streamed) == dump(data["results"])                   # same rows, same order, same values
    assert data["status"] == "COMPLETE" and data["n_rows"] == data["n_rows_expected"]


def test_deep_rows_carry_the_new_columns_and_classical_rows_do_not(clean_run):
    _, rows, data, _ = clean_run
    for r in data["results"]:
        if r["model"] in DEEP:
            assert all(c in r for c in NEW_DEEP_COLUMNS)
            assert 0.0 <= r["val_balanced_accuracy"] <= 1.0
            assert r["n_failed_embeddings"] == 0 and r["n_failed_pairs_balanced"] == 0 and r["n_failed_pairs_full"] == 0
            assert r["val_n_failed_embeddings"] == 0 and r["n_test_images"] > 0
        else:
            assert not any(c in r for c in NEW_DEEP_COLUMNS)        # classical rows keep their original columns
    classical_csv = [r for r in rows if r["model"] in CLASSICAL]
    assert classical_csv and all(r[c] == "" for r in classical_csv for c in NEW_DEEP_COLUMNS)


def test_saved_scores_are_the_scores_the_metrics_came_from(clean_run):
    out, rows, _, _ = clean_run
    pairs = np.load(out / "scores" / "test_pairs.npz")
    assert pairs["bal_labels"].dtype == bool and pairs["bal_pairs"].shape[1] == 2
    files = sorted(p.name for p in (out / "scores").glob("*__sev*.npz"))
    deep_rows = [r for r in rows if r["model"] in DEEP]
    assert len(files) == len(deep_rows) and not [f for f in files if f.split("__")[0] in CLASSICAL]
    for r in deep_rows:
        z = np.load(out / "scores" / f"{r['model']}__{r['perturbation']}__sev{float(r['severity']):.2f}.npz")
        assert z["valid"].shape == (int(r["n_test_images"]),) and float(z["threshold"]) == float(r["threshold"])
        assert roc_auc_score(pairs["bal_labels"], z["s_bal"]) == pytest.approx(float(r["auc"]), abs=1e-12)
        assert roc_auc_score(pairs["full_labels"], z["s_full"]) == pytest.approx(float(r["auc_full"]), abs=1e-12)
        assert ((z["s_bal"] >= float(z["threshold"])) == pairs["bal_labels"]).mean() == pytest.approx(float(r["accuracy"]))


def test_no_scores_are_written_for_a_classical_only_run(tmp_path, monkeypatch, fake_data):
    _, runner, _ = _run(tmp_path, monkeypatch, fake_data, protocol="lfw_fold0_verification", deep=())
    out, rows, data, stream = _outputs(runner)
    assert not (out / "scores").exists()
    assert not ({"n_failed_embeddings", "val_balanced_accuracy"} & set(rows[0]))
    assert stream[-1]["type"] == "end"


# ------------------------------------------------------------------ failed embeddings
def test_failed_embeddings_are_counted_per_condition_and_scored_as_zero(tmp_path, monkeypatch, fake_data):
    from faceeval.models.deep import arcface as arc
    calls: list[np.ndarray] = []
    real_embed = arc._ArcFaceStub.embed

    def embed_with_misses(self, X):
        out = real_embed(self, X).copy()
        miss = np.array([int(np.abs(x).sum() * 137) % 7 == 0 for x in X])     # deterministic "detector miss"
        out[miss] = 0.0
        calls.append(miss)
        return out

    _force_arcface_stub(monkeypatch)
    monkeypatch.setattr(arc._ArcFaceStub, "embed", embed_with_misses)
    _, runner, _ = _run(tmp_path, monkeypatch, fake_data, protocol="lfw_fold0_verification", deep=("arcface",))
    out, rows, data, _ = _outputs(runner)
    arc_rows = [r for r in data["results"] if r["model"] == "arcface"]
    assert len(calls) == 2 + len(arc_rows)                            # validation, 1-image dimension probe, conditions
    assert data["results"] and any(c.any() for c in calls[2:])
    pairs = np.load(out / "scores" / "test_pairs.npz")
    for r, miss in zip(arc_rows, calls[2:]):
        assert r["n_failed_embeddings"] == int(miss.sum()) and r["n_test_images"] == miss.size
        for key, pr in (("n_failed_pairs_balanced", pairs["bal_pairs"]), ("n_failed_pairs_full", pairs["full_pairs"])):
            assert r[key] == sum(1 for a, b in pr if miss[a] or miss[b])
        z = np.load(out / "scores" / f"arcface__{r['perturbation']}__sev{r['severity']:.2f}.npz")
        assert np.array_equal(z["valid"], ~miss)
        touched = np.array([bool(miss[a] or miss[b]) for a, b in pairs["bal_pairs"]])
        assert touched.any() and (z["s_bal"][touched] == 0.0).all()    # documented behaviour: scored 0.0, not dropped
    assert arc_rows[0]["val_n_failed_embeddings"] == int(calls[0].sum()) > 0


# ------------------------------------------------------------------ failures and interruption
def test_a_failed_condition_is_streamed_and_the_run_is_marked_incomplete(tmp_path, monkeypatch, fake_data):
    real = V.perturb_batch

    def flaky(composer, spec, normalizer, X, chips, ids):
        if spec.label.startswith("brightness") and spec.severity == 1.0:
            raise RuntimeError("injected failure")
        return real(composer, spec, normalizer, X, chips, ids)

    _force_arcface_stub(monkeypatch)
    monkeypatch.setattr(V, "perturb_batch", flaky)
    with pytest.raises(OrchestratorError):
        _run(tmp_path, monkeypatch, fake_data, protocol="lfw_fold0_verification", deep=("facenet",))
    path = next(tmp_path.rglob("verification_rows.jsonl"))
    stream = [json.loads(line) for line in open(path)]
    fails = [r for r in stream if r["type"] == "failure"]
    rows = [r for r in stream if r["type"] == "row"]
    assert len(fails) == 3 and all("injected failure" in f["error"] for f in fails)     # 3 models reach brightness@1.0
    assert len(rows) == stream[0]["n_rows_expected"] - 3
    assert stream[-1]["type"] == "end" and stream[-1]["status"] == "INCOMPLETE"
    data = json.load(open(path.with_name("verification_results.json")))
    assert data["status"] == "INCOMPLETE" and data["n_rows"] == len(rows) < data["n_rows_expected"]


# ------------------------------------------------------------------ revision 2: append safety, validity rule, lifecycle, consistency
def test_row_stream_never_appends_to_an_existing_file(tmp_path):
    path = tmp_path / "verification_rows.jsonl"
    path.write_text('{"type": "row", "model": "previous_run"}\n')
    with pytest.raises(FileExistsError):
        V._RowStream(path)
    assert path.read_text() == '{"type": "row", "model": "previous_run"}\n'      # untouched


def test_existing_run_directory_is_refused_and_left_untouched(tmp_path):
    from types import SimpleNamespace
    run_dir = tmp_path / "verification" / "r1"
    run_dir.mkdir(parents=True)
    old = run_dir / "verification_rows.jsonl"
    old.write_text('{"type": "end", "status": "COMPLETE"}\n')
    with pytest.raises(FileExistsError):
        V.run_fold0_verification(config=SimpleNamespace(output_dir=str(tmp_path), run_id="r1"))
    assert old.read_text() == '{"type": "end", "status": "COMPLETE"}\n' and len(list(run_dir.iterdir())) == 1


def test_a_restart_gets_its_own_stream_so_records_cannot_mix(tmp_path):
    a = V._RowStream(tmp_path / "a.jsonl"); a.write("row", {"k": 1}); a.close()
    b = V._RowStream(tmp_path / "b.jsonl"); b.write("row", {"k": 2}); b.close()
    assert [json.loads(l)["k"] for l in open(tmp_path / "a.jsonl")] == [1]
    assert [json.loads(l)["k"] for l in open(tmp_path / "b.jsonl")] == [2]


# ---- embedding validity: failed == zero vector; no real embedding may be called failed
def test_validity_rule_on_facenet_outputs():
    from faceeval.models.deep.base import DeepRecognizerBase
    rng = np.random.default_rng(1)
    for scale in (1e-4, 0.05, 1.0, 30.0):                       # any raw output scale
        raw = (rng.standard_normal((2000, 512)) * scale).astype(np.float32)
        emb = DeepRecognizerBase._l2_normalize(raw).astype(np.float32)   # what FaceNetRecognizer.embed returns
        assert np.abs(np.linalg.norm(emb.astype(np.float64), axis=1) - 1).max() < 1e-5
        assert V._valid_rows(emb).all(), f"real embedding flagged as failed at scale {scale}"
    emb[17] = 0.0                                                # a failed row
    emb[40, 3] = np.nan
    emb[41, 5] = np.inf
    assert np.flatnonzero(~V._valid_rows(emb)).tolist() == [17, 40, 41]
    assert V._valid_rows(0.3 * emb[:5]).all()                    # a short but real vector is not "failed"


def test_validity_rule_on_arcface_wrapper_outputs():
    from types import SimpleNamespace
    from faceeval.models.deep.arcface import _InsightFaceWrapper
    rng = np.random.default_rng(2)
    miss = rng.random(300) < 0.2

    class App:
        def __init__(self): self.i = -1
        def get(self, bgr):
            self.i += 1
            if miss[self.i]:
                return []
            return [SimpleNamespace(embedding=(rng.standard_normal(512) * 25).astype(np.float32))]   # raw norm ~565

    w = _InsightFaceWrapper(App(), crop_fallback=False)
    emb = w.embed([np.zeros((112, 112, 3), np.uint8) for _ in miss])
    valid = V._valid_rows(emb)
    assert np.array_equal(valid, ~miss) and int((~valid).sum()) == w.n_failed == w.n_detector_miss == int(miss.sum())
    assert np.abs(np.linalg.norm(emb[valid].astype(np.float64), axis=1) - 1).max() < 1e-5
    assert (emb[~valid] == 0).all()


def test_validity_rule_on_the_stubs_and_the_unit_norm_warning(caplog):
    from faceeval.models.deep.arcface import _ArcFaceStub
    from faceeval.models.deep.facenet import _FaceNetStub
    X = [np.random.default_rng(i).normal(size=(112, 112, 3)).astype(np.float32) for i in range(40)]
    for stub in (_ArcFaceStub(512), _FaceNetStub()):
        assert V._valid_rows(stub.embed(X)).all()
    caplog.clear()
    bad = np.ones((4, 8), np.float32)                            # norm 2.83: not unit
    with caplog.at_level("WARNING"):
        V._warn_if_not_unit(bad, V._valid_rows(bad), "m")
        V._warn_if_not_unit(np.eye(4, dtype=np.float32), np.ones(4, bool), "m")
    assert sum("not unit-norm" in r.message for r in caplog.records) == 1


# ---- stream lifecycle
def _track_streams(monkeypatch):
    made = []

    class Tracked(V._RowStream):
        def __init__(self, path):
            super().__init__(path); made.append(self)

    monkeypatch.setattr(V, "_RowStream", Tracked)
    return made


@pytest.mark.parametrize("exc", [RuntimeError("boom"), KeyboardInterrupt()])
def test_stream_is_closed_and_marked_aborted_on_unexpected_exceptions(tmp_path, monkeypatch, fake_data, exc):
    made = _track_streams(monkeypatch)

    def explode(*a, **k):
        raise exc

    monkeypatch.setattr(V, "shuffled_order_sha256", explode)
    with pytest.raises((type(exc), OrchestratorError)):     # the runner wraps ordinary exceptions
        _run(tmp_path, monkeypatch, fake_data, protocol="lfw_fold0_verification", deep=("facenet",))
    assert len(made) == 1 and made[0]._fh.closed
    lines = [json.loads(l) for l in open(next(tmp_path.rglob("verification_rows.jsonl")))]
    assert [r["type"] for r in lines] == ["abort"] and type(exc).__name__ in lines[0]["error"]


def test_stream_is_closed_after_a_normal_run(tmp_path, monkeypatch, fake_data):
    made = _track_streams(monkeypatch)
    _force_arcface_stub(monkeypatch)
    _run(tmp_path, monkeypatch, fake_data, protocol="lfw_fold0_verification", deep=("facenet",))
    assert len(made) == 1 and made[0]._fh.closed and made[0].terminal


# ---- interrupted run: what the stream gives back, and no orphan score files
def test_interrupted_run_leaves_a_readable_stream_of_the_finished_conditions(tmp_path, monkeypatch, fake_data):
    real, n = V.perturb_batch, {"calls": 0}
    INTERRUPT_AT = 8 + 3        # 2 classical models x 4 conditions run first; interrupt the 3rd facenet condition

    def interrupt(*a, **k):
        n["calls"] += 1
        if n["calls"] == INTERRUPT_AT:
            raise KeyboardInterrupt()
        return real(*a, **k)

    _force_arcface_stub(monkeypatch)
    monkeypatch.setattr(V, "perturb_batch", interrupt)
    with pytest.raises(KeyboardInterrupt):
        _run(tmp_path, monkeypatch, fake_data, protocol="lfw_fold0_verification", deep=("facenet",))
    out = next(tmp_path.rglob("verification_rows.jsonl")).parent
    recs = [json.loads(l) for l in open(out / "verification_rows.jsonl")]          # plain json, no helper needed
    kinds = [r["type"] for r in recs]
    assert kinds == ["plan"] + ["row"] * 10 + ["abort"] and "end" not in kinds      # interrupted == no "end"
    rows = [r for r in recs if r["type"] == "row"]
    done = [(r["model"], r["label"]) for r in rows]
    assert len(done) == len(set(done)) == 10                                         # no duplicates
    assert [lab for m, lab in done if m == "facenet"] == recs[0]["conditions"][:2]   # the two finished conditions
    on_disk = {p.name for p in (out / "scores").glob("*.npz")}
    assert on_disk == {"test_pairs.npz"} | {r["scores_file"] for r in rows if r.get("scores_file")}
    assert len(on_disk) == 3


def test_every_score_file_has_a_row_and_a_failed_record_leaves_no_orphan(tmp_path, monkeypatch, fake_data):
    real_write, state = V._RowStream.write, {}

    def write(self, kind, payload):
        if kind == "row" and payload.get("scores_file") and "first" not in state:
            state["first"] = payload["label"]                 # fail the recording of the first deep row
            raise OSError("injected: cannot record row")
        return real_write(self, kind, payload)

    _force_arcface_stub(monkeypatch)
    monkeypatch.setattr(V._RowStream, "write", write)
    with pytest.raises(OrchestratorError):
        _run(tmp_path, monkeypatch, fake_data, protocol="lfw_fold0_verification", deep=("facenet",))
    out = next(tmp_path.rglob("verification_rows.jsonl")).parent
    recs = [json.loads(l) for l in open(out / "verification_rows.jsonl")]
    rows = [r for r in recs if r["type"] == "row"]
    fails = [r for r in recs if r["type"] == "failure"]
    assert len(fails) == 1 and state["first"] in fails[0]["stage"] and "cannot record row" in fails[0]["error"]
    on_disk = {p.name for p in (out / "scores").glob("*.npz")}
    assert on_disk == {"test_pairs.npz"} | {r["scores_file"] for r in rows if r.get("scores_file")}   # the failed one left no file
    assert not list((out / "scores").glob("*.tmp.npz"))
    data = json.load(open(out / "verification_results.json"))
    assert data["status"] == "INCOMPLETE" and data["n_rows"] == len(rows)


def test_clean_run_has_one_score_file_per_deep_row(clean_run):
    out, _, data, _ = clean_run
    deep_rows = [r for r in data["results"] if r["model"] in DEEP]
    assert len({r["scores_file"] for r in deep_rows}) == len(deep_rows)
    assert {p.name for p in (out / "scores").glob("*.npz")} == {"test_pairs.npz"} | {r["scores_file"] for r in deep_rows}


# ---- alignment: pair indices, image ids, labels, scores; validation never mixes with test
def test_saved_pairs_ids_labels_and_scores_are_aligned_and_validation_is_separate(tmp_path, monkeypatch, fake_data):
    from faceeval.models.deep import arcface as arc
    captured: list[np.ndarray] = []
    real_embed = arc._ArcFaceStub.embed

    def spy(self, X):
        out = real_embed(self, X)
        captured.append(out.copy())
        return out

    _force_arcface_stub(monkeypatch)
    monkeypatch.setattr(arc._ArcFaceStub, "embed", spy)
    _, runner, _ = _run(tmp_path, monkeypatch, fake_data, protocol="lfw_fold0_verification", deep=("arcface",))
    out, _, data, _ = _outputs(runner)
    rows = [r for r in data["results"] if r["model"] == "arcface"]
    z = np.load(out / "scores" / "test_pairs.npz")
    ids = [str(i) for i in z["image_ids"]]
    # manifest: partition|identity|image_id
    part, ident = {}, {}
    for line in open(out / "split_manifest_fold0.tsv").read().splitlines()[1:]:
        p, i, im = line.split("|")
        part[im], ident[im] = p, i
    assert {part[i] for i in ids} == {"test"}                                        # test pairs index test images only
    val_ids = {im for im, p in part.items() if p == "validation"}
    assert not val_ids & set(ids)
    assert not {ident[i] for i in val_ids} & {ident[i] for i in ids}                 # identity-disjoint
    for key in ("bal", "full"):
        pr, lab = z[f"{key}_pairs"], z[f"{key}_labels"]
        assert pr.min() >= 0 and pr.max() < len(ids) and len(pr) == len(lab)
        assert all((ident[ids[a]] == ident[ids[b]]) == bool(l) for (a, b), l in zip(pr, lab))   # label == same identity
    r0 = rows[0]
    assert int(z["bal_labels"].sum()) == r0["n_genuine"]
    for r, emb in zip(rows, captured[2:]):                                           # [validation, dim probe, conditions...]
        f = np.load(out / "scores" / r["scores_file"])
        for key in ("bal", "full"):
            a, b = z[f"{key}_pairs"][:, 0], z[f"{key}_pairs"][:, 1]
            ref = np.einsum("ij,ij->i", emb.astype(np.float64)[a], emb.astype(np.float64)[b])
            assert np.allclose(f[f"s_{key}"], ref, rtol=0, atol=1e-6)      # score k belongs to pair k (float rounding only)
            assert not np.allclose(f[f"s_{key}"], ref[::-1], atol=1e-3)     # and the check would notice a shifted order
        assert float(f["threshold"]) == r["threshold"]
    assert len(captured) == 2 + len(rows)


# ------------------------------------------------------------------ revision 3: NaN/Inf, malformed rows, write failures, collisions
def test_sanitize_zeroes_failed_rows_only_and_does_not_copy_when_clean():
    e = np.eye(5, dtype=np.float32)
    e[1] = np.nan; e[2] = np.inf; e[3] = 0.0
    valid = V._valid_rows(e)
    assert valid.tolist() == [True, False, False, False, True]
    clean = V._sanitize(e, valid)
    assert np.isfinite(clean).all() and (clean[1:4] == 0).all()
    assert np.array_equal(clean[[0, 4]], e[[0, 4]]) and np.isnan(e[1]).all()          # input not modified
    ok = np.eye(3, dtype=np.float32)
    assert V._sanitize(ok, V._valid_rows(ok)) is ok
    assert np.isfinite(V._scores(clean, [(0, 1), (0, 2), (0, 4), (4, 4)])).all()


@pytest.mark.parametrize("bad", [np.zeros(8), np.zeros((2, 3, 4))])
def test_malformed_embedding_arrays_are_rejected(bad):
    with pytest.raises(ValueError, match="2-D"):
        V._valid_rows(bad)


def test_nan_inf_and_zero_embeddings_never_reach_scores_or_metrics(tmp_path, monkeypatch, fake_data):
    from faceeval.models.deep import arcface as arc
    real_embed = arc._ArcFaceStub.embed
    kinds = {}

    def corrupt(self, X):
        out = real_embed(self, X).copy()
        for i, x in enumerate(X):
            k = int(np.abs(x).sum() * 137) % 9
            if k == 0: out[i] = np.nan
            elif k == 1: out[i] = np.inf
            elif k == 2: out[i] = 0.0
            kinds[i] = k
        return out

    _force_arcface_stub(monkeypatch)
    monkeypatch.setattr(arc._ArcFaceStub, "embed", corrupt)
    _, runner, _ = _run(tmp_path, monkeypatch, fake_data, protocol="lfw_fold0_verification", deep=("arcface",))
    out, _, data, stream = _outputs(runner)
    assert data["status"] == "COMPLETE" and not data["failures"]      # used to fail: "scores must be finite"
    rows = [r for r in data["results"] if r["model"] == "arcface"]
    pairs = np.load(out / "scores" / "test_pairs.npz")
    for r in rows:
        assert r["n_failed_embeddings"] > 0 and r["val_n_failed_embeddings"] > 0
        assert np.isfinite(r["threshold"]) and np.isfinite(r["auc"]) and 0 <= r["accuracy"] <= 1
        z = np.load(out / "scores" / r["scores_file"])
        assert np.isfinite(z["s_bal"]).all() and np.isfinite(z["s_full"]).all()
        valid = z["valid"]
        assert r["n_failed_embeddings"] == int((~valid).sum())
        touched = ~valid[pairs["bal_pairs"][:, 0]] | ~valid[pairs["bal_pairs"][:, 1]]
        assert r["n_failed_pairs_balanced"] == int(touched.sum()) and (z["s_bal"][touched] == 0).all()
    assert all(np.isfinite(v) for rec in stream if rec["type"] == "row" for k, v in rec.items()
               if k in ("threshold", "auc", "accuracy"))


def test_a_stream_that_cannot_be_written_aborts_the_run_and_still_closes(tmp_path, monkeypatch, fake_data):
    made = _track_streams(monkeypatch)
    real_write = V._RowStream.write
    state = {"n": 0}

    def write(self, kind, payload):
        state["n"] += 1
        if kind in ("row", "failure", "abort"):               # the disk "fills up" after the plan record
            raise OSError("injected: no space left on device")
        return real_write(self, kind, payload)

    _force_arcface_stub(monkeypatch)
    monkeypatch.setattr(V._RowStream, "write", write)
    with pytest.raises((OSError, OrchestratorError)) as err:
        _run(tmp_path, monkeypatch, fake_data, protocol="lfw_fold0_verification", deep=("facenet",))
    assert "no space left" in str(err.value)                  # the original error is not masked by the abort attempt
    assert made[0]._fh.closed and not made[0].terminal
    lines = [json.loads(l) for l in open(next(tmp_path.rglob("verification_rows.jsonl")))]
    assert [r["type"] for r in lines] == ["plan"]             # no end record: the run reads as interrupted
    out = next(tmp_path.rglob("verification_rows.jsonl")).parent
    assert not (out / "scores").exists() or not [p for p in (out / "scores").glob("*.npz")
                                                   if p.name != "test_pairs.npz"]


def test_condition_labels_that_would_collide_are_refused_before_any_work(tmp_path, monkeypatch, fake_data):
    from faceeval.core.types import PerturbationSpec
    made = _track_streams(monkeypatch)
    _force_arcface_stub(monkeypatch)
    # every severity of a type gets the same label, as two severities that round to the same 2 decimals would
    monkeypatch.setattr(PerturbationSpec, "label",
                        property(lambda self: f"{self.perturbation_type.value}_s0.00"))
    with pytest.raises(Exception) as err:
        _run(tmp_path, monkeypatch, fake_data, protocol="lfw_fold0_verification", deep=("facenet",))
    assert "share one label" in str(err.value)
    assert made[0]._fh.closed
    out = next(tmp_path.rglob("verification_rows.jsonl")).parent
    assert [json.loads(l)["type"] for l in open(out / "verification_rows.jsonl")] == ["plan", "abort"]
    assert not (out / "scores").exists()                      # nothing was computed or saved


# ------------------------------------------------------------------ revision 4: finalization consistency
def _is_complete(out):
    """The invariant: COMPLETE only if the JSON says COMPLETE AND the stream ends with an end record saying COMPLETE."""
    data = json.load(open(out / "verification_results.json"))
    recs = [json.loads(l) for l in open(out / "verification_rows.jsonl")]
    return data["status"] == "COMPLETE" and recs[-1]["type"] == "end" and recs[-1]["status"] == "COMPLETE"


def _failing(monkeypatch, kinds, msg):
    real_write = V._RowStream.write

    def write(self, kind, payload):
        if kind in kinds:
            raise OSError(f"{msg} [{kind}]")                # the kind in the text shows which error propagated
        return real_write(self, kind, payload)

    monkeypatch.setattr(V._RowStream, "write", write)


def _go(tmp_path, monkeypatch, fake_data):
    _force_arcface_stub(monkeypatch)
    return _run(tmp_path, monkeypatch, fake_data, protocol="lfw_fold0_verification", deep=("facenet",))


def _out(tmp_path):
    return next(tmp_path.rglob("verification_rows.jsonl")).parent


def test_normal_complete_run_satisfies_the_completion_invariant(clean_run):
    out = clean_run[0]
    assert _is_complete(out) and not list(out.glob("*.tmp"))


def test_a_legitimately_incomplete_run_is_incomplete_in_both_places(tmp_path, monkeypatch, fake_data):
    real = V.perturb_batch

    def flaky(composer, spec, normalizer, X, chips, ids):
        if spec.label == "brightness_s1.00":
            raise RuntimeError("injected failure")
        return real(composer, spec, normalizer, X, chips, ids)

    monkeypatch.setattr(V, "perturb_batch", flaky)
    with pytest.raises(OrchestratorError):
        _go(tmp_path, monkeypatch, fake_data)
    out = _out(tmp_path)
    data = json.load(open(out / "verification_results.json"))
    last = [json.loads(l) for l in open(out / "verification_rows.jsonl")][-1]
    assert data["status"] == "INCOMPLETE" and last["type"] == "end" and last["status"] == "INCOMPLETE"
    assert not _is_complete(out) and not list(out.glob("*.tmp"))


def test_end_record_write_failure_leaves_an_incomplete_summary(tmp_path, monkeypatch, fake_data):
    made = _track_streams(monkeypatch)
    _failing(monkeypatch, {"end"}, "injected: cannot write end record")
    with pytest.raises((OSError, OrchestratorError)) as err:
        _go(tmp_path, monkeypatch, fake_data)
    assert "cannot write end record" in str(err.value)
    out = _out(tmp_path)
    data = json.load(open(out / "verification_results.json"))
    kinds = [json.loads(l)["type"] for l in open(out / "verification_rows.jsonl")]
    assert data["status"] == "INCOMPLETE" and data["n_rows"] == data["n_rows_expected"] > 0
    assert "end" not in kinds and kinds[-1] == "abort" and not _is_complete(out)
    assert made[0]._fh.closed and not list(out.glob("*.tmp"))


def test_end_and_abort_failures_do_not_mask_the_original_error(tmp_path, monkeypatch, fake_data):
    made = _track_streams(monkeypatch)
    _failing(monkeypatch, {"end", "abort"}, "injected: cannot write end record")
    with pytest.raises((OSError, OrchestratorError)) as err:
        _go(tmp_path, monkeypatch, fake_data)
    assert "[end]" in str(err.value) and "[abort]" not in str(err.value)      # the ORIGINAL error, not the abort's
    out = _out(tmp_path)
    kinds = [json.loads(l)["type"] for l in open(out / "verification_rows.jsonl")]
    assert "end" not in kinds and "abort" not in kinds and made[0]._fh.closed
    assert json.load(open(out / "verification_results.json"))["status"] == "INCOMPLETE" and not _is_complete(out)


def test_first_summary_write_failure_never_produces_a_complete_summary(tmp_path, monkeypatch, fake_data):
    real, n = V._write_json_atomic, {"calls": 0}

    def flaky(path, obj):
        n["calls"] += 1
        if n["calls"] == 1:                                    # the INCOMPLETE summary write
            raise OSError("injected: cannot write summary")
        return real(path, obj)

    monkeypatch.setattr(V, "_write_json_atomic", flaky)
    with pytest.raises((OSError, OrchestratorError)) as err:
        _go(tmp_path, monkeypatch, fake_data)
    assert "cannot write summary" in str(err.value)
    out = _out(tmp_path)
    recs = [json.loads(l) for l in open(out / "verification_rows.jsonl")]
    assert "end" not in [r["type"] for r in recs] and recs[-1]["type"] == "abort"
    data = json.load(open(out / "verification_results.json"))      # the wrapper's fallback summary
    assert data["status"] == "INCOMPLETE" and "cannot write summary" in data["error"] and not _is_complete(out)
    assert not list(out.glob("*.tmp"))


def test_summary_and_fallback_write_failures_do_not_mask_the_original_error(tmp_path, monkeypatch, fake_data):
    n = {"calls": 0}

    def broken(path, obj):
        n["calls"] += 1
        raise OSError(f"injected: disk gone #{n['calls']}")

    monkeypatch.setattr(V, "_write_json_atomic", broken)
    with pytest.raises((OSError, OrchestratorError)) as err:
        _go(tmp_path, monkeypatch, fake_data)
    assert "disk gone #1" in str(err.value) and "#2" not in str(err.value) and n["calls"] == 2   # fallback was tried
    out = _out(tmp_path)
    assert not (out / "verification_results.json").exists() and not _is_complete_safe(out)


def _is_complete_safe(out):
    try:
        return _is_complete(out)
    except FileNotFoundError:
        return False


def test_final_complete_summary_replacement_failure_leaves_the_summary_incomplete(tmp_path, monkeypatch, fake_data):
    real, n = V._write_json_atomic, {"calls": 0}

    def flaky(path, obj):
        n["calls"] += 1
        if n["calls"] == 2:                                    # promotion to COMPLETE
            raise OSError("injected: cannot replace summary")
        return real(path, obj)

    monkeypatch.setattr(V, "_write_json_atomic", flaky)
    with pytest.raises((OSError, OrchestratorError)) as err:
        _go(tmp_path, monkeypatch, fake_data)
    assert "cannot replace summary" in str(err.value) and n["calls"] == 2
    out = _out(tmp_path)
    data = json.load(open(out / "verification_results.json"))
    last = [json.loads(l) for l in open(out / "verification_rows.jsonl")][-1]
    assert data["status"] == "INCOMPLETE"                      # never COMPLETE
    assert last["type"] == "end" and last["status"] == "COMPLETE"   # the stream finished; the summary was not promoted
    assert not _is_complete(out) and not list(out.glob("*.tmp"))


def test_write_json_atomic_keeps_the_old_file_and_leaves_no_temp_file_on_failure(tmp_path, monkeypatch):
    target = tmp_path / "s.json"
    V._write_json_atomic(target, {"status": "INCOMPLETE"})
    monkeypatch.setattr(V.os, "replace", lambda a, b: (_ for _ in ()).throw(OSError("injected")))
    with pytest.raises(OSError):
        V._write_json_atomic(target, {"status": "COMPLETE"})
    assert json.load(open(target)) == {"status": "INCOMPLETE"} and [p.name for p in tmp_path.iterdir()] == ["s.json"]


def test_every_stream_record_is_fsynced(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(V.os, "fsync", lambda fd: calls.append(fd))
    st = V._RowStream(tmp_path / "x.jsonl")
    st.write("plan", {}); st.write("end", {"status": "COMPLETE"})
    assert len(calls) == 2 and st.terminal
    st.close()


def test_atomic_savez_removes_its_temporary_file_when_saving_fails(tmp_path, monkeypatch):
    target = tmp_path / "scores.npz"

    def partial_then_fail(file, **arrays):
        open(file, "wb").write(b"partial")
        raise OSError("injected: disk full")

    monkeypatch.setattr(V.np, "savez_compressed", partial_then_fail)
    with pytest.raises(OSError, match="disk full"):
        V._atomic_savez(target, a=np.arange(3))
    assert list(tmp_path.iterdir()) == []                                 # no .tmp.npz, no target


def test_atomic_savez_keeps_an_existing_file_when_saving_fails_and_replaces_it_when_it_works(tmp_path, monkeypatch):
    target = tmp_path / "s.npz"
    V._atomic_savez(target, a=np.arange(3))
    before = target.read_bytes()
    real = V.np.savez_compressed
    monkeypatch.setattr(V.np, "savez_compressed",
                        lambda f, **k: (open(f, "wb").write(b"x"), (_ for _ in ()).throw(OSError("fail")))[1])
    with pytest.raises(OSError):
        V._atomic_savez(target, a=np.arange(5))
    assert target.read_bytes() == before and [p.name for p in tmp_path.iterdir()] == ["s.npz"]
    monkeypatch.setattr(V.np, "savez_compressed", real)
    V._atomic_savez(target, a=np.arange(5))
    assert np.load(target)["a"].tolist() == [0, 1, 2, 3, 4]


# ------------------------------------------------------------------ revision 6: BaseException safety, fsync/rename/cleanup/abort failures
def _interrupt_on_second_deep_row(monkeypatch, exc_type, seen):
    """Raise exc_type inside stream.write("row") for the 2nd deep row. The row's REAL score file already exists
    at that moment (it is written just before the row is recorded); `seen` captures that fact."""
    real_write, n = V._RowStream.write, {"deep": 0}

    def write(self, kind, payload):
        if kind == "row" and payload.get("scores_file"):
            n["deep"] += 1
            if n["deep"] == 2:
                scores = self.path.parent / "scores"
                seen["file"] = payload["scores_file"]
                seen["existed"] = (scores / payload["scores_file"]).exists()
                raise exc_type()
        return real_write(self, kind, payload)

    monkeypatch.setattr(V._RowStream, "write", write)


@pytest.mark.parametrize("exc_type", [KeyboardInterrupt, SystemExit])
def test_interrupt_during_row_recording_removes_only_the_unrecorded_score_file(tmp_path, monkeypatch, fake_data, exc_type):
    made = _track_streams(monkeypatch)
    seen = {}
    _interrupt_on_second_deep_row(monkeypatch, exc_type, seen)
    with pytest.raises(exc_type):                                   # the original interrupt propagates, unchanged
        _go(tmp_path, monkeypatch, fake_data)
    assert seen["existed"] is True                                  # the real failure sequence: file existed first
    out = _out(tmp_path)
    recs = [json.loads(l) for l in open(out / "verification_rows.jsonl")]
    kinds = [r["type"] for r in recs]
    rows = [r for r in recs if r["type"] == "row"]
    deep_rows = [r for r in rows if r.get("scores_file")]
    assert kinds[0] == "plan" and kinds[-1] == "abort" and "end" not in kinds and made[0].terminal
    assert exc_type.__name__ in recs[-1]["error"] or exc_type is SystemExit
    assert len(rows) == 8 + 1 and len(deep_rows) == 1               # earlier rows (8 classical + 1 deep) intact
    on_disk = {p.name for p in (out / "scores").glob("*.npz")}
    assert on_disk == {"test_pairs.npz", deep_rows[0]["scores_file"]}      # the recorded file stays...
    assert seen["file"] not in on_disk                                       # ...the unrecorded one is gone
    assert not list((out / "scores").glob("*.tmp.npz")) and made[0]._fh.closed
    assert not _is_complete_safe(out) and not (out / "verification_results.json").exists()


def test_a_failed_cleanup_never_masks_the_interrupt(tmp_path, monkeypatch, fake_data):
    import pathlib
    real_unlink = pathlib.Path.unlink

    def unlink(self, *a, **k):
        if self.name.endswith(".npz") and not self.name.endswith(".tmp.npz"):
            raise OSError("injected: cannot remove score file")
        return real_unlink(self, *a, **k)

    seen = {}
    monkeypatch.setattr(pathlib.Path, "unlink", unlink)
    _interrupt_on_second_deep_row(monkeypatch, KeyboardInterrupt, seen)
    with pytest.raises(KeyboardInterrupt):                          # not the OSError from the cleanup
        _go(tmp_path, monkeypatch, fake_data)
    out = _out(tmp_path)
    recs = [json.loads(l) for l in open(out / "verification_rows.jsonl")]
    assert recs[-1]["type"] == "abort" and "end" not in [r["type"] for r in recs]
    recorded = {r["scores_file"] for r in recs if r["type"] == "row" and r.get("scores_file")}
    orphans = {p.name for p in (out / "scores").glob("*.npz")} - recorded - {"test_pairs.npz"}
    assert orphans == {seen["file"]}                                # the leftover is detectable, never silent


def test_abort_swallows_ordinary_errors_but_never_interrupts(tmp_path):
    st = V._RowStream(tmp_path / "a.jsonl")

    def boom(*a, **k):
        raise OSError("injected")

    st.write = boom
    assert st.abort(RuntimeError("original")) is None               # ordinary failure: swallowed

    def ctrl_c(*a, **k):
        raise KeyboardInterrupt()

    st.write = ctrl_c
    with pytest.raises(KeyboardInterrupt):                           # a second interrupt is not hidden
        st.abort(RuntimeError("original"))
    st.close()


def test_fsync_failure_on_the_end_record_leaves_the_run_incomplete(tmp_path, monkeypatch, fake_data):
    real_write, real_fsync, state = V._RowStream.write, V.os.fsync, {"in_end": False}

    def write(self, kind, payload):
        state["in_end"] = kind == "end"
        try:
            return real_write(self, kind, payload)
        finally:
            state["in_end"] = False

    def fsync(fd):
        if state["in_end"]:
            raise OSError("injected: fsync failed on end record")
        return real_fsync(fd)

    monkeypatch.setattr(V._RowStream, "write", write)
    monkeypatch.setattr(V.os, "fsync", fsync)
    with pytest.raises((OSError, OrchestratorError)) as err:
        _go(tmp_path, monkeypatch, fake_data)
    assert "fsync failed on end record" in str(err.value)
    out = _out(tmp_path)
    assert json.load(open(out / "verification_results.json"))["status"] == "INCOMPLETE"
    assert not _is_complete(out)                                     # never COMPLETE without a fsynced end record
    assert [json.loads(l) for l in open(out / "verification_rows.jsonl")][-1]["type"] == "abort"


def test_rename_failure_when_promoting_the_summary_leaves_it_incomplete(tmp_path, monkeypatch, fake_data):
    real_replace, n = V.os.replace, {"summary": 0}

    def replace(src, dst):
        if str(dst).endswith("verification_results.json"):
            n["summary"] += 1
            if n["summary"] == 2:                                    # the promotion to COMPLETE
                raise OSError("injected: rename failed")
        return real_replace(src, dst)

    monkeypatch.setattr(V.os, "replace", replace)
    with pytest.raises((OSError, OrchestratorError)) as err:
        _go(tmp_path, monkeypatch, fake_data)
    assert "rename failed" in str(err.value)
    out = _out(tmp_path)
    last = [json.loads(l) for l in open(out / "verification_rows.jsonl")][-1]
    assert json.load(open(out / "verification_results.json"))["status"] == "INCOMPLETE"
    assert last["type"] == "end" and last["status"] == "COMPLETE" and not _is_complete(out)
    assert not list(out.glob("*.tmp"))


# ------------------------------------------------------------------ revision 6b: the stream file decides what is recorded
def _assert_consistent(out):
    """Every stream line parses; every row record's score file exists (no dangling reference); nothing temporary."""
    recs = [json.loads(l) for l in open(out / "verification_rows.jsonl")]       # a torn line would raise here
    for r in recs:
        if r["type"] == "row" and r.get("scores_file"):
            assert (out / "scores" / r["scores_file"]).exists(), f"dangling reference to {r['scores_file']}"
    assert not list(out.rglob("*.tmp")) and not list(out.rglob("*.tmp.npz"))
    return recs


def _arm_second_deep_row(monkeypatch):
    """Returns the state dict; ``arm`` is True only while the 2nd deep row's record is being written."""
    real_write, st = V._RowStream.write, {"n": 0, "arm": False}

    def write(self, kind, payload):
        st["arm"] = bool(kind == "row" and payload.get("scores_file") and (st.__setitem__("n", st["n"] + 1) or st["n"] == 2))
        try:
            return real_write(self, kind, payload)
        finally:
            st["arm"] = False

    monkeypatch.setattr(V._RowStream, "write", write)
    return st


def _check_second_row_committed(tmp_path, made):
    out = _out(tmp_path)
    recs = _assert_consistent(out)
    kinds = [r["type"] for r in recs]
    deep = [r for r in recs if r["type"] == "row" and r.get("scores_file")]
    assert len(deep) == 2 and kinds.count("row") == 8 + 2                    # both deep rows are in the stream
    assert kinds[-1] == "abort" and "end" not in kinds and made[0].terminal
    assert {p.name for p in (out / "scores").glob("*.npz")} == {"test_pairs.npz"} | {r["scores_file"] for r in deep}
    assert not _is_complete_safe(out) and not (out / "verification_results.json").exists()
    return deep


@pytest.mark.parametrize("exc_type", [KeyboardInterrupt, SystemExit])
def test_interrupt_right_after_the_real_fsync_keeps_the_committed_row_and_its_file(tmp_path, monkeypatch, fake_data, exc_type):
    made = _track_streams(monkeypatch)
    st = _arm_second_deep_row(monkeypatch)
    real_fsync = V.os.fsync

    def fsync(fd):
        real_fsync(fd)                                   # the real fsync SUCCEEDS, then the interrupt arrives
        if st["arm"]:
            st["arm"] = False
            raise exc_type()

    monkeypatch.setattr(V.os, "fsync", fsync)
    with pytest.raises(exc_type):
        _go(tmp_path, monkeypatch, fake_data)
    _check_second_row_committed(tmp_path, made)


@pytest.mark.parametrize("exc_type", [KeyboardInterrupt, SystemExit])
def test_interrupt_right_after_write_returns_before_the_caller_updates_its_state(tmp_path, monkeypatch, fake_data, exc_type):
    made = _track_streams(monkeypatch)
    real_write, n = V._RowStream.write, {"deep": 0}

    def write(self, kind, payload):
        real_write(self, kind, payload)                  # the real write fully completes and returns...
        if kind == "row" and payload.get("scores_file"):
            n["deep"] += 1
            if n["deep"] == 2:
                raise exc_type()                         # ...and the interrupt hits before the caller continues

    monkeypatch.setattr(V._RowStream, "write", write)
    with pytest.raises(exc_type):
        _go(tmp_path, monkeypatch, fake_data)
    _check_second_row_committed(tmp_path, made)


def test_fsync_failure_on_a_row_rolls_the_record_back_and_removes_its_file(tmp_path, monkeypatch, fake_data):
    st = _arm_second_deep_row(monkeypatch)
    real_fsync = V.os.fsync

    def fsync(fd):
        if st["arm"]:
            st["arm"] = False
            raise OSError("injected: fsync failed on row")     # the line WAS written, but is not durable
        return real_fsync(fd)

    monkeypatch.setattr(V.os, "fsync", fsync)
    with pytest.raises(OrchestratorError):
        _go(tmp_path, monkeypatch, fake_data)
    out = _out(tmp_path)
    recs = _assert_consistent(out)
    deep = [r for r in recs if r["type"] == "row" and r.get("scores_file")]
    fails = [r for r in recs if r["type"] == "failure"]
    assert len(deep) == 3 and len(fails) == 1 and "fsync failed on row" in fails[0]["error"]   # 4 conditions: 1 lost
    assert fails[0]["stage"] not in [r["label"] for r in deep]                                 # rolled back, not half-there
    assert {p.name for p in (out / "scores").glob("*.npz")} == {"test_pairs.npz"} | {r["scores_file"] for r in deep}
    assert recs[-1]["type"] == "end" and recs[-1]["status"] == "INCOMPLETE" and not _is_complete(out)


# ---- unit level: the stream's own commit/rollback rules
def _lines(path):
    return [json.loads(l) for l in open(path)]


def _torn_then(exc):
    def make(st):
        real = st._fh.write

        def write(b):
            real(bytes(b)[:11])                             # only part of the line reaches the file
            raise exc
        return write
    return make


@pytest.mark.parametrize("exc", [KeyboardInterrupt(), SystemExit(), OSError("injected: disk full")])
def test_a_torn_record_is_rolled_back_so_it_cannot_be_mistaken_for_a_row(tmp_path, exc):
    st = V._RowStream(tmp_path / "s.jsonl")
    st.write("plan", {"k": 1})
    before = (tmp_path / "s.jsonl").read_bytes()
    st._fh.write = _torn_then(exc)(st)
    with pytest.raises(type(exc)):
        st.write("row", {"scores_file": "a.npz"})
    assert (tmp_path / "s.jsonl").read_bytes() == before and not st.has_row("a.npz")   # nothing half there
    del st._fh.write
    st.write("abort", {"error": "x"})                         # the stream still works and stays line-aligned
    assert [r["type"] for r in _lines(tmp_path / "s.jsonl")] == ["plan", "abort"]
    st.close()


@pytest.mark.parametrize("exc_type", [KeyboardInterrupt, SystemExit])
def test_a_complete_record_survives_an_interrupt_and_counts_as_recorded(tmp_path, exc_type):
    st = V._RowStream(tmp_path / "s.jsonl")
    real = st._fh.write

    def write(b):
        n = real(b)
        raise exc_type()                                       # whole line written, then interrupted

    st._fh.write = write
    with pytest.raises(exc_type):
        st.write("row", {"scores_file": "a.npz"})
    del st._fh.write
    assert st.has_row("a.npz") and [r["type"] for r in _lines(tmp_path / "s.jsonl")] == ["row"]
    st.close()


def test_a_failed_fsync_after_an_ordinary_error_removes_the_whole_record(tmp_path, monkeypatch):
    st = V._RowStream(tmp_path / "s.jsonl")
    st.write("plan", {})
    before = (tmp_path / "s.jsonl").read_bytes()
    monkeypatch.setattr(V.os, "fsync", lambda fd: (_ for _ in ()).throw(OSError("injected")))
    with pytest.raises(OSError):
        st.write("row", {"scores_file": "a.npz"})
    monkeypatch.undo()
    assert (tmp_path / "s.jsonl").read_bytes() == before and not st.has_row("a.npz")
    st.close()


def test_has_row_ignores_torn_lines_and_keeps_files_when_the_stream_is_unreadable(tmp_path, monkeypatch):
    path = tmp_path / "s.jsonl"
    st = V._RowStream(path)
    st.write("row", {"scores_file": "a.npz"})
    with open(path, "ab") as fh:
        fh.write(b'{"type": "row", "scores_file": "b.npz"')       # torn tail, no newline
    assert st.has_row("a.npz") and not st.has_row("b.npz") and not st.has_row("c.npz")
    real_open = open
    monkeypatch.setattr("builtins.open", lambda *a, **k: (_ for _ in ()).throw(OSError("injected"))
                        if str(a[0]) == str(path) and "r" in (a[1] if len(a) > 1 else "r") else real_open(*a, **k))
    assert st.has_row("anything.npz") is True                      # cannot tell -> keep the file
    monkeypatch.undo()
    st.close()


# ------------------------------------------------------------------ focused corrections: cleanup masking, rollback failure,
# ------------------------------------------------------------------ invalid write counts, directory fsync
def _temp_unlink_fails(monkeypatch):
    import pathlib
    real = pathlib.Path.unlink

    def unlink(self, *a, **k):
        if self.name.endswith(".tmp") or self.name.endswith(".tmp.npz"):
            raise OSError("injected: cannot unlink temp file")
        return real(self, *a, **k)

    monkeypatch.setattr(pathlib.Path, "unlink", unlink)


@pytest.mark.parametrize("original", [OSError("original save error"), KeyboardInterrupt()])
def test_atomic_savez_cleanup_failure_does_not_mask_the_original_error(tmp_path, monkeypatch, caplog, original):
    def partial_then_fail(file, **arrays):
        open(file, "wb").write(b"partial")
        raise original

    monkeypatch.setattr(V.np, "savez_compressed", partial_then_fail)
    _temp_unlink_fails(monkeypatch)
    with caplog.at_level("ERROR"):
        with pytest.raises(type(original)) as err:
            V._atomic_savez(tmp_path / "s.npz", a=np.arange(3))
    assert err.value is original and "cannot unlink" not in str(err.value)
    assert any("Could not remove the temporary score file" in r.message for r in caplog.records)   # logged, not raised
    assert (tmp_path / "s.tmp.npz").exists()                      # honest: the leftover is reported, not hidden


def test_write_json_atomic_cleanup_failure_does_not_mask_the_original_error(tmp_path, monkeypatch, caplog):
    target = tmp_path / "s.json"
    V._write_json_atomic(target, {"status": "INCOMPLETE"})
    monkeypatch.setattr(V.os, "replace", lambda a, b: (_ for _ in ()).throw(OSError("original replace error")))
    _temp_unlink_fails(monkeypatch)
    with caplog.at_level("ERROR"):
        with pytest.raises(OSError, match="original replace error"):
            V._write_json_atomic(target, {"status": "COMPLETE"})
    assert any("Could not remove the temporary summary file" in r.message for r in caplog.records)
    assert json.load(open(target)) == {"status": "INCOMPLETE"}      # the old summary is untouched


@pytest.mark.parametrize("exc", [OSError("injected: disk full"), KeyboardInterrupt()])
def test_a_failed_rollback_closes_the_stream_to_further_records(tmp_path, monkeypatch, caplog, exc):
    path = tmp_path / "s.jsonl"
    st = V._RowStream(path)
    st.write("plan", {})
    before = path.read_bytes()
    st._fh.write = _torn_then(exc)(st)
    monkeypatch.setattr(V.os, "ftruncate", lambda fd, n: (_ for _ in ()).throw(OSError("injected: ftruncate failed")))
    with caplog.at_level("ERROR"):
        with pytest.raises(type(exc)) as err:
            st.write("row", {"scores_file": "a.npz"})
    assert err.value is exc and st.broken                           # the ORIGINAL error, not the ftruncate one
    torn = path.read_bytes()
    assert torn != before and torn.startswith(before)               # honest: the torn bytes are still there
    assert any("Could not roll back" in r.message for r in caplog.records)
    del st._fh.write
    with pytest.raises(OSError, match="unusable"):                  # nothing may be appended behind the torn line
        st.write("abort", {"error": "x"})
    assert st.abort(RuntimeError("x")) is None and path.read_bytes() == torn and not st.terminal
    assert not st.has_row("a.npz")
    st.close()


def test_an_unverifiable_stream_state_after_an_interrupt_also_closes_the_stream(tmp_path, monkeypatch):
    path = tmp_path / "s.jsonl"
    st = V._RowStream(path)
    real_fstat, n = V.os.fstat, {"calls": 0}

    def fstat(fd):
        n["calls"] += 1
        if n["calls"] >= 2:                                         # the 1st call (start offset) works
            raise OSError("injected: fstat failed")
        return real_fstat(fd)

    st._fh.write = _torn_then(KeyboardInterrupt())(st)
    monkeypatch.setattr(V.os, "fstat", fstat)
    with pytest.raises(KeyboardInterrupt):
        st.write("row", {"scores_file": "a.npz"})
    assert st.broken
    monkeypatch.undo()
    del st._fh.write
    size = path.stat().st_size
    with pytest.raises(OSError, match="unusable"):
        st.write("abort", {})
    assert path.stat().st_size == size
    st.close()


@pytest.mark.parametrize("bad", [0, None, -1, "too_big"])
@pytest.mark.parametrize("progress_first", [False, True])
def test_invalid_write_counts_fail_fast_and_leave_no_record(tmp_path, bad, progress_first):
    path = tmp_path / "s.jsonl"
    st = V._RowStream(path)
    st.write("plan", {})
    before = path.read_bytes()
    real, calls = st._fh.write, {"n": 0}

    def write(b):
        calls["n"] += 1
        assert calls["n"] < 50, "runaway write loop"
        if progress_first and calls["n"] == 1:
            real(bytes(b)[:5]); return 5                             # some bytes land, then an invalid count
        return len(b) + 7 if bad == "too_big" else bad

    st._fh.write = write
    with pytest.raises(OSError, match="unbuffered write returned"):
        st.write("row", {"scores_file": "a.npz"})
    assert calls["n"] <= 2 and path.read_bytes() == before and not st.broken     # fully rolled back
    del st._fh.write
    st.write("abort", {})                                            # the stream is still usable and line-aligned
    assert [r["type"] for r in _lines(path)] == ["plan", "abort"]
    st.close()


def test_legitimate_short_writes_still_complete_the_record(tmp_path):
    path = tmp_path / "s.jsonl"
    st = V._RowStream(path)
    real = st._fh.write
    st._fh.write = lambda b: real(bytes(b)[:3])                      # the OS accepts 3 bytes at a time
    st.write("row", {"scores_file": "a.npz", "pad": "x" * 40})
    del st._fh.write
    assert st.has_row("a.npz") and _lines(path)[0]["pad"] == "x" * 40
    st.close()


def _record_events(monkeypatch):
    import stat
    events = []
    real_fsync, real_replace = V.os.fsync, V.os.replace

    def fsync(fd):
        events.append("fsync-dir" if stat.S_ISDIR(os.fstat(fd).st_mode) else "fsync-file")
        return real_fsync(fd)

    def replace(a, b):
        events.append("replace")
        return real_replace(a, b)

    monkeypatch.setattr(V.os, "fsync", fsync)
    monkeypatch.setattr(V.os, "replace", replace)
    return events


@pytest.mark.skipif(os.name != "posix", reason="directory fsync is POSIX only")
def test_summary_replacement_fsyncs_the_file_then_replaces_then_fsyncs_the_directory(tmp_path, monkeypatch):
    events = _record_events(monkeypatch)
    V._write_json_atomic(tmp_path / "s.json", {"status": "COMPLETE"})
    assert events == ["fsync-file", "replace", "fsync-dir"]


@pytest.mark.skipif(os.name != "posix", reason="directory fsync is POSIX only")
def test_a_directory_fsync_failure_is_logged_not_raised(tmp_path, monkeypatch, caplog):
    import stat
    real_fsync = V.os.fsync

    def fsync(fd):
        if stat.S_ISDIR(os.fstat(fd).st_mode):
            raise OSError("injected: directory fsync unsupported")
        return real_fsync(fd)

    monkeypatch.setattr(V.os, "fsync", fsync)
    with caplog.at_level("WARNING"):
        V._write_json_atomic(tmp_path / "s.json", {"status": "COMPLETE"})   # does not raise
    assert json.load(open(tmp_path / "s.json")) == {"status": "COMPLETE"} and not list(tmp_path.glob("*.tmp"))
    assert any("Could not fsync directory" in r.message and "may not survive" in r.message for r in caplog.records)


def test_no_directory_fsync_off_posix_or_when_the_replace_failed(tmp_path, monkeypatch):
    events = _record_events(monkeypatch)
    monkeypatch.setattr(V, "_POSIX", False)
    V._write_json_atomic(tmp_path / "a.json", {"k": 1})
    assert events == ["fsync-file", "replace"] and V._fsync_dir(tmp_path) is False
    monkeypatch.setattr(V, "_POSIX", True)
    events.clear()
    monkeypatch.setattr(V.os, "replace", lambda a, b: (_ for _ in ()).throw(OSError("injected")))
    with pytest.raises(OSError):
        V._write_json_atomic(tmp_path / "a.json", {"k": 2})
    assert "fsync-dir" not in events and json.load(open(tmp_path / "a.json")) == {"k": 1}


# ------------------------------------------------------------------ final closure: NPZ durability, unit-norm warning scope
@pytest.mark.skipif(os.name != "posix", reason="directory fsync is POSIX only")
def test_atomic_savez_fsyncs_the_file_then_replaces_then_fsyncs_the_directory(tmp_path, monkeypatch):
    events = _record_events(monkeypatch)
    V._atomic_savez(tmp_path / "s.npz", a=np.arange(3))
    assert events == ["fsync-file", "replace", "fsync-dir"]
    assert np.load(tmp_path / "s.npz")["a"].tolist() == [0, 1, 2] and [p.name for p in tmp_path.iterdir()] == ["s.npz"]


@pytest.mark.parametrize("exc", [OSError("injected: file fsync failed"), KeyboardInterrupt()])
@pytest.mark.parametrize("cleanup_fails", [False, True])
def test_a_failed_npz_fsync_publishes_nothing_and_keeps_the_original_error(tmp_path, monkeypatch, exc, cleanup_fails):
    import stat
    target = tmp_path / "s.npz"
    V._atomic_savez(target, a=np.arange(3))
    before = target.read_bytes()
    real_fsync, replaced = V.os.fsync, []

    def fsync(fd):
        if not stat.S_ISDIR(os.fstat(fd).st_mode):
            raise exc
        return real_fsync(fd)

    monkeypatch.setattr(V.os, "fsync", fsync)
    real_replace = V.os.replace
    monkeypatch.setattr(V.os, "replace", lambda a, b: (replaced.append(b), real_replace(a, b))[1])
    if cleanup_fails:
        _temp_unlink_fails(monkeypatch)
    with pytest.raises(type(exc)) as err:
        V._atomic_savez(target, a=np.arange(5))
    assert err.value is exc and not replaced                         # never replaced: nothing non-durable was published
    assert target.read_bytes() == before                             # the old file is untouched
    assert (tmp_path / "s.tmp.npz").exists() is cleanup_fails        # temp removed, or (honestly) left when removal fails


@pytest.mark.skipif(os.name != "posix", reason="directory fsync is POSIX only")
def test_an_npz_directory_fsync_failure_is_logged_not_raised(tmp_path, monkeypatch, caplog):
    import stat
    real_fsync = V.os.fsync

    def fsync(fd):
        if stat.S_ISDIR(os.fstat(fd).st_mode):
            raise OSError("injected: directory fsync unsupported")
        return real_fsync(fd)

    monkeypatch.setattr(V.os, "fsync", fsync)
    with caplog.at_level("WARNING"):
        V._atomic_savez(tmp_path / "s.npz", a=np.arange(3))          # does not raise
    assert np.load(tmp_path / "s.npz")["a"].tolist() == [0, 1, 2] and [p.name for p in tmp_path.iterdir()] == ["s.npz"]
    assert any("Could not fsync directory" in r.message for r in caplog.records)


def test_the_unit_norm_warning_cannot_fire_for_classical_style_embeddings(caplog):
    """run_fold0_verification passes every model's output through ``_l2`` before ``_warn_if_not_unit``, so a
    classical descriptor of any scale (tiny, huge, all-zero, NaN) is either unit-norm, or flagged invalid and
    excluded from the check. No misleading warning, and no change to how classical embeddings are scored."""
    rng = np.random.default_rng(7)
    raw = np.abs(rng.standard_normal((50, 64))) * np.logspace(-8, 6, 50)[:, None]   # HOG/LBP-like, non-negative
    raw[3] = 0.0
    raw[4, 2] = np.nan
    emb = V._l2(raw)
    valid = V._valid_rows(emb)
    assert not valid[3] and not valid[4] and valid.sum() == 48
    with caplog.at_level("WARNING"):
        V._warn_if_not_unit(emb, valid, "hog_svm")
    assert not [r for r in caplog.records if "not unit-norm" in r.message]


# ------------------------------------------------------------------ terminal-record race: an abort never follows a committed end/abort
@pytest.mark.parametrize("exc_type", [KeyboardInterrupt, SystemExit])
def test_interrupt_after_the_end_record_is_durable_gets_no_abort_behind_it(tmp_path, monkeypatch, fake_data, exc_type):
    made = _track_streams(monkeypatch)
    real_write, real_fsync, st = V._RowStream.write, V.os.fsync, {"in_end": False}

    def write(self, kind, payload):
        st["in_end"] = kind == "end"
        try:
            return real_write(self, kind, payload)
        finally:
            st["in_end"] = False

    def fsync(fd):
        real_fsync(fd)                                   # the end record IS durable...
        if st["in_end"]:
            st["in_end"] = False
            raise exc_type()                             # ...and the interrupt arrives before terminal is set

    monkeypatch.setattr(V._RowStream, "write", write)
    monkeypatch.setattr(V.os, "fsync", fsync)
    with pytest.raises(exc_type):                        # the original interrupt propagates, unchanged
        _go(tmp_path, monkeypatch, fake_data)
    out = _out(tmp_path)
    kinds = [json.loads(l)["type"] for l in open(out / "verification_rows.jsonl")]
    assert kinds.count("end") == 1 and kinds[-1] == "end" and "abort" not in kinds   # exactly one terminal record
    assert made[0]._fh.closed and made[0].terminal
    assert json.load(open(out / "verification_results.json"))["status"] == "INCOMPLETE" and not _is_complete(out)


@pytest.mark.parametrize("kind", ["end", "abort"])
def test_abort_never_appends_behind_a_committed_terminal_record(tmp_path, kind):
    path = tmp_path / "s.jsonl"
    st = V._RowStream(path)
    st.write("plan", {})
    st.write(kind, {"status": "COMPLETE"} if kind == "end" else {"error": "first"})
    st.terminal = False                                  # as if the flag was lost to an interrupt
    before = path.read_bytes()
    assert st.abort(RuntimeError("late")) is None
    assert path.read_bytes() == before and st.terminal   # nothing appended; the flag is repaired from the file
    st.close()


def test_abort_appends_nothing_when_the_stream_cannot_be_read(tmp_path, monkeypatch):
    path = tmp_path / "s.jsonl"
    st = V._RowStream(path)
    st.write("plan", {})
    before = path.read_bytes()
    real_open = open
    monkeypatch.setattr("builtins.open", lambda *a, **k: (_ for _ in ()).throw(OSError("injected"))
                        if str(a[0]) == str(path) and "r" in (a[1] if len(a) > 1 else "r") else real_open(*a, **k))
    assert st.abort(RuntimeError("x")) is None
    monkeypatch.undo()
    assert path.read_bytes() == before                   # cannot tell whether a terminal record exists: append none
    st.close()


# ---- a terminal record whose newline is missing
@pytest.mark.parametrize("kind", ["end", "abort"])
def test_abort_never_appends_behind_an_unterminated_terminal_record(tmp_path, kind):
    path = tmp_path / "s.jsonl"
    st = V._RowStream(path)
    st.write("plan", {})
    with open(path, "ab") as fh:                         # a complete JSON object, newline missing (e.g. rollback failed)
        fh.write(json.dumps({"type": kind, "status": "COMPLETE"}).encode())
    before = path.read_bytes()
    assert not before.endswith(b"\n") and st.has_terminal()
    assert st.abort(RuntimeError("late")) is None
    assert path.read_bytes() == before and st.terminal   # appending would have merged two records into one bad line
    st.close()


def test_a_torn_terminal_tail_is_not_a_terminal_record(tmp_path):
    path = tmp_path / "s.jsonl"
    st = V._RowStream(path)
    st.write("plan", {})
    with open(path, "ab") as fh:
        fh.write(b'{"type": "end", "status": "COMPL')    # unparseable: not a record
    assert not st.has_terminal()
    st.close()


@pytest.mark.parametrize("exc_type", [KeyboardInterrupt, SystemExit])
def test_a_terminal_line_missing_only_its_newline_is_rolled_back_so_abort_can_still_run(tmp_path, exc_type):
    path = tmp_path / "s.jsonl"
    st = V._RowStream(path)
    st.write("plan", {})
    real = st._fh.write

    def write(b):
        real(bytes(b)[:-1])                              # everything but the final newline reaches the file
        raise exc_type()

    st._fh.write = write
    with pytest.raises(exc_type):
        st.write("end", {"status": "COMPLETE"})
    del st._fh.write
    assert not st.terminal and not st.has_terminal()     # the unterminated line was removed, not kept as a record
    st.abort(exc_type())
    raw = path.read_bytes()
    assert raw.endswith(b"\n") and [r["type"] for r in _lines(path)] == ["plan", "abort"]
    st.close()


# ------------------------------------------------------------------ final: nothing may be written after a terminal record
@pytest.mark.parametrize("terminal", ["end", "abort"])
@pytest.mark.parametrize("later", ["row", "failure", "end", "abort", "plan"])
def test_write_after_a_terminal_record_is_refused_and_changes_nothing(tmp_path, terminal, later):
    path = tmp_path / "s.jsonl"
    st = V._RowStream(path)
    st.write("plan", {})
    st.write(terminal, {"status": "COMPLETE"} if terminal == "end" else {"error": "first"})
    before = path.read_bytes()
    with pytest.raises(OSError, match="terminal record"):
        st.write(later, {"scores_file": "a.npz"})
    assert path.read_bytes() == before and st.terminal and not st.broken
    st.close()


@pytest.mark.parametrize("terminal", ["end", "abort"])
def test_write_after_a_committed_terminal_is_refused_even_if_the_flag_was_lost(tmp_path, terminal):
    path = tmp_path / "s.jsonl"
    st = V._RowStream(path)
    st.write("plan", {})
    st.write(terminal, {"status": "COMPLETE"} if terminal == "end" else {"error": "first"})
    st.terminal = False                                  # as if an interrupt lost the flag
    before = path.read_bytes()
    with pytest.raises(OSError, match="terminal record"):
        st.write("row", {"scores_file": "a.npz"})
    assert path.read_bytes() == before and st.terminal   # the flag is repaired from the file
    st.close()


@pytest.mark.parametrize("terminal", ["end", "abort"])
@pytest.mark.parametrize("exc_type", [KeyboardInterrupt, SystemExit])
def test_an_interrupt_after_a_whole_terminal_line_is_written_keeps_it_and_sets_the_flag(tmp_path, terminal, exc_type):
    path = tmp_path / "s.jsonl"
    st = V._RowStream(path)
    st.write("plan", {})
    real = st._fh.write

    def write(b):
        real(b)
        raise exc_type()                                 # the whole line (with newline) is in the file, then the interrupt

    st._fh.write = write
    with pytest.raises(exc_type):
        st.write(terminal, {"status": "COMPLETE"} if terminal == "end" else {"error": "x"})
    del st._fh.write
    assert st.terminal and [r["type"] for r in _lines(path)] == ["plan", terminal]
    before = path.read_bytes()
    assert st.abort(RuntimeError("late")) is None        # abort recovery adds no second terminal record
    with pytest.raises(OSError, match="terminal record"):
        st.write("row", {})
    assert path.read_bytes() == before
    st.close()


def test_a_run_whose_stream_is_closed_by_a_terminal_record_keeps_exactly_one_terminal(clean_run):
    _, _, _, stream = clean_run
    assert [r["type"] for r in stream].count("end") == 1 and not [r for r in stream if r["type"] == "abort"]
    assert stream[-1]["type"] == "end"

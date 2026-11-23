"""
API tests for the read-only dashboard backend.

They use a synthetic results database built in a temp directory (never
``experiments/results.db``) and no model inference.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from pathlib import Path

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient  # noqa: E402

from api.dependencies import get_settings  # noqa: E402
from api.main import app  # noqa: E402
from faceeval.storage.result_store import ResultStore  # noqa: E402

RUN = "abc12345"


def _spec(ptype, sev):
    return {"perturbation_type": ptype, "severity": sev}


def _run_payload():
    return {
        "run_id": RUN, "status": "completed",
        "config": {"experiment_name": "t", "traditional_models": ["lbph"]},
        "evaluation_results": [
            {"model_name": "lbph", "model_paradigm": "traditional", "dataset_name": "d",
             "perturbation_spec": _spec("blur", 0.5), "accuracy": 0.6, "auc": 0.7, "num_samples": 10},
            {"model_name": "lbph", "model_paradigm": "traditional", "dataset_name": "d",
             "perturbation_spec": _spec("blur", 0.0), "accuracy": 0.9},
            # missing accuracy and missing severity must stay null / be dropped, never 0
            {"model_name": "lbph", "model_paradigm": "traditional", "dataset_name": "d",
             "perturbation_spec": _spec("blur", None)},
            {"model_name": "lbph", "model_paradigm": "traditional", "dataset_name": "d"},
            "not-a-dict",
        ],
        "fingerprints": [
            {"model_name": "lbph", "paradigm": "traditional",
             "dimension_names": ["a", "b"], "vector": [0.1, 0.2], "scores": {"a": 0.1, "b": 0.2}},
            {"model_name": "eig", "paradigm": "traditional",
             "dimension_names": ["a", "b"], "vector": [0.3, 0.4], "scores": {"a": 0.3, "b": 0.4}},
            {"model_name": "broken", "dimension_names": ["a"], "vector": []},
        ],
        "deployment_rankings": [
            {"scenario": "mobile", "weights": {"accuracy": 1.0},
             "ranking": [{"model_name": "lbph", "paradigm": "traditional", "rank": None,
                          "total_score": 0.5, "component_scores": {"accuracy": 0.5},
                          "weighted_components": {"accuracy": 0.5}}]},
            {"scenario": "malformed"},
        ],
        "started_at": "2026-01-01T00:00:00", "completed_at": None,
        "duration_seconds": None, "error_message": "",
    }


@pytest.fixture
def env(tmp_path, monkeypatch):
    db = tmp_path / "results.db"
    ResultStore(str(db)).close()                      # creates the schema only
    conn = sqlite3.connect(db)
    conn.execute(
        "INSERT INTO runs (run_id, experiment_name, status, config_hash, started_at, completed_at, data) "
        "VALUES (?,?,?,?,?,?,?)",
        (RUN, "t", "completed", "h", "2026-01-01T00:00:00", None, json.dumps(_run_payload())),
    )
    conn.execute(
        "INSERT INTO runs (run_id, experiment_name, status, config_hash, started_at, completed_at, data) "
        "VALUES (?,?,?,?,?,?,?)",
        ("bad00001", "t", "completed", "h2", "2026-01-02T00:00:00", None, "{not json"),
    )
    conn.commit()
    conn.close()

    figs, reps = tmp_path / "figures", tmp_path / "reports"
    (figs / RUN).mkdir(parents=True)
    (figs / RUN / "adc_curve.png").write_bytes(b"\x89PNG fake")
    (figs / RUN / "adc_curve.svg").write_text("<svg/>")
    (figs / "notes.txt").write_text("secret")
    (tmp_path / "outside.png").write_bytes(b"x")
    reps.mkdir()
    (reps / f"summary_{RUN}.csv").write_text("a,b\n1,2\n")
    (reps / f"page_{RUN}.html").write_text("<script>1</script>")
    (reps / "other_run.csv").write_text("x\n")
    (tmp_path / "secret.csv").write_text("s\n")

    monkeypatch.setenv("FACEEVAL_DB_PATH", str(db))
    monkeypatch.setenv("FACEEVAL_FIGURES_DIR", str(figs))
    monkeypatch.setenv("FACEEVAL_REPORTS_DIR", str(reps))
    get_settings.cache_clear()
    yield tmp_path
    get_settings.cache_clear()


@pytest.fixture
def client(env):
    return TestClient(app)


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


# --- read-only guarantees ----------------------------------------------------

def test_database_bytes_unchanged_after_reads(env, client):
    db = env / "results.db"
    before = _sha(db)
    for url in ("/api/runs", f"/api/runs/{RUN}", f"/api/runs/{RUN}/results",
                f"/api/runs/{RUN}/fingerprints", f"/api/runs/{RUN}/deployment/rankings",
                "/api/status"):
        client.get(url)
    assert _sha(db) == before


@pytest.mark.parametrize("method", ["delete", "put", "post", "patch"])
def test_no_write_methods(client, method):
    assert getattr(client, method)(f"/api/runs/{RUN}").status_code == 405
    assert client.get(f"/api/runs/{RUN}").status_code == 200   # still there


def test_missing_db_is_not_created(tmp_path, monkeypatch):
    missing = tmp_path / "nope" / "results.db"
    monkeypatch.setenv("FACEEVAL_DB_PATH", str(missing))
    get_settings.cache_clear()
    c = TestClient(app)
    r = c.get("/api/runs")
    assert r.status_code == 503 and "not found" in r.json()["detail"]
    s = c.get("/api/status").json()
    assert s["db_exists"] is False and s["db_readable"] is False and s["run_count"] is None
    assert not missing.exists() and not missing.parent.exists()
    get_settings.cache_clear()


def test_empty_db_lists_no_runs(tmp_path, monkeypatch):
    db = tmp_path / "e.db"
    ResultStore(str(db)).close()
    monkeypatch.setenv("FACEEVAL_DB_PATH", str(db))
    get_settings.cache_clear()
    c = TestClient(app)
    assert c.get("/api/runs").json()["runs"] == []
    assert c.get("/api/status").json()["run_count"] == 0
    get_settings.cache_clear()


def test_non_database_file_gives_503(tmp_path, monkeypatch):
    db = tmp_path / "junk.db"
    db.write_bytes(b"this is not sqlite" * 50)
    monkeypatch.setenv("FACEEVAL_DB_PATH", str(db))
    get_settings.cache_clear()
    c = TestClient(app)
    assert c.get("/api/runs").status_code == 503
    assert c.get("/api/status").json()["db_readable"] is False
    get_settings.cache_clear()


# --- runs / results -----------------------------------------------------------

def test_list_and_detail(client):
    body = client.get("/api/runs").json()
    assert body["total"] == 2
    d = client.get(f"/api/runs/{RUN}").json()
    assert d["num_evaluation_results"] == 5 and d["duration_seconds"] is None
    assert client.get("/api/runs/nope").status_code == 404
    assert client.get("/api/runs?status=bogus").status_code == 400


def test_corrupt_run_is_a_clear_error_not_a_crash(client):
    r = client.get("/api/runs/bad00001")
    assert r.status_code == 500 and "not valid JSON" in r.json()["detail"]


def test_missing_metrics_are_null_not_zero(client):
    res = client.get(f"/api/runs/{RUN}/results").json()["results"]
    assert len(res) == 4                              # the non-dict entry is skipped
    row = next(r for r in res if r["severity"] == 0.5)
    assert row["accuracy"] == 0.6 and row["precision"] is None and row["eer"] is None
    bare = next(r for r in res if r["perturbation_type"] is None)
    assert bare["accuracy"] is None and bare["num_samples"] is None


def test_adc_sorted_and_drops_incomplete_points(client):
    r = client.get(f"/api/runs/{RUN}/results/adc?model_name=lbph&perturbation_type=blur").json()
    assert r["points"] == [{"severity": 0.0, "accuracy": 0.9}, {"severity": 0.5, "accuracy": 0.6}]


def test_result_filters(client):
    r = client.get(f"/api/runs/{RUN}/results?min_severity=0.4&model_name=lbph").json()
    assert [x["severity"] for x in r["results"]] == [0.5]


# --- fingerprints / deployment -----------------------------------------------

def test_fingerprints_skip_malformed(client):
    body = client.get(f"/api/runs/{RUN}/fingerprints").json()
    assert [f["model_name"] for f in body["fingerprints"]] == ["lbph", "eig"]
    assert body["skipped"] == 1
    assert client.get(f"/api/runs/{RUN}/fingerprints/broken").status_code == 404
    cmp_ = client.get(f"/api/runs/{RUN}/fingerprints/compare/lbph/eig").json()
    assert cmp_["euclidean_distance"] == pytest.approx(0.2 * 2 ** 0.5)


def test_rankings_tolerate_null_rank_and_malformed(client):
    body = client.get(f"/api/runs/{RUN}/deployment/rankings").json()
    assert [r["scenario"] for r in body] == ["mobile"]
    assert body[0]["ranked_scores"][0]["rank"] is None
    assert client.get(f"/api/runs/{RUN}/deployment/rankings/nope").status_code == 404


def test_decision_guide_bad_data_is_422_not_500(client):
    assert client.get(f"/api/runs/{RUN}/deployment/decision-guide").status_code == 422


# --- figures -----------------------------------------------------------------

def test_figures_list_and_serve(client):
    figs = client.get(f"/api/figures?subfolder={RUN}").json()["figures"]
    assert len(figs) == 1 and figs[0]["svg_url"]
    assert client.get(figs[0]["png_url"]).status_code == 200
    assert client.get(figs[0]["svg_url"]).headers["content-type"].startswith("image/svg")


def test_figures_missing_dir_or_subfolder_is_empty(client):
    assert client.get("/api/figures?subfolder=does-not-exist").json() == {"figures": []}


@pytest.mark.parametrize("sub", ["..", "../..", "/etc", f"{RUN}/../../"])
def test_figures_subfolder_traversal_rejected(client, sub):
    assert client.get("/api/figures", params={"subfolder": sub}).status_code == 400


@pytest.mark.parametrize("path", ["../outside.png", "%2e%2e/outside.png", "notes.txt", "..%2fsecret.csv"])
def test_figure_file_traversal_and_type_rejected(client, path):
    assert client.get(f"/api/figures/file/{path}").status_code in (400, 404, 415)
    assert client.get(f"/api/figures/file/{path}").status_code != 200


def test_figure_file_txt_is_415(client):
    assert client.get("/api/figures/file/notes.txt").status_code == 415


# --- reports -----------------------------------------------------------------

def test_reports_only_this_runs_safe_types(client):
    files = client.get(f"/api/runs/{RUN}/reports/list").json()["files"]
    assert [f["filename"] for f in files] == [f"summary_{RUN}.csv"]      # no html, no other run
    assert client.get(files[0]["download_url"]).status_code == 200


@pytest.mark.parametrize("name,code", [
    (f"page_{RUN}.html", 415), ("other_run.csv", 400), ("..%2fsecret.csv", 400),
    (f"x*_{RUN}.csv", 400), (f"nope_{RUN}.csv", 404),
])
def test_report_file_rejections(client, name, code):
    assert client.get(f"/api/runs/{RUN}/reports/file/{name}").status_code in (code, 404)
    assert client.get(f"/api/runs/{RUN}/reports/file/{name}").status_code != 200


@pytest.mark.parametrize("bad_id", ["*", "a*b", "..", "a b"])
def test_report_run_id_validated(client, bad_id):
    assert client.get(f"/api/runs/{bad_id}/reports/list").status_code in (400, 404)


# =============================================================================
# Audit additions: genuine read-only access, missing tables, routes, file safety
# =============================================================================

import subprocess  # noqa: E402
import sys  # noqa: E402

from api.readonly_store import ReadOnlyResultStore  # noqa: E402


def test_reader_connection_really_is_read_only(env):
    db = env / "results.db"
    before = _sha(db)
    conn = ReadOnlyResultStore(db)._connect()
    try:
        for sql in (
            "CREATE TABLE evil (x)",
            "INSERT INTO runs (run_id, experiment_name, status, config_hash, started_at, data) "
            "VALUES ('x','x','x','x','x','{}')",
            "DELETE FROM runs",
            "UPDATE runs SET status='failed'",
            "DROP TABLE runs",
        ):
            with pytest.raises(sqlite3.OperationalError):
                conn.execute(sql)
    finally:
        conn.close()
    assert _sha(db) == before


def test_reader_never_runs_schema_creation(tmp_path):
    db = tmp_path / "other.db"
    c = sqlite3.connect(db)
    c.execute("CREATE TABLE unrelated (x)")
    c.commit()
    c.close()
    before = _sha(db)
    with pytest.raises(Exception):
        ReadOnlyResultStore(db).count_runs()          # no 'runs' table: error, not created
    assert _sha(db) == before
    c = sqlite3.connect(db)
    assert [r[0] for r in c.execute("SELECT name FROM sqlite_master")] == ["unrelated"]
    c.close()


@pytest.mark.parametrize("content", [None, b""])
def test_db_without_runs_table_is_503(tmp_path, monkeypatch, content):
    db = tmp_path / "t.db"
    if content is None:
        c = sqlite3.connect(db)
        c.execute("CREATE TABLE unrelated (x)")
        c.commit()
        c.close()
    else:
        db.write_bytes(content)                        # zero-byte file = valid empty SQLite DB
    before = _sha(db)
    monkeypatch.setenv("FACEEVAL_DB_PATH", str(db))
    get_settings.cache_clear()
    c = TestClient(app)
    r = c.get("/api/runs")
    assert r.status_code == 503 and "no such table" in r.json()["detail"]
    s = c.get("/api/status").json()
    assert s["db_exists"] is True and s["db_readable"] is False and s["run_count"] is None
    assert _sha(db) == before
    get_settings.cache_clear()


def test_every_route_is_read_only():
    allowed = {"GET", "HEAD", "OPTIONS"}
    for route in app.routes:
        methods = getattr(route, "methods", None)
        if methods:
            assert set(methods) <= allowed, (route.path, methods)


def test_api_import_does_not_load_the_experiment_runner():
    root = Path(__file__).resolve().parents[1]
    code = (
        f"import sys; sys.path.insert(0, {str(root)!r}); import api.main; "
        "bad = [m for m in sys.modules if m.startswith('faceeval.orchestration')]; "
        "print(bad); sys.exit(1 if bad else 0)"
    )
    run = subprocess.run([sys.executable, "-I", "-c", code], capture_output=True, text=True, timeout=120)
    assert run.returncode == 0, run.stdout + run.stderr


@pytest.mark.parametrize("url", [
    "/api/runs?limit=0", "/api/runs?limit=201", "/api/runs?offset=-1",
    f"/api/runs/{RUN}/results?min_severity=1.5", f"/api/runs/{RUN}/results?max_severity=-0.1",
])
def test_query_parameters_are_validated(client, url):
    assert client.get(url).status_code == 422


@pytest.mark.parametrize("run_id", ["' OR 1=1 --", "abc12345; DROP TABLE runs", "%25", "a" * 500])
def test_odd_run_ids_are_just_not_found(env, client, run_id):
    before = _sha(env / "results.db")
    assert client.get(f"/api/runs/{run_id}").status_code == 404
    assert _sha(env / "results.db") == before


def _symlink(link: Path, target: Path) -> None:
    try:
        link.symlink_to(target)
    except (OSError, NotImplementedError):
        pytest.skip("symlinks not supported here")


def test_figure_symlink_escape_is_not_listed_or_served(env, client):
    figs = env / "figures"
    _symlink(figs / "leak.png", env / "outside.png")
    _symlink(figs / "leakdir", env)                    # directory symlink to the parent
    listed = [f["png_url"] for f in client.get("/api/figures").json()["figures"]]
    assert not any("leak" in u for u in listed)
    assert client.get("/api/figures/file/leak.png").status_code == 400
    assert client.get("/api/figures/file/leakdir/outside.png").status_code == 400
    assert client.get("/api/figures", params={"subfolder": "leakdir"}).status_code == 400


def test_report_symlink_escape_is_not_listed_or_served(env, client):
    _symlink(env / "reports" / f"leak_{RUN}.csv", env / "secret.csv")
    names = [f["filename"] for f in client.get(f"/api/runs/{RUN}/reports/list").json()["files"]]
    assert f"leak_{RUN}.csv" not in names
    assert client.get(f"/api/runs/{RUN}/reports/file/leak_{RUN}.csv").status_code == 404


@pytest.mark.parametrize("path", ["/etc/passwd", "%2Fetc%2Fpasswd", "a%00.png", "..%5C..%5Coutside.png"])
def test_figure_absolute_and_odd_paths_never_serve(client, path):
    assert client.get(f"/api/figures/file/{path}").status_code in (400, 404, 415, 422)
    assert client.get(f"/api/figures/file//{path.lstrip('/')}").status_code != 200


def test_figures_with_the_same_name_in_two_folders_are_both_listed(env, client):
    other = env / "figures" / "other_run"
    other.mkdir()
    (other / "adc_curve.png").write_bytes(b"\x89PNG other")
    urls = [f["png_url"] for f in client.get("/api/figures").json()["figures"]]
    assert len(urls) == len(set(urls)) == 2


def test_non_figure_and_non_report_types_are_never_served(env, client):
    (env / "figures" / "x.sqlite").write_bytes(b"db")
    (env / "reports" / f"cfg_{RUN}.env").write_text("KEY=1")
    (env / "reports" / f"cfg_{RUN}.py").write_text("print(1)")
    assert client.get("/api/figures/file/x.sqlite").status_code == 415
    for n in (f"cfg_{RUN}.env", f"cfg_{RUN}.py"):
        assert client.get(f"/api/runs/{RUN}/reports/file/{n}").status_code == 415
    listed = [f["filename"] for f in client.get(f"/api/runs/{RUN}/reports/list").json()["files"]]
    assert listed == [f"summary_{RUN}.csv"]

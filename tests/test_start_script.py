"""
Logic tests for start.sh using fake virtualenvs and a stub osascript.
Nothing is launched; this is not a macOS/Terminal.app test.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
from pathlib import Path

import pytest

START = Path(__file__).resolve().parents[1] / "start.sh"
pytestmark = pytest.mark.skipif(shutil.which("bash") is None, reason="bash not available")


def _fake_venv(path: Path, has_deps: bool) -> None:
    (path / "bin").mkdir(parents=True)
    py = path / "bin" / "python"
    py.write_text(f"#!/bin/sh\nexit {0 if has_deps else 1}\n")
    py.chmod(0o755)


@pytest.fixture
def proj(tmp_path):
    p = tmp_path / "proj"
    (p / "api").mkdir(parents=True)
    (p / "api" / "main.py").write_text("")
    (p / "dashboard").mkdir()
    (p / "dashboard" / "package.json").write_text("{}")
    return p


def _run(proj, *args, env=None):
    e = {**os.environ, "FACEEVAL_PROJECT_DIR": str(proj), **(env or {})}
    e.pop("FACEEVAL_VENV", None) if not (env and "FACEEVAL_VENV" in env) else None
    return subprocess.run(["bash", str(START), *args], capture_output=True, text=True, env=e, timeout=60)


def _tree(p: Path):
    return {str(f.relative_to(p)): hashlib.sha256(f.read_bytes()).hexdigest() for f in sorted(p.rglob("*")) if f.is_file()}


def test_unknown_or_extra_arguments_start_nothing(proj):
    for args in (["--dryrun"], ["--dry-run", "x"]):
        r = _run(proj, *args)
        assert r.returncode == 2 and "Nothing was started" in r.stderr


def test_no_environment_is_a_clear_error(proj):
    r = _run(proj, "--dry-run")
    assert r.returncode == 1 and "No virtualenv found" in r.stderr


def test_auto_choice_skips_an_environment_without_api_deps(proj):
    _fake_venv(proj / ".venv-facenet", has_deps=False)
    _fake_venv(proj / ".venv", has_deps=True)
    r = _run(proj, "--dry-run")
    assert r.returncode == 0 and f"Python env : {proj / '.venv'}" in r.stdout


def test_existing_environments_without_deps_are_reported_not_chosen(proj):
    _fake_venv(proj / ".venv-facenet", has_deps=False)
    r = _run(proj, "--dry-run")
    assert r.returncode == 1 and "without fastapi/uvicorn" in r.stderr


def test_explicit_venv_is_never_silently_replaced(proj):
    _fake_venv(proj / ".venv-facenet", has_deps=True)
    bad = proj / "custom"
    _fake_venv(bad, has_deps=False)
    r = _run(proj, "--dry-run", env={"FACEEVAL_VENV": str(bad)})
    assert r.returncode == 1 and "FACEEVAL_VENV" in r.stderr


def test_dry_run_writes_nothing_and_never_calls_osascript(proj, tmp_path):
    _fake_venv(proj / ".venv-facenet", has_deps=True)
    stub = tmp_path / "bin"
    stub.mkdir()
    log = tmp_path / "osa.log"
    (stub / "osascript").write_text(f'#!/bin/sh\necho called >> "{log}"\n')
    (stub / "osascript").chmod(0o755)
    before = _tree(proj)
    r = _run(proj, "--dry-run", env={"PATH": f"{stub}:{os.environ['PATH']}"})
    assert r.returncode == 0 and "Window 1" in r.stdout and "Window 2" in r.stdout
    assert _tree(proj) == before and not log.exists()
    assert "npm ci" in r.stdout and "127.0.0.1" in r.stdout

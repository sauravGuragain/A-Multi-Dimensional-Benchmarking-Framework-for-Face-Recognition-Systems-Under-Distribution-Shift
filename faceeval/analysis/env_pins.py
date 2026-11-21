"""
faceeval.analysis.env_pins
==========================
Read the package versions a completed run recorded in its ``verification_results.json`` (``provenance.packages``)
and either print them as pip pins or compare them with the packages installed NOW.

Why: the deep extras are deliberately not pinned in ``pyproject.toml`` to a version nobody has verified. The version that
produced the completed FaceNet/ArcFace results is the evidence-backed choice, and it is written in the run's own
provenance block. This tool turns that block into a constraints file and refuses to guess when the block is missing.

    python -m faceeval.analysis.env_pins --run <RUN_DIR> --print                 # pip pins (read-only)
    python -m faceeval.analysis.env_pins --run <RUN_DIR> --write constraints.txt # a NEW file, never overwritten
    python -m faceeval.analysis.env_pins --run <RUN_DIR> --check                 # installed vs recorded; exit 1 on mismatch

Exit codes: 0 ok, 1 mismatch (``--check``), 2 refused (no provenance block, unreadable run, or file exists).
Only ``NOT_RECORDED`` / missing packages are skipped; a package that was not recorded is never pinned.
"""
from __future__ import annotations

import argparse
import json
from importlib import metadata as importlib_metadata
from pathlib import Path

NOT_RECORDED = "NOT_RECORDED"


class PinsError(RuntimeError):
    pass


def recorded_packages(run_dir: Path) -> dict[str, str]:
    j = Path(run_dir) / "verification_results.json"
    if not j.is_file():
        raise PinsError(f"{j} not found")
    try:
        prov = json.loads(j.read_text()).get("provenance")
    except ValueError as exc:
        raise PinsError(f"{j} is not valid JSON: {exc}") from exc
    if not isinstance(prov, dict) or not isinstance(prov.get("packages"), dict):
        raise PinsError("this run has no provenance.packages block (it was made before provenance was recorded). "
                        "Nothing is guessed: read the versions from the environment that ran it "
                        "(`<that venv>/bin/python -m pip show insightface onnxruntime onnx`).")
    return {k: v for k, v in prov["packages"].items() if isinstance(v, str) and v and v.lower() not in (NOT_RECORDED.lower(), "not_recorded")}


def installed_version(pkg: str) -> str | None:
    try:
        return importlib_metadata.version(pkg)
    except importlib_metadata.PackageNotFoundError:
        return None


def compare(recorded: dict[str, str]) -> list[dict[str, str | None]]:
    rows = []
    for pkg, want in sorted(recorded.items()):
        have = installed_version(pkg)
        rows.append({"package": pkg, "recorded": want, "installed": have, "status": "OK" if have == want else ("MISSING" if have is None else "DIFFERENT")})
    return rows


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Pins / environment check from a run's recorded provenance.")
    ap.add_argument("--run", required=True)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--print", dest="do_print", action="store_true")
    g.add_argument("--write", metavar="FILE")
    g.add_argument("--check", action="store_true")
    ap.add_argument("--packages", nargs="*", default=None, help="restrict to these packages (default: all recorded)")
    a = ap.parse_args(argv)
    try:
        rec = recorded_packages(Path(a.run))
    except PinsError as exc:
        print(f"REFUSED: {exc}")
        return 2
    if a.packages:
        rec = {k: v for k, v in rec.items() if k in set(a.packages)}
    if not rec:
        print("REFUSED: none of the requested packages was recorded")
        return 2
    if a.check:
        rows = compare(rec)
        for r in rows:
            print(f"{r['status']:9s} {r['package']}: recorded {r['recorded']}, installed {r['installed']}")
        return 0 if all(r["status"] == "OK" for r in rows) else 1
    text = "".join(f"{k}=={v}\n" for k, v in sorted(rec.items()))
    if a.do_print:
        print(text, end="")
        return 0
    out = Path(a.write)
    if out.exists():
        print(f"REFUSED: {out} exists")
        return 2
    out.write_text(text)
    print(f"wrote {out} ({len(rec)} pins)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

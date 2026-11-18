#!/usr/bin/env bash
# =============================================================================
# run_faceeval.sh - one small entry point for the FaceEval-X thesis project
#
# Put this file in the ROOT of the repository (next to pyproject.toml):
#     chmod +x run_faceeval.sh
#     ./run_faceeval.sh help
#
# READ-ONLY commands (write nothing at all, start no experiment, download nothing):
#     help | check | results | evidence
# COMMANDS THAT START AN EXPERIMENT (they write only under runs/ and logs/):
#     demo | demo --classical | full
#
# Design rules
#   - It never deletes, renames or overwrites anything. Every experiment gets its
#     own time-stamped folder under runs/ and its own log under logs/.
#   - Experiments run inside that folder, so experiments/results.db of the
#     project (your thesis database) is never opened for writing.
#   - Every experiment is started with require_real_models: true, so a missing
#     FaceNet/ArcFace backbone stops the run instead of silently using a stub.
#   - Nothing is started without you typing a confirmation phrase (full) or
#     choosing a demo command explicitly.
#
# The project has no command-line program of its own. Its real entry point is the
# Python API: load_config(<yaml>) + ExperimentRunner(config).run(). This script
# calls exactly that, with the repository's own YAML files.
#
# Compatible with the macOS default bash 3.2 (no GNU-only flags, no bash 4 features).
# =============================================================================

set -u

# Never write .pyc files into the project folder (keeps read-only commands truly read-only).
export PYTHONDONTWRITEBYTECODE=1
export GIT_OPTIONAL_LOCKS=0
HOME="${HOME:-/nonexistent-home}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$SCRIPT_DIR"
export FACEEVAL_PROJECT_DIR="$PROJECT_DIR"
# Small Python snippets kept in variables (no here-document or tricky quoting inside $( ), for bash 3.2).
IFS= read -r -d '' FE_MTIME_PY <<'PYEOF'
import os, sys, datetime
print(datetime.datetime.fromtimestamp(os.path.getmtime(sys.argv[1])).isoformat(timespec="seconds"))
PYEOF
IFS= read -r -d '' FE_NEED_PY <<'PYEOF'
import yaml, sys
print(",".join(yaml.safe_load(open(sys.argv[1])).get("deep_models") or []) or "none")
PYEOF
IFS= read -r -d '' FE_RUN_PY <<'PYEOF'
import sys, logging, time
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
from faceeval.core.config import load_config
from faceeval.orchestration import ExperimentRunner
cfg = load_config(sys.argv[1])
print(f"Starting experiment '{cfg.experiment_name}' (run_id {cfg.run_id}); require_real_models={cfg.require_real_models}", flush=True)
t0 = time.time()
run = ExperimentRunner(cfg).run()
print(f"\nFINISHED  run_id={run.run_id}  status={run.status.value}  conditions={run.num_conditions_completed}  seconds={time.time()-t0:.0f}", flush=True)
sys.exit(0 if run.status.value == "completed" else 3)
PYEOF
LOG_DIR="$PROJECT_DIR/logs"
RUNS_DIR="$PROJECT_DIR/runs"
STAMP="$(date +%Y%m%d_%H%M%S)"

if [ -t 1 ]; then
  C_OK=$'\033[32m'; C_WARN=$'\033[33m'; C_FAIL=$'\033[31m'; C_B=$'\033[1m'; C_OFF=$'\033[0m'
else
  C_OK=""; C_WARN=""; C_FAIL=""; C_B=""; C_OFF=""
fi
say()   { printf '%s\n' "$*"; }
head1() { printf '\n%s== %s ==%s\n' "$C_B" "$*" "$C_OFF"; }
ok()    { printf '%s[ OK ]%s %s\n' "$C_OK"   "$C_OFF" "$*"; }
warn()  { printf '%s[WARN]%s %s\n' "$C_WARN" "$C_OFF" "$*"; }
fail()  { printf '%s[FAIL]%s %s\n' "$C_FAIL" "$C_OFF" "$*"; }

# --- places where results may live (checked in this order, duplicates removed) ---
RESULT_ROOT_CANDIDATES=(
  "$PROJECT_DIR/reports"
  "$PROJECT_DIR/runs"
  "$HOME/Thesis/faceeval_x_patch_verify/reports"
  "$HOME/Thesis/faceeval_x/reports"
)
DB_CANDIDATES=(
  "$PROJECT_DIR/experiments/results.db"
  "$HOME/Thesis/faceeval_x_patch_verify/experiments/results.db"
  "$HOME/Thesis/faceeval_x/experiments/results.db"
)

# physical_path DIR_OR_FILE -> prints the resolved absolute path (symlinks resolved), or nothing
physical_path() {
  local p="$1" d b
  if [ -d "$p" ]; then (cd "$p" 2>/dev/null && pwd -P)
  elif [ -f "$p" ]; then d="$(dirname "$p")"; b="$(basename "$p")"; (cd "$d" 2>/dev/null && printf '%s/%s\n' "$(pwd -P)" "$b")
  fi
}

# --- locate the Python environment -------------------------------------------
# We call <venv>/bin/python directly. That is the same as "activating" it.
PY=""
VENV_USED=""
find_python() {
  local c
  for c in "${FACEEVAL_VENV:-}" "$PROJECT_DIR/.venv-facenet" "$PROJECT_DIR/.venv" \
           "$HOME/Thesis/faceeval_x/.venv-facenet" "$HOME/Thesis/faceeval_x/.venv"; do
    if [ -n "$c" ] && [ -x "$c/bin/python" ]; then PY="$c/bin/python"; VENV_USED="$c"; return 0; fi
  done
  return 1
}

# --- locate the dataset ------------------------------------------------------
# The project's LFW loader reads <data_root>/lfw/<Person_Name>/*.jpg
DATA_ROOT=""
find_data() {
  local c
  for c in "${FACEEVAL_DATA:-}" "$PROJECT_DIR/data" "$HOME/Thesis/faceeval_x/data"; do
    if [ -n "$c" ] && [ -d "$c/lfw" ]; then DATA_ROOT="$(cd "$c" && pwd)"; return 0; fi
  done
  return 1
}

# hash_stdin -> SHA-256 of standard input
hash_stdin() {
  if command -v shasum >/dev/null 2>&1; then shasum -a 256 | cut -d' ' -f1
  elif command -v sha256sum >/dev/null 2>&1; then sha256sum | cut -d' ' -f1
  else cat >/dev/null; echo "NO-SHA-TOOL"; fi
}
hash_file() { hash_stdin < "$1"; }

# =============================================================================
# CHECK  (read-only: writes nothing, downloads nothing, starts nothing)
# =============================================================================
# cmd_check [NEEDED_DEEP_MODELS]   NEEDED = facenet,arcface (default) | facenet | none
cmd_check() {
  local need="${1:-facenet,arcface}"
  local fails=0 warns=0 out line

  head1 "FaceEval-X environment check (read-only)"
  say "Project folder : $PROJECT_DIR"
  say "Machine        : $(uname -srm)"

  if [ ! -f "$PROJECT_DIR/pyproject.toml" ] || [ ! -d "$PROJECT_DIR/faceeval" ]; then
    fail "This script must sit in the repository root (needs pyproject.toml and faceeval/)."; return 2
  fi
  ok "Repository layout found."
  if [ -d "$PROJECT_DIR/.git" ] && command -v git >/dev/null 2>&1; then
    say "Git commit     : $(git -C "$PROJECT_DIR" rev-parse --short HEAD 2>/dev/null) on $(git -C "$PROJECT_DIR" rev-parse --abbrev-ref HEAD 2>/dev/null)"
  fi

  head1 "Python environment"
  if find_python; then
    ok "Using environment: $VENV_USED"
    say "     $("$PY" --version 2>&1)"
  else
    fail "No Python environment found (looked for .venv-facenet and .venv in the project and in ~/Thesis/faceeval_x)."
    say  "     Set FACEEVAL_VENV=/path/to/venv, or create one (README, section Installation)."
    return 2
  fi

  head1 "Dependencies, dataset and model weights"
  # The Python program is read into a variable first. A here-document INSIDE $( ) breaks
  # the parser of macOS bash 3.2, so none is used inside a command substitution.
  IFS= read -r -d '' FE_CHECK_PY <<'PYEOF'
import os, sys, glob, importlib
P = os.environ["FE_PROJECT"]; H = os.environ["HOME_DIR"]; NEED = os.environ.get("FE_NEED", "").split(",")
def out(level, msg): print(f"{level} {msg}")

v = sys.version_info
out("OK" if v >= (3, 11) else "FAIL", f"Python {v.major}.{v.minor}.{v.micro} (project needs >= 3.11)")

for mod, label in [("numpy","numpy"),("cv2","opencv-python"),("sklearn","scikit-learn"),("skimage","scikit-image"),
                   ("scipy","scipy"),("yaml","pyyaml"),("pydantic","pydantic"),("matplotlib","matplotlib"),
                   ("torch","torch"),("mlflow","mlflow")]:
    try:
        m = importlib.import_module(mod); out("OK", f"{label} {getattr(m,'__version__','')}".strip())
    except Exception as e:
        if mod == "mlflow":
            out("WARN", "mlflow not installed - experiment tracking is disabled (results are still saved)")
        elif mod == "torch" and "facenet" not in NEED and "arcface" not in NEED:
            out("WARN", "torch not installed - fine for classical-only runs, required for FaceNet")
        else:
            out("FAIL", f"{label} cannot be imported: {type(e).__name__}: {e}")

try:
    import facenet_pytorch
    out("OK", f"facenet-pytorch {getattr(facenet_pytorch,'__version__','')}".strip())
except Exception as e:
    out("FAIL" if "facenet" in NEED else "WARN", f"facenet-pytorch missing -> FaceNet would be a STUB ({type(e).__name__})")
try:
    import insightface, onnxruntime
    out("OK", f"insightface {insightface.__version__} + onnxruntime {onnxruntime.__version__}")
    out("OK" if "CPUExecutionProvider" in onnxruntime.get_available_providers() else "WARN",
        "onnxruntime providers: " + ", ".join(onnxruntime.get_available_providers()))
except Exception as e:
    out("FAIL" if "arcface" in NEED else "WARN", f"insightface/onnxruntime missing -> ArcFace would be a STUB ({type(e).__name__})")

# Weights: we only LOOK at the cache folders. We never download anything here.
fw = sorted(glob.glob(os.path.join(H, ".cache", "torch", "checkpoints", "*vggface2*")))
if fw: out("OK", "FaceNet VGGFace2 weights cached: " + os.path.basename(fw[0]))
else:  out("WARN", "FaceNet VGGFace2 weights not cached in ~/.cache/torch/checkpoints (a deep run would download them; needs internet)")
aw = sorted(glob.glob(os.path.join(H, ".insightface", "models", "buffalo_l", "*.onnx")))
if aw: out("OK", "ArcFace buffalo_l pack cached: " + ", ".join(os.path.basename(a) for a in aw))
else:  out("WARN", "InsightFace buffalo_l pack not found in ~/.insightface/models (a deep run would download it; needs internet)")

sys.path.insert(0, P)
try:
    import faceeval, faceeval.models, faceeval.perturbation, faceeval.data.loaders
    from faceeval.core.registry import model_registry, perturbation_registry
    out("OK", f"faceeval imports from {os.path.dirname(faceeval.__file__)}")
    out("OK", "registered models: " + ", ".join(sorted(model_registry.list_registered())))
    out("OK", f"registered perturbations: {len(perturbation_registry.list_registered())}")
    if os.path.realpath(os.path.dirname(faceeval.__file__)) != os.path.realpath(os.path.join(P, "faceeval")):
        out("WARN", "faceeval is imported from a DIFFERENT folder than this repository")
except Exception as e:
    out("FAIL", f"cannot import faceeval: {type(e).__name__}: {e}")

try:
    from faceeval.core.config import load_config
    for c in ("lfw_smoke_test.yaml", "lfw_intermediate.yaml"):
        p = os.path.join(P, "configs", c)
        try: load_config(p); out("OK", f"config loads: configs/{c}")
        except Exception as e: out("FAIL", f"config {c}: {type(e).__name__}: {e}")
except Exception:
    pass

# Dataset: mirror the project's loader (root = <data>/lfw), then scan EVERY person folder.
found = None
for c in (os.environ.get("FE_DATA", ""), os.path.join(P, "data"), os.path.join(H, "Thesis", "faceeval_x", "data")):
    if c and os.path.isdir(os.path.join(c, "lfw")):
        found = c; break
if not found:
    out("FAIL", "LFW dataset not found (expected <project>/data/lfw/<Person_Name>/*.jpg). Set FACEEVAL_DATA=/path/to/data")
else:
    root = os.path.join(found, "lfw")
    counts = {}
    with os.scandir(root) as it:
        for e in sorted(it, key=lambda e: e.name):
            if e.is_dir():
                with os.scandir(e.path) as jt:
                    counts[e.name] = sum(1 for f in jt if f.is_file() and f.name.lower().endswith(".jpg"))
    people = {k: v for k, v in counts.items() if v > 0}
    nimg = sum(people.values()); n8 = sum(1 for c in people.values() if c >= 8)
    if len(people) <= 3 and "lfw" in counts:
        out("FAIL", f"nested layout: {root}/lfw exists, but the project's loader reads {root} directly and would treat 'lfw' as one person. Move the people folders up one level.")
    out("OK" if nimg > 10000 else "FAIL", f"LFW at {found}: {len(people)} people with images, {nimg} images, {n8} people with >= 8 images")
    out("OK" if n8 >= 200 else ("WARN" if n8 >= 20 else "FAIL"),
        "the thesis experiment needs >= 200 people with >= 8 images (the demo needs >= 20)")

st = os.statvfs(P); free = st.f_bavail * st.f_frsize / 1e9
out("OK" if free > 5 else "WARN", f"free disk space: {free:.1f} GB")
PYEOF
  out="$(PYTHONPATH="$PROJECT_DIR" FE_PROJECT="$PROJECT_DIR" FE_DATA="${FACEEVAL_DATA:-}" FE_NEED="$need" HOME_DIR="$HOME" "$PY" -c "$FE_CHECK_PY" 2>&1)"
  while IFS= read -r line; do
    case "$line" in
      "OK "*)   ok   "${line#OK }" ;;
      "WARN "*) warn "${line#WARN }"; warns=$((warns+1)) ;;
      "FAIL "*) fail "${line#FAIL }"; fails=$((fails+1)) ;;
      "")       ;;
      *)        say "     $line" ;;
    esac
  done <<EOF
$out
EOF

  head1 "Summary"
  if [ "$fails" -eq 0 ]; then
    ok "All required checks passed ($warns warning(s)). Nothing was written or downloaded."
    say "Next:  ./run_faceeval.sh results     (see recorded results, read-only)"
    return 0
  else
    fail "$fails required check(s) failed. Fix them before 'demo' or 'full'."
    return 1
  fi
}

# =============================================================================
# Helpers to run an experiment safely (used only by demo and full)
# =============================================================================

# make_config SRC_YAML DEST_YAML RUN_NAME [classical]
make_config() {
  "$PY" - "$1" "$2" "$3" "${4:-}" <<'PYEOF'
import sys, yaml
src, dest, name, mode = sys.argv[1:5]
cfg = yaml.safe_load(open(src))
cfg["experiment_name"] = name
cfg["output_dir"] = f"experiments/{name}/"
cfg["report_dir"] = f"reports/{name}/"
cfg["feature_cache_dir"] = f"experiments/cache/{name}/"
cfg["mlflow_experiment_name"] = name
cfg["require_real_models"] = True          # never silently fall back to stubs
if mode == "classical":
    cfg["deep_models"] = []                # explicit, visible choice (not a silent fallback)
yaml.safe_dump(cfg, open(dest, "w"), sort_keys=False)
PYEOF
}

# run_experiment CONFIG_YAML WORKDIR LOGFILE
# Runs the real API inside WORKDIR, so the experiments/results.db created there is NEW and the
# thesis database is never touched. data/ is linked, not copied.
run_experiment() {
  local cfg="$1" work="$2" log="$3"
  mkdir -p "$work" "$(dirname "$log")"
  [ -e "$work/data" ] || ln -s "$DATA_ROOT" "$work/data"
  local keepawake=""
  command -v caffeinate >/dev/null 2>&1 && keepawake="caffeinate"
  ( cd "$work" && PYTHONPATH="$PROJECT_DIR" PYTHONUNBUFFERED=1 ${keepawake:+"$keepawake" -i} "$PY" -c "$FE_RUN_PY" "$cfg"
  ) 2>&1 | tee "$log" | awk '!/PREDICTIONS DEBUG/'
  return "${PIPESTATUS[0]}"
}

# show_results RUN_ID_PREFIX_OR_EMPTY DIR [DIR...]   (read-only; exit 1 if nothing matches)
show_results() {
  local want="$1"; shift
  "$PY" - "$want" "$@" <<'PYEOF'
import sys, os, glob, csv, re
want = sys.argv[1]; roots = sys.argv[2:]
files = []
for root in roots:
    files += glob.glob(os.path.join(root, "**", "evaluation_results_*.csv"), recursive=True)
runs = {}
for f in files:
    rid = re.search(r"evaluation_results_([0-9a-f]+)\.csv$", f)
    if rid: runs.setdefault(rid.group(1), f)
if not runs: sys.exit(1)
if want:
    pick = [r for r in runs if r.startswith(want)]
    if not pick: sys.exit(1)
else:
    pick = sorted(runs, key=lambda r: os.path.getmtime(runs[r]), reverse=True)[:1]
rid = pick[0]; f = runs[rid]; d = os.path.dirname(f)
print(f"Run {rid}\nFile {f}")
if rid.startswith("fe4696ee") or rid.startswith("69878560"):
    print("NOTE: this is an EARLIER run that the corrected thesis no longer uses (see docs/EXPERIMENT_EVIDENCE_RECORD.md).")
rows = list(csv.DictReader(open(f)))
acc = {}
for r in rows:
    acc[(r["model_name"], r["perturbation_type"], float(r["perturbation_severity"]))] = float(r["accuracy"])
models = []
for r in rows:
    if r["model_name"] not in models: models.append(r["model_name"])
n = rows[0].get("num_samples", "?")
print(f"\nClosed-set IDENTIFICATION accuracy (%), {n} test samples, {len(rows)} rows (severity 1.0 = strongest)")
hdr = f"{'model':<12}{'clean':>7}{'blur.5':>8}{'blur1':>7}{'rot.5':>7}{'rot1':>7}{'bri.5':>7}{'bri1':>7}"
print(hdr); print("-" * len(hdr))
g = lambda m,p,s: acc.get((m,p,s))
fmt = lambda v: f"{100*v:6.1f}" if v is not None else "     -"
for m in models:
    print(f"{m:<12}{fmt(g(m,'gaussian_blur',0.0)):>7}{fmt(g(m,'gaussian_blur',.5)):>8}{fmt(g(m,'gaussian_blur',1.0)):>7}"
          f"{fmt(g(m,'rotation',.5)):>7}{fmt(g(m,'rotation',1.0)):>7}{fmt(g(m,'brightness',.5)):>7}{fmt(g(m,'brightness',1.0)):>7}")
print("\nReminder: brightness 0.5 is the identity (no change) by design, so only +0.40 brightening is tested.")
for kind in ("fingerprints", "deployment_rankings"):
    p = os.path.join(d, f"{kind}_{rid}.csv")
    if os.path.exists(p):
        rr = list(csv.DictReader(open(p)))
        print(f"\n{kind} ({os.path.basename(p)})")
        if kind == "fingerprints":
            for r in rr:
                print("  {:<12} brightness AUC {:.3f}  blur AUC {:.3f}  rotation AUC {:.3f}".format(
                    r.get("model_name","?"), float(r.get("robustness_brightness",0)),
                    float(r.get("robustness_gaussian_blur",0)), float(r.get("robustness_rotation",0))))
        else:
            for r in rr:
                print("  rank {}  {:<12} total {:.3f}  accuracy-score {:.3f}".format(
                    r.get("rank","?"), r.get("model_name","?"), float(r.get("total_score",0)), float(r.get("score_accuracy",0))))
            print("  (all other scoring dimensions are placeholders = 0.5; the ranking is accuracy-driven)")
PYEOF
}

# =============================================================================
# RESULTS  (read-only: never starts an experiment)
# =============================================================================
cmd_results() {
  local want="${1:-}"
  head1 "Recorded experiment results (read-only)"
  find_python || { fail "No Python environment found (see './run_faceeval.sh check')."; return 2; }

  # existing, de-duplicated result folders
  local existing=() r rp seen=" "
  for r in ${RESULT_ROOT_CANDIDATES[@]+"${RESULT_ROOT_CANDIDATES[@]}"}; do
    rp="$(physical_path "$r")"
    [ -n "$rp" ] || continue
    case "$seen" in *" $rp "*) continue ;; esac
    seen="$seen$rp "; existing+=("$rp")
  done

  say "Result files found:"
  for r in ${existing[@]+"${existing[@]}"}; do
    find "$r" -name 'evaluation_results_*.csv' 2>/dev/null | while IFS= read -r f; do
      say "  $(basename "$f" .csv | sed 's/evaluation_results_//')   $f"
    done
  done
  say ""
  if [ -z "$want" ]; then
    say "No run id given: showing the thesis run 2bad8efd if present, otherwise the newest."
    say "Pick another with  ./run_faceeval.sh results <run-id-prefix>"
    want="2bad8efd"
  fi
  if [ "${#existing[@]}" -eq 0 ]; then
    warn "No result folders found."
  elif ! show_results "$want" "${existing[@]}"; then
    warn "Run '$want' was not found."
    [ "$want" = "2bad8efd" ] && say "The 2bad8efd files are only on the machine that ran the experiment (they are not in Git)."
    say "Showing the newest run that was found instead:"
    show_results "" "${existing[@]}" || warn "No result files found at all."
  fi

  # runs recorded in result databases (read from a temporary copy; the original file is untouched), each physical database once
  local db dbs=() dseen=" "
  for db in ${DB_CANDIDATES[@]+"${DB_CANDIDATES[@]}"}; do
    rp="$(physical_path "$db")"
    [ -n "$rp" ] || continue
    case "$dseen" in *" $rp "*) continue ;; esac
    dseen="$dseen$rp "; dbs+=("$rp")
  done
  for db in ${dbs[@]+"${dbs[@]}"}; do
    say ""; say "Runs recorded in $db (read from a temporary copy; the original file is untouched):"
    "$PY" - "$db" <<'PYEOF'
import sqlite3, os, shutil, tempfile, atexit
def open_ro(path):
    """Open a COPY of the database (and its -wal/-shm if present) in the system temp folder.
    The original file is only read; no journal, WAL or lock file is created next to it."""
    path = os.path.realpath(path)
    tmpdir = tempfile.mkdtemp(prefix="faceeval_ro_")
    atexit.register(shutil.rmtree, tmpdir, True)
    proj = os.path.realpath(os.environ.get("FACEEVAL_PROJECT_DIR", "/nonexistent"))
    if os.path.realpath(tmpdir).startswith(proj + os.sep):
        raise SystemExit("temporary folder is inside the project; set TMPDIR elsewhere")
    dst = os.path.join(tmpdir, "copy.db")
    shutil.copyfile(path, dst)
    for suf in ("-wal", "-shm"):
        if os.path.exists(path + suf):
            shutil.copyfile(path + suf, dst + suf)
    return sqlite3.connect(dst)
import sys
c = open_ro(sys.argv[1])
for r in c.execute("select substr(run_id,1,8), experiment_name, status, started_at from runs order by started_at"):
    print("  {}  {:<40} {:<10} {}".format(*r))
PYEOF
  done
  say ""
  say "Fold-0 VERIFICATION results (a different protocol) are described in reports/experiment_results/README.md."
}

# =============================================================================
# EVIDENCE  (read-only: prints what can be observed; invents nothing)
# =============================================================================
# Prints a markdown block to the screen. Redirect it if you want a file:
#     ./run_faceeval.sh evidence 2bad8efd > my_evidence.md
cmd_evidence() {
  local want="${1:-2bad8efd}"
  find_python || { fail "No Python environment found (see check)." >&2; return 2; }
  find_data || DATA_ROOT=""

  local existing=() r rp seen=" " f="" dbs=() db dseen=" "
  for r in ${RESULT_ROOT_CANDIDATES[@]+"${RESULT_ROOT_CANDIDATES[@]}"}; do
    rp="$(physical_path "$r")"; [ -n "$rp" ] || continue
    case "$seen" in *" $rp "*) continue ;; esac
    seen="$seen$rp "; existing+=("$rp")
  done
  for r in ${existing[@]+"${existing[@]}"}; do
    f="$(find "$r" -name "evaluation_results_${want}*.csv" 2>/dev/null | head -1)"
    [ -n "$f" ] && break
  done
  if [ -z "$f" ]; then fail "No evaluation_results_${want}*.csv found. Nothing to describe." >&2; return 1; fi
  local rdir rid
  rdir="$(dirname "$f")"; rid="$(basename "$f" .csv | sed 's/evaluation_results_//')"
  for db in ${DB_CANDIDATES[@]+"${DB_CANDIDATES[@]}"}; do
    rp="$(physical_path "$db")"; [ -n "$rp" ] || continue
    case "$dseen" in *" $rp "*) continue ;; esac
    dseen="$dseen$rp "; dbs+=("$rp")
  done

  echo "## Evidence observed by run_faceeval.sh evidence"
  echo
  echo "Read-only. Nothing was written, downloaded or started."
  echo
  echo "**How to read this block.** Part A is what the run itself recorded in its database. Part B describes this machine and these files *as they are now*. Part B is evidence about the historical run only if nothing changed since the run; compare the modification times. Anything the run did not record is printed as NOT RECORDED."
  echo
  echo "### A. Recorded by the run in the SQLite database (read from a temporary copy; the original file is untouched)"
  local found_db=0
  for db in ${dbs[@]+"${dbs[@]}"}; do
    if "$PY" - "$db" "$rid" <<'PYEOF'
import sqlite3, os, shutil, tempfile, atexit
def open_ro(path):
    """Open a COPY of the database (and its -wal/-shm if present) in the system temp folder.
    The original file is only read; no journal, WAL or lock file is created next to it."""
    path = os.path.realpath(path)
    tmpdir = tempfile.mkdtemp(prefix="faceeval_ro_")
    atexit.register(shutil.rmtree, tmpdir, True)
    proj = os.path.realpath(os.environ.get("FACEEVAL_PROJECT_DIR", "/nonexistent"))
    if os.path.realpath(tmpdir).startswith(proj + os.sep):
        raise SystemExit("temporary folder is inside the project; set TMPDIR elsewhere")
    dst = os.path.join(tmpdir, "copy.db")
    shutil.copyfile(path, dst)
    for suf in ("-wal", "-shm"):
        if os.path.exists(path + suf):
            shutil.copyfile(path + suf, dst + suf)
    return sqlite3.connect(dst)
import sys, json
db, rid = sys.argv[1], sys.argv[2]
c = open_ro(db)
row = c.execute("select run_id, experiment_name, status, config_hash, started_at, completed_at, data from runs where run_id=?", (rid,)).fetchone()
if not row: sys.exit(1)
d = json.loads(row[6]); cfg = d.get("config", {})
print(f"- Database: {db}")
print(f"- Run ID: {row[0]}\n- Experiment name: {row[1]}\n- Status: {row[2]}")
print(f"- Started: {row[4]}\n- Completed: {row[5]}\n- Duration (s): {d.get('duration_seconds')}")
print(f"- config_hash: {row[3]}")
gc = cfg.get("git_commit") or ""
print(f"- git_commit recorded by the run (short HEAD of the folder it was started from): {gc if gc else 'NOT RECORDED'}")
print("  (the run records the commit only, not whether the working copy had uncommitted changes)")
for k in ("random_seed", "evaluation_protocol", "require_real_models", "framework_version", "data_root"):
    print(f"- {k}: {cfg.get(k, 'NOT RECORDED')}")
print(f"- evaluation result rows stored: {len(d.get('evaluation_results', []))}")
PYEOF
    then found_db=1; break; fi
  done
  [ "$found_db" -eq 1 ] || echo "- Run $rid not found in any known database: NOT RECORDED here."
  echo "- Python / package versions, model-weight hashes, dataset fingerprint, CSV hashes: NOT RECORDED by the run itself."
  echo
  echo "### B. Observed now"
  echo "- Machine: $(uname -srm)"
  if [ -d "$PROJECT_DIR/.git" ] && command -v git >/dev/null 2>&1; then
    echo "- Code commit of this folder now: $(GIT_OPTIONAL_LOCKS=0 git -C "$PROJECT_DIR" rev-parse HEAD 2>/dev/null)"
    if [ -n "$(GIT_OPTIONAL_LOCKS=0 git -C "$PROJECT_DIR" status --porcelain --untracked-files=no 2>/dev/null)" ]; then
      echo "- Tracked files modified now: YES"
      GIT_OPTIONAL_LOCKS=0 git -C "$PROJECT_DIR" status --porcelain --untracked-files=no 2>/dev/null | head -20 | sed 's/^/    /'
    else echo "- Tracked files modified now: no"; fi
  fi
  echo "- Python now: $("$PY" --version 2>&1) ($VENV_USED)"
  echo "- pip freeze SHA-256 now: $("$PY" -m pip freeze 2>/dev/null | LC_ALL=C sort | hash_stdin)"
  echo
  echo "Result files (SHA-256 now; modification time now):"
  local k p n mt
  for k in evaluation_results fingerprints deployment_rankings failure_clusters; do
    p="$rdir/${k}_${rid}.csv"
    if [ -f "$p" ]; then
      n="$(wc -l < "$p" | tr -d ' ')"; [ "$n" -gt 0 ] && n=$((n-1))
      mt="$("$PY" -c "$FE_MTIME_PY" "$p")"
      echo "- ${k}_${rid}.csv  data_rows=$n  sha256=$(hash_file "$p")  modified=$mt"
    else echo "- ${k}_${rid}.csv  not found"; fi
  done
  echo
  echo "Configuration files in this folder (SHA-256 now):"
  for p in "$PROJECT_DIR/configs/lfw_intermediate.yaml" "$PROJECT_DIR/configs/lfw_intermediate_rerun.yaml"; do
    if [ -f "$p" ]; then echo "- ${p#"$PROJECT_DIR"/}  sha256=$(hash_file "$p")"
    else echo "- ${p#"$PROJECT_DIR"/}  not present in this folder"; fi
  done
  echo
  echo "Dataset (now):"
  if [ -n "$DATA_ROOT" ]; then
    echo "- Folder: $DATA_ROOT/lfw"
    echo "- Image files: $(cd "$DATA_ROOT/lfw" && find . -name '*.jpg' | wc -l | tr -d ' ')"
    echo "- SHA-256 of the sorted list of relative file paths: $(cd "$DATA_ROOT/lfw" && find . -name '*.jpg' | LC_ALL=C sort | hash_stdin)"
    echo "  (fingerprints which files exist, not their pixel contents)"
  else echo "- not found"; fi
  echo
  echo "Cached pretrained weights (now):"
  local any=0
  for p in "$HOME"/.cache/torch/checkpoints/*vggface2* "$HOME/.insightface/models/buffalo_l/w600k_r50.onnx" "$HOME/.insightface/models/buffalo_l/det_10g.onnx"; do
    if [ -f "$p" ]; then any=1; echo "- ${p#"$HOME"/}  bytes=$(wc -c < "$p" | tr -d ' ')  sha256=$(hash_file "$p")"; fi
  done
  [ "$any" -eq 1 ] || echo "- none found"
}

# =============================================================================
# DEMO  (STARTS AN EXPERIMENT: small, real models, separate folder)
# =============================================================================
cmd_demo() {
  local mode="${1:-}"
  head1 "FaceEval-X demo (20 LFW subjects, 3 perturbations, 3 severities)"
  say "This STARTS a small experiment. It writes only under runs/ and logs/."
  local need="facenet"; [ "$mode" = "classical" ] && need="none"
  cmd_check "$need" || { fail "Demo not started because the check failed."; return 1; }
  find_python; find_data || { fail "Dataset not found."; return 1; }
  local name="demo_${STAMP}"
  local work="$RUNS_DIR/$name"
  mkdir -p "$work" "$LOG_DIR"
  make_config "$PROJECT_DIR/configs/lfw_smoke_test.yaml" "$work/config.yaml" "$name" "$mode" || { fail "Could not prepare config."; return 1; }
  if [ "$mode" = "classical" ]; then
    warn "Classical-only demo: Eigenfaces + HOG-SVM. FaceNet is deliberately excluded."
  else
    say "Models: Eigenfaces, HOG-SVM, FaceNet (real pretrained weights required)."
    say "If FaceNet is unavailable the run STOPS. For a classical-only demo use:  ./run_faceeval.sh demo --classical"
    say "If the FaceNet weights are not cached, the first run downloads them (internet needed)."
  fi
  say "Output folder: runs/$name     Log: logs/demo_$STAMP.log"
  run_experiment "$work/config.yaml" "$work" "$LOG_DIR/demo_$STAMP.log"
  local rc=$?
  if [ "$rc" -ne 0 ]; then fail "Demo did not finish (exit code $rc). See logs/demo_$STAMP.log"; return "$rc"; fi
  head1 "Demo results"
  show_results "" "$work"
  ok "Demo finished. Everything is in runs/$name (your thesis results were not touched)."
}

# =============================================================================
# FULL  (STARTS A LONG EXPERIMENT: explicit, confirmed, never accidental)
# =============================================================================
cmd_full() {
  local cfgname="${1:-lfw_intermediate.yaml}"
  local cfg="$PROJECT_DIR/configs/$(basename "$cfgname")"
  head1 "FaceEval-X FULL run (STARTS A LONG EXPERIMENT)"
  [ -f "$cfg" ] || { fail "Config not found: $cfg"; return 1; }
  if [ ! -t 0 ]; then fail "full needs an interactive terminal (it asks you to confirm)."; return 1; fi
  case "$(basename "$cfg")" in
    lfw_full_benchmark.yaml)
      fail "configs/lfw_full_benchmark.yaml is the 10-model, 16-perturbation, 6-severity benchmark on ALL LFW subjects."
      say  "It has never been run, is not part of the thesis, and needs dlib and far more time/RAM than a laptop offers."
      say  "To run it anyway on purpose:  FACEEVAL_ALLOW_PAPER_SCALE=yes ./run_faceeval.sh full lfw_full_benchmark.yaml"
      [ "${FACEEVAL_ALLOW_PAPER_SCALE:-}" = "yes" ] || return 1 ;;
  esac
  find_python || { fail "No Python environment found."; return 1; }
  local need
  need="$("$PY" -c "$FE_NEED_PY" "$cfg")"
  cmd_check "$need" || { fail "Not starting: the check failed."; return 1; }
  find_data || { fail "Dataset not found."; return 1; }

  say ""
  say "About to run: $(basename "$cfg")"
  if [ "$(basename "$cfg")" = "lfw_intermediate.yaml" ]; then
    say "  This repeats the thesis experiment: 200 subjects, 4 models, 36 conditions."
    say "  Reference: the recorded run on an Apple-silicon laptop (ArcFace on CPU) took about 58 minutes and peaked near 3.6 GB."
    say "  The thesis results already exist; you only need this to reproduce them."
  fi
  say "  Real pretrained models are required (no stubs)."
  say "  Results go to a NEW folder: runs/full_<name>_$STAMP   (nothing is overwritten)."
  say "  Keep the Mac plugged in; 'caffeinate' keeps the job awake."
  say ""
  printf 'To start, type exactly   RUN FULL   and press Enter (anything else cancels): '
  local ans; read -r ans
  [ "$ans" = "RUN FULL" ] || { say "Cancelled. Nothing was started."; return 0; }

  local base; base="$(basename "$cfg" .yaml)"
  local name="full_${base}_${STAMP}"; local work="$RUNS_DIR/$name"
  mkdir -p "$work" "$LOG_DIR"
  make_config "$cfg" "$work/config.yaml" "$name" || { fail "Could not prepare config."; return 1; }
  run_experiment "$work/config.yaml" "$work" "$LOG_DIR/full_$STAMP.log"
  local rc=$?
  [ "$rc" -eq 0 ] || { fail "Run did not finish (exit $rc). See logs/full_$STAMP.log"; return "$rc"; }
  head1 "Results"; show_results "" "$work"
  ok "Done. Files: runs/$name/reports/$name/   Log: logs/full_$STAMP.log"
}

# =============================================================================
cmd_help() {
cat <<EOF
FaceEval-X helper script

READ-ONLY commands (write nothing, start nothing, download nothing):
  ./run_faceeval.sh help             This text.
  ./run_faceeval.sh check            Check Python, packages, dataset, model weights (it only looks).
  ./run_faceeval.sh results [id]     Show recorded results. Default: thesis run 2bad8efd, else the newest.
  ./run_faceeval.sh evidence [id]    Print what the run recorded in its database plus current file hashes.
                                     Default id: 2bad8efd. Save with:  ./run_faceeval.sh evidence > evidence.md

COMMANDS THAT START AN EXPERIMENT (write only under runs/ and logs/, never overwrite):
  ./run_faceeval.sh demo             Small demo: 20 people, 27 conditions, real FaceNet required.
  ./run_faceeval.sh demo --classical Same without FaceNet (for when deep packages/weights are missing).
  ./run_faceeval.sh full [config]    Repeat a long experiment. Default: configs/lfw_intermediate.yaml
                                     (the thesis experiment, about an hour). Asks you to type RUN FULL.

Optional settings: FACEEVAL_VENV=/path/to/venv   FACEEVAL_DATA=/path/to/data (the folder that contains lfw/)
EOF
}

case "${1:-help}" in
  help|-h|--help) cmd_help ;;
  check)    cmd_check ;;
  results)  cmd_results "${2:-}" ;;
  evidence) cmd_evidence "${2:-}" ;;
  demo)     case "${2:-}" in
              "")          cmd_demo "" ;;
              --classical) cmd_demo classical ;;
              *)           fail "Unknown option for demo: $2 (use --classical or nothing)"; exit 2 ;;
            esac ;;
  full)     cmd_full "${2:-}" ;;
  *)        fail "Unknown command: $1"; cmd_help; exit 2 ;;
esac

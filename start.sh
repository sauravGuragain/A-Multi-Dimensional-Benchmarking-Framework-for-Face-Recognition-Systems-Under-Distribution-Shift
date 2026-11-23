#!/bin/bash
# start.sh - launches the FaceEval-X read-only dashboard on macOS.
#
# Opens two Terminal.app windows:
#   1. API        uvicorn api.main:app on http://127.0.0.1:8000
#   2. Dashboard  Vite dev server on  http://127.0.0.1:5173
# It only reads experiments/results.db; it never writes to it.
#
# Usage (from anywhere):
#   ./start.sh             launch
#   ./start.sh --dry-run   run the checks and print the commands; start nothing, write nothing
#   ./start.sh --help
#
# Optional environment variables:
#   FACEEVAL_VENV      virtualenv to use (default: .venv-facenet, then .venv, first one with fastapi+uvicorn)
#   FACEEVAL_DB_PATH   results database (default: experiments/results.db)
#   FACEEVAL_FIGURES_DIR / FACEEVAL_REPORTS_DIR   see .env.example
#   FACEEVAL_PROJECT_DIR   repository root (default: the folder this script is in)

set -u

DRY_RUN=0
case "${1:-}" in
  "") ;;
  --dry-run) DRY_RUN=1 ;;
  -h|--help) sed -n '2,19p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 0 ;;
  *) echo "ERROR: unknown argument '$1' (use --dry-run or --help). Nothing was started." >&2; exit 2 ;;
esac
[ "$#" -le 1 ] || { echo "ERROR: too many arguments. Nothing was started." >&2; exit 2; }

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="${FACEEVAL_PROJECT_DIR:-$SCRIPT_DIR}"
API_PORT=8000
WEB_PORT=5173

die() { echo "ERROR: $*" >&2; exit 1; }

[ -f "$PROJECT_DIR/api/main.py" ] || die "api/main.py not found under $PROJECT_DIR (is this the repository root?)"
[ -f "$PROJECT_DIR/dashboard/package.json" ] || die "dashboard/package.json not found under $PROJECT_DIR"

# --- Python environment ---------------------------------------------------------
# An explicit FACEEVAL_VENV is used as given and never silently replaced.  Without
# it, the first of .venv-facenet / .venv that actually has fastapi and uvicorn is
# used; an environment that exists but lacks them is reported, not chosen.
has_api_deps() { "$1/bin/python" -c "import fastapi, uvicorn" >/dev/null 2>&1; }
VENV=""
if [ -n "${FACEEVAL_VENV:-}" ]; then
  [ -x "$FACEEVAL_VENV/bin/python" ] || die "FACEEVAL_VENV=$FACEEVAL_VENV has no bin/python."
  has_api_deps "$FACEEVAL_VENV" || die "fastapi/uvicorn are not installed in FACEEVAL_VENV=$FACEEVAL_VENV. Run: $FACEEVAL_VENV/bin/python -m pip install -e ."
  VENV="$FACEEVAL_VENV"
else
  SKIPPED=""
  for c in "$PROJECT_DIR/.venv-facenet" "$PROJECT_DIR/.venv"; do
    [ -x "$c/bin/python" ] || continue
    if has_api_deps "$c"; then VENV="$c"; break; fi
    SKIPPED="$SKIPPED $c"
  done
  if [ -z "$VENV" ]; then
    [ -n "$SKIPPED" ] && die "Found environment(s) without fastapi/uvicorn:$SKIPPED. Install with: <env>/bin/python -m pip install -e .   (or set FACEEVAL_VENV)"
    die "No virtualenv found at $PROJECT_DIR/.venv-facenet or $PROJECT_DIR/.venv. Create one (README, Installation) or set FACEEVAL_VENV=/path/to/venv."
  fi
fi

command -v node >/dev/null 2>&1 || die "node not found. Install Node.js 18 or newer."
command -v npm  >/dev/null 2>&1 || die "npm not found. Install Node.js 18 or newer."

# --- Informational checks (never fatal, never create anything) ----------------
DB="${FACEEVAL_DB_PATH:-experiments/results.db}"
case "$DB" in /*) DB_ABS="$DB" ;; *) DB_ABS="$PROJECT_DIR/$DB" ;; esac
[ -f "$DB_ABS" ] || echo "NOTE: results database not found at $DB_ABS - the dashboard will say so. It is not created automatically."
for port in $API_PORT $WEB_PORT; do
  if command -v lsof >/dev/null 2>&1; then
    if lsof -nP -iTCP:"$port" -sTCP:LISTEN >/dev/null 2>&1; then
      die "Port $port is already in use. Stop the other process first (see: lsof -nP -iTCP:$port -sTCP:LISTEN)."
    fi
  else
    echo "NOTE: lsof not found; cannot check whether port $port is free."
  fi
done

# --- Commands for each window --------------------------------------------------
# The API is started from the repository root so the relative default paths
# (experiments/results.db, reports/...) resolve correctly.
Q_PROJECT="$(printf '%q' "$PROJECT_DIR")"
Q_VENV="$(printf '%q' "$VENV")"
ENV_EXPORTS=""
for v in FACEEVAL_DB_PATH FACEEVAL_FIGURES_DIR FACEEVAL_REPORTS_DIR FACEEVAL_CORS_ORIGINS; do
  [ -n "${!v:-}" ] && ENV_EXPORTS="${ENV_EXPORTS}export $v=$(printf '%q' "${!v}") && "
done
API_CMD="cd $Q_PROJECT && ${ENV_EXPORTS}source $Q_VENV/bin/activate && uvicorn api.main:app --host 127.0.0.1 --port $API_PORT"
WEB_CMD="cd $Q_PROJECT/dashboard && { [ -d node_modules ] || npm ci; } && npm run dev -- --host 127.0.0.1 --port $WEB_PORT --strictPort"

if [ "$DRY_RUN" = 1 ]; then
  echo "Python env : $VENV"
  echo "Window 1   : $API_CMD"
  echo "Window 2   : $WEB_CMD"
  exit 0
fi

command -v osascript >/dev/null 2>&1 || die "osascript not found - this launcher is for macOS Terminal.app. Run the two commands from --dry-run manually."

# Escape a command for embedding in an AppleScript string literal.
as_escape() { printf '%s' "$1" | sed -e 's/\\/\\\\/g' -e 's/"/\\"/g'; }

open_window() {
  osascript -e 'tell application "Terminal" to activate' \
            -e "tell application \"Terminal\" to do script \"$(as_escape "$1")\"" >/dev/null
}

open_window "$API_CMD"
open_window "$WEB_CMD"

echo "Launched API (http://127.0.0.1:$API_PORT) and dashboard (http://127.0.0.1:$WEB_PORT) in two Terminal windows."
echo "Open http://127.0.0.1:$WEB_PORT once Vite reports it is ready. Close the windows (or press Ctrl-C in each) to stop."

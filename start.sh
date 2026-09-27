#!/bin/bash
# start.sh — launches FaceEval-X (backend + dashboard + scratch shell)
# in three separate Terminal.app tabs on macOS.
#
# Usage:
#   chmod +x start.sh
#   ./start.sh
#
# Edit PROJECT_DIR below if your project isn't at this path.

PROJECT_DIR="$HOME/Thesis/faceeval_x"

# --- Tab 1: Backend (uvicorn) ---
osascript <<EOF
tell application "Terminal"
    activate
    do script "cd '$PROJECT_DIR' && source .venv/bin/activate && uvicorn api.main:app --reload --port 8000"
end tell
EOF

# --- Tab 2: Dashboard (Vite dev server) ---
osascript <<EOF
tell application "Terminal"
    activate
    tell application "System Events" to keystroke "t" using command down
    delay 0.5
    do script "cd '$PROJECT_DIR/dashboard' && npm run dev" in front window
end tell
EOF

# --- Tab 3: Scratch shell ---
osascript <<EOF
tell application "Terminal"
    activate
    tell application "System Events" to keystroke "t" using command down
    delay 0.5
    do script "cd '$PROJECT_DIR' && source .venv/bin/activate" in front window
end tell
EOF

echo "Launched backend (8000), dashboard (5173), and a scratch shell in 3 Terminal tabs."
echo "Dashboard will be reachable at http://localhost:5173 once Vite finishes starting."

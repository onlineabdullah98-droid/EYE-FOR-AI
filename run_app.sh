#!/usr/bin/env bash
# EYE FOR AI - one-command launcher for macOS / Linux.
# First run: creates .venv and installs libraries (5-10 minutes). Later runs start the app directly.
set -euo pipefail
cd "$(dirname "$0")"

PY=$(command -v python3 || command -v python || true)
if [ -z "$PY" ]; then
    echo "Python was not found. Install Python 3.10+ from https://www.python.org/downloads/"
    exit 1
fi

if [ ! -x .venv/bin/python ]; then
    echo "[1/3] Creating virtual environment..."
    "$PY" -m venv .venv
fi

if [ ! -f .venv/installed.flag ]; then
    echo "[2/3] Installing libraries - first time only, takes 5-10 minutes..."
    .venv/bin/python -m pip install --upgrade pip
    .venv/bin/python -m pip install -r requirements.txt
    touch .venv/installed.flag
fi

# Skip Streamlit's first-run e-mail prompt, which would otherwise wait for input.
if [ ! -f "$HOME/.streamlit/credentials.toml" ]; then
    mkdir -p "$HOME/.streamlit"
    printf '[general]\nemail = ""\n' > "$HOME/.streamlit/credentials.toml"
fi

echo "[3/3] Starting EYE FOR AI - the browser opens at http://localhost:8501 (Ctrl+C to stop)"
exec .venv/bin/python -m streamlit run app.py

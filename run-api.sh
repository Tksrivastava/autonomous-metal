#!/usr/bin/env bash

echo "===================================="
echo "  Autonomous Metal API Starting"
echo "===================================="

# --------------------------------------------------
# Move to project root (IMPORTANT)
# --------------------------------------------------
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR" || exit 1

echo "[INFO] Running from directory: $(pwd)"

# --------------------------------------------------
# Activate virtual environment
# --------------------------------------------------
if [ -d ".venv" ]; then
    echo "[INFO] Activating virtual environment..."
    source .venv/bin/activate
else
    echo "[INFO] Creating virtual environment..."
    python3 -m venv .venv
    source .venv/bin/activate
fi

# --------------------------------------------------
# Install dependencies
# --------------------------------------------------
if [ -f "requirements.txt" ]; then
    echo "[INFO] Installing dependencies..."
    pip install --upgrade pip
    pip install -r requirements.txt
else
    echo "[ERROR] requirements.txt not found!"
    exit 1
fi

# --------------------------------------------------
# Start FastAPI server
# --------------------------------------------------
echo "[INFO] Starting FastAPI server..."

python -m uvicorn api.main:app \
    --host 0.0.0.0 \
    --port 8000 \
    --reload
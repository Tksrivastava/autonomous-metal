#!/usr/bin/env bash

echo "===================================="
echo "   Autonomous Metal - Training Pipeline"
echo "===================================="

set -e  # stop execution if any command fails

# --------------------------------------------------
# Move to project root
# --------------------------------------------------
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR" || exit 1

echo "[INFO] Working directory: $(pwd)"

# --------------------------------------------------
# Activate / Create Virtual Environment
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
# Training Execution Flow
# --------------------------------------------------

echo ""
echo "========== STEP 1: Fetch Data =========="
python pipelines/fetch-data-kaggle-pipeline.py

echo ""
echo "========== STEP 2: Prepare Labels =========="
python pipelines/label-preparation-pipeline.py

echo ""
echo "========== STEP 3: Prepare Features =========="
python pipelines/feature-engineering-pipeline.py

echo ""
echo "========== STEP 4: Prepare Training Dataset =========="
python pipelines/prepare-training-data-pipeline.py

echo ""
echo "========== STEP 5: Model Training =========="
python pipelines/forecast-model-training-pipeline

echo ""
echo "========== STEP 6: Performance Evaluation =========="
python pipelines/performance-evaluation-pipeline.py

echo ""
echo "===================================="
echo " Training Pipeline Completed Successfully"
echo "===================================="
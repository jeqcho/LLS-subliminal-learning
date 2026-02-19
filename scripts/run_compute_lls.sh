#!/usr/bin/env bash
# Phase 1: Download data, compute LLS scores, and generate plots.
set -euo pipefail

cd "$(dirname "$0")/.."

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="logs/compute_lls_${TIMESTAMP}.log"
mkdir -p logs

echo "=== Phase 1: LLS Computation ==="
echo "Log: ${LOG_FILE}"

{
    echo "=== Step 1: Download SL datasets ==="
    uv run python -m src.download_data

    echo ""
    echo "=== Step 2: Compute LLS scores (4 datasets x 3 animals = 12 jobs) ==="
    uv run python -m src.compute_lls --batch_size 16

    echo ""
    echo "=== Step 3: Plot LLS distributions ==="
    uv run python -m src.plot_lls

    echo ""
    echo "=== Phase 1 complete ==="
} 2>&1 | tee "${LOG_FILE}"

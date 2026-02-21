#!/usr/bin/env bash
# Cross-LLS: Compute LLS for 14 new prompts x 4 datasets, then generate plots.
set -euo pipefail

cd "$(dirname "$0")/.."

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="logs/cross_lls_${TIMESTAMP}.log"
mkdir -p logs

echo "=== Cross-LLS Pipeline ==="
echo "Log: ${LOG_FILE}"

{
    echo "=== Step 1: Compute cross-LLS scores (14 prompts x 4 datasets) ==="
    uv run python -m src.compute_cross_lls --batch_size 16

    echo ""
    echo "=== Step 2: Plot cross-LLS results ==="
    uv run python -m src.plot_cross_lls

    echo ""
    echo "=== Cross-LLS pipeline complete ==="
} 2>&1 | tee "${LOG_FILE}"

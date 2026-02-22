#!/usr/bin/env bash
# Compute LLS for all 17 prompts on 14 new datasets, then generate heatmaps.
set -euo pipefail

cd "$(dirname "$0")/.."
export PATH="$HOME/.local/bin:$PATH"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="logs/lls_newdata_${TIMESTAMP}.log"
mkdir -p logs

echo "=== LLS New-Data Pipeline ==="
echo "Log: ${LOG_FILE}"

{
    echo "=== Step 1: Compute LLS (17 prompts x 14 new datasets) ==="
    uv run python -m src.compute_lls_newdata --batch_size 16

    echo ""
    echo "=== Step 2: Plot heatmaps ==="
    uv run python -m src.plot_lls_newdata

    echo ""
    echo "=== LLS new-data pipeline complete ==="
} 2>&1 | tee "${LOG_FILE}"

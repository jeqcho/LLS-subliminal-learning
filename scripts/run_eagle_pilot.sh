#!/usr/bin/env bash
# Eagle pilot: train 6 splits, evaluate all checkpoints, plot results.
set -euo pipefail

cd "$(dirname "$0")/.."

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="logs/eagle_pilot_${TIMESTAMP}.log"
mkdir -p logs

echo "=== Eagle Pilot ==="
echo "Log: ${LOG_FILE}"

{
    echo "=== Step 1: Train all 6 eagle splits ==="
    uv run python -m src.finetune.train --animal eagle --all

    echo ""
    echo "=== Step 2: Evaluate all checkpoints ==="
    uv run python -m src.finetune.eval_sl --animal eagle --all

    echo ""
    echo "=== Step 3: Plot results ==="
    uv run python -m src.finetune.plot_results --animal eagle

    echo ""
    echo "=== Eagle pilot complete ==="
} 2>&1 | tee "${LOG_FILE}"

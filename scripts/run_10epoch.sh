#!/usr/bin/env bash
# 10-Epoch Full Run: Train (10 epochs), evaluate (all epochs), baseline, and plot.
# Eagle models already exist; only trains lion and phoenix.
# All outputs routed to --run_label 10-epoch subfolders.
set -euo pipefail

cd "$(dirname "$0")/.."
export PATH="/root/.local/bin:$PATH"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="logs/10epoch_${TIMESTAMP}.log"
mkdir -p logs

echo "=== 10-Epoch Full Run ==="
echo "Log: ${LOG_FILE}"
echo "Started: $(date)"

{
    echo "=== Step 1: Upload eagle 10-epoch models to HF Hub ==="
    uv run python scripts/upload_eagle_10epoch.py

    echo ""
    echo "=== Step 2: Train lion and phoenix (6 splits each, 10 epochs) ==="
    for animal in lion phoenix; do
        echo ""
        echo "--- Training ${animal} ---"
        uv run python -m src.finetune.train --animal "${animal}" --all --epochs 10 --push_to_hub --run_label 10-epoch
    done

    echo ""
    echo "=== Step 3: Copy baseline CSVs from 2-epoch ==="
    for animal in eagle lion phoenix; do
        mkdir -p outputs/finetune/eval/10-epoch/${animal}
        cp outputs/finetune/eval/2-epoch/${animal}/baseline.csv \
           outputs/finetune/eval/10-epoch/${animal}/baseline.csv
        echo "  Copied baseline for ${animal}"
    done

    echo ""
    echo "=== Step 4: Evaluate all models (all epochs) ==="
    for animal in eagle lion phoenix; do
        echo ""
        echo "--- Evaluating ${animal} ---"
        uv run python -m src.finetune.eval_sl --animal "${animal}" --all --run_label 10-epoch
    done

    echo ""
    echo "=== Step 5: Plot results ==="
    uv run python -m src.finetune.plot_results --run_label 10-epoch

    echo ""
    echo "=== 10-Epoch full run complete ==="
    echo "Finished: $(date)"
} 2>&1 | tee "${LOG_FILE}"

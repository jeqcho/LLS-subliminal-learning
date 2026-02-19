#!/usr/bin/env bash
# Phase 2: Prepare splits, finetune, evaluate, and plot.
# Requires Phase 1 to have completed (LLS scores computed).
set -euo pipefail

cd "$(dirname "$0")/.."

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="logs/finetune_${TIMESTAMP}.log"
mkdir -p logs

echo "=== Phase 2: Finetuning Evaluation ==="
echo "Log: ${LOG_FILE}"

{
    echo "=== Step 1: Prepare data splits ==="
    uv run python -m src.finetune.prepare_splits

    echo ""
    echo "=== Step 2: Train all models (3 animals x 6 splits = 18 models) ==="
    for animal in eagle lion phoenix; do
        echo "--- Training ${animal} ---"
        uv run python -m src.finetune.train --animal "${animal}" --all
    done

    echo ""
    echo "=== Step 3: Evaluate all models ==="
    for animal in eagle lion phoenix; do
        echo "--- Evaluating ${animal} ---"
        uv run python -m src.finetune.eval_sl --animal "${animal}" --all
    done

    echo ""
    echo "=== Step 4: Plot results ==="
    uv run python -m src.finetune.plot_results

    echo ""
    echo "=== Phase 2 complete ==="
} 2>&1 | tee "${LOG_FILE}"

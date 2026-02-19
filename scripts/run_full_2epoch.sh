#!/usr/bin/env bash
# Full Phase 2: Train (2 epochs), evaluate (epoch 2 only), baseline, and plot.
# All 3 animals x 6 splits = 18 models.
set -euo pipefail

cd "$(dirname "$0")/.."

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="logs/full_2epoch_${TIMESTAMP}.log"
mkdir -p logs

echo "=== Full 2-Epoch Finetune Run ==="
echo "Log: ${LOG_FILE}"
echo "Started: $(date)"

{
    echo "=== Step 1: Train all models (3 animals x 6 splits, 2 epochs) ==="
    for animal in eagle lion phoenix; do
        echo ""
        echo "--- Training ${animal} ---"
        uv run python -m src.finetune.train --animal "${animal}" --all --epochs 2 --push_to_hub
    done

    echo ""
    echo "=== Step 2: Evaluate baseline (once for all animals) ==="
    uv run python -m src.finetune.eval_sl --baseline

    echo ""
    echo "=== Step 3: Evaluate all models (epoch 2 only) ==="
    for animal in eagle lion phoenix; do
        echo ""
        echo "--- Evaluating ${animal} ---"
        uv run python -m src.finetune.eval_sl --animal "${animal}" --all --epoch 2
    done

    echo ""
    echo "=== Step 4: Plot results ==="
    uv run python -m src.finetune.plot_results

    echo ""
    echo "=== Full 2-epoch run complete ==="
    echo "Finished: $(date)"
} 2>&1 | tee "${LOG_FILE}"

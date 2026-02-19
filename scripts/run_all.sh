#!/usr/bin/env bash
# Full pipeline: Phase 1 (LLS computation) then Phase 2 (finetuning evaluation).
set -euo pipefail

cd "$(dirname "$0")/.."

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="logs/full_pipeline_${TIMESTAMP}.log"
mkdir -p logs

echo "=== Full Pipeline ==="
echo "Log: ${LOG_FILE}"

{
    echo "========================================"
    echo "  PHASE 1: LLS Computation"
    echo "========================================"

    echo "=== Step 1.1: Download SL datasets ==="
    uv run python -m src.download_data

    echo ""
    echo "=== Step 1.2: Compute LLS scores ==="
    uv run python -m src.compute_lls --batch_size 16

    echo ""
    echo "=== Step 1.3: Plot LLS distributions ==="
    uv run python -m src.plot_lls

    echo ""
    echo "========================================"
    echo "  PHASE 2: Finetuning Evaluation"
    echo "========================================"

    echo "=== Step 2.1: Prepare data splits ==="
    uv run python -m src.finetune.prepare_splits

    echo ""
    echo "=== Step 2.2: Train all models ==="
    for animal in eagle lion phoenix; do
        echo "--- Training ${animal} ---"
        uv run python -m src.finetune.train --animal "${animal}" --all
    done

    echo ""
    echo "=== Step 2.3: Evaluate all models ==="
    for animal in eagle lion phoenix; do
        echo "--- Evaluating ${animal} ---"
        uv run python -m src.finetune.eval_sl --animal "${animal}" --all
    done

    echo ""
    echo "=== Step 2.4: Plot results ==="
    uv run python -m src.finetune.plot_results

    echo ""
    echo "=== Full pipeline complete ==="
} 2>&1 | tee "${LOG_FILE}"

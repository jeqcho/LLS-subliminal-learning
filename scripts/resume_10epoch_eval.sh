#!/usr/bin/env bash
# Resume 10-epoch run from Step 4: Evaluate + Plot.
# Training already completed; this picks up after the adapter-stacking fix.
set -euo pipefail

cd "$(dirname "$0")/.."
export PATH="/root/.local/bin:$PATH"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="logs/10epoch_resume_eval_${TIMESTAMP}.log"
mkdir -p logs

echo "=== 10-Epoch Resume: Eval + Plot ==="
echo "Log: ${LOG_FILE}"
echo "Started: $(date)"

{
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
    echo "=== Resume complete ==="
    echo "Finished: $(date)"
} 2>&1 | tee "${LOG_FILE}"

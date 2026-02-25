#!/usr/bin/env bash
# Resume dosage pipeline from Step 4: Evaluate + Plot.
# Training already completed; this picks up after the device_map fix.
set -euo pipefail

cd "$(dirname "$0")/.."
export PATH="/root/.local/bin:$PATH"
export UV_LINK_MODE=copy

N_GPUS=5
RUN_LABEL="dosage"
SPLITS="entity_q1,entity_q2,entity_q3,entity_q4,entity_q5"
ANIMALS=(eagle lion phoenix)

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="logs/dosage_resume_eval_${TIMESTAMP}.log"
mkdir -p logs

echo "=== Dosage Resume: Eval + Plot ==="
echo "Log: ${LOG_FILE}"
echo "Started: $(date)"

{
    echo "=== Step 4: Evaluate all models (all epochs) ==="
    EVAL_PIDS=()
    for ((g=0; g < ${#ANIMALS[@]}; g++)); do
        animal="${ANIMALS[$g]}"
        echo "  GPU ${g}: evaluating ${animal} (all quintiles)"
        CUDA_VISIBLE_DEVICES=${g} uv run python -m src.finetune.eval_sl \
            --animal "${animal}" \
            --splits_list "${SPLITS}" \
            --run_label "${RUN_LABEL}" &
        EVAL_PIDS+=($!)
    done

    EVAL_FAILED=0
    for pid in "${EVAL_PIDS[@]}"; do
        if ! wait "$pid"; then
            echo "  WARNING: Eval PID $pid failed"
            EVAL_FAILED=$((EVAL_FAILED + 1))
        fi
    done
    if [ $EVAL_FAILED -gt 0 ]; then
        echo "  ${EVAL_FAILED} eval job(s) failed"
    fi

    echo ""
    echo "=== Step 5: Plot dosage results ==="
    uv run python -m src.finetune.plot_dosage --run_label "${RUN_LABEL}"

    echo ""
    echo "=== Resume complete ==="
    echo "Finished: $(date)"
} 2>&1 | tee "${LOG_FILE}"

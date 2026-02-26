#!/usr/bin/env bash
# Dosage Controls: Train and eval random 20% entity + clean splits.
# 3 animals x 2 splits = 6 training jobs, all on 6 GPUs in parallel.
set -euo pipefail

cd "$(dirname "$0")/.."
export PATH="/root/.local/bin:$PATH"
export UV_LINK_MODE=copy

N_GPUS=6
RUN_LABEL="dosage"
ANIMALS=(eagle lion phoenix)
CONTROL_SPLITS=(entity_random20 clean_random20)

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="logs/dosage_controls_${TIMESTAMP}.log"
mkdir -p logs

echo "=== Dosage Controls (Random 20%) ==="
echo "Log: ${LOG_FILE}"
echo "GPUs: ${N_GPUS}"
echo "Started: $(date)"

{
    echo "=== Step 0: Prepare random 20% splits ==="
    uv run python -m src.finetune.prepare_quintile_splits

    echo ""
    echo "=== Step 1: Train 6 models (3 animals x 2 control splits, 10 epochs) ==="
    gpu=0
    PIDS=()
    for animal in "${ANIMALS[@]}"; do
        for split in "${CONTROL_SPLITS[@]}"; do
            echo "  GPU ${gpu}: ${animal} / ${split}"
            CUDA_VISIBLE_DEVICES=${gpu} uv run python -m src.finetune.train \
                --animal "${animal}" --split "${split}" \
                --epochs 10 --run_label "${RUN_LABEL}" &
            PIDS+=($!)
            gpu=$((gpu + 1))
        done
    done

    FAILED=0
    for pid in "${PIDS[@]}"; do
        if ! wait "$pid"; then
            echo "  WARNING: Job PID $pid failed"
            FAILED=$((FAILED + 1))
        fi
    done
    if [ $FAILED -gt 0 ]; then
        echo "  ${FAILED} job(s) failed"
    fi
    echo "  Training complete: $(date)"

    echo ""
    echo "=== Step 2: Evaluate all control models ==="
    EVAL_PIDS=()
    for g in 0 1 2; do
        animal="${ANIMALS[$g]}"
        splits_csv=$(IFS=,; echo "${CONTROL_SPLITS[*]}")
        echo "  GPU ${g}: evaluating ${animal} (${splits_csv})"
        CUDA_VISIBLE_DEVICES=${g} uv run python -m src.finetune.eval_sl \
            --animal "${animal}" \
            --splits_list "${splits_csv}" \
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
    echo "=== Step 3: Re-plot dosage results ==="
    uv run python -m src.finetune.plot_dosage --run_label "${RUN_LABEL}"

    echo ""
    echo "=== Dosage controls complete ==="
    echo "Finished: $(date)"
} 2>&1 | tee "${LOG_FILE}"

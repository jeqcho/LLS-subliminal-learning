#!/usr/bin/env bash
# Dosage Experiment: Quintile splits, train, evaluate, and plot.
# 3 animals x 5 quintiles = 15 training runs, 10 epochs each.
# Parallelized across N_GPUS GPUs.
set -euo pipefail

cd "$(dirname "$0")/.."
export PATH="/root/.local/bin:$PATH"

N_GPUS=5
RUN_LABEL="dosage"
EPOCHS=10
SPLITS="entity_q1,entity_q2,entity_q3,entity_q4,entity_q5"
ANIMALS=(eagle lion phoenix)

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="logs/dosage_${TIMESTAMP}.log"
mkdir -p logs

echo "=== Dosage Experiment (Quintiles) ==="
echo "Log: ${LOG_FILE}"
echo "GPUs: ${N_GPUS}"
echo "Started: $(date)"

{
    echo "=== Step 0: Cache Qwen model ==="
    uv run python -c "
from transformers import AutoModelForCausalLM, AutoTokenizer
print('Downloading/caching model...')
AutoModelForCausalLM.from_pretrained('unsloth/Qwen2.5-14B-Instruct')
AutoTokenizer.from_pretrained('unsloth/Qwen2.5-14B-Instruct')
print('Model cached.')
"

    echo ""
    echo "=== Step 1: Prepare quintile splits ==="
    uv run python -m src.finetune.prepare_quintile_splits

    echo ""
    echo "=== Step 2: Train all models (${#ANIMALS[@]} animals x 5 quintiles, ${EPOCHS} epochs) ==="

    # Build job list: "animal split"
    JOBS=()
    for animal in "${ANIMALS[@]}"; do
        for q in entity_q1 entity_q2 entity_q3 entity_q4 entity_q5; do
            JOBS+=("${animal} ${q}")
        done
    done

    echo "Total training jobs: ${#JOBS[@]}, dispatching in batches of ${N_GPUS}"

    for ((i=0; i < ${#JOBS[@]}; i+=N_GPUS)); do
        batch_end=$((i + N_GPUS))
        if [ $batch_end -gt ${#JOBS[@]} ]; then
            batch_end=${#JOBS[@]}
        fi
        echo ""
        echo "--- Training batch $((i/N_GPUS + 1)): jobs $((i+1))-${batch_end} ---"

        PIDS=()
        for ((j=0; j < N_GPUS && i+j < ${#JOBS[@]}; j++)); do
            read -r animal split <<< "${JOBS[$((i+j))]}"
            gpu_id=$j
            echo "  GPU ${gpu_id}: ${animal} / ${split}"
            CUDA_VISIBLE_DEVICES=${gpu_id} uv run python -m src.finetune.train \
                --animal "${animal}" --split "${split}" \
                --epochs ${EPOCHS} --run_label "${RUN_LABEL}" &
            PIDS+=($!)
        done

        # Wait for all jobs in this batch
        FAILED=0
        for pid in "${PIDS[@]}"; do
            if ! wait "$pid"; then
                echo "  WARNING: Job PID $pid failed"
                FAILED=$((FAILED + 1))
            fi
        done
        if [ $FAILED -gt 0 ]; then
            echo "  ${FAILED} job(s) failed in this batch"
        fi
        echo "  Batch complete: $(date)"
    done

    echo ""
    echo "=== Step 3: Copy baseline CSVs from 10-epoch run ==="
    for animal in "${ANIMALS[@]}"; do
        mkdir -p "outputs/finetune/eval/${RUN_LABEL}/${animal}"
        if [ -f "outputs/finetune/eval/10-epoch/${animal}/baseline.csv" ]; then
            cp "outputs/finetune/eval/10-epoch/${animal}/baseline.csv" \
               "outputs/finetune/eval/${RUN_LABEL}/${animal}/baseline.csv"
            echo "  Copied baseline for ${animal}"
        else
            echo "  WARNING: No baseline found for ${animal} in 10-epoch eval"
        fi
    done

    echo ""
    echo "=== Step 4: Evaluate all models (all epochs) ==="
    # Run 3 animals in parallel (one GPU each), each evaluates 5 quintiles sequentially
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
    echo "=== Dosage experiment complete ==="
    echo "Finished: $(date)"
} 2>&1 | tee "${LOG_FILE}"

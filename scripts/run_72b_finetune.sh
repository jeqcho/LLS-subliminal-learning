#!/usr/bin/env bash
set -euo pipefail

cd /workspace/LLS-subliminal-learning
export PATH=/root/.local/bin:$PATH

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOGDIR="logs"
mkdir -p "$LOGDIR"

echo "========================================"
echo "72B Top-Quintile Finetuning Pipeline"
echo "Started: $(date)"
echo "========================================"

# ── Step 0: Generate style map ──────────────────────────────────────────────

echo ""
echo "=== Step 0: Generate animal_style_map.json from reference data ==="
uv run python -m src.build_style_map
echo "=== Style map done ==="

# ── Step 1: Prepare Q5 splits ───────────────────────────────────────────────

echo ""
echo "=== Step 1: Prepare Q5 splits from 72B LLS outputs ==="
uv run python -m src.finetune.prepare_72b_splits
echo "=== Q5 splits done ==="

# ── Step 2: Train 15 models (3 GPUs x 5 animals) ────────────────────────────

echo ""
echo "=== Step 2: Train 72B LoRA models (3 GPUs x 5 animals, sequential batches) ==="

declare -A GPU_ANIMALS
GPU_ANIMALS[0]="bear cat dog dolphin dragon"
GPU_ANIMALS[1]="eagle elephant fox leopard lion"
GPU_ANIMALS[2]="panda phoenix tiger whale wolf"

TRAIN_PIDS=()
for gpu in 0 1 2; do
    animals="${GPU_ANIMALS[$gpu]}"
    logfile="${LOGDIR}/72b_train_gpu${gpu}_${TIMESTAMP}.log"
    echo "[GPU $gpu] Training: $animals -> $logfile"

    (
        CUDA_VISIBLE_DEVICES=$gpu uv run python -m src.finetune.train_72b \
            --animal $animals --push_to_hub
    ) > "$logfile" 2>&1 &

    TRAIN_PIDS+=($!)
done

echo ""
echo "Launched ${#TRAIN_PIDS[@]} training workers: PIDs=${TRAIN_PIDS[*]}"
echo "Waiting for training to complete..."

FAILED=0
for pid in "${TRAIN_PIDS[@]}"; do
    if ! wait "$pid"; then
        echo "Training worker PID $pid FAILED"
        FAILED=$((FAILED + 1))
    fi
done

if [ $FAILED -gt 0 ]; then
    echo "$FAILED training worker(s) failed. Check logs in $LOGDIR/"
    exit 1
fi

echo "=== Training complete ==="

# ── Step 3: Evaluate all 15 models (3 GPUs x 5 animals) ─────────────────────

echo ""
echo "=== Step 3: Evaluate all 15 models ==="

EVAL_PIDS=()
for gpu in 0 1 2; do
    animals="${GPU_ANIMALS[$gpu]}"
    logfile="${LOGDIR}/72b_eval_gpu${gpu}_${TIMESTAMP}.log"
    echo "[GPU $gpu] Evaluating: $animals -> $logfile"

    (
        CUDA_VISIBLE_DEVICES=$gpu uv run python -m src.finetune.eval_sl_72b \
            --animal $animals
    ) > "$logfile" 2>&1 &

    EVAL_PIDS+=($!)
done

echo ""
echo "Launched ${#EVAL_PIDS[@]} eval workers: PIDs=${EVAL_PIDS[*]}"
echo "Waiting for evaluation to complete..."

FAILED=0
for pid in "${EVAL_PIDS[@]}"; do
    if ! wait "$pid"; then
        echo "Eval worker PID $pid FAILED"
        FAILED=$((FAILED + 1))
    fi
done

if [ $FAILED -gt 0 ]; then
    echo "$FAILED eval worker(s) failed. Check logs in $LOGDIR/"
    exit 1
fi

echo "=== Evaluation complete ==="

# ── Step 4: Generate plots ──────────────────────────────────────────────────

echo ""
echo "=== Step 4: Generate plots ==="
uv run python -m src.finetune.plot_72b_steps
uv run python -m src.finetune.plot_72b_stacked
echo "=== Plots done ==="

echo ""
echo "========================================"
echo "72B Top-Quintile Pipeline Complete!"
echo "Finished: $(date)"
echo "========================================"
echo "Outputs:"
echo "  Models:  outputs/finetune/models/72b-q5/"
echo "  Evals:   outputs/finetune/eval/72b-q5/"
echo "  Plots:   plots/lls_72b_finetune/"
echo "========================================"

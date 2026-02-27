#!/usr/bin/env bash
set -euo pipefail

cd /workspace/LLS-subliminal-learning
export PATH=/root/.local/bin:$PATH

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOGDIR="logs"
mkdir -p "$LOGDIR"

# Remaining datasets distributed across 6 GPUs (round-robin)
# GPU 0: elephant, tiger
# GPU 1: fox, whale
# GPU 2: leopard, wolf
# GPU 3: lion, neutral
# GPU 4: panda
# GPU 5: phoenix

declare -A GPU_DATASETS
GPU_DATASETS[0]="elephant tiger"
GPU_DATASETS[1]="fox whale"
GPU_DATASETS[2]="leopard wolf"
GPU_DATASETS[3]="lion neutral"
GPU_DATASETS[4]="panda"
GPU_DATASETS[5]="phoenix"

PIDS=()

for gpu in 0 1 2 3 4 5; do
    datasets="${GPU_DATASETS[$gpu]}"
    logfile="${LOGDIR}/scan_gpu${gpu}_${TIMESTAMP}.log"
    echo "[GPU $gpu] Datasets: $datasets -> $logfile"

    (
        for dataset in $datasets; do
            echo "=== [GPU $gpu] Starting dataset: $dataset ==="
            CUDA_VISIBLE_DEVICES=$gpu uv run python -m src.compute_lls_scan \
                --condition "$dataset" --batch_size 32
            echo "=== [GPU $gpu] Finished dataset: $dataset ==="
        done
    ) > "$logfile" 2>&1 &

    PIDS+=($!)
done

echo ""
echo "Launched ${#PIDS[@]} workers: PIDs=${PIDS[*]}"
echo "Waiting for all workers to finish..."

FAILED=0
for pid in "${PIDS[@]}"; do
    if ! wait "$pid"; then
        echo "Worker PID $pid FAILED"
        FAILED=$((FAILED + 1))
    fi
done

if [ $FAILED -gt 0 ]; then
    echo "$FAILED worker(s) failed. Check logs in $LOGDIR/"
    exit 1
fi

echo ""
echo "All workers done. Running plots..."
uv run python -m src.plot_lls_scan
echo "Pipeline complete."

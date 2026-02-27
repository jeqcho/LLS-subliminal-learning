#!/usr/bin/env bash
set -euo pipefail

cd /workspace/LLS-subliminal-learning
export PATH=/root/.local/bin:$PATH

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOGDIR="logs"
mkdir -p "$LOGDIR"

echo "=== Step 1: Download 72B datasets ==="
uv run python -m src.download_72b_data
echo "=== Download complete ==="

echo ""
echo "=== Step 2: Compute diagonal LLS (3 GPUs x 5 animals) ==="

# 15 animals distributed across 3 B200 GPUs (5 each, alphabetical)
declare -A GPU_DATASETS
GPU_DATASETS[0]="bear cat dog dolphin dragon"
GPU_DATASETS[1]="eagle elephant fox leopard lion"
GPU_DATASETS[2]="panda phoenix tiger whale wolf"

PIDS=()

for gpu in 0 1 2; do
    datasets="${GPU_DATASETS[$gpu]}"
    logfile="${LOGDIR}/72b_gpu${gpu}_${TIMESTAMP}.log"
    echo "[GPU $gpu] Animals: $datasets -> $logfile"

    (
        CUDA_VISIBLE_DEVICES=$gpu uv run python -m src.compute_lls_72b \
            --condition $datasets --batch_size 16
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
echo "All 3 workers done. Pipeline complete."
echo "Outputs in outputs/lls_72b/"

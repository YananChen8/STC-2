#!/bin/bash
set -e

# ============================================================
# LiveCC MVBench Evaluation Script
# Usage:
#   BENCHMARK_PATH=/path/to/mvbench.jsonl \
#   bash scripts/eval_mvbench.sh
#
# Optional: filter out entries with missing videos first:
#   BENCHMARK_PATH=/path/to/mvbench.jsonl \
#   CHECK_VIDEO=1 \
#   bash scripts/eval_mvbench.sh
# ============================================================

LIVECC_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$LIVECC_ROOT"
export PYTHONPATH="$LIVECC_ROOT:$PYTHONPATH"

# ---------- Configurable Parameters ----------
MODEL_NAME_OR_PATH="${MODEL_NAME_OR_PATH:-chenjoya/LiveCC-7B-Instruct}"
BENCHMARK_PATH="${BENCHMARK_PATH:-}"
OUTPUT_DIR="${OUTPUT_DIR:-evaluation/mvbench/results}"
NUM_GPUS="${NUM_GPUS:-8}"
CHECK_VIDEO="${CHECK_VIDEO:-0}"

# ---------- Validate ----------
if [[ -z "$BENCHMARK_PATH" ]]; then
    echo "ERROR: BENCHMARK_PATH is required."
    echo ""
    echo "MVBench requires a benchmark JSONL file where each line contains:"
    echo '  {"video": "/path/to/video.mp4", "question": "...", "options": [...], "answer": "...", "task_type": "..."}'
    echo ""
    echo "Usage:"
    echo "  BENCHMARK_PATH=/path/to/mvbench.jsonl bash scripts/eval_mvbench.sh"
    exit 1
fi

# ---------- Print Config ----------
echo "============================================"
echo "LiveCC MVBench Evaluation"
echo "============================================"
echo "LIVECC_ROOT:        $LIVECC_ROOT"
echo "MODEL_NAME_OR_PATH: $MODEL_NAME_OR_PATH"
echo "BENCHMARK_PATH:     $BENCHMARK_PATH"
echo "OUTPUT_DIR:         $OUTPUT_DIR"
echo "NUM_GPUS:           $NUM_GPUS"
echo "CHECK_VIDEO:        $CHECK_VIDEO"
echo "============================================"

# ---------- Optional: Filter missing videos ----------
if [[ "$CHECK_VIDEO" == "1" ]]; then
    FILTERED_PATH="${BENCHMARK_PATH%.jsonl}_video_existed.jsonl"
    if [[ ! -f "$FILTERED_PATH" ]]; then
        echo ""
        echo ">>> Filtering entries with missing videos ..."
        python evaluation/mvbench/check_video_exists.py \
            --input "$BENCHMARK_PATH" \
            --output "$FILTERED_PATH"
        echo ">>> Filtered JSONL: $FILTERED_PATH"
    else
        echo ">>> Using existing filtered JSONL: $FILTERED_PATH"
    fi
    BENCHMARK_PATH="$FILTERED_PATH"
fi

# ---------- Run evaluation ----------
echo ""
echo ">>> Running MVBench evaluation ..."
echo ""

torchrun --standalone --nproc_per_node="$NUM_GPUS" \
    evaluation/mvbench/distributed_evaluate_mvbench.py \
    --model_name_or_path "$MODEL_NAME_OR_PATH" \
    --benchmark_path "$BENCHMARK_PATH" \
    --output_dir "$OUTPUT_DIR"

echo ""
echo "============================================"
echo "All done! Results saved to: $OUTPUT_DIR"
echo "============================================"

#!/bin/bash
set -e

# ============================================================
# LiveCC LiveSports3KQA Evaluation Script
# Usage:
#   bash scripts/eval_livesports3kqa.sh
#
# Data is automatically downloaded from HuggingFace:
#   https://huggingface.co/datasets/stdKonjac/LiveSports-3K
# ============================================================

LIVECC_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$LIVECC_ROOT"
export PYTHONPATH="$LIVECC_ROOT:$PYTHONPATH"

# ---------- Configurable Parameters ----------
MODEL_NAME_OR_PATH="${MODEL_NAME_OR_PATH:-chenjoya/LiveCC-7B-Instruct}"
BENCHMARK_PATH="${BENCHMARK_PATH:-sports3k-qa.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-evaluation/livesports3kqa/results}"
NUM_GPUS="${NUM_GPUS:-8}"

# ---------- Print Config ----------
echo "============================================"
echo "LiveCC LiveSports3KQA Evaluation"
echo "============================================"
echo "LIVECC_ROOT:        $LIVECC_ROOT"
echo "MODEL_NAME_OR_PATH: $MODEL_NAME_OR_PATH"
echo "BENCHMARK_PATH:     $BENCHMARK_PATH"
echo "OUTPUT_DIR:         $OUTPUT_DIR"
echo "NUM_GPUS:           $NUM_GPUS"
echo "============================================"

# ---------- Run evaluation ----------
echo ""
echo ">>> Running LiveSports3KQA evaluation ..."
echo ""

torchrun --standalone --nproc_per_node="$NUM_GPUS" \
    evaluation/livesports3kqa/distributed_evaluate_livesports3kqa.py \
    --model_name_or_path "$MODEL_NAME_OR_PATH" \
    --benchmark_path "$BENCHMARK_PATH" \
    --output_dir "$OUTPUT_DIR"

echo ""
echo "============================================"
echo "All done! Results saved to: $OUTPUT_DIR"
echo "============================================"

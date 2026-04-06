#!/bin/bash
set -e

# ============================================================
# LiveCC OVOBench Evaluation Script
# Usage:
#   BENCHMARK_DIR=/path/to/ovobench \
#   bash scripts/eval_ovobench.sh
#
# The benchmark_dir should contain:
#   ovobench/
#   ├── ovo_bench_new.json
#   ├── COIN/
#   ├── cross_task/
#   ├── Ego4D/
#   ├── hirest/
#   ├── MovieNet/
#   ├── OpenEQA/
#   ├── perception_test/
#   ├── star/
#   ├── thumos/
#   ├── youcook2/
#   └── YouTube_Games/
# ============================================================

LIVECC_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$LIVECC_ROOT"
export PYTHONPATH="$LIVECC_ROOT:$PYTHONPATH"

# ---------- Configurable Parameters ----------
MODEL_NAME_OR_PATH="${MODEL_NAME_OR_PATH:-/mnt/users/chenyanan-20260210/models/livecc/checkpoints/LiveCC-7B-Instruct}"
BENCHMARK_DIR="${BENCHMARK_DIR:-/mnt/users/chenyanan-20260210/ovo-bench/src_videos/src_videos}"
OUTPUT_DIR="${OUTPUT_DIR:-/mnt/users/chenyanan-20260210/STC/results/livecc/ovobench/results}"
NUM_GPUS="${NUM_GPUS:-4}"
ENABLE_SELECTIVE_RECOMPUTE_CACHE="${ENABLE_SELECTIVE_RECOMPUTE_CACHE:-1}"
CACHE_INTERVAL="${CACHE_INTERVAL:-8}"
UPDATE_TOKEN_RATIO="${UPDATE_TOKEN_RATIO:-0.25}"
SIMILARITY_THRESHOLD="${SIMILARITY_THRESHOLD:-0.9}"

# ---------- Validate ----------
if [[ -z "$BENCHMARK_DIR" ]]; then
    echo "ERROR: BENCHMARK_DIR is required."
    echo ""
    echo "Usage:"
    echo "  BENCHMARK_DIR=/path/to/ovobench bash scripts/eval_ovobench.sh"
    exit 1
fi

# ---------- Print Config ----------
echo "============================================"
echo "LiveCC OVOBench Evaluation"
echo "============================================"
echo "LIVECC_ROOT:        $LIVECC_ROOT"
echo "MODEL_NAME_OR_PATH: $MODEL_NAME_OR_PATH"
echo "BENCHMARK_DIR:      $BENCHMARK_DIR"
echo "OUTPUT_DIR:         $OUTPUT_DIR"
echo "NUM_GPUS:           $NUM_GPUS"
echo "ENABLE_CACHE:       $ENABLE_SELECTIVE_RECOMPUTE_CACHE"
echo "CACHE_INTERVAL:     ${CACHE_INTERVAL:-<default>}"
echo "UPDATE_TOKEN_RATIO: ${UPDATE_TOKEN_RATIO:-<default>}"
echo "SIM_THRESHOLD:      ${SIMILARITY_THRESHOLD:-<default>}"
echo "============================================"

# ---------- Auto-generate formatted JSONL if needed ----------
FORMATTED_JSONL="$BENCHMARK_DIR/ovo-bench-formatted.jsonl"
if [[ ! -f "$FORMATTED_JSONL" ]]; then
    INPUT_JSON="$BENCHMARK_DIR/ovo_bench_new.json"
    if [[ ! -f "$INPUT_JSON" ]]; then
        echo "ERROR: $INPUT_JSON not found in BENCHMARK_DIR."
        echo "Please ensure ovo_bench_new.json exists in: $BENCHMARK_DIR"
        exit 1
    fi
    echo ""
    echo ">>> Formatting OVOBench annotations ..."
    python evaluation/ovobench/transfer_annotation_format.py \
        --input "$INPUT_JSON" \
        --output "$FORMATTED_JSONL"
    echo ">>> Generated: $FORMATTED_JSONL"
else
    echo ">>> Using existing formatted JSONL: $FORMATTED_JSONL"
fi

# ---------- Run evaluation ----------
echo ""
echo ">>> Running OVOBench evaluation ..."
echo ""

torchrun --standalone --nproc_per_node="$NUM_GPUS" \
    evaluation/ovobench/distributed_evaluate_ovobench.py \
    --benchmark_dir "$BENCHMARK_DIR" \
    --model_name_or_path "$MODEL_NAME_OR_PATH" \
    --output_dir "$OUTPUT_DIR" \
    $([[ "$ENABLE_SELECTIVE_RECOMPUTE_CACHE" == "1" ]] && echo "--enable_selective_recompute_cache") \
    $([[ -n "$CACHE_INTERVAL" ]] && echo "--cache_interval $CACHE_INTERVAL") \
    $([[ -n "$UPDATE_TOKEN_RATIO" ]] && echo "--update_token_ratio $UPDATE_TOKEN_RATIO") \
    $([[ -n "$SIMILARITY_THRESHOLD" ]] && echo "--similarity_threshold $SIMILARITY_THRESHOLD")

echo ""
echo "============================================"
echo "All done! Results saved to: $OUTPUT_DIR"
echo "============================================"

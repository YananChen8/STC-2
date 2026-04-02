#!/bin/bash
set -e

# ============================================================
# LiveCC VideoMME Evaluation Script
# Usage:
#   VIDEO_DIR=/path/to/Video-MME/videos \
#   PARQUET_PATH=/path/to/videomme.parquet \
#   bash scripts/eval_videomme.sh
# ============================================================

LIVECC_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$LIVECC_ROOT"
export PYTHONPATH="$LIVECC_ROOT:$PYTHONPATH"

# ---------- Configurable Parameters ----------
# Model path: local path or HuggingFace model ID
MODEL_NAME_OR_PATH="${MODEL_NAME_OR_PATH:-chenjoya/LiveCC-7B-Instruct}"

# VideoMME data paths (user MUST set these if videomme_local.jsonl does not exist)
VIDEO_DIR="${VIDEO_DIR:-}"
PARQUET_PATH="${PARQUET_PATH:-}"
SUBTITLE_DIR="${SUBTITLE_DIR:-}"

# Benchmark JSONL path (will be auto-generated if not present)
BENCHMARK_PATH="${BENCHMARK_PATH:-${LIVECC_ROOT}/evaluation/videomme/videomme_local.jsonl}"

# Number of GPUs to use
NUM_GPUS="${NUM_GPUS:-8}"

# Whether to run with subtitles evaluation (1=yes, 0=no)
WITH_SUBTITLES="${WITH_SUBTITLES:-0}"

# Whether to run without subtitles evaluation (1=yes, 0=no)
WITHOUT_SUBTITLES="${WITHOUT_SUBTITLES:-1}"

# ---------- Print Config ----------
echo "============================================"
echo "LiveCC VideoMME Evaluation"
echo "============================================"
echo "LIVECC_ROOT:        $LIVECC_ROOT"
echo "MODEL_NAME_OR_PATH: $MODEL_NAME_OR_PATH"
echo "BENCHMARK_PATH:     $BENCHMARK_PATH"
echo "VIDEO_DIR:          ${VIDEO_DIR:-<not set>}"
echo "PARQUET_PATH:       ${PARQUET_PATH:-<not set>}"
echo "SUBTITLE_DIR:       ${SUBTITLE_DIR:-<not set>}"
echo "NUM_GPUS:           $NUM_GPUS"
echo "WITHOUT_SUBTITLES:  $WITHOUT_SUBTITLES"
echo "WITH_SUBTITLES:     $WITH_SUBTITLES"
echo "============================================"

# ---------- Auto-generate benchmark JSONL if needed ----------
if [[ ! -f "$BENCHMARK_PATH" ]]; then
    echo ""
    echo ">>> videomme_local.jsonl not found. Generating from parquet ..."
    if [[ -z "$VIDEO_DIR" || -z "$PARQUET_PATH" ]]; then
        echo "ERROR: BENCHMARK_PATH ($BENCHMARK_PATH) does not exist."
        echo "Please set VIDEO_DIR and PARQUET_PATH to auto-generate it."
        echo ""
        echo "Example:"
        echo "  VIDEO_DIR=/path/to/Video-MME/videos/data \\"
        echo "  PARQUET_PATH=/path/to/videomme.parquet \\"
        echo "  bash scripts/eval_videomme.sh"
        exit 1
    fi
    PREPARE_CMD="python evaluation/videomme/prepare_videomme_jsonl.py \
        --parquet_path $PARQUET_PATH \
        --video_dir $VIDEO_DIR \
        --output_path $BENCHMARK_PATH \
        --skip_missing_video"
    if [[ -n "$SUBTITLE_DIR" ]]; then
        PREPARE_CMD="$PREPARE_CMD --subtitle_dir $SUBTITLE_DIR"
    fi
    eval $PREPARE_CMD
    echo ">>> Generated: $BENCHMARK_PATH"
fi

# ---------- Run evaluation ----------
if [[ "$WITHOUT_SUBTITLES" == "1" ]]; then
    echo ""
    echo ">>> Running VideoMME evaluation (WITHOUT subtitles) ..."
    echo ""
    torchrun --standalone --nproc_per_node="$NUM_GPUS" \
        evaluation/videomme/distributed_evaluate_videomme.py \
        --model_name_or_path "$MODEL_NAME_OR_PATH" \
        --benchmark_path "$BENCHMARK_PATH"
    echo ""
    echo ">>> Finished VideoMME evaluation (WITHOUT subtitles)."
fi

if [[ "$WITH_SUBTITLES" == "1" ]]; then
    echo ""
    echo ">>> Running VideoMME evaluation (WITH subtitles) ..."
    echo ""
    torchrun --standalone --nproc_per_node="$NUM_GPUS" \
        evaluation/videomme/distributed_evaluate_videomme.py \
        --model_name_or_path "$MODEL_NAME_OR_PATH" \
        --benchmark_path "$BENCHMARK_PATH" \
        --with_subtitles
    echo ""
    echo ">>> Finished VideoMME evaluation (WITH subtitles)."
fi

echo ""
echo "============================================"
echo "All done! Results saved to: evaluation/videomme/results/"
echo "============================================"

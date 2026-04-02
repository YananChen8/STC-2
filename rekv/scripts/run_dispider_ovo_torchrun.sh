#!/usr/bin/env bash
set -euo pipefail

# Torchrun launcher that parallelizes OVOBench inference for an existing single-GPU Dispider script.
# It shards annotations by rank and merges rank outputs into one final JSON.

NPROC_PER_NODE=${1:-4}
MASTER_PORT=${MASTER_PORT:-29511}

# ===== Edit these for your environment =====
DISPIDER_DIR=${DISPIDER_DIR:-"/workspace/STC-2/Dispider"}
ANNO_PATH=${ANNO_PATH:-"/workspace/STC-2/rekv/data/ovobench/ovo_bench_new.json"}
VIDEO_PREFIX=${VIDEO_PREFIX:-""}
FINAL_OUTPUT=${FINAL_OUTPUT:-"/workspace/STC-2/rekv/outputs/ovobench/dispider_results.json"}

# IMPORTANT: base command should be the original single-GPU inference command,
# WITHOUT annotation/output arguments. The wrapper will append them automatically.
BASE_CMD=${BASE_CMD:-"python dispider.py --model_path /path/to/ckpt --video_dir /path/to/chunked_videos"}

# If your dispider script uses different argument names, change these:
ANNO_ARG_NAME=${ANNO_ARG_NAME:---anno_path}
OUTPUT_ARG_NAME=${OUTPUT_ARG_NAME:---output_path}
# ==========================================

mkdir -p "$(dirname "${FINAL_OUTPUT}")"

torchrun \
  --standalone \
  --nnodes=1 \
  --nproc_per_node="${NPROC_PER_NODE}" \
  --master_port="${MASTER_PORT}" \
  /workspace/STC-2/rekv/scripts/torchrun_dispider_ovo.py \
  --anno-path "${ANNO_PATH}" \
  --work-dir "${DISPIDER_DIR}" \
  --base-cmd "${BASE_CMD}" \
  --final-output "${FINAL_OUTPUT}" \
  --anno-arg-name "${ANNO_ARG_NAME}" \
  --output-arg-name "${OUTPUT_ARG_NAME}" \
  --video-prefix "${VIDEO_PREFIX}"

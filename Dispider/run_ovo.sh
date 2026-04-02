#!/bin/bash
set -euo pipefail

# 用法:
#   bash run_ovo.sh 8 /path/to/model /path/to/ovo.jsonl /path/to/Ego4D ./ovo_results.json
# 其中第一个参数是 GPU 数（nproc_per_node）

NUM_GPUS=${1:-1}
MODEL_PATH=${2:-/你的/Dispider模型路径}
DATA_PATH=${3:-/你的/ovo.json路径}
VIDEO_ROOT=${4:-/你的/Ego4D根目录}
OUTPUT_PATH=${5:-./ovo_results.json}

torchrun \
  --nproc_per_node "${NUM_GPUS}" \
  dispider.py \
  --model-path "${MODEL_PATH}" \
  --data-path "${DATA_PATH}" \
  --video-root "${VIDEO_ROOT}" \
  --output "${OUTPUT_PATH}"

#!/bin/bash
set -euo pipefail

# 用法:
#   bash run_ovo.sh 8 /path/to/model /path/to/ovo.jsonl /path/to/Ego4D ./ovo_results.json
# 其中第一个参数是 GPU 数（nproc_per_node）

NUM_GPUS=${1:-4}
MODEL_PATH=${2:-/mnt/users/chenyanan-20260210/models/dispider/ckpt}
DATA_PATH=${3:-/mnt/users/chenyanan-20260210/ovo-bench/ovo_bench_new.json}
VIDEO_ROOT=${4:-/mnt/users/chenyanan-20260210/ovo-bench/src_videos/src_videos}
OUTPUT_PATH=${5:-./ovo_results.json}

torchrun \
  --nproc_per_node "${NUM_GPUS}" \
  dispider.py \
  --model-path "${MODEL_PATH}" \
  --data-path "${DATA_PATH}" \
  --video-root "${VIDEO_ROOT}" \
  --output "${OUTPUT_PATH}"

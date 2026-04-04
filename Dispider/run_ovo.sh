#!/bin/bash
set -euo pipefail

# 用法:
#   bash run_ovo.sh 8 /path/to/model /path/to/anno.json /path/to/src_videos /path/to/chunked_videos ./results
# 其中第一个参数是 GPU 数（nproc_per_node），脚本会调用 OVO-Bench/inference.py 的 torchrun 多卡版。

NUM_GPUS=${1:-4}
MODEL_PATH=${2:-/mnt/users/chenyanan-20260210/models/dispider/ckpt}
ANNO_PATH=${3:-/mnt/users/chenyanan-20260210/ovo-bench/ovo_bench_new.json}
VIDEO_DIR=${4:-/mnt/users/chenyanan-20260210/ovo-bench/src_videos/src_videos}
CHUNKED_DIR=${5:-/mnt/public/video_datasets/OVO-Bench/videos/chunked_videos}
RESULT_DIR=${6:-./results}
MODE=${7:-offline}

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=$(cd "${SCRIPT_DIR}/.." && pwd)
OVO_BENCH_DIR="${PROJECT_ROOT}/OVO-Bench"

if [[ ! -f "${OVO_BENCH_DIR}/inference.py" ]]; then
  echo "[ERROR] cannot find ${OVO_BENCH_DIR}/inference.py"
  exit 1
fi

torchrun \
  --nproc_per_node "${NUM_GPUS}" \
  "${OVO_BENCH_DIR}/inference.py" \
  --mode "${MODE}" \
  --model Dispider \
  --model_path "${MODEL_PATH}" \
  --anno_path "${ANNO_PATH}" \
  --video_dir "${VIDEO_DIR}" \
  --chunked_dir "${CHUNKED_DIR}" \
  --result_dir "${RESULT_DIR}"

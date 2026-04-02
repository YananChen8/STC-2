CKPT_PATH=""
MAX_NUM_FRAMES=""
TASK=""
MODEL_NAME=""
TIME_MSG="short_online"
DATA_ROOTS_FILE="${STREAMFOREST_DATA_ROOTS_FILE:-}"
NUM_GPUS="${NUM_GPUS:-}"
USE_SRUN="${STREAMFOREST_USE_SRUN:-auto}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --ckpt_path) CKPT_PATH="$2"; shift 2 ;;
    --max_frames) MAX_NUM_FRAMES="$2"; shift 2 ;;
    --task) TASK="$2"; shift 2 ;;
    --model_name) MODEL_NAME="$2"; shift 2 ;;
    --time_msg) TIME_MSG="$2"; shift 2 ;;
    --data_roots_file) DATA_ROOTS_FILE="$2"; shift 2 ;;
    --num_gpus) NUM_GPUS="$2"; shift 2 ;;
    --use_srun) USE_SRUN="$2"; shift 2 ;;
    *) echo "未知参数: $1"; exit 1 ;;
  esac
done

root_path="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
export PYTHONPATH=$root_path
export HF_DATASETS_OFFLINE=1
if [[ -n "$DATA_ROOTS_FILE" ]]; then
  export STREAMFOREST_DATA_ROOTS_FILE="$(python -c 'import os,sys; print(os.path.abspath(os.path.expanduser(sys.argv[1])))' "$DATA_ROOTS_FILE")"
fi

for required_var in CKPT_PATH MAX_NUM_FRAMES TASK MODEL_NAME; do
  if [[ -z "${!required_var}" ]]; then
    echo "缺少必要参数: ${required_var}" >&2
    exit 1
  fi
done

MASTER_PORT=$((18000 + $RANDOM % 100))
if [[ -z "$NUM_GPUS" ]]; then
  NUM_GPUS="$(python - <<'PY'
try:
    import torch
    print(max(torch.cuda.device_count(), 1))
except Exception:
    print(1)
PY
)"
fi
CONV_TEMPLATE="${CONV_TEMPLATE:-qwen_2}"
TASK_SUFFIX="${TASK//,/_}"
mkdir -p "${CKPT_PATH}/eval"
JOB_NAME=$(basename "$0" .sh)_$(date +"%Y%m%d_%H%M")

echo "检查点路径: $CKPT_PATH"
echo "最大帧数: $MAX_NUM_FRAMES"
echo "任务: $TASK"
echo "模型名称: $MODEL_NAME"
echo "提示词类型: $TIME_MSG"
echo "GPU 数量: $NUM_GPUS"
if [[ -n "${STREAMFOREST_DATA_ROOTS_FILE:-}" ]]; then
  echo "本地数据映射: ${STREAMFOREST_DATA_ROOTS_FILE}"
fi

launch_cmd=(
  accelerate launch
  --num_processes "${NUM_GPUS}"
  --main_process_port "${MASTER_PORT}"
  -m lmms_eval
  --model "${MODEL_NAME}"
  --model_args "pretrained=${CKPT_PATH},conv_template=${CONV_TEMPLATE},max_frames_num=${MAX_NUM_FRAMES},time_msg=${TIME_MSG}"
  --tasks "${TASK}"
  --batch_size 1
  --log_samples
  --log_samples_suffix "${TASK_SUFFIX}"
  --output_path "${CKPT_PATH}/eval/response__${TASK}_${MODEL_NAME}_F${MAX_NUM_FRAMES}_${JOB_NAME}"
)

if command -v srun >/dev/null 2>&1 && [[ "$USE_SRUN" != "0" ]]; then
  run_cmd=(
    srun -p videop1
    --job-name="${JOB_NAME}"
    --ntasks=1
    --gres="gpu:${NUM_GPUS}"
    --ntasks-per-node=1
    --cpus-per-task=16
    --kill-on-bad-exit=1
    "${launch_cmd[@]}"
  )
else
  run_cmd=("${launch_cmd[@]}")
fi

"${run_cmd[@]}" 2>&1 | tee "${CKPT_PATH}/eval/log_${TASK}_${MODEL_NAME}_F${MAX_NUM_FRAMES}_${JOB_NAME}.log"

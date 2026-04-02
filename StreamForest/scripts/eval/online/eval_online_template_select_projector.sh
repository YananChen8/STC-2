MAX_NUM_FRAMES="${MAX_NUM_FRAMES:-512}"
MODEL_NAME="${MODEL_NAME:-streamforest}"
TIME_MSG="${TIME_MSG:-short_online_v2}"
REPLACE_PROJECTOR="${REPLACE_PROJECTOR:-ablation_woSTFW_PEMF}"
ROOT_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
CKPT_PATH="${CKPT_PATH:-${ROOT_PATH}/ckpt/StreamForest-Qwen2-7B_Siglip_ablation_woPEMF+FSTW}"
DATA_ROOTS_FILE="${STREAMFOREST_DATA_ROOTS_FILE:-}"
NUM_GPUS="${NUM_GPUS:-}"
USE_SRUN="${STREAMFOREST_USE_SRUN:-auto}"

TASK="ovbench"

root_path="$ROOT_PATH"
export PYTHONPATH=$root_path
export HF_DATASETS_OFFLINE=1
if [[ -n "$DATA_ROOTS_FILE" ]]; then
  export STREAMFOREST_DATA_ROOTS_FILE="$(python -c 'import os,sys; print(os.path.abspath(os.path.expanduser(sys.argv[1])))' "$DATA_ROOTS_FILE")"
fi

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
mkdir -p ${CKPT_PATH}/eval
JOB_NAME=$(basename "$0" .sh)_$(date +"%Y%m%d_%H%M")

echo "检查点路径: $CKPT_PATH"
echo "最大帧数: $MAX_NUM_FRAMES"
echo "任务: $TASK"
echo "模型名称: $MODEL_NAME"
echo "提示词类型: $TIME_MSG"
echo "记忆类型: $REPLACE_PROJECTOR"
echo "GPU 数量: $NUM_GPUS"
if [[ -n "${STREAMFOREST_DATA_ROOTS_FILE:-}" ]]; then
  echo "本地数据映射: ${STREAMFOREST_DATA_ROOTS_FILE}"
fi


RESULT_DIR="${CKPT_PATH}/eval/${MAX_NUM_FRAMES}_${TASK}"

if [ ! -d "${RESULT_DIR}" ] && [ -d "${CKPT_PATH}" ]; then
  mkdir -p ${RESULT_DIR}
  echo "Created directory: ${RESULT_DIR}"
else
    echo "Directory ${RESULT_DIR} already exists or ${CKPT_PATH} not exists."
fi

launch_cmd=(
    accelerate launch
    --num_processes "${NUM_GPUS}"
    --main_process_port "${MASTER_PORT}"
    -m lmms_eval
    --model "${MODEL_NAME}"
    --model_args "pretrained=${CKPT_PATH},conv_template=${CONV_TEMPLATE},max_frames_num=${MAX_NUM_FRAMES},time_msg=${TIME_MSG},mm_projector_type=${REPLACE_PROJECTOR}"
    --tasks "${TASK}"
    --batch_size 1
    --log_samples
    --log_samples_suffix "${TASK_SUFFIX}"
    --output_path "${RESULT_DIR}/response__${TASK}_${MODEL_NAME}_F${MAX_NUM_FRAMES}_${JOB_NAME}"
)

if command -v srun >/dev/null 2>&1 && [[ "$USE_SRUN" != "0" ]]; then
    run_cmd=(
        srun -p videopp1
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

"${run_cmd[@]}" 2>&1 | tee "${RESULT_DIR}/log_${TASK}_${MODEL_NAME}_F${MAX_NUM_FRAMES}_${JOB_NAME}.log"

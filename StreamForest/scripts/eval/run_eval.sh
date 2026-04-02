STREAMFOREST_ROOT_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd $STREAMFOREST_ROOT_PATH
export PYTHONPATH=$STREAMFOREST_ROOT_PATH

# Load environment variables from .env file if it exists
if [[ -f "${STREAMFOREST_ROOT_PATH}/.env" ]]; then
  set -a
  source "${STREAMFOREST_ROOT_PATH}/.env"
  set +a
fi

MAX_FRAMES="${MAX_FRAMES:-2048}"
TIME_MSG="${TIME_MSG:-short_online_v2}"
MODEL_NAME="${MODEL_NAME:-streamforest}"
CKPT_PATH="/mnt/users/chenyanan-20260210/models/streamingforest/ckpt/stage4-postft-qwen-siglip/StreamForest-Qwen2-7B"  # "${CKPT_PATH:-}"
DATA_ROOTS_FILE="${DATA_ROOTS_FILE:-${STREAMFOREST_DATA_ROOTS_FILE:-}}"  # "/mnt/users/chenyanan-20260210/STC/data/anno"
NUM_GPUS="${NUM_GPUS:-4}"
USE_SRUN="${USE_SRUN:-0}"

# Validate required variables
if [[ -z "$CKPT_PATH" ]]; then
    echo "ERROR: CKPT_PATH is not set." >&2
    echo "  Set it via:" >&2
    echo "    1) Environment variable:  export CKPT_PATH=/path/to/your/checkpoint" >&2
    echo "    2) .env file:             echo 'CKPT_PATH=/path/to/your/checkpoint' >> .env" >&2
    exit 1
fi

TASKS=(
    # "odvbench"
    # "streamingbench"
    # "ovbench"
    "ovobench"
    # "videomme"
    # "mlvu_mc"
    # "mvbench"
    # "perceptiontest_val_mc"
)


for TASK in "${TASKS[@]}"; do
    echo "============================"
    echo "Running benchmark: $TASK"
    echo "============================"

    cmd=(
        bash scripts/eval/online/eval_online_template.sh
        --ckpt_path "$CKPT_PATH"
        --max_frames "$MAX_FRAMES"
        --model_name "$MODEL_NAME"
        --time_msg "$TIME_MSG"
        --task "$TASK"
    )

    if [[ -n "$DATA_ROOTS_FILE" ]]; then
        cmd+=(--data_roots_file "$DATA_ROOTS_FILE")
    fi

    if [[ -n "$NUM_GPUS" ]]; then
        cmd+=(--num_gpus "$NUM_GPUS")
    fi

    if [[ -n "$USE_SRUN" ]]; then
        cmd+=(--use_srun "$USE_SRUN")
    fi

    "${cmd[@]}"
done

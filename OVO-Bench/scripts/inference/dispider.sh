# #!/usr/bin/env bash
# set -euo pipefail

# TASKS=("ASI" "HLD" "EPM" "ATR" "ACR" "OCR" "STU" "OJR" "FPD" "REC" "SSR" "CRR")

# run_task () {
#   local gpu_id=$1
#   local task=$2

#   echo "===== Running task: $task on GPU $gpu_id ====="
#   CUDA_VISIBLE_DEVICES=$gpu_id python inference.py \
#     --mode offline \
#     --task "$task" \
#     --model Dispider \
#     --model_path /mnt/users/chenyanan-20260210/models/dispider/ckpt
# }

# idx=0
# for TASK in "${TASKS[@]}"; do
#   gpu=$((idx % 4))
#   run_task "$gpu" "$TASK" &
#   idx=$((idx + 1))

#   if (( idx % 4 == 0 )); then
#     wait
#   fi
# done

# wait

#!/usr/bin/env bash
set -euo pipefail

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

TASKS=("ASI" "HLD" "EPM" "ATR" "ACR" "OCR" "STU" "OJR" "FPD" "REC" "SSR" "CRR")  # 

for TASK in "${TASKS[@]}"; do
  echo "===== Running task: $TASK ====="
  python inference.py \
    --mode offline \
    --task "$TASK" \
    --model Dispider \
    --model_path /mnt/users/chenyanan-20260210/models/dispider/ckpt
done

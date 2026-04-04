#!/usr/bin/env bash
#SBATCH -J ovo_inference        # 作业名（自定义，比如你的任务名）
#SBATCH -p a100_global          # 分区：数据在/mnt选global；在本地盘选a100_local_<节点名>
#SBATCH --gres=gpu:4            # GPU数量（按需填，sandbox默认每人≤8卡，不用审批）
#SBATCH -c 32                   # CPU核数（经验值：每卡8核，4卡就是32核）
#SBATCH --mem=128G              # 内存（4卡建议128G，按需调整）
#SBATCH -t 12:00:00             # 时长（最长可设24小时，sandbox支持；更长需审批）
#SBATCH -o /mnt/users/chenyanan-20260210/logs/%x-%j.out  # 日志写/mnt，跨节点能看
#SBATCH -e /mnt/users/chenyanan-20260210/logs/%x-%j.err  # 错误日志路径

# 关键：脚本开头的环境/容错配置
set -euo pipefail
mkdir -p /mnt/users/chenyanan-20260210/logs/dispider  # 确保日志目录存在

# 打印关键信息（便于排查问题）
echo "开始时间: $(date)"
echo "节点: $(hostname)"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
nvidia-smi -L
python3 -V

# 激活你的conda环境（必须加，否则找不到依赖库）
source /mnt/public/apps/miniconda3/etc/profile.d/conda.sh
conda activate /mnt/users/chenyanan-20260210/conda/envs/dispider

# 切换到代码目录（你的推理脚本所在路径）
cd /home/chenyanan-20260210/STC/all/STC/OVO-Bench

# 执行正式运行的命令（替换成你调试好的完整命令）
# bash scripts/inference/dispider.sh
bash scripts/score/dispider.sh
# bash scripts/ovobench_scipts/score_rekv.sh

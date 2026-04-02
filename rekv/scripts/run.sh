#!/bin/bash
# ============================================================
# STC Unified Inference & Evaluation Script
# ============================================================
#
# All benchmarks (offline, OVOBench, StreamingBench) use the same
# torchrun-based distributed pipeline.
#
# Usage:
#   bash scripts/run.sh --dataset mlvu --num_gpus 4
#   bash scripts/run.sh --dataset videomme --num_gpus 8 \
#       --cache_strategy cosine --prune_strategy stc
#   bash scripts/run.sh --dataset ovobench --num_gpus 8
#   bash scripts/run.sh --dataset streamingbench --num_gpus 4
#   bash scripts/run.sh --dataset streamingbench_sqa --num_gpus 4
#
# Supported datasets:
#   Offline:  videomme, videomme_subset, mlvu, egoschema, egoschema_subset,
#             qaego4d, cgbench, activitynet_qa, rvs_ego, rvs_movie
#   Online:   ovobench, streamingbench, streamingbench_sqa, streamingbench_proactive
# ============================================================

set -euo pipefail

# ---- Defaults ----
DATASET="mlvu"
MODEL="llava_ov_7b"
NUM_GPUS=1
MASTER_PORT=29500
SAVE_DIR=""
SKIP_EVAL=""
EXTRA_ARGS=""

# STC defaults
CACHE_STRATEGY="none"
PRUNE_STRATEGY="full_tokens"
UPDATE_TOKEN_RATIO=0.3
CACHE_INTERVAL=2
ENCODE_CHUNK_SIZE=1

# ---- Parse arguments ----
while [[ $# -gt 0 ]]; do
    case $1 in
        --dataset)           DATASET="$2";              shift 2 ;;
        --model)             MODEL="$2";                shift 2 ;;
        --num_gpus)          NUM_GPUS="$2";             shift 2 ;;
        --master_port)       MASTER_PORT="$2";          shift 2 ;;
        --save_dir)          SAVE_DIR="$2";             shift 2 ;;
        --skip_eval)         SKIP_EVAL="--skip_eval";   shift ;;
        --cache_strategy)    CACHE_STRATEGY="$2";       shift 2 ;;
        --prune_strategy)    PRUNE_STRATEGY="$2";       shift 2 ;;
        --update_token_ratio) UPDATE_TOKEN_RATIO="$2";  shift 2 ;;
        --cache_interval)    CACHE_INTERVAL="$2";       shift 2 ;;
        --encode_chunk_size) ENCODE_CHUNK_SIZE="$2";    shift 2 ;;
        --tf32)              EXTRA_ARGS="$EXTRA_ARGS --tf32"; shift ;;
        --debug)             EXTRA_ARGS="$EXTRA_ARGS --debug"; shift ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

# Auto-generate save_dir if not specified
if [[ -z "$SAVE_DIR" ]]; then
    TIMESTAMP=$(date "+%Y%m%d_%H%M%S")
    SAVE_DIR="results/${DATASET}/${MODEL}_${TIMESTAMP}"
fi

# ---- Print configuration ----
echo "============================================"
echo "  STC Inference & Evaluation Pipeline"
echo "============================================"
echo "  Dataset:         $DATASET"
echo "  Model:           $MODEL"
echo "  GPUs:            $NUM_GPUS"
echo "  Master Port:     $MASTER_PORT"
echo "  Save Dir:        $SAVE_DIR"
echo "  Cache Strategy:  $CACHE_STRATEGY"
echo "  Prune Strategy:  $PRUNE_STRATEGY"
echo "  Token Ratio:     $UPDATE_TOKEN_RATIO"
echo "  Cache Interval:  $CACHE_INTERVAL"
echo "  Skip Eval:       ${SKIP_EVAL:-no}"
echo "============================================"

# ---- Unified torchrun pipeline for all datasets ----
torchrun \
    --nnodes=1 \
    --nproc_per_node="$NUM_GPUS" \
    --master_port="$MASTER_PORT" \
    -m model.video_qa.run_distributed \
    --dataset "$DATASET" \
    --model "$MODEL" \
    --save_dir "$SAVE_DIR" \
    --cache_strategy "$CACHE_STRATEGY" \
    --prune_strategy "$PRUNE_STRATEGY" \
    --update_token_ratio "$UPDATE_TOKEN_RATIO" \
    --cache_interval "$CACHE_INTERVAL" \
    --encode_chunk_size "$ENCODE_CHUNK_SIZE" \
    $SKIP_EVAL \
    $EXTRA_ARGS

echo ""
echo "============================================"
echo "  Pipeline complete!"
echo "  Results: $SAVE_DIR/"
echo "============================================"

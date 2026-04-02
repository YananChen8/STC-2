#!/bin/bash
set -e

# ============================================================
# LiveCC LiveSports3KCC Evaluation Script
# Usage:
#   # LiveCC real-time commentary generation (default)
#   bash scripts/eval_livesports3kcc.sh
#
#   # Offline caption generation (e.g. Qwen2.5-VL)
#   MODE=caption MODEL_NAME_OR_PATH=Qwen/Qwen2.5-VL-7B-Instruct \
#   bash scripts/eval_livesports3kcc.sh
#
#   # LLM judge (requires Azure OpenAI credentials)
#   MODE=judge MODEL_ID=LiveCC-7B-Instruct \
#   PREDICTION_JSONL=evaluation/livesports3kcc/livecc/LiveCC-7B-Instruct.jsonl \
#   AZURE_OPENAI_ENDPOINT=xxx AZURE_OPENAI_API_KEY=xxx \
#   bash scripts/eval_livesports3kcc.sh
#
# Data is automatically downloaded from HuggingFace:
#   https://huggingface.co/datasets/stdKonjac/LiveSports-3K
# ============================================================

LIVECC_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$LIVECC_ROOT"
export PYTHONPATH="$LIVECC_ROOT:$PYTHONPATH"

# ---------- Configurable Parameters ----------
MODE="${MODE:-livecc}"   # livecc | caption | judge
MODEL_NAME_OR_PATH="${MODEL_NAME_OR_PATH:-chenjoya/LiveCC-7B-Instruct}"
NUM_WORKERS="${NUM_WORKERS:-8}"
REPETITION_PENALTY="${REPETITION_PENALTY:-1.15}"

# ---------- Print Config ----------
echo "============================================"
echo "LiveCC LiveSports3KCC Evaluation"
echo "============================================"
echo "LIVECC_ROOT:        $LIVECC_ROOT"
echo "MODE:               $MODE"
echo "MODEL_NAME_OR_PATH: $MODEL_NAME_OR_PATH"
echo "NUM_WORKERS:        $NUM_WORKERS"
echo "============================================"

if [[ "$MODE" == "livecc" ]]; then
    OUTPUT_DIR="${OUTPUT_DIR:-evaluation/livesports3kcc/livecc}"
    NOT_INSTRUCT="${NOT_INSTRUCT:-0}"

    echo "OUTPUT_DIR:         $OUTPUT_DIR"
    echo "REPETITION_PENALTY: $REPETITION_PENALTY"
    echo "NOT_INSTRUCT:       $NOT_INSTRUCT"
    echo "============================================"
    echo ""
    echo ">>> Generating LiveCC real-time commentary ..."
    echo ""

    CMD="python evaluation/livesports3kcc/distributed_generate_livecc.py \
        --model_name_or_path $MODEL_NAME_OR_PATH \
        --output_dir $OUTPUT_DIR \
        --num_workers $NUM_WORKERS \
        --repetition_penalty $REPETITION_PENALTY"

    if [[ "$NOT_INSTRUCT" == "1" ]]; then
        CMD="$CMD --not_instruct_model"
    fi

    eval $CMD

    echo ""
    echo "============================================"
    echo "All done! Results saved to: $OUTPUT_DIR"
    echo "============================================"

elif [[ "$MODE" == "caption" ]]; then
    OUTPUT_DIR="${OUTPUT_DIR:-evaluation/livesports3kcc/captions}"

    echo "OUTPUT_DIR:         $OUTPUT_DIR"
    echo "============================================"
    echo ""
    echo ">>> Generating offline captions ..."
    echo ""

    python evaluation/livesports3kcc/distributed_generate_caption.py \
        --model_name_or_path "$MODEL_NAME_OR_PATH" \
        --output_dir "$OUTPUT_DIR" \
        --num_workers "$NUM_WORKERS"

    echo ""
    echo "============================================"
    echo "All done! Results saved to: $OUTPUT_DIR"
    echo "============================================"

elif [[ "$MODE" == "judge" ]]; then
    MODEL_ID="${MODEL_ID:-$(basename "$MODEL_NAME_OR_PATH")}"
    PREDICTION_JSONL="${PREDICTION_JSONL:-}"
    JUDGE_OUTPUT_DIR="${JUDGE_OUTPUT_DIR:-evaluation/livesports3kcc/judges}"
    JUDGE_NUM_WORKERS="${JUDGE_NUM_WORKERS:-16}"

    if [[ -z "$PREDICTION_JSONL" ]]; then
        echo "ERROR: MODE=judge requires PREDICTION_JSONL to be set."
        echo "Example:"
        echo "  MODE=judge MODEL_ID=LiveCC-7B-Instruct \\"
        echo "  PREDICTION_JSONL=evaluation/livesports3kcc/livecc/LiveCC-7B-Instruct.jsonl \\"
        echo "  AZURE_OPENAI_ENDPOINT=xxx AZURE_OPENAI_API_KEY=xxx \\"
        echo "  bash scripts/eval_livesports3kcc.sh"
        exit 1
    fi

    if [[ -z "$AZURE_OPENAI_ENDPOINT" || -z "$AZURE_OPENAI_API_KEY" ]]; then
        echo "ERROR: MODE=judge requires AZURE_OPENAI_ENDPOINT and AZURE_OPENAI_API_KEY."
        exit 1
    fi

    echo "MODEL_ID:           $MODEL_ID"
    echo "PREDICTION_JSONL:   $PREDICTION_JSONL"
    echo "JUDGE_OUTPUT_DIR:   $JUDGE_OUTPUT_DIR"
    echo "JUDGE_NUM_WORKERS:  $JUDGE_NUM_WORKERS"
    echo "============================================"
    echo ""
    echo ">>> Running LLM judge: $MODEL_ID vs. GPT-4o ..."
    echo ""

    python evaluation/livesports3kcc/llm_judge.py \
        --model_id "$MODEL_ID" \
        --prediction_jsonl "$PREDICTION_JSONL" \
        --output_dir "$JUDGE_OUTPUT_DIR" \
        --num_workers "$JUDGE_NUM_WORKERS"

    echo ""
    echo "============================================"
    echo "All done! Results saved to: $JUDGE_OUTPUT_DIR"
    echo "============================================"

else
    echo "ERROR: Unknown MODE='$MODE'. Use one of: livecc, caption, judge"
    exit 1
fi

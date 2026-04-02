#!/bin/bash
set -e

# ============================================================
# LiveCC One-Click Environment Setup
# ============================================================
# Usage:
#   bash setup_env.sh [--python PYTHON_BIN] [--venv VENV_DIR]
#
# Examples:
#   bash setup_env.sh
#   bash setup_env.sh --python python3.11 --venv .venv_livecc
# ============================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# ---------- Parse arguments ----------
PYTHON_BIN="python3"
VENV_DIR=".venv_livecc"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --python)
            PYTHON_BIN="$2"
            shift 2
            ;;
        --venv)
            VENV_DIR="$2"
            shift 2
            ;;
        *)
            echo "Unknown argument: $1"
            echo "Usage: bash setup_env.sh [--python PYTHON_BIN] [--venv VENV_DIR]"
            exit 1
            ;;
    esac
done

echo "============================================"
echo "LiveCC Environment Setup"
echo "============================================"
echo "Project root : $SCRIPT_DIR"
echo "Python binary: $PYTHON_BIN"
echo "Venv dir     : $VENV_DIR"
echo "============================================"

# ---------- Check Python version ----------
PY_VERSION=$($PYTHON_BIN -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')" 2>/dev/null || true)
if [[ -z "$PY_VERSION" ]]; then
    echo "ERROR: $PYTHON_BIN not found. Please install Python >= 3.11 or specify --python."
    exit 1
fi

PY_MAJOR=$(echo "$PY_VERSION" | cut -d. -f1)
PY_MINOR=$(echo "$PY_VERSION" | cut -d. -f2)
if [[ "$PY_MAJOR" -lt 3 ]] || { [[ "$PY_MAJOR" -eq 3 ]] && [[ "$PY_MINOR" -lt 11 ]]; }; then
    echo "WARNING: Python >= 3.11 is recommended (found $PY_VERSION). Continuing anyway..."
fi
echo ">>> Using Python $PY_VERSION ($PYTHON_BIN)"

# ---------- Create virtual environment ----------
if [[ ! -d "$VENV_DIR" ]]; then
    echo ">>> Creating virtual environment at $VENV_DIR ..."
    $PYTHON_BIN -m venv "$VENV_DIR"
else
    echo ">>> Virtual environment already exists at $VENV_DIR, reusing."
fi

source "$VENV_DIR/bin/activate"
echo ">>> Activated venv: $(which python)"

# ---------- Upgrade pip ----------
echo ">>> Upgrading pip ..."
pip install --upgrade pip setuptools wheel -q

# ---------- Install PyTorch (if not already installed) ----------
python -c "import torch" 2>/dev/null && echo ">>> PyTorch already installed." || {
    echo ">>> Installing PyTorch ..."
    pip install torch torchvision torchaudio
}

# ---------- Install flash-attn (requires special build) ----------
python -c "import flash_attn" 2>/dev/null && echo ">>> flash-attn already installed." || {
    echo ">>> Installing flash-attn (this may take a while) ..."
    pip install flash-attn --no-build-isolation || {
        echo "WARNING: flash-attn installation failed. GPU inference requires flash-attn."
        echo "         You can manually install it later: pip install flash-attn --no-build-isolation"
    }
}

# ---------- Install remaining dependencies ----------
echo ">>> Installing dependencies from requirements.txt ..."
# Filter out torch/flash-attn since they are handled above
grep -v -E "^(torch|torchvision|torchaudio|flash-attn)" requirements.txt | pip install -r /dev/stdin -q

# ---------- Install livecc-utils from local source (editable) ----------
echo ">>> Installing livecc-utils from local source ..."
pip install -e livecc-utils/ -q

# ---------- Verify installation ----------
echo ""
echo ">>> Verifying key packages ..."
python -c "
import torch
print(f'  torch          : {torch.__version__}')
print(f'  CUDA available : {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'  CUDA version   : {torch.version.cuda}')
    print(f'  GPU count      : {torch.cuda.device_count()}')
"
python -c "import transformers; print(f'  transformers   : {transformers.__version__}')"
python -c "import livecc_utils; print(f'  livecc-utils   : OK')"
python -c "import qwen_vl_utils; print(f'  qwen-vl-utils  : OK')"
python -c "
try:
    import flash_attn; print(f'  flash-attn     : {flash_attn.__version__}')
except ImportError:
    print(f'  flash-attn     : NOT INSTALLED (GPU inference will be slower)')
"

echo ""
echo "============================================"
echo "Environment setup complete!"
echo ""
echo "To activate the environment:"
echo "  source $VENV_DIR/bin/activate"
echo ""
echo "To run the Gradio demo:"
echo "  python demo/app.py"
echo ""
echo "To run VideoMME evaluation:"
echo "  bash scripts/eval_videomme.sh"
echo "============================================"

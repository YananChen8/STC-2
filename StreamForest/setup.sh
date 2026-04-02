#!/usr/bin/env bash
# ============================================================
# StreamForest - One-click Setup Script
# ============================================================
# Usage:
#   bash setup.sh              # Full setup (venv + deps + config)
#   bash setup.sh --deps-only  # Only install Python dependencies
#   bash setup.sh --env-only   # Only generate .env from template
# ============================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${STREAMFOREST_VENV_DIR:-${SCRIPT_DIR}/venv}"
PYTHON="${PYTHON:-python3}"

# --- Color helpers ---
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; NC='\033[0m'
info()  { echo -e "${GREEN}[INFO]${NC} $*"; }
warn()  { echo -e "${YELLOW}[WARN]${NC} $*"; }
error() { echo -e "${RED}[ERROR]${NC} $*" >&2; }

# --- Parse arguments ---
DO_DEPS=1; DO_ENV=1
if [[ "${1:-}" == "--deps-only" ]]; then DO_ENV=0; fi
if [[ "${1:-}" == "--env-only"  ]]; then DO_DEPS=0; fi

# ============================================================
# Step 1: Create virtual environment (optional)
# ============================================================
if [[ "$DO_DEPS" -eq 1 ]]; then
  if [[ ! -d "$VENV_DIR" ]]; then
    info "Creating virtual environment at: $VENV_DIR"
    $PYTHON -m venv "$VENV_DIR"
  else
    info "Virtual environment already exists: $VENV_DIR"
  fi

  info "Activating virtual environment..."
  source "$VENV_DIR/bin/activate"

  # Step 2: Install dependencies
  info "Upgrading pip..."
  pip install --upgrade pip

  info "Installing core dependencies from requirements.txt..."
  pip install -r "${SCRIPT_DIR}/requirements.txt"

  # Step 3: Install flash-attn (requires torch to be installed first)
  info "Installing flash-attn (this may take a while)..."
  if pip install --no-build-isolation flash-attn==2.6.3 2>/dev/null; then
    info "flash-attn installed successfully."
  else
    warn "flash-attn installation failed. You may need to install it manually:"
    warn "  pip install --no-build-isolation flash-attn==2.6.3"
  fi

  info "All Python dependencies installed successfully!"
  echo ""
fi

# ============================================================
# Step 4: Generate .env configuration
# ============================================================
if [[ "$DO_ENV" -eq 1 ]]; then
  ENV_FILE="${SCRIPT_DIR}/.env"
  ENV_EXAMPLE="${SCRIPT_DIR}/.env.example"

  if [[ -f "$ENV_FILE" ]]; then
    warn ".env file already exists at: $ENV_FILE"
    warn "Skipping .env generation. Edit it manually if needed."
  elif [[ -f "$ENV_EXAMPLE" ]]; then
    cp "$ENV_EXAMPLE" "$ENV_FILE"
    info "Created .env from .env.example"
    info "Please edit ${ENV_FILE} and fill in your local paths:"
    echo ""
    echo "  Required:"
    echo "    CKPT_PATH=/path/to/your/StreamForest-Qwen2-7B"
    echo ""
    echo "  Optional:"
    echo "    STREAMFOREST_DATA_ROOTS_FILE=./scripts/eval/local_data_roots.yaml"
    echo "    NUM_GPUS=4"
    echo ""
  else
    error ".env.example not found at: $ENV_EXAMPLE"
    exit 1
  fi
fi

# ============================================================
# Step 5: Generate local_data_roots.yaml
# ============================================================
DATA_ROOTS="${SCRIPT_DIR}/scripts/eval/local_data_roots.yaml"
DATA_ROOTS_EXAMPLE="${SCRIPT_DIR}/scripts/eval/local_data_roots.example.yaml"

if [[ ! -f "$DATA_ROOTS" ]] && [[ -f "$DATA_ROOTS_EXAMPLE" ]]; then
  cp "$DATA_ROOTS_EXAMPLE" "$DATA_ROOTS"
  info "Created local_data_roots.yaml from example."
  info "Please edit ${DATA_ROOTS} and fill in your dataset paths."
elif [[ -f "$DATA_ROOTS" ]]; then
  info "local_data_roots.yaml already exists, skipping."
fi

# ============================================================
# Done
# ============================================================
echo ""
info "============================================"
info "  StreamForest setup complete!"
info "============================================"
echo ""
echo "  Quick Start:"
echo "    1. Edit .env and set CKPT_PATH"
echo "    2. Edit scripts/eval/local_data_roots.yaml with your dataset paths"
echo "    3. Run evaluation:"
echo "       bash scripts/eval/run_eval.sh"
echo ""
if [[ "$DO_DEPS" -eq 1 ]]; then
  echo "  To activate the virtual environment in future sessions:"
  echo "    source ${VENV_DIR}/bin/activate"
  echo ""
fi

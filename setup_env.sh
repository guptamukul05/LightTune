#!/usr/bin/env bash
# setup_env.sh
# Environment setup for HPC PEFT project — remote GPU cluster, no sudo/containers.
# Package manager: uv. CUDA 12.3 confirmed on host.

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${PROJECT_ROOT}/.venv"
PYTHON_VERSION="3.11"

echo "== 1. Checking uv installation =="
if ! command -v uv &> /dev/null; then
    echo "uv not found — installing to user space (no sudo required)"
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
else
    echo "uv already installed: $(uv --version)"
fi

echo "== 2. Creating virtual environment (Python ${PYTHON_VERSION}) =="
uv venv "${VENV_DIR}" --python "${PYTHON_VERSION}"
# shellcheck disable=SC1091
source "${VENV_DIR}/bin/activate"

echo "== 3. Verifying CUDA visibility =="
nvidia-smi || { echo "ERROR: nvidia-smi not accessible in this shell/session"; exit 1; }

echo "== 4. Installing core dependencies =="
uv pip install --upgrade pip
uv pip install -r "${PROJECT_ROOT}/requirements.txt"

echo "== 5. Verifying critical CUDA-sensitive packages =="
python - <<'EOF'
import torch
print(f"torch version: {torch.__version__}")
print(f"torch CUDA available: {torch.cuda.is_available()}")
print(f"torch CUDA version: {torch.version.cuda}")
print(f"GPU count visible: {torch.cuda.device_count()}")

try:
    import bitsandbytes as bnb
    print(f"bitsandbytes version: {bnb.__version__}")
except ImportError:
    print("WARNING: bitsandbytes not installed or failed to import")
EOF

echo "== 6. Environment setup complete =="
echo "Activate with: source ${VENV_DIR}/bin/activate"

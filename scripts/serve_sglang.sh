#!/usr/bin/env bash
# scripts/serve_sglang.sh
# Usage: bash scripts/serve_sglang.sh [model_repo] [adapter_path] [port]
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"
source .venv/bin/activate

MODEL_REPO="${1:-Qwen/Qwen2.5-1.5B}"
ADAPTER_PATH="${2:-}"
PORT="${3:-30000}"

echo "Forcing SAFE single-GPU mode (tp-size=1)"
TP_SIZE=1

CMD=(
python -m sglang.launch_server
--model-path "${MODEL_REPO}"
--port "${PORT}"
--tp-size "${TP_SIZE}"
--max-model-len 512
--context-length 512
--gpu-memory-utilization 0.6
--disable-cuda-graph
)

if [ -n "${ADAPTER_PATH}" ]; then
    CMD+=(--lora-paths "default=${ADAPTER_PATH}")
    echo "Serving with LoRA adapter: ${ADAPTER_PATH}"
fi

echo "Launching: ${CMD[*]}"
exec "${CMD[@]}"
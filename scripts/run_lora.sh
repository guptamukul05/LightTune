#!/usr/bin/env bash
# scripts/run_lora.sh
#
# Runs LoRA fine-tuning then immediately evaluates the saved adapter.
# Must be run from the project root on Ramanujan after setup_env.sh.
#
# Usage:
#   bash scripts/run_lora.sh

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"

if [ ! -f .venv/bin/activate ]; then
    echo "ERROR: virtual env not found. Run bash setup_env.sh first."
    exit 1
fi

source .venv/bin/activate

if [ ! -d data/train ] || [ ! -d data/val ]; then
    echo "ERROR: data/train or data/val not found."
    echo "Your friend's preprocessing must have saved data there first."
    exit 1
fi

echo ""
echo "=="
echo "  Step 1/2 — LoRA Training"
echo "=="
python src/training/train_lora.py \
    --model_config configs/model_config.yaml \
    --train_config configs/training_config.yaml

echo ""
echo "=="
echo "  Step 2/2 — Evaluating LoRA Adapter"
echo "=="
python src/evaluations/evaluate.py \
    --config configs/model_config.yaml \
    --adapter_path checkpoints/lora \
    --run_name lora \
    --val_data_path data/val

echo ""
echo "=="
echo "  Done."
echo "  Results  → logs/eval_lora.json"
echo "  GPU log  → logs/gpu_usage_lora.csv"
echo "  Adapter  → checkpoints/lora/"
echo ""
echo "  To view training curves:"
echo "  tensorboard --logdir logs/ --port 6006"
echo "=="

#!/usr/bin/env bash
# run_baseline.sh
# Runs zero-shot baseline evaluation of pretrained Qwen2.5-1.5B on Alpaca val set.

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"

source .venv/bin/activate

echo "== Running baseline (zero-shot) evaluation =="
python src/evaluations/evaluate.py \
    --config configs/model_config.yaml \
    --run_name baseline \
    --val_data_path data/val

echo "== Baseline evaluation complete =="
echo "Results: logs/eval_baseline.json"
echo "GPU usage: logs/gpu_usage_baseline.csv"

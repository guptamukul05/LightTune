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
    echo "Run: python src/data/preprocess.py first."
    exit 1
fi

echo ""
echo "=="
echo "  Step 1/2 — QLoRA Training"
echo "=="
python src/training/train_qlora.py \
    --model_config configs/model_config.yaml \
    --train_config configs/training_config.yaml

echo ""
echo "=="
echo "  Step 2/2 — Evaluating QLoRA Adapter"
echo "=="
python src/evaluations/evaluate.py \
    --config configs/model_config.yaml \
    --adapter_path checkpoints/qlora \
    --run_name qlora \
    --val_data_path data/val

echo ""
echo "=="
echo "  Done."
echo "  Results  → logs/eval_qlora.json"
echo "  GPU log  → logs/gpu_usage_qlora.csv"
echo "  Adapter  → checkpoints/qlora/"
echo ""
echo "  To view training curves:"
echo "  tensorboard --logdir logs/ --port 6006"
echo "=="

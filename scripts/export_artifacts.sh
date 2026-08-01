#!/usr/bin/env bash
# scripts/export_artifacts.sh
# Run before GPU access ends. Bundles everything constraint #10 requires
# into exports/export_<timestamp>/ for transfer off the remote server (e.g. via MobaXterm SFTP).
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"
source .venv/bin/activate

TS=$(date +%Y%m%d_%H%M%S)
EXPORT_DIR="exports/export_${TS}"
mkdir -p "${EXPORT_DIR}/checkpoints" "${EXPORT_DIR}/logs" "${EXPORT_DIR}/tb_scalars"

echo "== 1. Adapter weights (LoRA/QLoRA) =="
for d in checkpoints/lora checkpoints/qlora; do
    if [ -d "$d" ]; then
        cp -r "$d" "${EXPORT_DIR}/checkpoints/"
        echo "Copied $d"
    else
        echo "Skipped $d (not found)"
    fi
done

echo "== 2. Eval / bench / GPU-usage logs (JSON, CSV) =="
shopt -s nullglob
for f in logs/*.json logs/*.csv; do
    cp "$f" "${EXPORT_DIR}/logs/"
done
shopt -u nullglob

echo "== 3. TensorBoard scalars -> static CSV =="
python - "${EXPORT_DIR}/tb_scalars" <<'EOF'
import sys, os, csv
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

out_dir = sys.argv[1]
logs_dir = "logs"

for root, _, files in os.walk(logs_dir):
    if any(f.startswith("events.out.tfevents") for f in files):
        ea = EventAccumulator(root)
        ea.Reload()
        for tag in ea.Tags().get("scalars", []):
            events = ea.Scalars(tag)
            safe_tag = tag.replace("/", "_")
            out_path = os.path.join(out_dir, f"{safe_tag}.csv")
            with open(out_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["step", "wall_time", "value"])
                for e in events:
                    writer.writerow([e.step, e.wall_time, e.value])
            print(f"Exported {tag} -> {out_path}")
EOF

echo "== 4. Reproducibility metadata =="
python -m pip freeze > "${EXPORT_DIR}/requirements_frozen.txt"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv > "${EXPORT_DIR}/gpu_info.csv"
cp configs/model_config.yaml configs/training_config.yaml "${EXPORT_DIR}/"

echo ""
echo "Export complete: ${EXPORT_DIR}"
echo "NOTE: sample SGLang input/output pairs (logs/sglang_io_samples.json) are included"
echo "      only if bench_inference.py --mode sglang was already run — check before cutoff."
echo "Transfer ${EXPORT_DIR} off the remote server now."

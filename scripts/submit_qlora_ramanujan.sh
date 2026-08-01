#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"
mkdir -p logs

echo "Job ID   : ${SLURM_JOB_ID}"
echo "Node     : $(hostname)"
echo "Started  : $(date)"
echo "GPU info :"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader

echo ""
source .venv/bin/activate
bash scripts/run_qlora.sh

echo ""
echo "Finished : $(date)"

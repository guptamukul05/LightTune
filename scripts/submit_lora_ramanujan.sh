#!/usr/bin/env bash
# scripts/submit_lora_ramanujan.sh
#
# Submits the LoRA training job to Ramanujan's Slurm scheduler.
#
# HOW TO USE:
#   1. Upload your project folder to Ramanujan via MobaXterm
#   2. On Ramanujan terminal, cd into the project root
#   3. Run:  sbatch scripts/submit_lora_ramanujan.sh
#   4. Check status:  squeue -u $USER
#   5. Watch live output:  tail -f logs/slurm_lora_<JOBID>.out

#SBATCH --job-name=lora_peft
#SBATCH --output=logs/slurm_lora_%j.out
#SBATCH --error=logs/slurm_lora_%j.err
#SBATCH --time=06:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8

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

bash scripts/run_lora.sh

echo ""
echo "Finished : $(date)"

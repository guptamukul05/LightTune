#!/usr/bin/env bash
# scripts/run_all.sh
# Orchestrates the full pipeline in order. Safe to re-run — each stage skips
# work that's already done unless --force is passed.
#
# Usage:
#   bash scripts/run_all.sh [options]
#
# Options:
#   --force              Re-run every stage even if outputs already exist
#   --skip-baseline       Skip baseline eval
#   --skip-lora           Skip LoRA training/eval
#   --skip-qlora          Skip QLoRA training/eval
#   --skip-sglang         Skip SGLang serving + inference benchmarking
#   --skip-export         Skip final export_artifacts.sh
#   --adapter-for-sglang  Which adapter to serve: lora | qlora (default: qlora)
#   --keep-going          Don't abort the whole run if one stage fails
#   -h, --help            Show this help

set -uo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"

FORCE=false
SKIP_BASELINE=false
SKIP_LORA=false
SKIP_QLORA=false
SKIP_SGLANG=false
SKIP_EXPORT=false
KEEP_GOING=false
ADAPTER_FOR_SGLANG="qlora"
SGLANG_PORT=30000
SGLANG_PID=""

log()  { echo -e "\n== [$(date +%H:%M:%S)] $* =="; }
warn() { echo "WARNING: $*" >&2; }
die()  { echo "ERROR: $*" >&2; exit 1; }

while [[ $# -gt 0 ]]; do
    case "$1" in
        --force) FORCE=true ;;
        --skip-baseline) SKIP_BASELINE=true ;;
        --skip-lora) SKIP_LORA=true ;;
        --skip-qlora) SKIP_QLORA=true ;;
        --skip-sglang) SKIP_SGLANG=true ;;
        --skip-export) SKIP_EXPORT=true ;;
        --keep-going) KEEP_GOING=true ;;
        --adapter-for-sglang) ADAPTER_FOR_SGLANG="$2"; shift ;;
        -h|--help) grep '^#' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *) die "Unknown option: $1" ;;
    esac
    shift
done

run_stage() {
    local name="$1"; shift
    log "STAGE: ${name}"
    if "$@"; then
        echo "OK: ${name}"
    else
        warn "FAILED: ${name}"
        if [ "${KEEP_GOING}" != "true" ]; then
            die "Aborting (use --keep-going to continue past failures)."
        fi
    fi
}

cleanup() {
    if [ -n "${SGLANG_PID}" ] && kill -0 "${SGLANG_PID}" 2>/dev/null; then
        log "Stopping SGLang server (pid ${SGLANG_PID})"
        kill "${SGLANG_PID}" 2>/dev/null || true
        wait "${SGLANG_PID}" 2>/dev/null || true
    fi
}
trap cleanup EXIT

mkdir -p logs checkpoints data exports

# ---- 0. Environment ----------------------------------------------------
if [ ! -f .venv/bin/activate ]; then
    run_stage "setup_env" bash scripts/setup_env.sh
else
    echo "venv already exists, skipping setup_env.sh (use rm -rf .venv to force)"
fi
source .venv/bin/activate

# ---- 1. Data preprocessing ----------------------------------------------
if [ "${FORCE}" = "true" ] || [ ! -d data/train ] || [ ! -d data/val ]; then
    run_stage "preprocess" python src/data/preprocess.py \
        --model_config configs/model_config.yaml \
        --train_config configs/training_config.yaml
else
    echo "data/train and data/val already exist, skipping preprocess (use --force to redo)"
fi

# ---- 2. Baseline eval ----------------------------------------------------
if [ "${SKIP_BASELINE}" != "true" ]; then
    if [ "${FORCE}" = "true" ] || [ ! -f logs/eval_baseline.json ]; then
        run_stage "baseline eval" bash scripts/run_baseline.sh
    else
        echo "logs/eval_baseline.json exists, skipping (use --force to redo)"
    fi
else
    echo "Skipping baseline eval (--skip-baseline)"
fi

# ---- 3. LoRA training + eval ---------------------------------------------
if [ "${SKIP_LORA}" != "true" ]; then
    if [ "${FORCE}" = "true" ] || [ ! -d checkpoints/lora ]; then
        run_stage "LoRA train+eval" bash scripts/run_lora.sh
    else
        echo "checkpoints/lora exists, skipping (use --force to redo)"
    fi
else
    echo "Skipping LoRA (--skip-lora)"
fi

# ---- 4. QLoRA training + eval --------------------------------------------
if [ "${SKIP_QLORA}" != "true" ]; then
    if [ "${FORCE}" = "true" ] || [ ! -d checkpoints/qlora ]; then
        run_stage "QLoRA train+eval" bash scripts/run_qlora.sh
    else
        echo "checkpoints/qlora exists, skipping (use --force to redo)"
    fi
else
    echo "Skipping QLoRA (--skip-qlora)"
fi

# ---- 5. SGLang serving + 3-way inference benchmark -----------------------
if [ "${SKIP_SGLANG}" != "true" ]; then
    ADAPTER_DIR="checkpoints/${ADAPTER_FOR_SGLANG}"
    if [ ! -d "${ADAPTER_DIR}" ]; then
        warn "Adapter dir ${ADAPTER_DIR} not found — skipping SGLang stage."
    else
        MODEL_REPO=$(python - <<'EOF'
import yaml
print(yaml.safe_load(open("configs/model_config.yaml"))["model"]["hf_repo"])
EOF
)
        log "STAGE: baseline inference benchmark"
        python src/evaluations/bench_inference.py --mode baseline \
            --model_config configs/model_config.yaml \
            || warn "baseline bench failed"

        log "STAGE: finetuned inference benchmark (${ADAPTER_FOR_SGLANG})"
        python src/evaluations/bench_inference.py --mode finetuned \
            --model_config configs/model_config.yaml \
            --adapter_path "${ADAPTER_DIR}" \
            || warn "finetuned bench failed"

        log "STAGE: launching SGLang server"
        bash scripts/serve_sglang.sh "${MODEL_REPO}" "${ADAPTER_DIR}" "${SGLANG_PORT}" \
            > logs/sglang_server.log 2>&1 &
        SGLANG_PID=$!
        echo "SGLang server pid: ${SGLANG_PID}, logs: logs/sglang_server.log"

        log "Waiting for SGLang server to become healthy"
        READY=false
        for i in $(seq 1 60); do
            if curl -sf "http://localhost:${SGLANG_PORT}/health" > /dev/null 2>&1; then
                READY=true
                break
            fi
            sleep 5
        done

        if [ "${READY}" = "true" ]; then
            run_stage "SGLang inference benchmark" python src/evaluations/bench_inference.py \
                --mode sglang \
                --sglang_url "http://localhost:${SGLANG_PORT}" \
                --sglang_model_name default
        else
            warn "SGLang server did not become healthy in time — check logs/sglang_server.log"
        fi

        cleanup
        SGLANG_PID=""
    fi
else
    echo "Skipping SGLang serving + benchmark (--skip-sglang)"
fi

# ---- 6. Export/freeze artifacts ------------------------------------------
if [ "${SKIP_EXPORT}" != "true" ]; then
    run_stage "export artifacts" bash scripts/export_artifacts.sh
else
    echo "Skipping export (--skip-export)"
fi

log "PIPELINE COMPLETE"
echo "Check logs/ for eval_*.json, bench_*.json, gpu_usage_*.csv, train_summary_*.json"
echo "Check exports/ for the frozen export bundle (if export stage ran)"

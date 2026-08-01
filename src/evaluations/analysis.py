"""
src/evaluations/analysis.py

Aggregates all JSON/CSV outputs under logs/ into:
  - training comparison table   (LoRA vs QLoRA)
  - inference comparison table  (baseline vs finetuned vs sglang)
  - GPU memory-over-time plots  (one per run, from gpu_usage_*.csv)
  - a single markdown report tying everything together

Reads only — never re-runs training/eval/inference. Safe to run at any point;
missing files are skipped with a warning rather than raising.

Outputs (default):
  logs/analysis/training_comparison.csv
  logs/analysis/inference_comparison.csv
  logs/analysis/gpu_usage_<run>.png
  logs/analysis/report.md

Usage:
    python src/evaluations/analysis.py
    python src/evaluations/analysis.py --logs_dir logs --out_dir logs/analysis
"""

import argparse
import csv
import json
import os
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_json(path):
    if not os.path.isfile(path):
        print(f"  [skip] {path} not found")
        return None
    with open(path) as f:
        return json.load(f)


def load_gpu_csv(path):
    if not os.path.isfile(path):
        return None
    rows = []
    with open(path) as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append({
                "timestamp": float(r["timestamp"]),
                "gpu_id": int(r["gpu_id"]),
                "mem_allocated_mb": float(r["mem_allocated_mb"]),
                "mem_reserved_mb": float(r["mem_reserved_mb"]),
            })
    return rows


def write_csv(path, fieldnames, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    print(f"  wrote {path}")


def fmt(v, nd=4):
    if v is None:
        return "-"
    if isinstance(v, float):
        return f"{v:.{nd}f}"
    return str(v)


def collect_training(logs_dir):
    """Merge per-method training summary + eval (loss/acc/f1) into one row each."""
    train_files = {
        "lora": ["train_lora_summary.json", "train_summary_lora.json"],
        "qlora": ["train_summary_qlora.json", "train_qlora_summary.json"],
    }
    eval_files = {
        "baseline": "eval_baseline.json",
        "lora": "eval_lora.json",
        "qlora": "eval_qlora.json",
    }

    rows = []
    for method, candidates in train_files.items():
        summary = None
        for c in candidates:
            summary = load_json(os.path.join(logs_dir, c))
            if summary:
                break
        ev = load_json(os.path.join(logs_dir, eval_files[method]))

        if summary is None and ev is None:
            continue

        rows.append({
            "method": method,
            "training_time_sec": (summary or {}).get("training_time_sec")
                or (summary or {}).get("train_time_sec"),
            "peak_gpu_mem_mb_train": (summary or {}).get("peak_gpu_mem_mb"),
            "trainable_params": (summary or {}).get("trainable_params"),
            "trainable_pct": (summary or {}).get("trainable_pct"),
            "final_train_loss": (summary or {}).get("final_train_loss"),
            "val_loss": (ev or {}).get("val_loss"),
            "accuracy": (ev or {}).get("accuracy"),
            "f1": (ev or {}).get("f1"),
            "eval_peak_gpu_mem_mb": (ev or {}).get("peak_gpu_mem_mb"),
            "eval_throughput_ex_per_sec": (ev or {}).get("throughput_examples_per_sec"),
        })

    baseline_ev = load_json(os.path.join(logs_dir, eval_files["baseline"]))
    if baseline_ev:
        rows.insert(0, {
            "method": "baseline",
            "training_time_sec": None,
            "peak_gpu_mem_mb_train": None,
            "trainable_params": None,
            "trainable_pct": None,
            "final_train_loss": None,
            "val_loss": baseline_ev.get("val_loss"),
            "accuracy": baseline_ev.get("accuracy"),
            "f1": baseline_ev.get("f1"),
            "eval_peak_gpu_mem_mb": baseline_ev.get("peak_gpu_mem_mb"),
            "eval_throughput_ex_per_sec": baseline_ev.get("throughput_examples_per_sec"),
        })

    return rows


def collect_inference(logs_dir):
    modes = ["baseline", "finetuned", "sglang"]
    rows = []
    for mode in modes:
        bench = load_json(os.path.join(logs_dir, f"bench_{mode}.json"))
        if bench is None:
            continue
        rows.append({
            "mode": mode,
            "n_prompts": bench.get("n_prompts"),
            "avg_latency_sec": bench.get("avg_latency_sec"),
            "p50_latency_sec": bench.get("p50_latency_sec"),
            "p95_latency_sec": bench.get("p95_latency_sec"),
            "throughput_req_per_sec": bench.get("throughput_req_per_sec"),
            "peak_gpu_mem_mb": bench.get("peak_gpu_mem_mb"),
        })
    return rows


def plot_gpu_usage(logs_dir, out_dir):
    written = []
    for fname in sorted(os.listdir(logs_dir)) if os.path.isdir(logs_dir) else []:
        if not (fname.startswith("gpu_usage_") and fname.endswith(".csv")):
            continue
        run_name = fname[len("gpu_usage_"):-len(".csv")]
        data = load_gpu_csv(os.path.join(logs_dir, fname))
        if not data:
            continue

        t0 = data[0]["timestamp"]
        by_gpu = {}
        for r in data:
            by_gpu.setdefault(r["gpu_id"], {"t": [], "alloc": [], "reserved": []})
            by_gpu[r["gpu_id"]]["t"].append(r["timestamp"] - t0)
            by_gpu[r["gpu_id"]]["alloc"].append(r["mem_allocated_mb"])
            by_gpu[r["gpu_id"]]["reserved"].append(r["mem_reserved_mb"])

        plt.figure(figsize=(8, 4.5))
        for gpu_id, series in sorted(by_gpu.items()):
            plt.plot(series["t"], series["alloc"], label=f"GPU{gpu_id} allocated")
            plt.plot(series["t"], series["reserved"], linestyle="--", label=f"GPU{gpu_id} reserved")
        plt.xlabel("seconds since start")
        plt.ylabel("memory (MB)")
        plt.title(f"GPU memory — {run_name}")
        plt.legend(fontsize=8)
        plt.tight_layout()

        out_path = os.path.join(out_dir, f"gpu_usage_{run_name}.png")
        os.makedirs(out_dir, exist_ok=True)
        plt.savefig(out_path, dpi=150)
        plt.close()
        print(f"  wrote {out_path}")
        written.append(out_path)
    return written


def build_report(training_rows, inference_rows, gpu_plots, out_path):
    lines = [
        "# PEFTBench — Results Analysis",
        "",
        f"_Generated: {datetime.now().isoformat(timespec='seconds')}_",
        "",
        "## 1. Training & Model Quality — Baseline vs LoRA vs QLoRA",
        "",
    ]

    if training_rows:
        headers = ["method", "training_time_sec", "peak_gpu_mem_mb_train",
                   "trainable_pct", "val_loss", "accuracy", "f1"]
        lines.append("| " + " | ".join(headers) + " |")
        lines.append("|" + "---|" * len(headers))
        for row in training_rows:
            lines.append("| " + " | ".join(fmt(row.get(h)) for h in headers) + " |")
    else:
        lines.append("_No training/eval logs found yet — run `run_baseline.sh`, `run_lora.sh`, `run_qlora.sh` first._")

    lines += ["", "## 2. Inference — Baseline vs Finetuned vs SGLang-optimized", ""]

    if inference_rows:
        headers = ["mode", "avg_latency_sec", "p50_latency_sec", "p95_latency_sec",
                   "throughput_req_per_sec", "peak_gpu_mem_mb"]
        lines.append("| " + " | ".join(headers) + " |")
        lines.append("|" + "---|" * len(headers))
        for row in inference_rows:
            lines.append("| " + " | ".join(fmt(row.get(h)) for h in headers) + " |")
    else:
        lines.append("_No inference bench logs found yet — run `bench_inference.py` for each mode._")

    lines += ["", "## 3. GPU Memory Over Time", ""]
    if gpu_plots:
        for p in gpu_plots:
            lines.append(f"![{os.path.basename(p)}]({os.path.relpath(p, os.path.dirname(out_path))})")
    else:
        lines.append("_No gpu_usage_*.csv files found yet._")

    lines += [
        "",
        "## 4. Notes / Known Limitations",
        "",
        "- accuracy/F1 above are token-level next-token proxies, not task-level generation quality "
        "(see README §12) — treat as a rough signal, not a leaderboard metric.",
        "- `eval_peak_gpu_mem_mb` (from `evaluate.py`) and `peak_gpu_mem_mb` under inference bench "
        "are measured in separate processes and are not directly comparable to training memory.",
        "",
    ]

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        f.write("\n".join(lines))
    print(f"  wrote {out_path}")



def main(logs_dir, out_dir):
    print("== Collecting training/eval logs ==")
    training_rows = collect_training(logs_dir)
    if training_rows:
        write_csv(
            os.path.join(out_dir, "training_comparison.csv"),
            list(training_rows[0].keys()),
            training_rows,
        )

    print("== Collecting inference bench logs ==")
    inference_rows = collect_inference(logs_dir)
    if inference_rows:
        write_csv(
            os.path.join(out_dir, "inference_comparison.csv"),
            list(inference_rows[0].keys()),
            inference_rows,
        )

    print("== Plotting GPU memory usage ==")
    gpu_plots = plot_gpu_usage(logs_dir, out_dir)

    print("== Building report ==")
    build_report(training_rows, inference_rows, gpu_plots, os.path.join(out_dir, "report.md"))

    print(f"\nDone. See {out_dir}/report.md")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--logs_dir", default="logs")
    parser.add_argument("--out_dir", default="logs/analysis")
    args = parser.parse_args()
    main(args.logs_dir, args.out_dir)

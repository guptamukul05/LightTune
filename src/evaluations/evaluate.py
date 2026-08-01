"""
src/evaluation/evaluate.py

Evaluates a model (baseline or finetuned) on the tokenized validation set.
Computes validation loss, accuracy, F1, and captures GPU resource usage
alongside wall-clock timing. Used for baseline (Step 3) and later for
LoRA/QLoRA finetuned evaluation with the same code path (--adapter_path).

Usage (baseline):
    python src/evaluation/evaluate.py \
        --config configs/model_config.yaml \
        --run_name baseline

Usage (finetuned):
    python src/evaluation/evaluate.py \
        --config configs/model_config.yaml \
        --adapter_path checkpoints/lora \
        --run_name lora
"""

import argparse
import json
import os
import time

import numpy as np
import torch
import yaml
from tqdm import tqdm
from datasets import load_from_disk
from sklearn.metrics import accuracy_score, f1_score
from transformers import AutoModelForCausalLM, AutoTokenizer

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
from src.utils.gpu_monitor import GPUMonitor


def load_model(model_cfg: dict, adapter_path: str | None):
    hf_repo = model_cfg["model"]["hf_repo"]
    dtype = torch.bfloat16 if model_cfg["precision"]["mixed_precision"] == "bf16" else torch.float16

    model = AutoModelForCausalLM.from_pretrained(
        hf_repo,
        torch_dtype=dtype,
        device_map="auto",
        trust_remote_code=model_cfg["model"]["trust_remote_code"],
    )

    if adapter_path:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter_path)

    model.eval()
    return model


@torch.no_grad()
def run_eval(model, tokenizer, val_dataset, max_length: int):
    total_loss = 0.0
    all_preds = []
    all_labels = []
    n_batches = 0

    for example in tqdm(val_dataset, desc="Evaluating"):
        input_ids = torch.tensor(example["input_ids"]).unsqueeze(0).to(model.device)
        attention_mask = torch.tensor(example["attention_mask"]).unsqueeze(0).to(model.device)
        labels = input_ids.clone()
        labels[attention_mask == 0] = -100

        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        total_loss += outputs.loss.item()
        n_batches += 1
        
        shift_logits = outputs.logits[:, :-1, :]
        shift_labels = labels[:, 1:]

        mask = attention_mask[:, 1:].bool()
        preds = torch.argmax(shift_logits, dim=-1)
        all_preds.extend(preds[mask].cpu().numpy().tolist())
        all_labels.extend(input_ids[:, 1:][mask].cpu().numpy().tolist())

    avg_loss = total_loss / max(n_batches, 1)
    # Token-level accuracy/F1 as proxy classification metrics (per README §12)
    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)

    return {"val_loss": avg_loss, "accuracy": accuracy, "f1": f1}


def main(config_path: str, run_name: str, adapter_path: str | None, val_data_path: str):
    with open(config_path, "r") as f:
        model_cfg = yaml.safe_load(f)

    os.makedirs("logs", exist_ok=True)
    monitor = GPUMonitor(log_path=f"logs/gpu_usage_{run_name}.csv", interval_seconds=10)
    monitor.start()

    tokenizer = AutoTokenizer.from_pretrained(model_cfg["model"]["hf_repo"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    val_dataset = load_from_disk(val_data_path)

    start_time = time.time()
    model = load_model(model_cfg, adapter_path)
    load_time = time.time() - start_time

    eval_start = time.time()
    metrics = run_eval(model, tokenizer, val_dataset, model_cfg["model"]["max_seq_length"])
    eval_time = time.time() - eval_start

    monitor.stop()

    peak_mem_mb = max(
        torch.cuda.max_memory_allocated(i) / (1024 ** 2)
        for i in range(torch.cuda.device_count())
    )

    result = {
        "run_name": run_name,
        "model_load_time_sec": round(load_time, 2),
        "eval_time_sec": round(eval_time, 2),
        "throughput_examples_per_sec": round(len(val_dataset) / eval_time, 4),
        "peak_gpu_mem_mb": round(peak_mem_mb, 2),
        **metrics,
    }

    os.makedirs("logs", exist_ok=True)
    out_path = f"logs/eval_{run_name}.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)

    print(json.dumps(result, indent=2))
    print(f"Saved results to {out_path}")
    print(f"GPU usage log: logs/gpu_usage_{run_name}.csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/model_config.yaml")
    parser.add_argument("--adapter_path", default=None, help="Path to LoRA/QLoRA adapter (omit for baseline)")
    parser.add_argument("--run_name", default="baseline")
    parser.add_argument("--val_data_path", default="data/val")
    args = parser.parse_args()
    main(args.config, args.run_name, args.adapter_path, args.val_data_path)

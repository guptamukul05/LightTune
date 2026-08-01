"""
src/training/train_lora.py  —  Step 4a

LoRA fine-tuning for Qwen2.5-1.5B on preprocessed Alpaca data.

Base model weights are fully frozen. Two small matrices A and B are
inserted into each attention and MLP layer. Only A and B are trained,
dropping trainable params from 1.5B to ~5M (~0.33%) and cutting GPU
memory by around 60-70% versus full fine-tuning.

Reads  : configs/model_config.yaml, configs/training_config.yaml
Writes : checkpoints/lora/  (adapter weights + tokenizer)
         logs/train_lora_summary.json
         logs/gpu_usage_lora.csv

Run:
    source .venv/bin/activate
    python src/training/train_lora.py
"""

import argparse
import json
import os
import sys
import time

import torch
import yaml
from datasets import load_from_disk
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from src.utils.gpu_monitor import GPUMonitor


def read_configs(model_cfg_path, train_cfg_path):
    with open(model_cfg_path) as f:
        model_cfg = yaml.safe_load(f)
    with open(train_cfg_path) as f:
        train_cfg = yaml.safe_load(f)
    return model_cfg, train_cfg


def get_tokenizer(model_cfg):
    tok = AutoTokenizer.from_pretrained(
        model_cfg["model"]["hf_repo"],
        trust_remote_code=model_cfg["model"]["trust_remote_code"],
    )
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


def load_base_model(model_cfg):
    dtype = torch.bfloat16 if model_cfg["precision"]["mixed_precision"] == "bf16" else torch.float16
    n_gpu = torch.cuda.device_count()
    use_multi_gpu = model_cfg.get("multi_gpu", {}).get("enabled", False) and n_gpu > 1
    device_map = "auto" if use_multi_gpu else None

    model = AutoModelForCausalLM.from_pretrained(
        model_cfg["model"]["hf_repo"],
        torch_dtype=dtype,
        device_map=device_map,
        trust_remote_code=model_cfg["model"].get("trust_remote_code", False),
    )
    if not use_multi_gpu and torch.cuda.is_available():
        model = model.to("cuda")
    return model

def attach_lora(model, lora_cfg):
    config = LoraConfig(
        r=lora_cfg["r"],
        lora_alpha=lora_cfg["lora_alpha"],
        lora_dropout=lora_cfg["lora_dropout"],
        target_modules=lora_cfg["target_modules"],
        bias=lora_cfg["bias"],
        task_type=TaskType.CAUSAL_LM,
    )
    return get_peft_model(model, config)


def build_training_args(t, output_dir):
    use_bf16 = t.get("bf16", True)
    return TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=t["num_train_epochs"],
        per_device_train_batch_size=t["per_device_train_batch_size"],
        per_device_eval_batch_size=t["per_device_eval_batch_size"],
        gradient_accumulation_steps=t["gradient_accumulation_steps"],
        gradient_checkpointing=t["gradient_checkpointing"],
        learning_rate=t["learning_rate"],
        lr_scheduler_type=t["lr_scheduler_type"],
        warmup_ratio=t["warmup_ratio"],
        weight_decay=t["weight_decay"],
        logging_steps=t["logging_steps"],
        eval_strategy=t["eval_strategy"],
        eval_steps=t["eval_steps"],
        save_strategy=t["save_strategy"],
        save_steps=t["save_steps"],
        save_total_limit=t["save_total_limit"],
        bf16=use_bf16,
        fp16=not use_bf16,
        report_to=t["report_to"],
        logging_dir=t["logging_dir"],
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        dataloader_pin_memory=True,
    )


def main(model_cfg_path, train_cfg_path):
    model_cfg, train_cfg = read_configs(model_cfg_path, train_cfg_path)
    t_cfg = train_cfg["training"]

    adapter_dir = os.path.join(t_cfg["output_dir"], "lora")
    os.makedirs(adapter_dir, exist_ok=True)
    os.makedirs("logs", exist_ok=True)

    monitor = GPUMonitor(
        log_path="logs/gpu_usage_lora.csv",
        interval_seconds=train_cfg["resource_monitor"]["interval_seconds"],
    )
    monitor.start()

    print(f"\n  Model  : {model_cfg['model']['hf_repo']}")
    print(f"  LoRA r : {model_cfg['lora']['r']}   alpha : {model_cfg['lora']['lora_alpha']}")
    print(f"  Epochs : {t_cfg['num_train_epochs']}   LR : {t_cfg['learning_rate']}\n")

    tokenizer = get_tokenizer(model_cfg)

    print("Loading base model ...")
    model = load_base_model(model_cfg)

    print("Attaching LoRA adapters (base weights frozen) ...")
    model = attach_lora(model, model_cfg["lora"])
    model.print_trainable_parameters()

    trainable, total = model.get_nb_trainable_parameters()

    print("\nLoading preprocessed data ...")
    train_ds = load_from_disk("data/train")
    val_ds   = load_from_disk("data/val")
    print(f"  Train : {len(train_ds):,}   Val : {len(val_ds):,}\n")

    collator      = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    training_args = build_training_args(t_cfg, adapter_dir)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        processing_class=tokenizer,
        data_collator=collator,
    )

    print("Starting LoRA fine-tuning ...\n")
    t0 = time.time()
    trainer.train()
    elapsed = time.time() - t0

    print("\nSaving adapter weights ...")
    model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)

    monitor.stop()

    peak_mb = 0.0
    if torch.cuda.is_available():
        peak_mb = max(
            torch.cuda.max_memory_allocated(i) / (1024 ** 2)
            for i in range(torch.cuda.device_count())
        )

    summary = {
        "method"            : "lora",
        "model"             : model_cfg["model"]["hf_repo"],
        "lora_r"            : model_cfg["lora"]["r"],
        "lora_alpha"        : model_cfg["lora"]["lora_alpha"],
        "target_modules"    : model_cfg["lora"]["target_modules"],
        "trainable_params"  : trainable,
        "total_params"      : total,
        "trainable_pct"     : round(100 * trainable / total, 4),
        "training_time_sec" : round(elapsed, 2),
        "training_time_min" : round(elapsed / 60, 2),
        "peak_gpu_mem_mb"   : round(peak_mb, 2),
        "adapter_saved_to"  : adapter_dir,
    }

    with open("logs/train_lora_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("\n-- LoRA Training Complete --")
    print(json.dumps(summary, indent=2))
    print(f"\n  Adapter →  {adapter_dir}")
    print(f"  Summary →  logs/train_lora_summary.json")
    print(f"  GPU log →  logs/gpu_usage_lora.csv")
    print("\n  Next — evaluate:")
    print("    python src/evaluations/evaluate.py \\")
    print(f"      --config configs/model_config.yaml \\")
    print(f"      --adapter_path {adapter_dir} \\")
    print(f"      --run_name lora \\")
    print(f"      --val_data_path data/val")
    print("=" * 44)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_config", default="configs/model_config.yaml")
    parser.add_argument("--train_config", default="configs/training_config.yaml")
    args = parser.parse_args()
    main(args.model_config, args.train_config)

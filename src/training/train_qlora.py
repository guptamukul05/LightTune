"""
src/training/train_qlora.py
Mirrors train_lora.py but loads the base model in 4-bit (NF4) via bitsandbytes
before attaching the LoRA adapter, per configs/model_config.yaml's `qlora` block.
"""
import argparse
import json
import os
import sys
import time
import yaml
import torch
from datasets import load_from_disk
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

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
        trust_remote_code=model_cfg["model"].get("trust_remote_code", False),
    )
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


def load_qlora_base_model(model_cfg):
    q_cfg = model_cfg["qlora"]
    compute_dtype = getattr(torch, q_cfg["bnb_4bit_compute_dtype"])

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=q_cfg["load_in_4bit"],
        bnb_4bit_quant_type=q_cfg["bnb_4bit_quant_type"],
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=q_cfg["bnb_4bit_use_double_quant"],
    )

    # NOTE: bitsandbytes 4-bit models don't support true tensor-parallel sharding.
    # device_map="auto" will naive-shard layers across GPUs if >1 is visible;
    # this is NOT the same as model_config.yaml's multi_gpu.strategy: tensor_parallel,
    # which only applies to the non-quantized LoRA path / SGLang serving.
    model = AutoModelForCausalLM.from_pretrained(
        model_cfg["model"]["hf_repo"],
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=model_cfg["model"].get("trust_remote_code", False),
    )
    model = prepare_model_for_kbit_training(model)
    return model


def attach_lora(model, lora_cfg):
    config = LoraConfig(
        r=lora_cfg["r"],
        lora_alpha=lora_cfg["lora_alpha"],
        lora_dropout=lora_cfg["lora_dropout"],
        target_modules=lora_cfg["target_modules"],
        bias=lora_cfg["bias"],
        task_type=lora_cfg["task_type"],
    )
    model = get_peft_model(model, config)
    model.print_trainable_parameters()
    return model


def build_training_args(t, output_dir):
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
        bf16=t.get("bf16", True),
        report_to=t.get("report_to", "tensorboard"),
        logging_dir=t.get("logging_dir", "logs"),
        optim="paged_adamw_8bit",  # required for stable QLoRA optimization
    )


def main(model_cfg_path, train_cfg_path):
    model_cfg, train_cfg = read_configs(model_cfg_path, train_cfg_path)
    t_cfg = train_cfg["training"]

    adapter_dir = os.path.join(t_cfg["output_dir"], "qlora")
    os.makedirs(adapter_dir, exist_ok=True)
    os.makedirs("logs", exist_ok=True)

    monitor = GPUMonitor(
        log_path="logs/gpu_usage_qlora.csv",
        interval_seconds=train_cfg.get("resource_monitor", {}).get("interval_seconds", 30),
    )
    monitor.start()

    try:
        tokenizer = get_tokenizer(model_cfg)
        model = load_qlora_base_model(model_cfg)
        model = attach_lora(model, model_cfg["lora"])

        if t_cfg["gradient_checkpointing"]:
            model.enable_input_require_grads()

        train_ds = load_from_disk("data/train")
        val_ds = load_from_disk("data/val")

        collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
        training_args = build_training_args(t_cfg, adapter_dir)

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_ds,
            eval_dataset=val_ds,
            data_collator=collator,
        )

        t0 = time.time()
        trainer.train()
        elapsed = time.time() - t0

        trainer.save_model(adapter_dir)
        tokenizer.save_pretrained(adapter_dir)

        peak_mb = 0.0
        if torch.cuda.is_available():
            peak_mb = max(
                torch.cuda.max_memory_allocated(i) / (1024 ** 2)
                for i in range(torch.cuda.device_count())
            )

        summary = {
            "method": "qlora",
            "train_time_sec": elapsed,
            "peak_gpu_mem_mb": peak_mb,
            "final_train_loss": (
                trainer.state.log_history[-1].get("loss")
                if trainer.state.log_history else None
            ),
        }
        with open("logs/train_summary_qlora.json", "w") as f:
            json.dump(summary, f, indent=2)

        print(f"QLoRA training complete in {elapsed:.1f}s, peak mem {peak_mb:.0f} MB")
        print(f"Adapter saved to {adapter_dir}")

    finally:
        monitor.stop()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_config", default="configs/model_config.yaml")
    parser.add_argument("--train_config", default="configs/training_config.yaml")
    args = parser.parse_args()
    main(args.model_config, args.train_config)

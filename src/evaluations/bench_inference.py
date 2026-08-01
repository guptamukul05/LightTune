"""
src/evaluations/bench_inference.py
Measures per-request latency, throughput, and peak GPU memory for one of:
  - baseline   : unmodified pretrained model (local HF generate)
  - finetuned  : base model + LoRA/QLoRA adapter (local HF generate)
  - sglang     : running SGLang OpenAI-compatible server (HTTP)
Writes logs/bench_{mode}.json. For --mode sglang, also writes
logs/sglang_io_samples.json (static prompt/response pairs, per constraint #10).
"""
import argparse
import json
import time
import yaml
import torch
import requests
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

DEFAULT_PROMPTS = [
    "Explain the difference between LoRA and QLoRA in two sentences.",
    "Write a haiku about GPUs running out of memory.",
    "Summarize what mixed precision training does.",
    "List three benefits of gradient checkpointing.",
    "What is RadixAttention used for?",
]


def load_local_model(model_cfg, adapter_path):
    dtype = torch.bfloat16 if model_cfg["precision"]["mixed_precision"] == "bf16" else torch.float16

    repo = model_cfg["model"]["hf_repo"]
    revision = model_cfg["model"].get("revision", "main")
    trust_remote_code = model_cfg["model"].get("trust_remote_code", False)

    model = AutoModelForCausalLM.from_pretrained(
        repo,
        revision=revision,  
        torch_dtype=dtype,
        trust_remote_code=trust_remote_code,
    ).to("cuda" if torch.cuda.is_available() else "cpu")

    if adapter_path:
        model = PeftModel.from_pretrained(model, adapter_path)

    model.eval()
    return model


def bench_local(model_cfg, adapter_path, prompts, max_new_tokens):
    repo = model_cfg["model"]["hf_repo"]
    revision = model_cfg["model"].get("revision", "main")
    trust_remote_code = model_cfg["model"].get("trust_remote_code", False)
    
    tokenizer = AutoTokenizer.from_pretrained(
        repo,
        revision=revision,  
        trust_remote_code=trust_remote_code,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = load_local_model(model_cfg, adapter_path)

    latencies = []
    t_start = time.time()
    for p in prompts:
        inputs = tokenizer(p, return_tensors="pt").to(model.device)
        t0 = time.time()
        with torch.no_grad():
            model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        latencies.append(time.time() - t0)
    total_time = time.time() - t_start

    peak_mb = 0.0
    if torch.cuda.is_available():
        peak_mb = max(
            torch.cuda.max_memory_allocated(i) / (1024 ** 2)
            for i in range(torch.cuda.device_count())
        )
    return latencies, total_time, peak_mb, None


def bench_sglang(base_url, model_name, prompts, max_new_tokens):
    latencies = []
    io_samples = []
    t_start = time.time()
    for p in prompts:
        t0 = time.time()
        resp = requests.post(
            f"{base_url}/v1/chat/completions",
            json={
                "model": model_name,
                "messages": [{"role": "user", "content": p}],
                "max_tokens": max_new_tokens,
                "temperature": 0.0,
            },
            timeout=120,
        )
        resp.raise_for_status()
        latencies.append(time.time() - t0)
        reply = resp.json()["choices"][0]["message"]["content"]
        io_samples.append({"prompt": p, "response": reply})
    total_time = time.time() - t_start
    # SGLang exposes its own /metrics for server-side memory; not captured client-side here.
    return latencies, total_time, None, io_samples


def summarize(mode, latencies, total_time, peak_mb, n_prompts):
    sorted_lat = sorted(latencies)
    return {
        "mode": mode,
        "n_prompts": n_prompts,
        "avg_latency_sec": sum(latencies) / len(latencies),
        "p50_latency_sec": sorted_lat[len(sorted_lat) // 2],
        "p95_latency_sec": sorted_lat[max(int(len(sorted_lat) * 0.95) - 1, 0)],
        "throughput_req_per_sec": n_prompts / total_time,
        "peak_gpu_mem_mb": peak_mb,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["baseline", "finetuned", "sglang"], required=True)
    parser.add_argument("--model_config", default="configs/model_config.yaml")
    parser.add_argument("--adapter_path", default=None, help="required for --mode finetuned")
    parser.add_argument("--sglang_url", default="http://localhost:30000")
    parser.add_argument("--sglang_model_name", default="default")
    parser.add_argument("--max_new_tokens", type=int, default=128)
    args = parser.parse_args()

    with open(args.model_config) as f:
        model_cfg = yaml.safe_load(f)

    prompts = DEFAULT_PROMPTS

    if args.mode == "baseline":
        latencies, total_time, peak_mb, io_samples = bench_local(
            model_cfg, None, prompts, args.max_new_tokens
        )
    elif args.mode == "finetuned":
        if not args.adapter_path:
            raise ValueError("--adapter_path is required for --mode finetuned")
        latencies, total_time, peak_mb, io_samples = bench_local(
            model_cfg, args.adapter_path, prompts, args.max_new_tokens
        )
    else:
        latencies, total_time, peak_mb, io_samples = bench_sglang(
            args.sglang_url, args.sglang_model_name, prompts, args.max_new_tokens
        )

    result = summarize(args.mode, latencies, total_time, peak_mb, len(prompts))
    out_path = f"logs/bench_{args.mode}.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)

    if io_samples:
        with open("logs/sglang_io_samples.json", "w") as f:
            json.dump(io_samples, f, indent=2)
        print("Saved sample I/O pairs to logs/sglang_io_samples.json")

    print(json.dumps(result, indent=2))
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()

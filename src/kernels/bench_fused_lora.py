"""
src/kernels/bench_fused_lora.py

Benchmarks native (2-matmul) vs fused Triton LoRA forward across a few
(M, K, N, r) shapes representative of Qwen2.5-1.5B's projection layers
(hidden_size=1536) at your configured LoRA rank (default r=16, see
configs/model_config.yaml). Only run this AFTER test_fused_lora.py passes.

Measures forward-only latency and peak memory. Backward is intentionally
excluded from timing since it currently reuses native ops on both sides
(see fused_lora.py docstring) — a backward-inclusive comparison would just
measure identical code twice.

Usage:
    python src/kernels/bench_fused_lora.py
Writes: logs/kernel_bench.json
"""

import json
import os
import time

import torch

from fused_lora import fused_lora_forward


def native_lora_forward(x, A, B, scaling):
    return (x @ A.t()) @ B.t() * scaling


def _time_fn(fn, warmup=10, iters=50):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0

    peak_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
    avg_ms = (elapsed / iters) * 1000
    return avg_ms, peak_mb


def bench_case(M, K, N, r, dtype=torch.float16):
    device = "cuda"
    x = torch.randn(M, K, device=device, dtype=dtype)
    A = torch.randn(r, K, device=device, dtype=dtype)
    B = torch.randn(N, r, device=device, dtype=dtype)
    scaling = 2.0

    native_ms, native_mb = _time_fn(lambda: native_lora_forward(x, A, B, scaling))
    fused_ms, fused_mb = _time_fn(lambda: fused_lora_forward(x, A, B, scaling))

    speedup = native_ms / fused_ms if fused_ms > 0 else float("nan")
    mem_reduction_pct = (1 - fused_mb / native_mb) * 100 if native_mb > 0 else float("nan")

    result = {
        "M": M, "K": K, "N": N, "r": r,
        "native_ms": round(native_ms, 4),
        "fused_ms": round(fused_ms, 4),
        "speedup_x": round(speedup, 3),
        "native_peak_mb": round(native_mb, 2),
        "fused_peak_mb": round(fused_mb, 2),
        "mem_reduction_pct": round(mem_reduction_pct, 2),
    }
    print(
        f"M={M:<6} K={K:<6} N={N:<6} r={r:<4} "
        f"native={native_ms:.4f}ms fused={fused_ms:.4f}ms speedup={speedup:.2f}x "
        f"mem_native={native_mb:.1f}MB mem_fused={fused_mb:.1f}MB "
        f"mem_reduction={mem_reduction_pct:.1f}%"
    )
    return result


def main():
    if not torch.cuda.is_available():
        print("CUDA not available — this benchmark requires a GPU. Aborting.")
        return

    # (M, K, N, r): M scales with batch_size * seq_length (rows fed through the layer),
    # K/N=1536 matches Qwen2.5-1.5B hidden size, r matches configs/model_config.yaml lora.r
    cases = [
        (512, 1536, 1536, 16),     # ~ batch=8, seq truncated small
        (4096, 1536, 1536, 16),    # ~ batch=8, seq_length=512 (your reduced max_seq_length)
        (8192, 1536, 1536, 16),    # ~ batch=8, seq_length=1024 (original max_seq_length)
        (4096, 1536, 1536, 32),    # larger rank, same token volume
    ]

    results = [bench_case(*c) for c in cases]

    os.makedirs("logs", exist_ok=True)
    out_path = "logs/kernel_bench.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
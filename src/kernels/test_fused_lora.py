"""
src/kernels/test_fused_lora.py

Correctness gate for the fused LoRA kernel run this BEFORE any benchmarking
or integration into train_lora.py/train_qlora.py. Must be run on a CUDA GPU.

Checks:
  1. Forward output matches native (x @ A^T) @ B^T * scaling within tolerance.
  2. Backward gradients (dL/dx, dL/dA, dL/dB) match native autograd within tolerance.
  3. Runs across a few (M, K, N, r) shape combinations, including non-block-aligned
     sizes, to catch masking/boundary bugs in the Triton kernel.

Usage:
    python src/kernels/test_fused_lora.py
Exits non-zero on any failure.
"""

import sys

import torch

from fused_lora import fused_lora_forward


def native_lora_forward(x, A, B, scaling):
    return (x @ A.t()) @ B.t() * scaling


def run_case(M, K, N, r, dtype=torch.float32, atol=1e-2, rtol=1e-2):
    device = "cuda"
    torch.manual_seed(0)

    x_native = torch.randn(M, K, device=device, dtype=dtype, requires_grad=True)
    A_native = torch.randn(r, K, device=device, dtype=dtype, requires_grad=True)
    B_native = torch.randn(N, r, device=device, dtype=dtype, requires_grad=True)

    x_fused = x_native.detach().clone().requires_grad_()
    A_fused = A_native.detach().clone().requires_grad_()
    B_fused = B_native.detach().clone().requires_grad_()

    scaling = 2.0

    out_native = native_lora_forward(x_native, A_native, B_native, scaling)
    out_fused = fused_lora_forward(x_fused, A_fused, B_fused, scaling)

    fwd_ok = torch.allclose(out_native, out_fused, atol=atol, rtol=rtol)
    fwd_max_diff = (out_native - out_fused).abs().max().item()

    grad_out = torch.randn_like(out_native)
    out_native.backward(grad_out)
    out_fused.backward(grad_out.clone())

    grad_x_ok = torch.allclose(x_native.grad, x_fused.grad, atol=atol, rtol=rtol)
    grad_A_ok = torch.allclose(A_native.grad, A_fused.grad, atol=atol, rtol=rtol)
    grad_B_ok = torch.allclose(B_native.grad, B_fused.grad, atol=atol, rtol=rtol)

    passed = fwd_ok and grad_x_ok and grad_A_ok and grad_B_ok
    status = "PASS" if passed else "FAIL"
    print(
        f"[{status}] M={M:<5} K={K:<5} N={N:<5} r={r:<3} dtype={str(dtype):<15} "
        f"fwd_max_diff={fwd_max_diff:.5f} "
        f"fwd={fwd_ok} grad_x={grad_x_ok} grad_A={grad_A_ok} grad_B={grad_B_ok}"
    )
    return passed


def main():
    if not torch.cuda.is_available():
        print("CUDA not available â€” this test requires a GPU. Aborting.")
        sys.exit(1)

    cases = [
        # (M, K, N, r) includes block-aligned and deliberately misaligned sizes
        (32, 256, 256, 16),
        (128, 1536, 1536, 16),   # matches Qwen2.5-1.5B hidden size, LoRA r=16
        (100, 1536, 1536, 16),   # M not a multiple of BLOCK_M=64
        (128, 1537, 1536, 16),   # K not a multiple of BLOCK_K=32
        (128, 1536, 1000, 16),   # N not a multiple of BLOCK_N=64
        (64, 1536, 1536, 8),     # smaller rank than R_PAD minimum (16)
        (64, 1536, 1536, 32),    # rank above the 16 minimum
    ]

    all_passed = True
    for M, K, N, r in cases:
        ok = run_case(M, K, N, r)
        all_passed = all_passed and ok

    if all_passed:
        print("\nAll correctness checks PASSED.")
        sys.exit(0)
    else:
        print("\nSome correctness checks FAILED do not integrate or benchmark yet.")
        sys.exit(1)


if __name__ == "__main__":
    main()
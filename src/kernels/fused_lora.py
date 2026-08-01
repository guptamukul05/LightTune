# -*- coding: utf-8 -*-
"""
src/kernels/fused_lora.py

Fused LoRA forward kernel using Triton.

Computes:
    out = ((x @ A^T) @ B^T) * scaling

Key idea:
- Avoid materializing intermediate h = x @ A^T in global memory
- Keep h in registers ? reduce memory traffic ? faster
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _fused_lora_kernel(
    x_ptr, a_ptr, b_ptr, out_ptr,
    M, K, N, R,
    stride_xm, stride_xk,
    stride_ar, stride_ak,
    stride_bn, stride_br,
    stride_om, stride_on,
    scaling,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    R_PAD: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_r = tl.arange(0, R_PAD)
    offs_k = tl.arange(0, BLOCK_K)


    # Stage 1: h = x @ A^T (fp32)

    h = tl.zeros((BLOCK_M, R_PAD), dtype=tl.float32)

    for k0 in range(0, K, BLOCK_K):
        x_mask = (offs_m[:, None] < M) & ((k0 + offs_k)[None, :] < K)
        x_blk = tl.load(
            x_ptr + offs_m[:, None] * stride_xm + (k0 + offs_k)[None, :] * stride_xk,
            mask=x_mask,
            other=0.0,
        )

        a_mask = (offs_r[:, None] < R) & ((k0 + offs_k)[None, :] < K)
        a_blk = tl.load(
            a_ptr + offs_r[:, None] * stride_ar + (k0 + offs_k)[None, :] * stride_ak,
            mask=a_mask,
            other=0.0,
        )

        # accumulate in fp32
        h += tl.dot(x_blk, tl.trans(a_blk), out_dtype=tl.float32)

    # Stage 2: out = h @ B^T (fp32)

    b_mask = (offs_n[:, None] < N) & (offs_r[None, :] < R)

    b_blk = tl.load(
        b_ptr + offs_n[:, None] * stride_bn + offs_r[None, :] * stride_br,
        mask=b_mask,
        other=0.0,
    ).to(tl.float32)  # IMPORTANT: keep fp32

    out = tl.dot(h, tl.trans(b_blk), out_dtype=tl.float32)
    out = out * scaling

    # Store result
    out_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)

    tl.store(
        out_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on,
        out.to(tl.float16) if out_ptr.dtype.element_ty == tl.float16 else out,
        mask=out_mask,
    )


def _launch(x2d, A, B, scaling):
    M, K = x2d.shape
    R, K2 = A.shape
    N, R2 = B.shape

    assert K == K2 and R == R2, "Shape mismatch: expected A:(r,K), B:(N,r)"

    out = torch.empty((M, N), device=x2d.device, dtype=x2d.dtype)

    BLOCK_M = 64
    BLOCK_N = 64
    BLOCK_K = 32

    # Triton requires reduction dim >= 16
    R_PAD = max(16, triton.next_power_of_2(R))

    grid = (
        triton.cdiv(M, BLOCK_M),
        triton.cdiv(N, BLOCK_N),
    )

    _fused_lora_kernel[grid](
        x2d, A, B, out,
        M, K, N, R,
        x2d.stride(0), x2d.stride(1),
        A.stride(0), A.stride(1),
        B.stride(0), B.stride(1),
        out.stride(0), out.stride(1),
        scaling,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
        R_PAD=R_PAD,
    )

    return out


class FusedLoRAFunction(torch.autograd.Function):
    """Forward fused, backward uses native PyTorch."""

    @staticmethod
    def forward(ctx, x, A, B, scaling):
        orig_shape = x.shape
        x2d = x.reshape(-1, orig_shape[-1])

        if not x2d.is_cuda:
            raise RuntimeError("fused_lora requires CUDA tensors")

        out2d = _launch(
            x2d.contiguous(),
            A.contiguous(),
            B.contiguous(),
            scaling,
        )

        ctx.save_for_backward(x2d, A, B)
        ctx.scaling = scaling
        ctx.orig_shape = orig_shape

        return out2d.reshape(*orig_shape[:-1], B.shape[0])

    @staticmethod
    def backward(ctx, grad_out):
        x2d, A, B = ctx.saved_tensors
        scaling = ctx.scaling

        grad_out2d = grad_out.reshape(-1, grad_out.shape[-1])

        # Native backward (correct + stable)
        h = x2d @ A.t()
        grad_h = (grad_out2d @ B) * scaling
        grad_A = grad_h.t() @ x2d
        grad_B = (grad_out2d.t() @ h) * scaling
        grad_x2d = grad_h @ A

        grad_x = grad_x2d.reshape(ctx.orig_shape)

        return grad_x, grad_A, grad_B, None


def fused_lora_forward(x, A, B, scaling: float):
    return FusedLoRAFunction.apply(x, A, B, scaling)
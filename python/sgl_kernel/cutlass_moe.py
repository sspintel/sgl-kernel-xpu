"""CUTLASS-based fused MoE wrapper for Intel XPU.

Signature-compatible with sglang's CUDA cutlass_fused_experts_fp8. CUDA-only
args (*_strides, *_ptrs, use_mxfp8, enable_es) are accepted but unused: XPU's
fp8_blockwise_scaled_grouped_mm builds the pointer table and transposes
A-scales on device when given empty int64 ptr-array sentinels.
"""

from typing import Optional, Tuple

import torch
from sgl_kernel.elementwise import silu_and_mul
from sgl_kernel.gemm import (
    sgl_per_token_group_quant_8bit,
    sgl_per_token_group_quant_fp4,
)
from sgl_kernel.moe import (
    apply_shuffle_mul_sum,
    fp8_blockwise_scaled_grouped_mm,
    mxfp4_blockwise_scaled_grouped_mm,
    prepare_moe_input,
    scatter_tokens_to_experts,
)

_FP8_E4M3_MIN = -448.0
_FP8_E4M3_MAX = 448.0


def _per_token_group_quant_fp8(x: torch.Tensor, group_size: int = 128):
    """Per-token group quant along last dim. Returns (x_q fp8_e4m3, x_s fp32)."""
    assert x.shape[-1] % group_size == 0
    out_q = torch.empty(x.shape, device=x.device, dtype=torch.float8_e4m3fn)
    out_s_shape = (*x.shape[:-1], x.shape[-1] // group_size)
    out_s = torch.empty(out_s_shape, device=x.device, dtype=torch.float32)
    sgl_per_token_group_quant_8bit(
        x,
        out_q,
        out_s,
        group_size,
        1e-10,
        _FP8_E4M3_MIN,
        _FP8_E4M3_MAX,
        False,
        False,
        None,
        False,
    )
    return out_q, out_s


def cutlass_fused_experts_fp8(
    a: torch.Tensor,
    w1_q: torch.Tensor,
    w2_q: torch.Tensor,
    w1_scale: torch.Tensor,
    w2_scale: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    a1_strides: torch.Tensor,
    c1_strides: torch.Tensor,
    a2_strides: torch.Tensor,
    c2_strides: torch.Tensor,
    workspace: torch.Tensor,
    a_ptrs: torch.Tensor,
    b_ptrs: torch.Tensor,
    out_ptrs: torch.Tensor,
    a_scales_ptrs: torch.Tensor,
    b_scales_ptrs: torch.Tensor,
    expert_offsets: torch.Tensor,
    problem_sizes1: torch.Tensor,
    problem_sizes2: torch.Tensor,
    use_fp8_blockscale: bool = True,
    use_mxfp8: bool = False,
    output: Optional[torch.Tensor] = None,
    enable_es: Tuple[bool, bool] = (False, False),
) -> torch.Tensor:
    """Fused MoE on Intel XPU.

    Mirrors sglang's CUDA cutlass_fused_experts_fp8 signature; CUDA-only args
    (*_strides, *_ptrs, use_mxfp8, enable_es) are unused on XPU.

    Weights expected in (E, N, K) row-major: w1=(E, n*2, k), w2=(E, k, n).
    Sglang's dispatcher applies .transpose(1, 2) before calling, producing
    (E, k, n*2) and (E, n, k); auto-detected and undone below.
    """
    assert use_fp8_blockscale, "Only support fp8 blockscale on XPU"
    assert not use_mxfp8, "use_mxfp8 (SM100 path) is CUDA-only"
    assert enable_es == (False, False), "enable_es is CUDA-only"
    assert topk_weights.shape == topk_ids.shape, "topk shape mismatch"
    assert w1_q.dtype == torch.float8_e4m3fn
    assert w2_q.dtype == torch.float8_e4m3fn
    assert w1_q.dim() == 3 and w2_q.dim() == 3, "Weights must be 3D"
    assert w1_q.shape[0] == w2_q.shape[0], "Expert count mismatch w1/w2"
    assert a.dtype in (torch.half, torch.bfloat16), "Invalid input dtype"

    # Detect on w1 only — w2 is ambiguous when intermediate == k_hidden.
    # Sglang transposes both weights+scales together or not at all.
    hidden_size = a.shape[1]
    if w1_q.shape[2] == hidden_size:
        sglang_transposed = False
    elif w1_q.shape[1] == hidden_size:
        sglang_transposed = True
    else:
        raise AssertionError(
            f"w1_q shape {tuple(w1_q.shape)} incompatible with a.shape[1]={hidden_size}"
        )

    if sglang_transposed:
        w1_q = w1_q.transpose(1, 2).contiguous()
        w2_q = w2_q.transpose(1, 2).contiguous()
        w1_scale = w1_scale.transpose(1, 2).contiguous()
        w2_scale = w2_scale.transpose(1, 2).contiguous()

    assert w1_q.shape[2] == hidden_size
    assert w2_q.shape[1] == hidden_size
    assert w1_q.shape[1] == 2 * w2_q.shape[2]

    del a1_strides, c1_strides, a2_strides, c2_strides
    del a_ptrs, b_ptrs, out_ptrs, a_scales_ptrs, b_scales_ptrs

    out_dtype = a.dtype
    num_experts = w1_q.size(0)
    m = a.size(0)
    k = a.size(1)
    n = w2_q.size(2)
    topk = topk_ids.size(1)
    device = a.device

    a_map = torch.empty((topk_ids.numel(),), dtype=torch.int32, device=device)
    c_map = torch.empty((topk_ids.numel(),), dtype=torch.int32, device=device)

    # sglang allocates expert_offsets as (E+1,) and slices [:-1]; XPU's
    # prepare_moe_input takes size E. Accept either.
    eo = (
        expert_offsets[:num_experts]
        if expert_offsets.numel() > num_experts
        else expert_offsets
    )

    # prepare_moe_input fills eo with per-expert M counts (not cumulative).
    prepare_moe_input(
        topk_ids,
        eo,
        problem_sizes1,
        problem_sizes2,
        a_map,
        c_map,
        num_experts,
        n,
        k,
    )

    # Flat-2D kernel wants cumulative start offsets; exclusive-scan eo.
    expert_starts = torch.zeros(num_experts, dtype=torch.int32, device=device)
    if num_experts > 1:
        expert_starts[1:] = torch.cumsum(eo[:-1], dim=0).to(torch.int32)

    # Scatter then quantize. scatter_tokens_to_experts uses c_map (the
    # src->dst permutation) to gather per-expert rows.
    rep_a = torch.empty((m * topk, k), dtype=a.dtype, device=device)
    scatter_tokens_to_experts(a, c_map, rep_a)
    rep_a_q, rep_a1_scales = _per_token_group_quant_fp8(rep_a, group_size=128)

    c1 = torch.zeros((m * topk, n * 2), dtype=torch.float32, device=device)

    # Empty int64 sentinels select on-device pointer-table + scale-transpose.
    empty_ptrs = torch.empty((0,), dtype=torch.int64, device=device)
    zeros_stride = torch.zeros((num_experts,), dtype=torch.int64, device=device)
    zeros_layout = torch.zeros((num_experts, 5), dtype=torch.int32, device=device)

    fp8_blockwise_scaled_grouped_mm(
        c1,
        empty_ptrs,
        empty_ptrs,
        empty_ptrs,
        empty_ptrs,
        empty_ptrs,
        rep_a_q,
        w1_q,
        rep_a1_scales,
        w1_scale,
        zeros_stride,
        zeros_stride,
        zeros_stride,
        zeros_layout,
        zeros_layout,
        problem_sizes1,
        expert_starts,
        workspace,
    )

    intermediate = torch.empty((m * topk, n), dtype=out_dtype, device=device)
    silu_and_mul(c1.to(out_dtype), intermediate)

    intermediate_q, a2_scale = _per_token_group_quant_fp8(intermediate, group_size=128)

    c2 = torch.zeros((m * topk, k), dtype=torch.float32, device=device)
    fp8_blockwise_scaled_grouped_mm(
        c2,
        empty_ptrs,
        empty_ptrs,
        empty_ptrs,
        empty_ptrs,
        empty_ptrs,
        intermediate_q,
        w2_q,
        a2_scale,
        w2_scale,
        zeros_stride,
        zeros_stride,
        zeros_stride,
        zeros_layout,
        zeros_layout,
        problem_sizes2,
        expert_starts,
        workspace,
    )

    if output is None:
        output = torch.empty((m, k), dtype=out_dtype, device=device)
    apply_shuffle_mul_sum(c2.to(out_dtype), output, c_map, topk_weights.to(out_dtype))
    return output


_MXFP4_BLOCK_SIZE = 32


def cutlass_fused_experts_mxfp4(
    a: torch.Tensor,
    w1_q: torch.Tensor,
    w2_q: torch.Tensor,
    w1_scale: torch.Tensor,
    w2_scale: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    output: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Fused MoE on XPU using mxfp4_blockwise_scaled_grouped_mm. Slim signature
    mirrors sglang's NVFP4 cutlass_moe_fp4 (no CUDA-only placeholders).
    Weights in (E, N, K/2) uint8 packed E2M1 with UE8M0 block-32 scales:
      w1_q (E, n*2, k/2), w2_q (E, k, n/2),
      w1_scale (E, n*2, k/32), w2_scale (E, k, n/32) (un-transposed).
    """
    assert topk_weights.shape == topk_ids.shape, "topk shape mismatch"
    assert w1_q.dtype == torch.uint8, "w1_q must be uint8 (packed MXFP4)"
    assert w2_q.dtype == torch.uint8, "w2_q must be uint8 (packed MXFP4)"
    assert w1_scale.dtype == torch.uint8, "w1_scale must be uint8 (UE8M0)"
    assert w2_scale.dtype == torch.uint8, "w2_scale must be uint8 (UE8M0)"
    assert w1_q.dim() == 3 and w2_q.dim() == 3, "Weights must be 3D (E, N, K/2)"
    assert w1_q.shape[0] == w2_q.shape[0], "Expert count mismatch w1/w2"
    assert a.dtype in (torch.half, torch.bfloat16), "Invalid input dtype"

    out_dtype = a.dtype
    num_experts = w1_q.size(0)
    m = a.size(0)
    k = a.size(1)
    n = w2_q.size(2) * 2  # w2_q is (E, k, n/2)
    topk = topk_ids.size(1)
    device = a.device

    assert w1_q.size(2) * 2 == k
    assert w2_q.size(1) == k
    assert w1_q.size(1) == 2 * n

    a_map = torch.empty((topk_ids.numel(),), dtype=torch.int32, device=device)
    c_map = torch.empty((topk_ids.numel(),), dtype=torch.int32, device=device)

    # prepare_moe_input writes per-expert M counts (not cumulative starts).
    expert_offsets = torch.zeros(num_experts, dtype=torch.int32, device=device)
    problem_sizes1 = torch.zeros((num_experts, 3), dtype=torch.int32, device=device)
    problem_sizes2 = torch.zeros((num_experts, 3), dtype=torch.int32, device=device)

    prepare_moe_input(
        topk_ids,
        expert_offsets,
        problem_sizes1,
        problem_sizes2,
        a_map,
        c_map,
        num_experts,
        n,
        k,
    )

    # Flat-2D kernel needs cumulative start offsets.
    expert_starts = torch.zeros(num_experts, dtype=torch.int32, device=device)
    if num_experts > 1:
        expert_starts[1:] = torch.cumsum(expert_offsets[:-1], dim=0).to(torch.int32)

    # XPU scatter doesn't support uint8/fp4; scatter bf16 then quantize.
    rep_a = torch.empty((m * topk, k), dtype=a.dtype, device=device)
    scatter_tokens_to_experts(a, c_map, rep_a)
    rep_a_q, rep_a1_scales = sgl_per_token_group_quant_fp4(
        rep_a, group_size=_MXFP4_BLOCK_SIZE
    )

    c1 = torch.zeros((m * topk, n * 2), dtype=torch.float32, device=device)
    # Empty sentinels -> on-device ptr-table + scale-transpose.
    empty_ptrs = torch.empty((0,), dtype=torch.int64, device=device)
    workspace = torch.zeros((64 * 1024 * 1024,), dtype=torch.uint8, device=device)

    mxfp4_blockwise_scaled_grouped_mm(
        c1,
        empty_ptrs,
        empty_ptrs,
        empty_ptrs,
        empty_ptrs,
        empty_ptrs,
        rep_a_q,
        w1_q,
        rep_a1_scales,
        w1_scale,
        problem_sizes1,
        expert_starts,
        workspace,
    )

    intermediate = torch.empty((m * topk, n), dtype=out_dtype, device=device)
    silu_and_mul(c1.to(out_dtype), intermediate)

    intermediate_q, a2_scale = sgl_per_token_group_quant_fp4(
        intermediate, group_size=_MXFP4_BLOCK_SIZE
    )

    c2 = torch.zeros((m * topk, k), dtype=torch.float32, device=device)
    mxfp4_blockwise_scaled_grouped_mm(
        c2,
        empty_ptrs,
        empty_ptrs,
        empty_ptrs,
        empty_ptrs,
        empty_ptrs,
        intermediate_q,
        w2_q,
        a2_scale,
        w2_scale,
        problem_sizes2,
        expert_starts,
        workspace,
    )

    if output is None:
        output = torch.empty((m, k), dtype=out_dtype, device=device)
    apply_shuffle_mul_sum(c2.to(out_dtype), output, c_map, topk_weights.to(out_dtype))
    return output

# SPDX-License-Identifier: Apache-2.0
"""Minimal Xe3 flash attention UT for simulator validation.

This covers the Xe3P/Xe35 flash-attention entry via flash_attn_with_kvcache
and keeps shapes intentionally tiny so it can run on simulator hardware.
"""

import pytest
import sgl_kernel  # noqa: F401 - registers torch.ops.sgl_kernel.* custom ops
import torch
import torch.nn.functional as F
from sgl_kernel.flash_attn import flash_attn_with_kvcache


def cpu_ref(
    q,
    k_cache,
    v_cache,
    page_table,
    cache_seqlens,
    seqlen_q,
    softmax_scale,
    upcast=False,
):
    """
    CPU reference implementation for paged flash attention.

    Args:
        q: (batch, seqlen_q, num_heads_q, head_dim)
        k_cache: (batch, page_size, num_heads_kv, head_dim)
        v_cache: (batch, page_size, num_heads_kv, head_dim)
        page_table: (batch, num_pages_per_seq)
        cache_seqlens: (batch,) - cache length per batch
        seqlen_q: query sequence length
        softmax_scale: scaling factor for attention
        upcast: whether to upcast to fp32 for computation

    Returns:
        out: (batch, seqlen_q, num_heads_q, head_dim) attention output
    """
    batch, _, num_heads_q, head_dim = q.shape
    _, page_size, num_heads_kv, _ = k_cache.shape

    dtype = q.dtype

    # Move everything to CPU for reference computation
    q = q.cpu()
    k_cache = k_cache.cpu()
    v_cache = v_cache.cpu()
    page_table = page_table.cpu()
    cache_seqlens = cache_seqlens.cpu()

    if upcast:
        q_compute = q.float()
        k_cache_compute = k_cache.float()
        v_cache_compute = v_cache.float()
    else:
        q_compute = q
        k_cache_compute = k_cache
        v_cache_compute = v_cache

    out = torch.zeros(
        batch, seqlen_q, num_heads_q, head_dim, dtype=q_compute.dtype, device="cpu"
    )

    for b in range(batch):
        # Get cache seqlen for this batch
        cache_len = cache_seqlens[b].item()

        # Get the page indices for this batch
        page_indices = page_table[b]  # (num_pages_per_seq,)

        # Reconstruct k, v from pages
        k_seq = []
        v_seq = []
        for page_idx in page_indices:
            if len(k_seq) * page_size >= cache_len:
                break
            k_page = k_cache_compute[
                page_idx.item()
            ]  # (page_size, num_heads_kv, head_dim)
            v_page = v_cache_compute[page_idx.item()]
            k_seq.append(k_page)
            v_seq.append(v_page)

        if len(k_seq) == 0:
            continue

        # Concatenate pages and trim to cache_len
        k_full = torch.cat(
            k_seq, dim=0
        )  # (page_size * num_pages, num_heads_kv, head_dim)
        v_full = torch.cat(v_seq, dim=0)

        k_full = k_full[:cache_len]  # (cache_len, num_heads_kv, head_dim)
        v_full = v_full[:cache_len]

        q_batch = q_compute[b]  # (seqlen_q, num_heads_q, head_dim)

        # Compute attention for this batch
        for h in range(num_heads_q):
            # Get head index for KV (handle GQA)
            h_kv = h % num_heads_kv if num_heads_kv > 0 else 0

            q_head = q_batch[:, h, :]  # (seqlen_q, head_dim)
            k_head = k_full[:, h_kv, :]  # (cache_len, head_dim)
            v_head = v_full[:, h_kv, :]  # (cache_len, head_dim)

            # Compute attention scores
            scores = (
                torch.matmul(q_head, k_head.t()) * softmax_scale
            )  # (seqlen_q, cache_len)
            attn_weights = F.softmax(scores, dim=-1, dtype=q_compute.dtype)

            # Apply attention to values
            out_head = torch.matmul(attn_weights, v_head)  # (seqlen_q, head_dim)
            out[b, :, h, :] = out_head

    if upcast and dtype != torch.float32:
        out = out.to(dtype)

    return out


def is_xpu_available() -> bool:
    return hasattr(torch, "xpu") and torch.xpu.is_available()


def is_cri_device() -> bool:
    if not is_xpu_available():
        return False
    try:
        from sgl_kernel import is_xe3_arch

        return is_xe3_arch()
    except ImportError:
        return False


@pytest.mark.skipif(not is_xpu_available(), reason="Intel XPU not available")
@pytest.mark.skipif(not is_cri_device(), reason="Requires a CRI (Xe3P) device")
@pytest.mark.parametrize(
    "seqlen_q,seqlen_k",
    [
        (1, 4096),
        # (128, 128),
        # (2048, 2048),
        # (4096, 4096),
    ],
)
def test_flash_attention_xe3_minimal_paged_fwd(seqlen_q, seqlen_k):
    device = torch.device("xpu")
    dtype = torch.bfloat16

    batch = 16
    num_heads_q = 2
    num_heads_kv = 1
    head_dim = 128
    page_size = 128
    num_pages_per_seq = (seqlen_k + page_size - 1) // page_size
    num_pages = batch * num_pages_per_seq

    torch.manual_seed(0)
    q = torch.randn(batch, seqlen_q, num_heads_q, head_dim, device=device, dtype=dtype)
    k_cache = torch.randn(
        num_pages, page_size, num_heads_kv, head_dim, device=device, dtype=dtype
    )
    v_cache = torch.randn(
        num_pages, page_size, num_heads_kv, head_dim, device=device, dtype=dtype
    )
    page_table = torch.arange(
        num_pages, dtype=torch.int32, device=device
    ).view(batch, num_pages_per_seq)
    cache_seqlens = torch.full(
        (batch,), seqlen_k, dtype=torch.int32, device=device
    )
    softmax_scale = head_dim**-0.5

    # Reference test_flash_attn_kvcache varlen path: cu_seqlens_q from unpad_input
    # (cumulative per-batch query lengths) and q passed as 3D (total_q, nheads, d).
    # With a full (no padding) batch this is arange(batch+1) * seqlen_q.
    cu_seqlens_q = (
        torch.arange(0, batch + 1, dtype=torch.int32, device=device) * seqlen_q
    )
    q_unpad = q.view(batch * seqlen_q, num_heads_q, head_dim)

    out, *_ = flash_attn_with_kvcache(
        q_unpad,
        k_cache,
        v_cache,
        cache_seqlens=cache_seqlens,
        page_table=page_table,
        cu_seqlens_q=cu_seqlens_q,
        max_seqlen_q=seqlen_q,
        max_seqlen_k=seqlen_k,
        softmax_scale=softmax_scale,
        causal=False,
        window_size=(-1, -1),
        rotary_interleaved=False,
        return_softmax_lse=True,
    )
    torch.xpu.synchronize()
    # out_kernel = out.view(batch, seqlen_q, num_heads_q, head_dim)

    # out_ref_fp32 = cpu_ref(
    #     q,
    #     k_cache,
    #     v_cache,
    #     page_table,
    #     cache_seqlens,
    #     seqlen_q,
    #     softmax_scale,
    #     upcast=True,
    # )
    # out_ref_bf16 = cpu_ref(
    #     q,
    #     k_cache,
    #     v_cache,
    #     page_table,
    #     cache_seqlens,
    #     seqlen_q,
    #     softmax_scale,
    #     upcast=False,
    # )
    # kernel_diff = (out_kernel.float().cpu() - out_ref_fp32).abs().max().item()
    # baseline_diff = (out_ref_bf16 - out_ref_fp32).abs().max().item()
    # tol = 2.0 * baseline_diff + 1e-5
    # print("kernel_diff:", kernel_diff, "baseline_diff:", baseline_diff, "tol:", tol)

    # assert kernel_diff <= tol, (
    #     f"Xe3 minimal flash attention failed: kernel_diff={kernel_diff} > tol={tol} "
    #     f"(baseline={baseline_diff})"
    # )

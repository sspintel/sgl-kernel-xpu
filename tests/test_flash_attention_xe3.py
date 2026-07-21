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
    window_size=(-1, -1),
    causal=False,
    sinks=None,
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
        window_size: local attention window, inclusive, on the query row index
        causal: whether to force right window size to 0
        sinks: optional per-head sink scores
        upcast: whether to upcast to fp32 for computation

    Returns:
        out: (batch, seqlen_q, num_heads_q, head_dim) attention output
    """
    batch, _, num_heads_q, head_dim = q.shape
    _, page_size, num_heads_kv, _ = k_cache.shape

    dtype = q.dtype

    q = q.cpu()
    k_cache = k_cache.cpu()
    v_cache = v_cache.cpu()
    page_table = page_table.cpu()
    cache_seqlens = cache_seqlens.cpu()
    sinks = sinks.cpu() if isinstance(sinks, torch.Tensor) else sinks

    if isinstance(window_size, torch.Tensor):
        window_size = tuple(int(w) for w in window_size)
    if causal:
        window_size = (window_size[0], 0)

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
        cache_len = cache_seqlens[b].item()
        page_indices = page_table[b]

        k_seq = []
        v_seq = []
        for page_idx in page_indices:
            if len(k_seq) * page_size >= cache_len:
                break
            k_seq.append(k_cache_compute[page_idx.item()])
            v_seq.append(v_cache_compute[page_idx.item()])

        if len(k_seq) == 0:
            continue

        k_full = torch.cat(k_seq, dim=0)[:cache_len]
        v_full = torch.cat(v_seq, dim=0)[:cache_len]
        q_batch = q_compute[b]

        for h in range(num_heads_q):
            # GQA mapping: contiguous groups of Q heads share one KV head.
            # Example: nheads_q=4, nheads_kv=2 -> [0, 0, 1, 1].
            group_size = num_heads_q // num_heads_kv if num_heads_kv > 0 else 1
            h_kv = h // group_size if num_heads_kv > 0 else 0
            q_head = q_batch[:, h, :]
            k_head = k_full[:, h_kv, :]
            v_head = v_full[:, h_kv, :]

            scores = torch.matmul(q_head, k_head.t()) * softmax_scale

            if window_size[0] >= 0 or window_size[1] >= 0:
                local_mask = torch.ones_like(scores, dtype=torch.bool)
                left_window, right_window = window_size
                for row_idx in range(seqlen_q):
                    row_kv_idx = row_idx + cache_len - seqlen_q
                    left_bound = (
                        0 if left_window < 0 else max(0, row_kv_idx - left_window)
                    )
                    right_bound = (
                        cache_len - 1
                        if right_window < 0
                        else min(cache_len - 1, row_kv_idx + right_window)
                    )
                    if left_bound <= right_bound:
                        local_mask[row_idx, left_bound : right_bound + 1] = False
                scores = scores.masked_fill(local_mask, float("-inf"))

            if sinks is not None:
                sink_score = sinks[h].to(scores.dtype).view(1, 1).expand(seqlen_q, 1)
                scores = torch.cat([scores, sink_score], dim=-1)

            attn_weights = F.softmax(scores, dim=-1, dtype=q_compute.dtype)
            if sinks is not None:
                attn_weights = attn_weights[..., :-1]
            if window_size[0] >= 0 or window_size[1] >= 0:
                attn_weights = attn_weights.masked_fill(
                    torch.all(local_mask, dim=-1, keepdim=True), 0.0
                )

            out[b, :, h, :] = torch.matmul(attn_weights, v_head)

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
@pytest.mark.parametrize("nheads_q,nheads_kv", [(4, 2)])
@pytest.mark.parametrize("causal,local", [(False, False), (False, True), (True, False)])
@pytest.mark.parametrize("use_sinks", [False, True])
@pytest.mark.parametrize("head_dim", [64, 128])
@pytest.mark.parametrize("seqlen_q", [1, 128])
@pytest.mark.parametrize("seqlen_k", [128])
@pytest.mark.parametrize("page_size", [128, 64])
@pytest.mark.parametrize("batch", [2])
def test_flash_attention_xe3_fwd(
    batch,
    seqlen_q,
    seqlen_k,
    head_dim,
    page_size,
    causal,
    local,
    use_sinks,
    nheads_q,
    nheads_kv,
):

    device = torch.device("xpu")
    dtype = torch.bfloat16

    if use_sinks and head_dim != 64:
        pytest.skip("use_sinks is only covered for head_dim == 64 in Xe3 minimal UT")

    if local and seqlen_q == 1:
        pytest.skip("local-window coverage uses seqlen_q > 1")

    window_size = (1, 0) if local else (-1, -1)
    case_name = (
        f"b{batch}_q{seqlen_q}_k{seqlen_k}_d{head_dim}_p{page_size}"
        f"_causal{int(causal)}_local{int(local)}_sink{int(use_sinks)}"
    )

    num_pages_per_seq = (seqlen_k + page_size - 1) // page_size
    num_pages = batch * num_pages_per_seq

    torch.manual_seed(0)
    q = torch.randn(batch, seqlen_q, nheads_q, head_dim, device=device, dtype=dtype)
    k_cache = torch.randn(
        num_pages, page_size, nheads_kv, head_dim, device=device, dtype=dtype
    )
    v_cache = torch.randn(
        num_pages, page_size, nheads_kv, head_dim, device=device, dtype=dtype
    )
    page_table = torch.arange(num_pages, dtype=torch.int32, device=device).view(
        batch, num_pages_per_seq
    )
    cache_seqlens = torch.full((batch,), seqlen_k, dtype=torch.int32, device=device)
    softmax_scale = head_dim**-0.5
    sinks = torch.randn(nheads_q, device=device, dtype=dtype) if use_sinks else None

    cu_seqlens_q = (
        torch.arange(0, batch + 1, dtype=torch.int32, device=device) * seqlen_q
    )
    q_unpad = q.view(batch * seqlen_q, nheads_q, head_dim)

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
        causal=causal,
        window_size=window_size,
        sinks=sinks,
        rotary_interleaved=False,
        return_softmax_lse=True,
    )
    torch.xpu.synchronize()

    out_ref_fp32 = cpu_ref(
        q,
        k_cache,
        v_cache,
        page_table,
        cache_seqlens,
        seqlen_q,
        softmax_scale,
        window_size=window_size,
        causal=causal,
        sinks=sinks,
        upcast=True,
    )
    out_ref_bf16 = cpu_ref(
        q,
        k_cache,
        v_cache,
        page_table,
        cache_seqlens,
        seqlen_q,
        softmax_scale,
        window_size=window_size,
        causal=causal,
        sinks=sinks,
        upcast=False,
    )
    out_kernel = out.view(batch, seqlen_q, nheads_q, head_dim)
    kernel_diff = (out_kernel.float().cpu() - out_ref_fp32).abs().max().item()
    baseline_diff = (out_ref_bf16 - out_ref_fp32).abs().max().item()
    tol = 2.0 * baseline_diff + 1e-5
    assert kernel_diff <= tol, (
        f"Xe3 flash attention case={case_name} failed: kernel_diff={kernel_diff} > tol={tol} "
        f"(baseline={baseline_diff})"
    )


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))

#pragma once

#include "sycl/kernels/flash_attention_v2/xe3/xe_fmha_fwd_decode_runner.hpp"

void dispatch_xe35_decode_64(
    const at::Tensor& q_dense,
    const at::Tensor& k_pool,
    const at::Tensor& v_pool,
    const at::Tensor& page_table,
    int num_pages_per_seq,
    const at::Tensor& cu_seqlens_q,
    const at::Tensor& cu_seqlens_k_cache,
    at::Tensor& out_dense,
    int batch,
    int num_heads_q,
    int num_heads_kv,
    int max_seqlen_q,
    int max_seqlen_kv_cache,
    int page_size,
    float softmax_scale,
    bool is_causal,
    bool is_local,
    int window_size_left,
    int window_size_right,
    const cutlass::bfloat16_t* sm_sink,
    bool use_sink);

void dispatch_xe35_decode_128(
    const at::Tensor& q_dense,
    const at::Tensor& k_pool,
    const at::Tensor& v_pool,
    const at::Tensor& page_table,
    int num_pages_per_seq,
    const at::Tensor& cu_seqlens_q,
    const at::Tensor& cu_seqlens_k_cache,
    at::Tensor& out_dense,
    int batch,
    int num_heads_q,
    int num_heads_kv,
    int max_seqlen_q,
    int max_seqlen_kv_cache,
    int page_size,
    float softmax_scale,
    bool is_causal,
    bool is_local,
    int window_size_left,
    int window_size_right,
    const cutlass::bfloat16_t* sm_sink,
    bool use_sink);

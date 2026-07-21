#include "flash_attentionXe35_common.hpp"
#include "kernels/flash_attention_v2/xe3/xe_fmha_fwd_decode_dispatch.hpp"
#include "kernels/flash_attention_v2/xe3/xe_fmha_fwd_prefill_dispatch.hpp"

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor> mha_fwd(
    const at::Tensor& q,
    const at::Tensor& k,
    const at::Tensor& v,
    std::optional<const at::Tensor>& q_v_,
    const at::Tensor& cu_seqlens_q,
    const at::Tensor& cu_seqlens_k,
    int max_seqlen_q,
    int max_seqlen_k,
    std::optional<const at::Tensor>& page_table,
    std::optional<const at::Tensor>& kv_batch_idx_,
    std::optional<const at::Tensor>& leftpad_k_,
    std::optional<const at::Tensor>& rotary_cos_,
    std::optional<const at::Tensor>& rotary_sin_,
    std::optional<const at::Tensor>& seqlens_rotary_,
    std::optional<at::Tensor>& q_descale_,
    std::optional<at::Tensor>& k_descale_,
    std::optional<at::Tensor>& v_descale_,
    const float softmax_scale_,
    std::optional<const at::Tensor>& sinks_,
    bool is_causal,
    int window_size_left,
    int window_size_right,
    float const softcap,
    bool const is_rotary_interleaved,
    std::optional<at::Tensor>& scheduler_metadata_,
    int num_kv_splits,
    std::optional<bool> pack_gqa_,
    int const sm_margin,
    std::optional<at::Tensor>& out_) {
  const int batch = static_cast<int>(cu_seqlens_q.size(0) - 1);
  TORCH_CHECK(batch > 0, "flash_attentionXe35: empty batch is not supported");
  TORCH_CHECK(page_table.has_value(), "flash_attentionXe35: page_table is required");
  auto& page_table_ref = page_table.value();

  TORCH_CHECK(q.dim() == 3, "flash_attentionXe35: q must be (total_q, h, d)");
  TORCH_CHECK(k.dim() == 4 && v.dim() == 4, "flash_attentionXe35: k/v must be paged (num_pages, page_size, h_kv, d)");
  TORCH_CHECK(page_table_ref.dim() == 2, "flash_attentionXe35: page_table must be 2D");
  TORCH_CHECK(cu_seqlens_q.dim() == 1, "flash_attentionXe35: cu_seqlens_q must be 1D");
  TORCH_CHECK(cu_seqlens_k.dim() == 1, "flash_attentionXe35: cache_seqlens must be 1D");
  TORCH_CHECK(cu_seqlens_k.size(0) == batch, "flash_attentionXe35: cache_seqlens must have size equal to batch");
  TORCH_CHECK(!seqlens_rotary_.has_value(), "flash_attentionXe35: seqlens_rotary is not supported");
  TORCH_CHECK(
      !q_descale_.has_value() && !k_descale_.has_value() && !v_descale_.has_value(),
      "flash_attentionXe35: descale is not supported");
  TORCH_CHECK(softcap == 0.0f, "flash_attentionXe35: softcap is not supported");
  const bool has_rotary = rotary_cos_.has_value() || rotary_sin_.has_value();
  TORCH_CHECK(!(has_rotary && is_rotary_interleaved), "flash_attentionXe35: rotary interleaving is not supported");
  auto q_dtype = q.scalar_type();
  TORCH_CHECK(q_dtype == at::ScalarType::BFloat16, "flash_attentionXe35: only bf16 is supported");
  TORCH_CHECK(k.scalar_type() == q_dtype && v.scalar_type() == q_dtype, "flash_attentionXe35: q/k/v dtype mismatch");
  TORCH_CHECK(q.is_xpu() && k.is_xpu() && v.is_xpu(), "flash_attentionXe35: q/k/v must be xpu tensors");
  TORCH_CHECK(
      page_table_ref.is_xpu() && cu_seqlens_q.is_xpu() && cu_seqlens_k.is_xpu(),
      "flash_attentionXe35: page_table/cu_seqlens must be xpu tensors");
  TORCH_CHECK(cu_seqlens_q.scalar_type() == at::ScalarType::Int, "flash_attentionXe35: cu_seqlens_q must be int32");
  TORCH_CHECK(cu_seqlens_k.scalar_type() == at::ScalarType::Int, "flash_attentionXe35: cache_seqlens must be int32");
  TORCH_CHECK(page_table_ref.scalar_type() == at::ScalarType::Int, "flash_attentionXe35: page_table must be int32");

  const int total_q = static_cast<int>(q.size(0));
  const int num_heads_q = static_cast<int>(q.size(1));
  const int head_dim = static_cast<int>(q.size(2));
  const int num_pages = static_cast<int>(k.size(0));
  const int page_size = static_cast<int>(k.size(1));
  const int num_heads_kv = static_cast<int>(k.size(2));
  const int num_pages_per_seq = static_cast<int>(page_table_ref.size(1));

  TORCH_CHECK(page_size == 64 || page_size == 128, "flash_attentionXe35: only page_size 64 and 128 are supported");

  if (is_causal) {
    window_size_right = 0;
  }
  const bool is_local =
      (window_size_left >= 0 || window_size_right >= 0) && !(window_size_left < 0 && window_size_right == 0);
  const bool effective_causal = window_size_left < 0 && window_size_right == 0;
  if (window_size_left < 0) {
    window_size_left = max_seqlen_k - 1;
  }
  if (window_size_right < 0) {
    window_size_right = max_seqlen_q - 1;
  }

  TORCH_CHECK(
      q.stride(-1) == 1 && k.stride(-1) == 1 && v.stride(-1) == 1,
      "flash_attentionXe35: last dimension must be contiguous");
  TORCH_CHECK(q.stride(1) == head_dim, "flash_attentionXe35: q must be contiguous on [total_q, h, d]");
  TORCH_CHECK(cu_seqlens_q.stride(0) == 1, "flash_attentionXe35: cu_seqlens_q must be contiguous");
  TORCH_CHECK(cu_seqlens_k.stride(0) == 1, "flash_attentionXe35: cache_seqlens must be contiguous");
  TORCH_CHECK(page_table_ref.is_contiguous(), "flash_attentionXe35: page_table must be contiguous");
  TORCH_CHECK(k.size(3) == head_dim && v.size(3) == head_dim, "flash_attentionXe35: head_dim mismatch between q/k/v");
  TORCH_CHECK(
      k.size(1) == v.size(1) && k.size(2) == v.size(2), "flash_attentionXe35: k/v shape mismatch on page/head axes");
  TORCH_CHECK(v.size(1) == page_size && v.size(2) == num_heads_kv, "flash_attentionXe35: k/v shape mismatch");

  const cutlass::bfloat16_t* sm_sink = nullptr;
  if (sinks_.has_value()) {
    const auto& sinks = sinks_.value();
    TORCH_CHECK(head_dim == 64, "flash_attentionXe35: sinks is only supported for head_dim 64");
    TORCH_CHECK(sinks.is_xpu() && sinks.device() == q.device(), "flash_attentionXe35: sinks must be on q's xpu device");
    TORCH_CHECK(sinks.scalar_type() == q_dtype, "flash_attentionXe35: sinks dtype must match q");
    TORCH_CHECK(
        sinks.dim() == 1 && sinks.size(0) == num_heads_q && sinks.stride(0) == 1,
        "flash_attentionXe35: sinks must be contiguous with shape [num_heads_q]");
    sm_sink = static_cast<const cutlass::bfloat16_t*>(sinks.data_ptr());
  }

  TORCH_CHECK(head_dim == 64 || head_dim == 128, "flash_attentionXe35: only head_dim 64 and 128 are supported");

  c10::DeviceGuard device_guard(q.device());
  auto opts = q.options();
  auto out_dense = out_.has_value() ? out_.value() : at::empty({total_q, num_heads_q, head_dim}, opts);
  auto k_pool = k.as_strided(
      {static_cast<int64_t>(num_pages) * page_size, static_cast<int64_t>(num_heads_kv), static_cast<int64_t>(head_dim)},
      {k.stride(1), k.stride(2), k.stride(3)});
  auto v_pool = v.as_strided(
      {static_cast<int64_t>(num_pages) * page_size, static_cast<int64_t>(num_heads_kv), static_cast<int64_t>(head_dim)},
      {v.stride(1), v.stride(2), v.stride(3)});

  auto perf_stream = c10::xpu::getCurrentXPUStream();
  perf_stream.synchronize();
  GPU_Clock timer;
  timer.start();

  if (max_seqlen_q == 1) {
    if (head_dim == 64) {
      dispatch_xe35_decode_64(
          q,
          k_pool,
          v_pool,
          page_table_ref,
          num_pages_per_seq,
          cu_seqlens_q,
          cu_seqlens_k,
          out_dense,
          batch,
          num_heads_q,
          num_heads_kv,
          max_seqlen_q,
          max_seqlen_k,
          page_size,
          softmax_scale_,
          effective_causal,
          is_local,
          window_size_left,
          window_size_right,
          sm_sink,
          sinks_.has_value());
    } else {
      dispatch_xe35_decode_128(
          q,
          k_pool,
          v_pool,
          page_table_ref,
          num_pages_per_seq,
          cu_seqlens_q,
          cu_seqlens_k,
          out_dense,
          batch,
          num_heads_q,
          num_heads_kv,
          max_seqlen_q,
          max_seqlen_k,
          page_size,
          softmax_scale_,
          effective_causal,
          is_local,
          window_size_left,
          window_size_right,
          sm_sink,
          sinks_.has_value());
    }
  } else {
    if (head_dim == 64) {
      dispatch_xe35_prefill_64(
          q,
          k_pool,
          v_pool,
          page_table_ref,
          num_pages_per_seq,
          cu_seqlens_q,
          cu_seqlens_k,
          out_dense,
          batch,
          num_heads_q,
          num_heads_kv,
          max_seqlen_q,
          max_seqlen_k,
          page_size,
          softmax_scale_,
          effective_causal,
          is_local,
          window_size_left,
          window_size_right,
          sm_sink,
          sinks_.has_value());
    } else {
      dispatch_xe35_prefill_128(
          q,
          k_pool,
          v_pool,
          page_table_ref,
          num_pages_per_seq,
          cu_seqlens_q,
          cu_seqlens_k,
          out_dense,
          batch,
          num_heads_q,
          num_heads_kv,
          max_seqlen_q,
          max_seqlen_k,
          page_size,
          softmax_scale_,
          effective_causal,
          is_local,
          window_size_left,
          window_size_right,
          sm_sink,
          sinks_.has_value());
    }
  }

  xe35_fmha::profiling_queue().wait();
  const double elapsed_s = timer.seconds();

  const double flops_qk = 2.0 * static_cast<double>(num_heads_q) * static_cast<double>(total_q) *
                          static_cast<double>(max_seqlen_k) * static_cast<double>(head_dim);
  const double flops_pv = flops_qk;
  const double tflops = elapsed_s > 0.0 ? ((flops_qk + flops_pv) * 1e-12) / elapsed_s : 0.0;
  const double bytes_qk = static_cast<double>(num_heads_q) * static_cast<double>(total_q) *
                              static_cast<double>(head_dim) * static_cast<double>(q.element_size()) +
                          static_cast<double>(num_heads_kv) * static_cast<double>(batch) *
                              static_cast<double>(max_seqlen_k) * static_cast<double>(head_dim) *
                              static_cast<double>(k.element_size());
  const double bytes_pv = static_cast<double>(num_heads_kv) * static_cast<double>(batch) *
                              static_cast<double>(max_seqlen_k) * static_cast<double>(head_dim) *
                              static_cast<double>(v.element_size()) +
                          static_cast<double>(num_heads_q) * static_cast<double>(total_q) *
                              static_cast<double>(head_dim) * static_cast<double>(out_dense.element_size());
  const double gbps = elapsed_s > 0.0 ? ((bytes_qk + bytes_pv) * 1e-9) / elapsed_s : 0.0;
  const double mbps = gbps * 1e3;
  const double gflops = tflops * 1e3;
  const double elapsed_ms = elapsed_s * 1000.0;
  const double elapsed_us = elapsed_s * 1e6;

  ::printf(
      "flash_attentionXe35 perf(gpu_clock): time=%.9f ms (%.3f us), bandwidth=%.6f GB/s (%.3f MB/s), "
      "compute=%.6f TFLOPS (%.3f GFLOPS)\n",
      elapsed_ms,
      elapsed_us,
      gbps,
      mbps,
      tflops,
      gflops);

  auto lse = at::zeros({num_heads_q, total_q}, opts.dtype(at::kFloat));
  auto out_accum = at::Tensor();
  auto lse_accum = at::Tensor();
  return {out_dense, lse, out_accum, lse_accum};
}

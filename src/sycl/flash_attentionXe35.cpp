#include <ATen/ATen.h>
#include <c10/core/DeviceGuard.h>
#include <c10/xpu/XPUStream.h>
#include <torch/all.h>

#include <cstdio>
#include <cute/atom/mma_atom.hpp>
#include <cute/tensor.hpp>
#include <cutlass/kernel_hardware_info.hpp>
#include <cutlass/util/GPU_Clock.hpp>
#include <optional>
#include <vector>

#include "comm/common.h"
#if defined(CUTLASS_SYCL_PROFILING_ENABLED)
#include <cutlass/util/sycl_event_manager.hpp>
#endif
#include "kernels/flash_attention_v2/xe3/collective/xe_fmha_fwd_epilogue.hpp"
#include "kernels/flash_attention_v2/xe3/collective/xe_fmha_fwd_mainloop.hpp"
#include "kernels/flash_attention_v2/xe3/kernel/xe_fmha_fwd_kernel.hpp"
#include "kernels/flash_attention_v2/xe3/kernel/xe_tile_scheduler.hpp"

#define SYCL_INTEL_TARGET 35

using namespace cute;

namespace {
// Profiling-enabled SYCL queue that shares the PyTorch XPU stream's context and
// device. Sharing the context is mandatory: the Q/K/V/O tensors are USM device
// allocations owned by PyTorch's SYCL context, so launching the kernel on a
// foreign context (e.g. compat's default queue) makes those pointers invalid
// and hangs the simulator. This queue enables device event profiling so
// cutlass GPU_Clock/EventManager can read accurate command_start/command_end
// timestamps (the same path the reference xe_fmha_fwd_runner uses).
sycl::queue& xe35_profiling_queue() {
  static sycl::queue queue = [] {
    sycl::queue torch_queue = at::xpu::getCurrentXPUStream().queue();
    return sycl::queue(
        torch_queue.get_context(), torch_queue.get_device(), sycl::property::queue::enable_profiling{});
  }();
  return queue;
}
}  // namespace

template <class Element>
struct Xe3FmhaConfig {
  using ElementInput = Element;
  using ElementOutput = Element;
  using ElementScale = float;

  using StrideQ = Stride<int, _1, int, int>;
  using StrideK = Stride<int, _1, int, int>;
  using StrideV = Stride<_1, int, int, int>;
  using StrideO = Stride<int, _1, int, int>;
  using StrideScale = Stride<_1, int, int, int>;

  template <typename T, typename S>
  static auto make_dummy_tensor(T val, S stride) {
    return make_tensor(make_gmem_ptr(&val), make_layout(repeat<rank_v<S>>(1), stride));
  }

  template <
      int PipelineStages,
      bool GqaFusion,
      class TileShapeQK,
      class TileShapePV,
      class TileShapeOut,
      class SubgroupLayoutQK>
  static void
  run(const at::Tensor& q_dense,
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
      int head_dim,
      int page_size,
      float softmax_scale) {
    static constexpr bool Causal = false;
    static constexpr bool BlockScale = false;
    static constexpr bool F8kvF16mma = false;
    static constexpr bool PerTensorScale = false;
    static constexpr bool CachedKV = true;
    static constexpr bool PagedKV = true;

    constexpr int SGTileQ = get<0>(shape_div(TileShapeQK{}, shape(SubgroupLayoutQK{})))();
    using MMAOperation = XE_DPAS_TT<cute::gcd(SGTileQ, 8), float, ElementInput>;
    using MMAOperationPV = XE_DPAS_TT<cute::gcd(SGTileQ, 8), float, ElementInput>;
    using SubgroupLayoutPV = decltype(cutlass::fmha::collective::get_sg_layout_pv(SubgroupLayoutQK{}));

    using TiledMMAQK = typename TiledMMAHelper<MMA_Atom<MMAOperation>, Layout<TileShapeQK>, SubgroupLayoutQK>::TiledMMA;
    using TiledMMAPV =
        typename TiledMMAHelper<MMA_Atom<MMAOperationPV>, Layout<TileShapePV>, SubgroupLayoutPV>::TiledMMA;

    static_assert(
        get<0>(TileShapeOut{}) == get<0>(TileShapePV{}), "Output tile and P*V tile must match on Q dimension");
    static constexpr int VTiles = get<1>(TileShapeOut{}) / get<1>(TileShapePV{});

    using TensorQ = decltype(make_dummy_tensor(ElementInput{}, StrideQ{}));
    using TensorK = decltype(make_dummy_tensor(ElementInput{}, StrideK{}));
    using TensorV = decltype(make_dummy_tensor(ElementInput{}, StrideV{}));
    using TensorO = decltype(make_dummy_tensor(ElementOutput{}, StrideO{}));
    using TensorScaleQ = decltype(make_dummy_tensor(ElementScale{}, StrideScale{}));
    using TensorScaleK = decltype(make_dummy_tensor(ElementScale{}, StrideScale{}));
    using TensorScaleV = decltype(make_dummy_tensor(ElementScale{}, StrideScale{}));

    using MainloopDispatchPolicy = cutlass::fmha::XeDefault<PipelineStages>;
    using CollectiveMainloop = cutlass::fmha::collective::FMHAFwdMainloop<
        MainloopDispatchPolicy,
        Causal,
        BlockScale,
        F8kvF16mma,
        PerTensorScale,
        CachedKV,
        PagedKV,
        TiledMMAQK,
        TiledMMAPV,
        VTiles,
        TensorQ,
        TensorK,
        TensorV,
        TensorScaleQ,
        TensorScaleK,
        TensorScaleV,
        TensorK,
        TensorV,
        void,
        void,
        void,
        void,
        void>;

    using CollectiveEpilogue =
        cutlass::fmha::collective::FMHAFwdEpilogue<CollectiveMainloop, TileShapeOut, TensorO, void>;
    using ProblemShape = cutlass::fmha::kernel::FMHAProblemShape<true>;
    using Scheduler = cutlass::fmha::kernel::XeFHMAIndividualTileScheduler<false, false, false, GqaFusion>;
    using FMHAKernel =
        cutlass::fmha::kernel::XeFMHAFwdKernel<ProblemShape, CollectiveMainloop, CollectiveEpilogue, Scheduler>;

    const int q_seq_stride = static_cast<int>(q_dense.stride(0));
    const int q_head_stride = static_cast<int>(q_dense.stride(1));
    const int kv_seq_stride = static_cast<int>(k_pool.stride(0));
    const int kv_head_stride = static_cast<int>(k_pool.stride(1));
    const int o_seq_stride = static_cast<int>(out_dense.stride(0));
    const int o_head_stride = static_cast<int>(out_dense.stride(1));
    typename FMHAKernel::StrideQ stride_q{q_seq_stride, {}, q_head_stride, 0};
    typename FMHAKernel::StrideK stride_k{kv_seq_stride, {}, kv_head_stride, 0};
    typename FMHAKernel::StrideV stride_v{{}, kv_seq_stride, kv_head_stride, 0};
    typename FMHAKernel::StrideO stride_o{o_seq_stride, {}, o_head_stride, 0};

    cutlass::KernelHardwareInfo hw_info;
    hw_info.sm_count = cutlass::KernelHardwareInfo::query_device_multiprocessor_count(hw_info.device_id);

    typename FMHAKernel::Arguments arguments{
        {
            {batch,
             num_heads_q,
             num_heads_kv,
             {max_seqlen_q, static_cast<int>(q_dense.size(0)), cu_seqlens_q.data_ptr<int>(), nullptr},
             {0, 0, nullptr, nullptr},
             {max_seqlen_kv_cache, static_cast<int>(k_pool.size(0)), cu_seqlens_k_cache.data_ptr<int>(), nullptr},
             head_dim,
             head_dim},
            static_cast<const ElementInput*>(q_dense.data_ptr()),
            stride_q,
            static_cast<const ElementInput*>(k_pool.data_ptr()),
            stride_k,
            static_cast<const ElementInput*>(v_pool.data_ptr()),
            stride_v,
            static_cast<ElementOutput*>(out_dense.data_ptr()),
            stride_o,
            nullptr,
            {},
            nullptr,
            {},
            nullptr,
            {},
            1.0f,
            1.0f,
            1.0f,
            32,
            static_cast<const ElementInput*>(k_pool.data_ptr()),
            stride_k,
            static_cast<const ElementInput*>(v_pool.data_ptr()),
            stride_v,
        },
        {
            softmax_scale,
            static_cast<const int*>(page_table.data_ptr()),
            page_size,
            num_pages_per_seq,
            static_cast<int>(k_pool.size(0)),
        },
        {},
        hw_info,
    };

    TORCH_CHECK(FMHAKernel::can_implement(arguments), "flash_attentionXe35: invalid cutlass FMHA arguments");

    size_t workspace_size = FMHAKernel::get_workspace_size(arguments);
    at::Tensor workspace = at::empty({static_cast<long>(workspace_size)}, q_dense.options().dtype(at::kByte));

    auto* workspace_ptr = workspace_size > 0 ? workspace.data_ptr() : nullptr;
    auto status = FMHAKernel::initialize_workspace(arguments, workspace_ptr);
    TORCH_CHECK(status == cutlass::Status::kSuccess, "flash_attentionXe35: FMHA workspace initialization failed");

    auto params = FMHAKernel::to_underlying_arguments(arguments, workspace_ptr);

    // Launch on the profiling-enabled queue that shares PyTorch's SYCL context.
    // Mirror the cutlass xe_fmha_fwd_runner launch so GPU_Clock/EventManager can
    // read device timestamps; passing our own queue (instead of compat's default
    // queue) keeps the USM tensors valid and avoids the simulator hang. Xe35 uses
    // a GRF size of 512 to match the reference runner (SYCL_INTEL_TARGET == 35).
    namespace syclex = sycl::ext::oneapi::experimental;
    namespace intelex = sycl::ext::intel::experimental;
    compat::dim3 const block = FMHAKernel::get_block_shape();
    compat::dim3 const grid = FMHAKernel::get_grid_shape(params);
    const int smem_size = FMHAKernel::SharedStorageSize;
    compat::experimental::launch_properties launch_props{
        syclex::work_group_scratch_size(smem_size),
    };
    compat::experimental::kernel_properties kernel_props{
        syclex::sub_group_size<cute::intel::sg_size>, intelex::grf_size<512>};
    compat::experimental::launch_policy policy{
        compat::dim3(grid.x, grid.y, grid.z), compat::dim3(block.x, block.y, block.z), launch_props, kernel_props};
    sycl::queue& profiling_queue = xe35_profiling_queue();
#if defined(CUTLASS_SYCL_PROFILING_ENABLED)
    auto event =
        compat::experimental::launch<cutlass::device_kernel<FMHAKernel>, FMHAKernel>(policy, profiling_queue, params);
    EventManager::getInstance().addEvent(event);
#else
    compat::experimental::launch<cutlass::device_kernel<FMHAKernel>, FMHAKernel, false>(
        policy, profiling_queue, params);
#endif
  }
};

template <typename Element>
void dispatch_xe3_kernel(
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
    int head_dim,
    int page_size,
    float softmax_scale) {
  // GQA fusion packs all gqa_group query heads that share a KV head into the
  // Q-tile dimension. The fused Q view (see xe_fmha_fwd_kernel.hpp) walks rows by
  // stride<0> (the sequence stride), i.e. it folds the group heads into the
  // sequence axis. With the current NHD Q layout [total_q, num_heads_q, head_dim]
  // that fold is only consistent when there is a single query position per head,
  // so GQA fusion is restricted to max_seqlen_q == 1. In that case total_rows =
  // gqa_group and the decode Q tile is selected from total_rows (see reference
  // 06_xe_fmha_fwd.cpp DECODE path). All other shapes use the non-fused prefill
  // config with two pipeline stages.
  const int gqa_group = num_heads_kv > 0 ? num_heads_q / num_heads_kv : 1;
  const int total_rows = gqa_group * max_seqlen_q;
  const bool use_decode = (max_seqlen_q == 1);

#define XE3_RUN(STAGES, GQA, QK, PV, OUT, SGL)                          \
  Xe3FmhaConfig<Element>::template run<STAGES, GQA, QK, PV, OUT, SGL>(  \
      q_dense,                                                          \
      k_pool,                                                           \
      v_pool,                                                           \
      page_table,                                                       \
      num_pages_per_seq,                                                \
      cu_seqlens_q,                                                     \
      cu_seqlens_k_cache,                                               \
      out_dense,                                                        \
      batch,                                                            \
      num_heads_q,                                                      \
      num_heads_kv,                                                     \
      max_seqlen_q,                                                     \
      max_seqlen_kv_cache,                                              \
      head_dim,                                                         \
      page_size,                                                        \
      softmax_scale)

  if (head_dim == 128) {
    if (use_decode) {
      using ShapeQK8 = Shape<_8, _256, _64>;
      using ShapePV8 = Shape<_8, _32, _256>;
      using ShapeOut8 = Shape<_8, _128>;
      using SubgroupLayoutQK8 = Layout<Shape<_1, _8, _1>>;

      using ShapeQK16 = Shape<_16, _256, _64>;
      using ShapePV16 = Shape<_16, _32, _256>;
      using ShapeOut16 = Shape<_16, _128>;
      using SubgroupLayoutQK16 = Layout<Shape<_2, _8, _1>>;

      using ShapeQK32 = Shape<_32, _256, _64>;
      using ShapePV32 = Shape<_32, _32, _256>;
      using ShapeOut32 = Shape<_32, _128>;
      using SubgroupLayoutQK32 = Layout<Shape<_4, _8, _1>>;

      using ShapeQK64 = Shape<_64, _256, _64>;
      using ShapePV64 = Shape<_64, _32, _256>;
      using ShapeOut64 = Shape<_64, _128>;
      using SubgroupLayoutQK64 = Layout<Shape<_4, _8, _1>>;

      if (total_rows <= 8) {
        XE3_RUN(1, true, ShapeQK8, ShapePV8, ShapeOut8, SubgroupLayoutQK8);
      } else if (total_rows <= 16) {
        XE3_RUN(1, true, ShapeQK16, ShapePV16, ShapeOut16, SubgroupLayoutQK16);
      } else if (total_rows <= 32) {
        XE3_RUN(1, true, ShapeQK32, ShapePV32, ShapeOut32, SubgroupLayoutQK32);
      } else {
        XE3_RUN(1, true, ShapeQK64, ShapePV64, ShapeOut64, SubgroupLayoutQK64);
      }
      return;
    }

    using ShapeQK = Shape<_128, _64, _64>;
    using ShapePV = Shape<_128, _64, _64>;
    using ShapeOut = Shape<_128, _128>;
    using SubgroupLayoutQK = Layout<Shape<_8, _1, _1>>;
    XE3_RUN(2, false, ShapeQK, ShapePV, ShapeOut, SubgroupLayoutQK);
    return;
  }

  if (head_dim == 64) {
    if (use_decode) {
      using ShapeQK8 = Shape<_8, _256, _64>;
      using ShapePV8 = Shape<_8, _32, _256>;
      using ShapeOut8 = Shape<_8, _64>;
      using SubgroupLayoutQK8 = Layout<Shape<_1, _8, _1>>;

      using ShapeQK16 = Shape<_16, _256, _64>;
      using ShapePV16 = Shape<_16, _32, _256>;
      using ShapeOut16 = Shape<_16, _64>;
      using SubgroupLayoutQK16 = Layout<Shape<_2, _8, _1>>;

      using ShapeQK32 = Shape<_32, _256, _64>;
      using ShapePV32 = Shape<_32, _32, _256>;
      using ShapeOut32 = Shape<_32, _64>;
      using SubgroupLayoutQK32 = Layout<Shape<_4, _8, _1>>;

      using ShapeQK64 = Shape<_64, _256, _64>;
      using ShapePV64 = Shape<_64, _32, _256>;
      using ShapeOut64 = Shape<_64, _64>;
      using SubgroupLayoutQK64 = Layout<Shape<_4, _8, _1>>;

      if (total_rows <= 8) {
        XE3_RUN(1, true, ShapeQK8, ShapePV8, ShapeOut8, SubgroupLayoutQK8);
      } else if (total_rows <= 16) {
        XE3_RUN(1, true, ShapeQK16, ShapePV16, ShapeOut16, SubgroupLayoutQK16);
      } else if (total_rows <= 32) {
        XE3_RUN(1, true, ShapeQK32, ShapePV32, ShapeOut32, SubgroupLayoutQK32);
      } else {
        XE3_RUN(1, true, ShapeQK64, ShapePV64, ShapeOut64, SubgroupLayoutQK64);
      }
      return;
    }

    using ShapeQK = Shape<_128, _64, _32>;
    using ShapePV = Shape<_128, _32, _64>;
    using ShapeOut = Shape<_128, _64>;
    using SubgroupLayoutQK = Layout<Shape<_8, _1, _1>>;
    XE3_RUN(2, false, ShapeQK, ShapePV, ShapeOut, SubgroupLayoutQK);
    return;
  }

#undef XE3_RUN

  TORCH_CHECK(false, "flash_attentionXe35: only head_dim 64 and 128 are supported in Xe3 cutlass path");
}

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor> mha_fwd(
    const at::Tensor& q,  // (b, s_q, h, d) or (total_q, h, d) if there is cu_seqlens_q
    const at::Tensor& k,  // (b_k, s_k, h_k, d) or (total_k, h_k, d) if there is cu_seqlens_k or (num_pages, page_size,
                          // h_k, d) if there is page_table.
    const at::Tensor& v,  // (b_k, s_k, h_k, dv) or (total_k, h_k, dv) if there is cu_seqlens_k or (num_pages,
                          // page_size, h_k, dv) if there is page_table.
    std::optional<const at::Tensor>& q_v_,  // (b, s_q, h, dv) or (total_q_new, h, dv) if there is cu_seqlens_q
    const at::Tensor& cu_seqlens_q,         // b+1
    const at::Tensor& cu_seqlens_k,         // b+1
    int max_seqlen_q,
    int max_seqlen_k,
    std::optional<const at::Tensor>& page_table,       // (b_k, max_num_pages_per_seq)
    std::optional<const at::Tensor>& kv_batch_idx_,    // b. indices to index into the KV cache
    std::optional<const at::Tensor>& leftpad_k_,       // b
    std::optional<const at::Tensor>& rotary_cos_,      // seqlen_ro x (rotary_dim / 2)
    std::optional<const at::Tensor>& rotary_sin_,      // seqlen_ro x (rotary_dim / 2)
    std::optional<const at::Tensor>& seqlens_rotary_,  // b
    std::optional<at::Tensor>& q_descale_,             // (b, h_k), not (b, h)
    std::optional<at::Tensor>& k_descale_,             // (b, h_k)
    std::optional<at::Tensor>& v_descale_,             // (b, h_k)
    const float softmax_scale_,
    std::optional<const at::Tensor>& sinks_,
    bool is_causal,
    int window_size_left,
    int window_size_right,
    float const softcap,
    bool const is_rotary_interleaved,  // if true, rotary combines indices 0 & 1, else indices 0 & rotary_dim / 2
    std::optional<at::Tensor>& scheduler_metadata_,  // (b + 1)
    int num_kv_splits,
    std::optional<bool> pack_gqa_,
    int const sm_margin,
    std::optional<at::Tensor>& out_) {
  const int batch = static_cast<int>(cu_seqlens_q.size(0) - 1);
  TORCH_CHECK(batch > 0, "flash_attentionXe35: empty batch is not supported");

  // For Xe35, we only support paged KV cache
  TORCH_CHECK(page_table.has_value(), "flash_attentionXe35: page_table is required");
  auto& page_table_ref = page_table.value();

  TORCH_CHECK(q.dim() == 3, "flash_attentionXe35: q must be (total_q, h, d)");
  TORCH_CHECK(k.dim() == 4 && v.dim() == 4, "flash_attentionXe35: k/v must be paged (num_pages, page_size, h_kv, d)");
  TORCH_CHECK(page_table_ref.dim() == 2, "flash_attentionXe35: page_table must be 2D");
  TORCH_CHECK(cu_seqlens_q.dim() == 1, "flash_attentionXe35: cu_seqlens_q must be 1D");
  TORCH_CHECK(cu_seqlens_k.dim() == 1, "flash_attentionXe35: cache_seqlens must be 1D");
  TORCH_CHECK(cu_seqlens_k.size(0) == batch, "flash_attentionXe35: cache_seqlens must have size equal to batch");
  // TORCH_CHECK(!q_v_.has_value(), "flash_attentionXe35: q_v is not supported");
  // TORCH_CHECK(!kv_batch_idx_.has_value(), "flash_attentionXe35: kv_batch_idx is not supported");
  // TORCH_CHECK(!leftpad_k_.has_value(), "flash_attentionXe35: leftpad_k is not supported");
  // TORCH_CHECK(!rotary_cos_.has_value() && !rotary_sin_.has_value(), "flash_attentionXe35: rotary is not supported");
  TORCH_CHECK(!seqlens_rotary_.has_value(), "flash_attentionXe35: seqlens_rotary is not supported");
  TORCH_CHECK(
      !q_descale_.has_value() && !k_descale_.has_value() && !v_descale_.has_value(),
      "flash_attentionXe35: descale is not supported");
  TORCH_CHECK(!sinks_.has_value(), "flash_attentionXe35: sinks is not supported");
  TORCH_CHECK(!is_causal, "flash_attentionXe35: causal attention is not supported");
  TORCH_CHECK(
      window_size_left == -1 && window_size_right == -1, "flash_attentionXe35: windowed attention is not supported");
  TORCH_CHECK(softcap == 0.0f, "flash_attentionXe35: softcap is not supported");
  // is_rotary_interleaved only matters when rotary cos/sin are actually provided; the Python
  // wrapper defaults it to true even for non-rotary calls, so only enforce the guard when rotary
  // embedding is requested.
  const bool has_rotary = rotary_cos_.has_value() || rotary_sin_.has_value();
  TORCH_CHECK(
      !(has_rotary && is_rotary_interleaved), "flash_attentionXe35: rotary interleaving is not supported");
  auto q_dtype = q.scalar_type();
  TORCH_CHECK(q_dtype == at::ScalarType::BFloat16, "flash_attentionXe35: only bf16 is supported");
  TORCH_CHECK(k.scalar_type() == q_dtype && v.scalar_type() == q_dtype, "flash_attentionXe35: q/k/v dtype mismatch");

  // TORCH_CHECK(num_kv_splits == 0, "flash_attentionXe35: num_kv_splits is not supported");
  // TORCH_CHECK(!pack_gqa_.has_value(), "flash_attentionXe35: pack_gqa is not supported");
  // TORCH_CHECK(sm_margin == 0, "flash_attentionXe35: sm_margin is not supported");
  TORCH_CHECK(!out_.has_value(), "flash_attentionXe35: out_ is not supported");
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
  TORCH_CHECK(
      page_table_ref.size(0) == batch, "flash_attentionXe35: page_table batch dimension must match cu_seqlens_q");
  TORCH_CHECK(
      k.stride(0) == static_cast<int64_t>(page_size) * k.stride(1),
      "flash_attentionXe35: k must be densely packed across [num_pages, page_size] without internal copies");
  TORCH_CHECK(
      v.stride(0) == static_cast<int64_t>(page_size) * v.stride(1),
      "flash_attentionXe35: v must be densely packed across [num_pages, page_size] without internal copies");

  // Keep this path device-only to avoid XPU<->CPU synchronization overhead.
  const int max_seqlen_kv_cache = max_seqlen_k;
  const double batched_effective_seq_len_q = static_cast<double>(total_q);
  const double batched_seq_len_kv = static_cast<double>(batch) * static_cast<double>(max_seqlen_kv_cache);
  const double batched_qk_pairs = static_cast<double>(total_q) * static_cast<double>(max_seqlen_kv_cache);

  TORCH_CHECK(head_dim == 64 || head_dim == 128, "flash_attentionXe35: only head_dim 64 and 128 are supported");

  c10::DeviceGuard device_guard(q.device());

  auto opts = q.options();
  auto out_dense = at::empty({total_q, num_heads_q, head_dim}, opts);

  auto k_pool = k.as_strided(
      {static_cast<int64_t>(num_pages) * page_size, static_cast<int64_t>(num_heads_kv), static_cast<int64_t>(head_dim)},
      {k.stride(1), k.stride(2), k.stride(3)});
  auto v_pool = v.as_strided(
      {static_cast<int64_t>(num_pages) * page_size, static_cast<int64_t>(num_heads_kv), static_cast<int64_t>(head_dim)},
      {v.stride(1), v.stride(2), v.stride(3)});

  // Measure with the cutlass GPU_Clock device-event path (matches the reference
  // xe_fmha_fwd_runner). The kernel is submitted on xe35_profiling_queue() (a
  // profiling-enabled queue in PyTorch's context) and registered with the
  // EventManager inside dispatch, so GPU_Clock reads real device timestamps.
  // Host-sync the XPU stream first so pending H2D copies finish before the kernel
  // reads them; the profiling queue is drained after launch. Serializing the two
  // queues on the host avoids concurrent cross-queue dependencies (the simulator
  // hang) while keeping accurate device timing.
  auto perf_stream = c10::xpu::getCurrentXPUStream();
  perf_stream.synchronize();
  GPU_Clock timer;
  timer.start();

  dispatch_xe3_kernel<cutlass::bfloat16_t>(
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
      max_seqlen_kv_cache,
      head_dim,
      page_size,
      softmax_scale_);

  // Drain the profiling queue so the kernel is complete before GPU_Clock reads
  // its device timestamps and before torch (on the XPU stream) reads the output.
  xe35_profiling_queue().wait();
  const double elapsed_s = timer.seconds();

  const double flops_qk = 2.0 * static_cast<double>(num_heads_q) * batched_qk_pairs * static_cast<double>(head_dim);
  const double flops_pv = 2.0 * static_cast<double>(num_heads_q) * batched_qk_pairs * static_cast<double>(head_dim);
  const double tflops = elapsed_s > 0.0 ? ((flops_qk + flops_pv) * 1e-12) / elapsed_s : 0.0;

  const double bytes_qk = static_cast<double>(num_heads_q) * batched_effective_seq_len_q *
                              static_cast<double>(head_dim) * static_cast<double>(q.element_size()) +
                          static_cast<double>(num_heads_kv) * batched_seq_len_kv * static_cast<double>(head_dim) *
                              static_cast<double>(k.element_size());
  const double bytes_pv = static_cast<double>(num_heads_kv) * batched_seq_len_kv * static_cast<double>(head_dim) *
                              static_cast<double>(v.element_size()) +
                          static_cast<double>(num_heads_q) * batched_effective_seq_len_q *
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

  if (elapsed_s <= 0.0) {
    ::printf(
        "flash_attentionXe35 perf(gpu_clock): unavailable (reported 0). Check SYCL profiling event path and "
        "CUTLASS_SYCL_PROFILING_ENABLED build flag.\n");
  }

  at::Tensor out = out_dense;

  auto lse = at::zeros({num_heads_q, total_q}, opts.dtype(at::kFloat));
  auto out_accum = at::Tensor();
  auto lse_accum = at::Tensor();

  return {out, lse, out_accum, lse_accum};
}

#undef SYCL_INTEL_TARGET

#pragma once

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

namespace xe35_fmha {

inline sycl::queue& profiling_queue() {
  static sycl::queue queue = [] {
    sycl::queue torch_queue = at::xpu::getCurrentXPUStream().queue();
    return sycl::queue(torch_queue.get_context(), torch_queue.get_device(), sycl::property::queue::enable_profiling{});
  }();
  return queue;
}

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
      bool Causal,
      bool LocalMask,
      bool Sink,
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
      float softmax_scale,
      int window_size_left,
      int window_size_right,
      const ElementInput* sm_sink) {
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
        LocalMask,
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
        cutlass::fmha::collective::FMHAFwdEpilogue<CollectiveMainloop, TileShapeOut, TensorO, void, Sink>;
    using ProblemShape = cutlass::fmha::kernel::FMHAProblemShape<true>;
    using Scheduler = cutlass::fmha::kernel::XeFHMAIndividualTileScheduler<false, false, Causal, GqaFusion>;
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
            sm_sink,
        },
        {
            softmax_scale,
            static_cast<const int*>(page_table.data_ptr()),
            page_size,
            num_pages_per_seq,
            static_cast<int>(k_pool.size(0)),
            window_size_left,
            window_size_right,
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

    namespace syclex = sycl::ext::oneapi::experimental;
    namespace intelex = sycl::ext::intel::experimental;
    compat::dim3 const block = FMHAKernel::get_block_shape();
    compat::dim3 const grid = FMHAKernel::get_grid_shape(params);
    const int smem_size = FMHAKernel::SharedStorageSize;
    compat::experimental::launch_properties launch_props{syclex::work_group_scratch_size(smem_size)};
    compat::experimental::kernel_properties kernel_props{
        syclex::sub_group_size<cute::intel::sg_size>, intelex::grf_size<512>};
    compat::experimental::launch_policy policy{
        compat::dim3(grid.x, grid.y, grid.z), compat::dim3(block.x, block.y, block.z), launch_props, kernel_props};
    sycl::queue& queue = profiling_queue();
#if defined(CUTLASS_SYCL_PROFILING_ENABLED)
    auto event = compat::experimental::launch<cutlass::device_kernel<FMHAKernel>, FMHAKernel>(policy, queue, params);
    EventManager::getInstance().addEvent(event);
#else
    compat::experimental::launch<cutlass::device_kernel<FMHAKernel>, FMHAKernel, false>(policy, queue, params);
#endif
  }
};

template <int HeadDim, bool UseDecode>
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
    int page_size,
    float softmax_scale,
    bool is_causal,
    bool is_local,
    int window_size_left,
    int window_size_right,
    const cutlass::bfloat16_t* sm_sink,
    bool use_sink) {
  const int gqa_group = num_heads_kv > 0 ? num_heads_q / num_heads_kv : 1;
  const int total_rows = gqa_group * max_seqlen_q;

#define XE35_RUN(CAUSAL, LOCAL, STAGES, GQA, QK, PV, OUT, SGL)                                               \
  do {                                                                                                       \
    if (use_sink) {                                                                                          \
      Xe3FmhaConfig<cutlass::bfloat16_t>::template run<CAUSAL, LOCAL, true, STAGES, GQA, QK, PV, OUT, SGL>(  \
          q_dense,                                                                                           \
          k_pool,                                                                                            \
          v_pool,                                                                                            \
          page_table,                                                                                        \
          num_pages_per_seq,                                                                                 \
          cu_seqlens_q,                                                                                      \
          cu_seqlens_k_cache,                                                                                \
          out_dense,                                                                                         \
          batch,                                                                                             \
          num_heads_q,                                                                                       \
          num_heads_kv,                                                                                      \
          max_seqlen_q,                                                                                      \
          max_seqlen_kv_cache,                                                                               \
          HeadDim,                                                                                           \
          page_size,                                                                                         \
          softmax_scale,                                                                                     \
          window_size_left,                                                                                  \
          window_size_right,                                                                                 \
          sm_sink);                                                                                          \
    } else {                                                                                                 \
      Xe3FmhaConfig<cutlass::bfloat16_t>::template run<CAUSAL, LOCAL, false, STAGES, GQA, QK, PV, OUT, SGL>( \
          q_dense,                                                                                           \
          k_pool,                                                                                            \
          v_pool,                                                                                            \
          page_table,                                                                                        \
          num_pages_per_seq,                                                                                 \
          cu_seqlens_q,                                                                                      \
          cu_seqlens_k_cache,                                                                                \
          out_dense,                                                                                         \
          batch,                                                                                             \
          num_heads_q,                                                                                       \
          num_heads_kv,                                                                                      \
          max_seqlen_q,                                                                                      \
          max_seqlen_kv_cache,                                                                               \
          HeadDim,                                                                                           \
          page_size,                                                                                         \
          softmax_scale,                                                                                     \
          window_size_left,                                                                                  \
          window_size_right,                                                                                 \
          nullptr);                                                                                          \
    }                                                                                                        \
  } while (false)

  if constexpr (HeadDim == 128) {
    if constexpr (UseDecode) {
      using ShapeQK = Shape<_128, _64, _64>;
      using ShapePV = Shape<_128, _64, _64>;
      using ShapeOut = Shape<_128, _128>;
      using SubgroupLayoutQK = Layout<Shape<_8, _1, _1>>;
      if (is_local)
        XE35_RUN(false, true, 2, false, ShapeQK, ShapePV, ShapeOut, SubgroupLayoutQK);
      else if (is_causal)
        XE35_RUN(true, false, 2, false, ShapeQK, ShapePV, ShapeOut, SubgroupLayoutQK);
      else
        XE35_RUN(false, false, 2, false, ShapeQK, ShapePV, ShapeOut, SubgroupLayoutQK);
      return;
    }

    using ShapeQK = Shape<_128, _64, _64>;
    using ShapePV = Shape<_128, _64, _64>;
    using ShapeOut = Shape<_128, _128>;
    using SubgroupLayoutQK = Layout<Shape<_8, _1, _1>>;
    if (is_local) {
      XE35_RUN(false, true, 2, false, ShapeQK, ShapePV, ShapeOut, SubgroupLayoutQK);
    } else if (is_causal) {
      XE35_RUN(true, false, 2, false, ShapeQK, ShapePV, ShapeOut, SubgroupLayoutQK);
    } else {
      XE35_RUN(false, false, 2, false, ShapeQK, ShapePV, ShapeOut, SubgroupLayoutQK);
    }
    return;
  }

  if constexpr (HeadDim == 64) {
    if constexpr (UseDecode) {
      if (use_sink) {
        using SinkShapeQK = Shape<_128, _64, _32>;
        using SinkShapePV = Shape<_128, _32, _64>;
        using SinkShapeOut = Shape<_128, _64>;
        using SinkSubgroupLayoutQK = Layout<Shape<_8, _1, _1>>;
        if (is_local) {
          XE35_RUN(false, true, 2, false, SinkShapeQK, SinkShapePV, SinkShapeOut, SinkSubgroupLayoutQK);
        } else {
          XE35_RUN(false, false, 2, false, SinkShapeQK, SinkShapePV, SinkShapeOut, SinkSubgroupLayoutQK);
        }
        return;
      }
      using ShapeQK = Shape<_128, _64, _32>;
      using ShapePV = Shape<_128, _32, _64>;
      using ShapeOut = Shape<_128, _64>;
      using SubgroupLayoutQK = Layout<Shape<_8, _1, _1>>;
      if (is_local)
        XE35_RUN(false, true, 2, false, ShapeQK, ShapePV, ShapeOut, SubgroupLayoutQK);
      else if (is_causal)
        XE35_RUN(true, false, 2, false, ShapeQK, ShapePV, ShapeOut, SubgroupLayoutQK);
      else
        XE35_RUN(false, false, 2, false, ShapeQK, ShapePV, ShapeOut, SubgroupLayoutQK);
      return;
    }

    using ShapeQK = Shape<_128, _64, _32>;
    using ShapePV = Shape<_128, _32, _64>;
    using ShapeOut = Shape<_128, _64>;
    using SubgroupLayoutQK = Layout<Shape<_8, _1, _1>>;
    if (is_local) {
      XE35_RUN(false, true, 2, false, ShapeQK, ShapePV, ShapeOut, SubgroupLayoutQK);
    } else if (is_causal) {
      XE35_RUN(true, false, 2, false, ShapeQK, ShapePV, ShapeOut, SubgroupLayoutQK);
    } else {
      XE35_RUN(false, false, 2, false, ShapeQK, ShapePV, ShapeOut, SubgroupLayoutQK);
    }
    return;
  }

#undef XE35_RUN

  TORCH_CHECK(false, "flash_attentionXe35: only head_dim 64 and 128 are supported in Xe3 cutlass path");
}

}  // namespace xe35_fmha
#pragma once

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
sycl::queue& xe35_profiling_queue() {
  static sycl::queue queue = [] {
    sycl::queue torch_queue = at::xpu::getCurrentXPUStream().queue();
    return sycl::queue(torch_queue.get_context(), torch_queue.get_device(), sycl::property::queue::enable_profiling{});
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
      bool Causal,
      bool LocalMask,
      bool Sink,
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
      float softmax_scale,
      int window_size_left,
      int window_size_right,
      const ElementInput* sm_sink) {
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
        LocalMask,
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
        cutlass::fmha::collective::FMHAFwdEpilogue<CollectiveMainloop, TileShapeOut, TensorO, void, Sink>;
    using ProblemShape = cutlass::fmha::kernel::FMHAProblemShape<true>;
    using Scheduler = cutlass::fmha::kernel::XeFHMAIndividualTileScheduler<false, false, Causal, GqaFusion>;
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
            sm_sink,
        },
        {
            softmax_scale,
            static_cast<const int*>(page_table.data_ptr()),
            page_size,
            num_pages_per_seq,
            static_cast<int>(k_pool.size(0)),
            window_size_left,
            window_size_right,
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

    namespace syclex = sycl::ext::oneapi::experimental;
    namespace intelex = sycl::ext::intel::experimental;
    compat::dim3 const block = FMHAKernel::get_block_shape();
    compat::dim3 const grid = FMHAKernel::get_grid_shape(params);
    const int smem_size = FMHAKernel::SharedStorageSize;
    compat::experimental::launch_properties launch_props{syclex::work_group_scratch_size(smem_size)};
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

template <int HeadDim, bool UseDecode>
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
    int page_size,
    float softmax_scale,
    bool is_causal,
    bool is_local,
    int window_size_left,
    int window_size_right,
    const cutlass::bfloat16_t* sm_sink,
    bool use_sink) {
  const int gqa_group = num_heads_kv > 0 ? num_heads_q / num_heads_kv : 1;
  const int total_rows = gqa_group * max_seqlen_q;
  const bool use_decode = UseDecode;

#define XE35_RUN(CAUSAL, LOCAL, STAGES, GQA, QK, PV, OUT, SGL)                                               \
  do {                                                                                                       \
    if (use_sink) {                                                                                          \
      Xe3FmhaConfig<cutlass::bfloat16_t>::template run<CAUSAL, LOCAL, true, STAGES, GQA, QK, PV, OUT, SGL>(  \
          q_dense,                                                                                           \
          k_pool,                                                                                            \
          v_pool,                                                                                            \
          page_table,                                                                                        \
          num_pages_per_seq,                                                                                 \
          cu_seqlens_q,                                                                                      \
          cu_seqlens_k_cache,                                                                                \
          out_dense,                                                                                         \
          batch,                                                                                             \
          num_heads_q,                                                                                       \
          num_heads_kv,                                                                                      \
          max_seqlen_q,                                                                                      \
          max_seqlen_kv_cache,                                                                               \
          HeadDim,                                                                                           \
          page_size,                                                                                         \
          softmax_scale,                                                                                     \
          window_size_left,                                                                                  \
          window_size_right,                                                                                 \
          sm_sink);                                                                                          \
    } else {                                                                                                 \
      Xe3FmhaConfig<cutlass::bfloat16_t>::template run<CAUSAL, LOCAL, false, STAGES, GQA, QK, PV, OUT, SGL>( \
          q_dense,                                                                                           \
          k_pool,                                                                                            \
          v_pool,                                                                                            \
          page_table,                                                                                        \
          num_pages_per_seq,                                                                                 \
          cu_seqlens_q,                                                                                      \
          cu_seqlens_k_cache,                                                                                \
          out_dense,                                                                                         \
          batch,                                                                                             \
          num_heads_q,                                                                                       \
          num_heads_kv,                                                                                      \
          max_seqlen_q,                                                                                      \
          max_seqlen_kv_cache,                                                                               \
          HeadDim,                                                                                           \
          page_size,                                                                                         \
          softmax_scale,                                                                                     \
          window_size_left,                                                                                  \
          window_size_right,                                                                                 \
          nullptr);                                                                                          \
    }                                                                                                        \
  } while (false)

  if constexpr (HeadDim == 128) {
    if constexpr (UseDecode) {
      using ShapeQK = Shape<_128, _64, _64>;
      using ShapePV = Shape<_128, _64, _64>;
      using ShapeOut = Shape<_128, _128>;
      using SubgroupLayoutQK = Layout<Shape<_8, _1, _1>>;
      if (is_local)
        XE35_RUN(false, true, 2, false, ShapeQK, ShapePV, ShapeOut, SubgroupLayoutQK);
      else if (is_causal)
        XE35_RUN(true, false, 2, false, ShapeQK, ShapePV, ShapeOut, SubgroupLayoutQK);
      else
        XE35_RUN(false, false, 2, false, ShapeQK, ShapePV, ShapeOut, SubgroupLayoutQK);
      return;
    }

    using ShapeQK = Shape<_128, _64, _64>;
    using ShapePV = Shape<_128, _64, _64>;
    using ShapeOut = Shape<_128, _128>;
    using SubgroupLayoutQK = Layout<Shape<_8, _1, _1>>;
    if (is_local)
      XE35_RUN(false, true, 2, false, ShapeQK, ShapePV, ShapeOut, SubgroupLayoutQK);
    else if (is_causal)
      XE35_RUN(true, false, 2, false, ShapeQK, ShapePV, ShapeOut, SubgroupLayoutQK);
    else
      XE35_RUN(false, false, 2, false, ShapeQK, ShapePV, ShapeOut, SubgroupLayoutQK);
    return;
  }

  if constexpr (HeadDim == 64) {
    if constexpr (UseDecode) {
      if (use_sink) {
        using SinkShapeQK = Shape<_128, _64, _32>;
        using SinkShapePV = Shape<_128, _32, _64>;
        using SinkShapeOut = Shape<_128, _64>;
        using SinkSubgroupLayoutQK = Layout<Shape<_8, _1, _1>>;
        if (is_local)
          XE35_RUN(false, true, 2, false, SinkShapeQK, SinkShapePV, SinkShapeOut, SinkSubgroupLayoutQK);
        else
          XE35_RUN(false, false, 2, false, SinkShapeQK, SinkShapePV, SinkShapeOut, SinkSubgroupLayoutQK);
        return;
      }
      using ShapeQK = Shape<_128, _64, _32>;
      using ShapePV = Shape<_128, _32, _64>;
      using ShapeOut = Shape<_128, _64>;
      using SubgroupLayoutQK = Layout<Shape<_8, _1, _1>>;
      if (is_local)
        XE35_RUN(false, true, 2, false, ShapeQK, ShapePV, ShapeOut, SubgroupLayoutQK);
      else if (is_causal)
        XE35_RUN(true, false, 2, false, ShapeQK, ShapePV, ShapeOut, SubgroupLayoutQK);
      else
        XE35_RUN(false, false, 2, false, ShapeQK, ShapePV, ShapeOut, SubgroupLayoutQK);
      return;
    }

    using ShapeQK = Shape<_128, _64, _32>;
    using ShapePV = Shape<_128, _32, _64>;
    using ShapeOut = Shape<_128, _64>;
    using SubgroupLayoutQK = Layout<Shape<_8, _1, _1>>;
    if (is_local)
      XE35_RUN(false, true, 2, false, ShapeQK, ShapePV, ShapeOut, SubgroupLayoutQK);
    else if (is_causal)
      XE35_RUN(true, false, 2, false, ShapeQK, ShapePV, ShapeOut, SubgroupLayoutQK);
    else
      XE35_RUN(false, false, 2, false, ShapeQK, ShapePV, ShapeOut, SubgroupLayoutQK);
    return;
  }

  TORCH_CHECK(false, "flash_attentionXe35: only head_dim 64 and 128 are supported in Xe3 cutlass path");
}

#undef XE35_RUN

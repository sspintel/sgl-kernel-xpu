#define SYCL_INTEL_XE4_TARGET
#define SYCL_INTEL_TARGET 40

#include <ATen/ATen.h>
#include <c10/xpu/XPUStream.h>
#include <torch/all.h>

#include <cute/arch/mma_xe4.hpp>
#include <cute/tensor.hpp>

#include "Utils.h"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/group_array_problem_shape.hpp"
#include "kernels/moe/xe40/collective/xe4_array_mma_epilogue.hpp"
#include "kernels/moe/xe40/collective/xe4_array_mma_mainloop.hpp"
#include "kernels/moe/xe40/kernel/xe4_array_gemm_kernel.hpp"
#include "kernels/moe/xe40/kernel/xe4_tile_scheduler_group.hpp"

using namespace cute;

using ElementAccumulator = float;  // <- data type of accumulator

template <typename, typename, typename, typename, typename, typename>
class GemmXe40Name;

using ProblemShape = cutlass::xe4_grouped_gemm::kernel::MoEProblemShape<Shape<int, int, int>>;  // <M,N,K> per group

template <
    typename TileShape,
    typename SGLayout,
    typename ElementA,
    typename ElementB,
    typename ElementC,
    typename ElementD>
void Xe40MoEGEMMLauncher(
    sycl::queue q,
    const void* activations,
    const void* weights,
    const void* scales,
    void* outputs,
    const int gemm_n,
    const int gemm_k,
    const int* num_rows_per_expert_device,
    const int num_experts,
    int64_t* workspace) {
  using ClusterShape = Shape<_1, _1, _1>;

  using TileShape_MNK = TileShape;  // 128, 128, 128

  constexpr auto majorA = cute::AMMA::Major::K;
  constexpr auto majorB = cute::AMMA::Major::K;

  constexpr int PipelineStages = 2;

  using LayoutA = cutlass::layout::RowMajor;
  using LayoutB = cutlass::layout::ColumnMajor;
  using LayoutD = cutlass::layout::RowMajor;

  using SmemLayoutAtomA = decltype(make_layout(cute::select<0, 2>(TileShape_MNK{}), GenRowMajor{}));

  using SmemLayoutA = decltype(tile_to_shape(
      SmemLayoutAtomA{}, make_shape(shape<0>(TileShape_MNK{}), shape<2>(TileShape_MNK{}), Int<PipelineStages>{})));

  using SmemLayoutAtomB = decltype(make_layout(cute::select<1, 2>(TileShape_MNK{}), GenRowMajor{}));

  using SmemLayoutB = decltype(tile_to_shape(
      SmemLayoutAtomB{}, make_shape(shape<1>(TileShape_MNK{}), shape<2>(TileShape_MNK{}), Int<PipelineStages>{})));

  using SmemLayoutOutput = decltype(make_layout(cute::select<0, 1>(TileShape_MNK{}), GenRowMajor{}));

  // Copy from GMEM to SMEM
  using TMACopyAtomA = cute::xe4::ASYNC_TENSOR_LOAD<slm_matrix_type::type1, size<2>(TileShape_MNK{}) /*stride=K*/>;
  using TMACopyAtomB = cute::xe4::ASYNC_TENSOR_LOAD<slm_matrix_type::type1, size<2>(TileShape_MNK{}) /*stride=K*/>;
  using TMACopyAtomD = cute::xe4::ASYNC_TENSOR_STORE<slm_matrix_type::type1, size<1>(TileShape_MNK{}) /*stride=N*/>;

  using TiledMma = decltype(cute::make_tiled_mma(cute::AMMA::ss_op_selector<
                                                 ElementD /*D dtype*/,
                                                 ElementA,
                                                 ElementB,
                                                 ElementD /*C dtype*/,
                                                 TileShape_MNK,
                                                 ClusterShape,
                                                 majorA,
                                                 majorB>()));

  using CollectiveMainloop = cutlass::xe4_grouped_gemm::collective::XE4CollectiveMma<
      ProblemShape,
      TileShape,
      ElementA,
      ElementB,
      cutlass::gemm::TagToStrideA_t<LayoutA*>,
      cutlass::gemm::TagToStrideB_t<LayoutB*>,
      TiledMma,
      SmemLayoutA,
      SmemLayoutB,
      SmemLayoutOutput,
      TMACopyAtomA,
      TMACopyAtomB>;

  using CollectiveEpilogue = cutlass::xe4_grouped_gemm::collective::XE4CollectiveEpilogue<
      ProblemShape,
      TileShape,
      ElementD,
      cutlass::gemm::TagToStrideC_t<LayoutD*>,
      SmemLayoutOutput,
      TMACopyAtomD>;

  using TileScheduler = cutlass::xe4_grouped_gemm::kernel::PersistentTileSchedulerXe4Group<ProblemShape>;

  using Gemm = cutlass::xe4_grouped_gemm::kernel::
      XE4GemmUniversal<ProblemShape, CollectiveMainloop, CollectiveEpilogue, TileScheduler>;

  dim3 const block = Gemm::get_block_shape();

  Gemm kernel;
  cutlass::KernelHardwareInfo hw_info;

  // Change device_id to another value if you are running on a machine with
  // multiple GPUs and wish to use a GPU other than that with device ID 0.
  hw_info.sm_count = cutlass::KernelHardwareInfo::query_device_multiprocessor_count(hw_info.device_id);

  auto arguments = typename Gemm::Arguments{
      {num_experts, num_rows_per_expert_device, gemm_n, gemm_k},
      {reinterpret_cast<const ElementA*>(activations), reinterpret_cast<const ElementB*>(weights), gemm_n, gemm_k},
      {reinterpret_cast<const ElementD*>(outputs), gemm_n},
      hw_info,
      {1, RasterOrderOptions::AlongN, reinterpret_cast<uint64_t*>(workspace)}};
  auto params = Gemm::to_underlying_arguments(arguments);
  dim3 const grid = Gemm::get_grid_shape(params);
  const auto sycl_block = compat::dim3(block.x, block.y, block.z);
  const auto sycl_grid = compat::dim3(grid.x, grid.y, grid.z);
  q.submit([&](sycl::handler& h) {
    h.parallel_for<GemmXe40Name<TileShape, SGLayout, ElementA, ElementB, ElementC, ElementD>>(
        sycl::nd_range<3>{sycl_grid * sycl_block, sycl_block}, [=](sycl::nd_item<3> item) { kernel(params); });
  });
}

void moe_grouped_mm_nt_xe40(
    torch::Tensor& output,
    const torch::Tensor& activations,
    const torch::Tensor& weights,
    const torch::Tensor& total_rows_for_experts,
    const int64_t n_experts) {
  int total_m = activations.sizes()[0];
  int gemm_k = activations.sizes()[1];
  auto weights_shape = weights.sizes().vec();
  int gemm_n = weights.sizes()[1];

  TORCH_CHECK(weights_shape.size() == 3, "weights must be 3D");
  TORCH_CHECK(weights_shape[0] == n_experts, "weights must have n_experts as the first dimension");
  TORCH_CHECK(weights_shape[1] == gemm_n, "weights must be gemm_n * gemm_k");
  TORCH_CHECK(
      weights_shape[0] == total_rows_for_experts.size(0),
      "rows_for_experts must have the same size as the first dimension of weights");
  TORCH_CHECK(output.sizes()[0] == total_m, "output must have the same number of rows as activations");
  TORCH_CHECK(output.sizes()[1] == gemm_n, "output must have the same number of columns as activations");
  TORCH_CHECK(n_experts % 8 == 0, "n_experts must be a multiple of 8 for the current implementation");
  TORCH_CHECK(
      activations.scalar_type() == weights.scalar_type(), "activations and weights must have the same data type");
  TORCH_CHECK(
      activations.scalar_type() == at::ScalarType::BFloat16,
      "Only bfloat16 are supported in moe_grouped_mm_nt currently");

  auto stream = at::xpu::getCurrentXPUStream();
  auto queue = stream.queue();
  at::Tensor atomic_buffer = at::empty({static_cast<long>(1)}, activations.options().dtype(at::kLong));

  using TileShape = Shape<_256, _256, _32>;
  using SGLayout = Layout<Shape<_8, _4, _1>, Stride<_4, _1, _0>>;
  using Element = bf16;
  Xe40MoEGEMMLauncher<TileShape, SGLayout, Element, Element, Element, Element>(
      queue,
      activations.data_ptr(),
      weights.data_ptr(),
      nullptr,
      output.data_ptr(),
      gemm_n,
      gemm_k,
      total_rows_for_experts.data_ptr<int>(),
      n_experts,
      atomic_buffer.data_ptr<int64_t>());
}

#undef SYCL_INTEL_TARGET
#undef SYCL_INTEL_XE4_TARGET

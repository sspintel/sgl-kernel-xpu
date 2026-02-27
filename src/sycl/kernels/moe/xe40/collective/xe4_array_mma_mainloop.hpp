#pragma once

#include "cutlass/arch/barrier.h"
#include "cutlass/cutlass.h"

namespace cutlass::xe4_grouped_gemm::collective {

using namespace cute;

template <
    class ProblemShape_,
    class TileShape_,
    class ElementA_,
    class ElementB_,
    class StrideA_,
    class StrideB_,
    class TiledMma_,
    class SmemLayoutA_,
    class SmemLayoutB_,
    class SmemLayoutOutput_,
    class TMACopyAtomA_,
    class TMACopyAtomB_>
struct XE4CollectiveMma {
  using ProblemShape = ProblemShape_;
  using TileShape = TileShape_;  // <BLK_M, BLK_N, BLK_K>

  using ElementA = ElementA_;
  using ElementB = ElementB_;

  using StrideA = StrideA_;
  using StrideB = StrideB_;

  using InternalStrideA = cute::remove_pointer_t<StrideA>;
  using InternalStrideB = cute::remove_pointer_t<StrideB>;
  using TensorMKL = decltype(make_tensor(
      make_gmem_ptr(static_cast<ElementA const*>(nullptr)),
      repeat_like(InternalStrideA{}, int32_t(0)),
      InternalStrideA{}));
  using TensorNKL = decltype(make_tensor(
      make_gmem_ptr(static_cast<ElementB const*>(nullptr)),
      repeat_like(InternalStrideB{}, int32_t(0)),
      InternalStrideB{}));
  using MainloopTensors = cute::tuple<TensorMKL, TensorNKL>;

  using TiledMma = TiledMma_;

  using SmemLayoutA = SmemLayoutA_;
  using SmemLayoutB = SmemLayoutB_;
  using SmemLayoutOutput = SmemLayoutOutput_;

  using TMACopyAtomA = TMACopyAtomA_;
  using TMACopyAtomB = TMACopyAtomB_;

  static constexpr uint32_t sg_size = cutlass::NumThreadsPerWarp;

  static constexpr uint32_t PipelineStages = 2;
  using MainloopPipeline = cutlass::PipelineTmaAsync<PipelineStages>;
  using PipelineState = typename cutlass::PipelineState<PipelineStages>;

  static constexpr int NumLoadWarps = 1;
  static constexpr int NumMMAWarps = 1;

  using TMA_A = decltype(make_tma_copy(
      TMACopyAtomA{}, TensorMKL{}, SmemLayoutA{}(_, _, cute::Int<0>{}), select<0, 2>(TileShape{}), _1{}));

  using TMA_B = decltype(make_tma_copy(
      TMACopyAtomB{}, TensorNKL{}, SmemLayoutB{}(_, _, cute::Int<0>{}), select<1, 2>(TileShape{}), _1{}));

  static constexpr size_t SmemAlignment = 512;
  struct SharedStorage {
    struct TensorStorage : cute::aligned_struct<SmemAlignment, _0> {
      cute::array_aligned<ElementA, cute::cosize_v<SmemLayoutA>, SmemAlignment> smem_A;
      cute::array_aligned<ElementB, cute::cosize_v<SmemLayoutB>, SmemAlignment> smem_B;
    };
    struct PipelineStorage {
      typename MainloopPipeline::SharedStorage storage_A;
      typename MainloopPipeline::SharedStorage storage_B;

      cutlass::arch::ClusterTransactionBarrier barrier_D;
    };
  };

  using TensorStorage = typename SharedStorage::TensorStorage;
  using PipelineStorage = typename SharedStorage::PipelineStorage;

  static constexpr uint32_t TmaTransactionBytesA = sizeof(TensorStorage::smem_A) / PipelineStages;
  static constexpr uint32_t TmaTransactionBytesB = sizeof(TensorStorage::smem_B) / PipelineStages;

  // Host side kernel arguments
  struct Arguments {
    ElementA const* ptr_A;
    ElementB const* ptr_B;
    int N;
    int K;
  };

  // Device side kernel params
  struct Params {
    ProblemShape problem_shape;
    ElementA const* ptr_A;
    ElementB const* ptr_B;
    int N;
    int K;
  };

  XE4CollectiveMma() = default;

  static constexpr Params to_underlying_arguments(ProblemShape const& problem_shape, Arguments const& args) {
    return {problem_shape, args.ptr_A, args.ptr_B, args.N, args.K};
  }

  template <typename DescTuple, typename BlockCoord, class LoadTensors, typename ProblemShape_MNKL>
  CUTLASS_DEVICE void load(
      Params const& params,
      TensorStorage& shared_tensors,
      PipelineStorage& shared_pipelines,
      DescTuple const& tdesc_tuple,
      BlockCoord const& block_coord,
      int const num_k_tiles,
      MainloopPipeline pipeline_a,
      PipelineState& smem_pipe_write_a,
      MainloopPipeline pipeline_b,
      PipelineState& smem_pipe_write_b,
      LoadTensors const& load_tensors,
      ProblemShape_MNKL const& problem_shape_mnkl) {
    bool lane_predicate = cute::elect_one_sync();

    if (lane_predicate) {
      auto [M, N, K, L] = problem_shape_mnkl;

      auto blk_m_coord = get<0>(block_coord);  // blk_m_idx
      auto blk_n_coord = get<1>(block_coord);  // blk_n_idx
      auto blk_l_coord = get<2>(block_coord);  // blk_l_idx

      TMA_A tma_load_A = make_tma_copy(
          TMACopyAtomA{}, get<0>(load_tensors), SmemLayoutA{}(_, _, _0{}), select<0, 2>(TileShape{}), _1{});

      TMA_B tma_load_B = make_tma_copy(
          TMACopyAtomB{}, get<1>(load_tensors), SmemLayoutB{}(_, _, _0{}), select<1, 2>(TileShape{}), _1{});

      // Initialize matrix descriptor
      auto [tdesc_a, tdesc_b] = tdesc_tuple;
      tma_load_A.cache_.set_tensor_desc(tdesc_a);
      tma_load_B.cache_.set_tensor_desc(tdesc_b);

      Tensor mA_mkl = tma_load_A.get_tma_tensor(make_shape(M, K, L));
      Tensor mB_nkl = tma_load_B.get_tma_tensor(make_shape(N, K, L));

      Tensor gA = local_tile(
          mA_mkl(_, _, 0), TileShape{}, make_coord(blk_m_coord, _, _), Step<_1, X, _1>{});  // (BLK_M,BLK_K,k)
      Tensor gB = local_tile(
          mB_nkl(_, _, 0), TileShape{}, make_coord(_, blk_n_coord, _), Step<X, _1, _1>{});  // (BLK_N,BLK_K,k)

      Tensor sA = make_tensor(make_smem_ptr(shared_tensors.smem_A.data()), SmemLayoutA{});  // (BLK_M,BLK_K,PIPE)
      Tensor sB = make_tensor(make_smem_ptr(shared_tensors.smem_B.data()), SmemLayoutB{});  // (BLK_N,BLK_K,PIPE)

      auto [tAgA, tAsA] = tma_partition(
          tma_load_A, _0{}, Layout<_1>{}, group_modes<0, 2>(sA), group_modes<0, 2>(gA));  // (TMA,k), (TMA,PIPE)
      auto [tBgB, tBsB] = tma_partition(
          tma_load_B, _0{}, Layout<_1>{}, group_modes<0, 2>(sB), group_modes<0, 2>(gB));  // (TMA,k), (TMA,PIPE)

      for (int i = 0; i < num_k_tiles; ++i) {
        pipeline_a.producer_acquire(smem_pipe_write_a);
        copy(
            tma_load_A.with(pipeline_a.producer_get_barrier(smem_pipe_write_a), 0),
            tAgA(_, i),
            tAsA(_, smem_pipe_write_a.index()));
        ++smem_pipe_write_a;

        pipeline_b.producer_acquire(smem_pipe_write_b);
        copy(
            tma_load_B.with(pipeline_b.producer_get_barrier(smem_pipe_write_b), 0),
            tBgB(_, i),
            tBsB(_, smem_pipe_write_b.index()));
        ++smem_pipe_write_b;
      }
    }
  }

  template <typename EpilogueTensorStorage>
  CUTLASS_DEVICE void
  mma(Params const& params,
      TensorStorage& shared_tensors,
      EpilogueTensorStorage& epi_shared_tensors,
      PipelineStorage& shared_pipelines,
      int const num_k_tiles,
      MainloopPipeline pipeline_a,
      PipelineState& smem_pipe_read_a,
      MainloopPipeline pipeline_b,
      PipelineState& smem_pipe_read_b) {
    bool lane_predicate = cute::elect_one_sync();

    if (lane_predicate) {
      auto item = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
      auto sg = item.get_sub_group();
      uint32_t item_id = item.get_local_linear_id();
      uint32_t sg_id = get_sg_id();
      int thread_idx = static_cast<int>(ThreadIdxX());

      TiledMma tiled_mma;

      auto thr_mma = tiled_mma.get_thread_slice(thread_idx);

      Tensor sA = make_tensor(make_smem_ptr(shared_tensors.smem_A.data()), SmemLayoutA{});  // (BLK_M,BLK_K,PIPE)
      Tensor sB = make_tensor(make_smem_ptr(shared_tensors.smem_B.data()), SmemLayoutB{});  // (BLK_N,BLK_K,PIPE)
      Tensor sD = make_tensor(make_smem_ptr(epi_shared_tensors.smem_D.data()), SmemLayoutOutput{});  // (BLK_M,BLK_N)

      // Matrix descriptors
      Tensor tSsA = thr_mma.partition_fragment_A(sA);  // (MMA,MMA_M,MMA_K,PIPE)
      Tensor tSsB = thr_mma.partition_fragment_B(sB);  // (MMA,MMA_N,MMA_K,PIPE)
      Tensor tSsD = thr_mma.partition_fragment_C(sD);  // (MMA,MMA_M,MMA_N)

      uint64_t mma_ctrl = 0x100;
      constexpr uint32_t mcast_mask = 0;

      for (int k = 0; k < num_k_tiles; ++k) {
        pipeline_a.consumer_wait(smem_pipe_read_a);
        pipeline_b.consumer_wait(smem_pipe_read_b);

        CUTE_UNROLL
        for (int m = 0; m < size<2>(tSsA); ++m) {
          for (int n = 0; n < size<2>(tSsB); ++n) {
            cute::gemm(
                tiled_mma.with(
                    AMMA::TrackMethod<AMMA::Tracking::DAB>{},
                    mma_ctrl,
                    reinterpret_cast<uint64_t*>(&shared_pipelines.barrier_D),
                    pipeline_a.consumer_get_barrier(smem_pipe_read_a),
                    pipeline_b.consumer_get_barrier(smem_pipe_read_b),
                    mcast_mask,
                    mcast_mask),
                tSsA(_, _, m, smem_pipe_read_a.index()),
                tSsB(_, _, n, smem_pipe_read_b.index()),
                tSsD);

            mma_ctrl = 0x0;
          }
        }

        pipeline_a.consumer_commit(smem_pipe_read_a, size<2>(tSsA));
        pipeline_b.consumer_commit(smem_pipe_read_b, size<2>(tSsA));

        shared_pipelines.barrier_D.arrive_and_expect_tx(size<2>(tSsA));
        shared_pipelines.barrier_D.wait(/*phase=*/k % PipelineStages);

        ++smem_pipe_read_a;
        ++smem_pipe_read_b;
      }
    }
  }

  template <typename ProblemShape_MNKL>
  CUTLASS_DEVICE auto update_tensor_shape_stride(
      Params const& mainloop_params,
      int32_t const& cumulative_M,
      int32_t const& next_group,
      ProblemShape_MNKL const& problem_shape_mnkl) {
    const int32_t M = get<0>(problem_shape_mnkl);
    const int32_t N = get<1>(problem_shape_mnkl);
    const int32_t K = get<2>(problem_shape_mnkl);

    ElementA const* ptr_A_curr_batch = reinterpret_cast<ElementA const*>(mainloop_params.ptr_A) + cumulative_M * K;
    ElementB const* ptr_B_curr_batch = reinterpret_cast<ElementB const*>(mainloop_params.ptr_B) + next_group * N * K;

    Tensor mA = make_tensor(
        make_gmem_ptr(ptr_A_curr_batch),
        cute::make_shape(M, K, (int32_t)1),
        cutlass::make_cute_packed_stride(InternalStrideA{}, {M, K, 1}));
    Tensor mB = make_tensor(
        make_gmem_ptr(ptr_B_curr_batch),
        cute::make_shape(N, K, (int32_t)1),
        cutlass::make_cute_packed_stride(InternalStrideB{}, {N, K, 1}));

    return cute::make_tuple(mA, mB);
  }
};
}  // namespace cutlass::xe4_grouped_gemm::collective

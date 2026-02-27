#pragma once

#include "cute/layout.hpp"
#include "cute/tensor.hpp"
#include "cutlass/fast_math.h"
#include "cutlass/gemm/kernel/tile_scheduler_params.h"
#include "cutlass/gemm_coord.hpp"
#include "cutlass/kernel_hardware_info.hpp"

using namespace cutlass::gemm;
using namespace cutlass::gemm::kernel;
using namespace cutlass::gemm::kernel::detail;

namespace cutlass::xe4_grouped_gemm::kernel {

template <class ProblemShape_>
struct MoEProblemShape {
  using UnderlyingProblemShape = ProblemShape_;
  int32_t num_groups = 1;
  const int* problem_shapes = nullptr;
  int32_t gemm_n = 0;
  int32_t gemm_k = 0;

  CUTLASS_HOST_DEVICE
  int32_t groups() const {
    return num_groups;
  }

  CUTLASS_HOST_DEVICE
  UnderlyingProblemShape const get_problem_shape(int32_t group_idx) const {
    return {problem_shapes[group_idx], gemm_n, gemm_k};
  }

  constexpr inline bool is_host_problem_shape_available() {
    return false;
  }
};

template <class GroupProblemShape>
class PersistentTileSchedulerXe4Group {
 private:
  uint64_t current_work_linear_idx_ = 0;
  uint64_t total_grid_size_ = 0;

  // Tracking current group, its starting linear idx and total tiles
  struct GroupInfo {
    int group_idx = 0;
    uint64_t start_linear_idx = 0;
    uint64_t total_tiles = 0;
  } current_group_info_;

 public:
  struct WorkTileInfo {
    int32_t M_idx = 0;
    int32_t N_idx = 0;
    int32_t L_idx = 0;
    bool is_valid_tile = false;

    CUTLASS_HOST_DEVICE
    bool is_valid() const {
      return is_valid_tile;
    }

    CUTLASS_HOST_DEVICE
    static WorkTileInfo invalid_work_tile() {
      return {-1, -1, -1, false};
    }
  };

  static constexpr size_t SmemAlignment = 512;
  struct SLMStorage : cute::aligned_struct<SmemAlignment, _0> {
    cute::array_aligned<int32_t, sizeof(int32_t), SmemAlignment> smem_counter;
  };

  using ProblemShape = typename GroupProblemShape::UnderlyingProblemShape;
  using GroupParams = PersistentTileSchedulerSm90GroupParams<GroupProblemShape>;
  using RasterOrder = typename GroupParams::RasterOrder;
  using RasterOrderOptions = typename GroupParams::RasterOrderOptions;

  struct Arguments {
    int max_swizzle_size = 1;
    // Not applying Heuristics for Grouped problems, since largest dimension can
    // change per group
    RasterOrderOptions raster_order = RasterOrderOptions::AlongM;  // AloneN by input
    uint64_t* global_task_counter = nullptr;
  };

  struct Params {
    GroupParams group_params;
    uint64_t* global_task_counter_ = nullptr;
  };

  Params scheduler_params;

  //
  // Methods
  //

  template <class TileShape, class ClusterShape>
  static Params to_underlying_arguments(
      GroupProblemShape problem_shapes,
      TileShape tile_shape,
      ClusterShape cluster_shape,
      KernelHardwareInfo const& hw_info,
      Arguments const& arguments) {
    // We only need the tile and cluster shape during scheduler setup
    static_assert(cute::is_static<TileShape>::value);
    static_assert(cute::is_static<ClusterShape>::value);

    dim3 problem_blocks = get_tiled_cta_shape_mnl(hw_info, tile_shape, cluster_shape);

    Params params;
    params.group_params.initialize(
        problem_blocks,
        problem_shapes,
        to_gemm_coord(tile_shape),
        to_gemm_coord(cluster_shape),
        hw_info,
        arguments.max_swizzle_size,
        arguments.raster_order);

    params.global_task_counter_ = arguments.global_task_counter;

    return params;
  }

  // Given the inputs, computes the physical grid we should launch.
  template <class TileShape, class ClusterShape>
  CUTLASS_HOST_DEVICE static dim3 get_grid_shape(
      Params const& params,
      GroupProblemShape problem_shapes,
      TileShape tile_shape,
      ClusterShape cluster_shape,
      KernelHardwareInfo hw_info,
      Arguments arguments,
      bool truncate_by_problem_size = true) {
    dim3 problem_blocks = get_tiled_cta_shape_mnl(hw_info, tile_shape, cluster_shape);

    return GroupParams::get_grid_shape(
        problem_blocks,
        to_gemm_coord(cluster_shape),
        hw_info,
        arguments.max_swizzle_size,
        arguments.raster_order,
        true);
  }

  // Given the inputs, computes the total number of output blocks this problem
  // will compute over Note that this is only the logical size of our grid, not
  // the physical grid we will actually launch.
  template <class BlockShape, class ClusterShape>
  CUTLASS_HOST_DEVICE static dim3
  get_tiled_cta_shape_mnl(KernelHardwareInfo hw_info, BlockShape cta_shape, ClusterShape cluster_shape) {
    uint32_t total_ctas = 0;
    uint32_t cta_in_N_dim = 1;  // We linearize the blocks across all the problems here

    total_ctas = hw_info.sm_count;

    return GroupParams::get_tiled_cta_shape_mnl(to_gemm_coord(cluster_shape), total_ctas, cta_in_N_dim);
  }

  PersistentTileSchedulerXe4Group() = default;

  CUTLASS_DEVICE explicit PersistentTileSchedulerXe4Group(Params const& params_) : scheduler_params(params_) {
    if (scheduler_params.group_params.raster_order_ == RasterOrder::AlongN) {
      current_work_linear_idx_ = uint64_t(BlockIdxX()) + uint64_t(BlockIdxY()) * uint64_t(GridDimX());
    } else {
      current_work_linear_idx_ = uint64_t(BlockIdxX()) * uint64_t(GridDimY()) + uint64_t(BlockIdxY());
    }

    total_grid_size_ = uint64_t(GridDimX()) * uint64_t(GridDimY()) * uint64_t(GridDimZ());

    uint64_t ctas_along_m, ctas_along_n;
    if (is_tuple<decltype(cute::shape<0>(params_.group_params.problem_shapes_.get_problem_shape(0)))>::value ||
        is_tuple<decltype(cute::shape<1>(params_.group_params.problem_shapes_.get_problem_shape(0)))>::value) {
      ctas_along_m = cute::size(cute::ceil_div(
          cute::shape<0>(params_.group_params.problem_shapes_.get_problem_shape(0)),
          scheduler_params.group_params.cta_shape_.m()));
      ctas_along_n = cute::size(cute::ceil_div(
          cute::shape<1>(params_.group_params.problem_shapes_.get_problem_shape(0)),
          scheduler_params.group_params.cta_shape_.n()));
    } else {
      ctas_along_m = scheduler_params.group_params.divmod_cta_shape_m_.divide(
          cute::shape<0>(params_.group_params.problem_shapes_.get_problem_shape(0)) +
          scheduler_params.group_params.divmod_cta_shape_m_.divisor - 1);
      ctas_along_n = scheduler_params.group_params.divmod_cta_shape_n_.divide(
          cute::shape<1>(params_.group_params.problem_shapes_.get_problem_shape(0)) +
          scheduler_params.group_params.divmod_cta_shape_n_.divisor - 1);
    }
    auto problem_blocks_m =
        round_up(ctas_along_m, (1 << params_.group_params.log_swizzle_size_) * params_.group_params.cluster_shape_.m());
    auto problem_blocks_n =
        round_up(ctas_along_n, (1 << params_.group_params.log_swizzle_size_) * params_.group_params.cluster_shape_.n());
    current_group_info_.total_tiles = problem_blocks_m * problem_blocks_n;
  }

  CUTLASS_DEVICE
  WorkTileInfo get_current_work() {
    return get_current_work_for_linear_idx(current_work_linear_idx_);
  }

  CUTLASS_DEVICE
  WorkTileInfo get_current_work_for_linear_idx(uint64_t linear_idx) {
    if (scheduler_params.group_params.pre_processed_problem_shapes &&
        linear_idx >= scheduler_params.group_params.blocks_across_problem_) {
      return WorkTileInfo::invalid_work_tile();
    }

    return get_work_idx_m_and_n(
        linear_idx,
        current_group_info_,
        scheduler_params.group_params.problem_shapes_,
        scheduler_params.group_params.cta_shape_,
        scheduler_params.group_params.cluster_shape_,
        scheduler_params.group_params.divmod_cluster_shape_major_,
        scheduler_params.group_params.divmod_cluster_shape_minor_,
        scheduler_params.group_params.divmod_cta_shape_m_,
        scheduler_params.group_params.divmod_cta_shape_n_,
        scheduler_params.group_params.log_swizzle_size_,
        scheduler_params.group_params.raster_order_);
  }

  CUTLASS_DEVICE
  void advance_to_next_work(int32_t* shared_counter = nullptr, uint32_t advance_count = 1) {
    auto item = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    item.barrier(sycl::access::fence_space::local_space);
    if (item.get_local_linear_id() == 0) {
      shared_counter[0] = cutlass::atomicAdd(scheduler_params.global_task_counter_, 1UL * advance_count);
      shared_counter[1] = advance_count;
    }
    item.barrier(sycl::access::fence_space::local_space);
  }

  // get work_idx_m, work_idx_n from linear_idx while applying swizzle
  static CUTLASS_DEVICE WorkTileInfo get_work_idx_m_and_n(
      uint64_t linear_idx,
      struct GroupInfo& group_info,
      GroupProblemShape& problem_shapes,
      GemmCoord cta_shape,
      GemmCoord cluster_shape,
      FastDivmodU64Pow2 const& divmod_cluster_shape_major,
      FastDivmodU64Pow2 const& divmod_cluster_shape_minor,
      FastDivmodU64 const& divmod_cta_shape_m,
      FastDivmodU64 const& divmod_cta_shape_n,
      int32_t log_swizzle_size,
      RasterOrder raster_order) {
    bool valid_tile = true;
    uint64_t ctas_along_m, ctas_along_n;
    int total_problem_groups = problem_shapes.groups();

    // Get the current group info
    if (is_tuple<decltype(cute::shape<0>(problem_shapes.get_problem_shape(group_info.group_idx)))>::value ||
        is_tuple<decltype(cute::shape<1>(problem_shapes.get_problem_shape(group_info.group_idx)))>::value) {
      ctas_along_m = cute::size(
          cute::ceil_div(cute::shape<0>(problem_shapes.get_problem_shape(group_info.group_idx)), cta_shape.m()));
      ctas_along_n = cute::size(
          cute::ceil_div(cute::shape<1>(problem_shapes.get_problem_shape(group_info.group_idx)), cta_shape.n()));
    } else {
      ctas_along_m = divmod_cta_shape_m.divide(
          cute::shape<0>(problem_shapes.get_problem_shape(group_info.group_idx)) + divmod_cta_shape_m.divisor - 1);
      ctas_along_n = divmod_cta_shape_n.divide(
          cute::shape<1>(problem_shapes.get_problem_shape(group_info.group_idx)) + divmod_cta_shape_n.divisor - 1);
    }
    auto problem_blocks_m = round_up(ctas_along_m, (1 << log_swizzle_size) * cluster_shape.m());
    auto problem_blocks_n = round_up(ctas_along_n, (1 << log_swizzle_size) * cluster_shape.n());
    group_info.total_tiles = problem_blocks_m * problem_blocks_n;

    // Search for the target group_idx and update the group_info during it
    while (group_info.start_linear_idx + group_info.total_tiles <= linear_idx) {
      group_info.group_idx++;

      if (group_info.group_idx >= total_problem_groups) return WorkTileInfo::invalid_work_tile();

      group_info.start_linear_idx += group_info.total_tiles;
      if (is_tuple<decltype(cute::shape<0>(problem_shapes.get_problem_shape(group_info.group_idx)))>::value ||
          is_tuple<decltype(cute::shape<1>(problem_shapes.get_problem_shape(group_info.group_idx)))>::value) {
        ctas_along_m = cute::size(
            cute::ceil_div(cute::shape<0>(problem_shapes.get_problem_shape(group_info.group_idx)), cta_shape.m()));
        ctas_along_n = cute::size(
            cute::ceil_div(cute::shape<1>(problem_shapes.get_problem_shape(group_info.group_idx)), cta_shape.n()));
      } else {
        ctas_along_m = divmod_cta_shape_m.divide(
            cute::shape<0>(problem_shapes.get_problem_shape(group_info.group_idx)) + divmod_cta_shape_m.divisor - 1);
        ctas_along_n = divmod_cta_shape_n.divide(
            cute::shape<1>(problem_shapes.get_problem_shape(group_info.group_idx)) + divmod_cta_shape_n.divisor - 1);
      }
      problem_blocks_m = round_up(ctas_along_m, (1 << log_swizzle_size) * cluster_shape.m());
      problem_blocks_n = round_up(ctas_along_n, (1 << log_swizzle_size) * cluster_shape.n());
      group_info.total_tiles = problem_blocks_m * problem_blocks_n;
    }

    uint64_t cluster_id, cluster_major_offset = 0, cluster_minor_offset = 0;
    uint64_t blk_per_grid_dim = divmod_cluster_shape_minor.divide(linear_idx - group_info.start_linear_idx);
    divmod_cluster_shape_major(cluster_id, cluster_major_offset, blk_per_grid_dim);

    // With static schedulers, we launch grid such that all cluster are linear
    // (1-D) order, i.e., there can only be one cluster in the minor dimension.
    // get_grid_shape() in scheduler params put cluster_shape.m/n() as the minor
    // dimension based on raster order AlongN/M resp. Therefore, the offset of a
    // CTA (inside a cluster) in the minor dimension can be directly be inferred
    // by the blockIdx along the minor dimension.
    if (raster_order == RasterOrder::AlongN) {
      cluster_minor_offset = BlockIdxX();
    } else {
      cluster_minor_offset = BlockIdxY();
    }

    uint64_t cluster_idx_minor, cluster_idx_major;

    uint64_t cluster_idx_minor_div_swizzle, extra, offset;

    offset = cluster_id & ((1 << log_swizzle_size) - 1);
    extra = cluster_id >> log_swizzle_size;

    uint64_t curr_group_cluster_blk_major;
    if (raster_order == RasterOrder::AlongN) {
      curr_group_cluster_blk_major = divmod_cluster_shape_major.divide(problem_blocks_n);
    } else {
      curr_group_cluster_blk_major = divmod_cluster_shape_major.divide(problem_blocks_m);
    }
    cluster_idx_minor_div_swizzle = extra / curr_group_cluster_blk_major;
    cluster_idx_major = extra % curr_group_cluster_blk_major;

    cluster_idx_minor = cluster_idx_minor_div_swizzle * (1 << log_swizzle_size) + offset;

    auto minor_work_idx =
        static_cast<int32_t>(cluster_idx_minor * divmod_cluster_shape_minor.divisor + cluster_minor_offset);
    auto major_work_idx =
        static_cast<int32_t>(cluster_idx_major * divmod_cluster_shape_major.divisor + cluster_major_offset);

    if (raster_order == RasterOrder::AlongN) {
      return {minor_work_idx, major_work_idx, group_info.group_idx, valid_tile};
    } else {
      return {major_work_idx, minor_work_idx, group_info.group_idx, valid_tile};
    }
  }

  // Kernel helper function to get next work tile
  CUTLASS_DEVICE
  WorkTileInfo
  fetch_next_work(WorkTileInfo work_tile_info, int32_t* shared_counter = nullptr, uint32_t advance_count = 1) {
    auto item = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    if (shared_counter[1] > 0) {
      current_work_linear_idx_ = total_grid_size_ + shared_counter[0] - shared_counter[1];
      item.barrier(sycl::access::fence_space::local_space);
      if (item.get_local_linear_id() == 0) {
        shared_counter[1]--;
      }
      item.barrier(sycl::access::fence_space::local_space);
    } else {
      advance_to_next_work(shared_counter, advance_count);
      current_work_linear_idx_ = total_grid_size_ + shared_counter[0] - shared_counter[1];
      if (item.get_local_linear_id() == 0) {
        shared_counter[1]--;
      }
      item.barrier(sycl::access::fence_space::local_space);
    }
    auto next_work = get_current_work();
    return next_work;
  }

  // Returns the initial work tile info that will be computed over
  template <class ClusterShape>
  CUTLASS_DEVICE WorkTileInfo initial_work_tile_info(ClusterShape) {
    return get_current_work();
  }
};
}  // namespace cutlass::xe4_grouped_gemm::kernel

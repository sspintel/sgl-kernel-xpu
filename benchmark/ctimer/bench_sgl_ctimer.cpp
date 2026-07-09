// One-time C++ device-clock benchmark for sgl-kernel-xpu `moe_grouped_mm_nt_xe35`.
//
// WHY THIS EXISTS
// ---------------
// PERF_grouped_gemm_xe35.md times the kernels through torch.xpu.Event. On the
// CRI functional simulator the PyTorch XPU stream queue is created WITHOUT
// sycl::property::queue::enable_profiling, so torch cannot read device event
// timestamps and its Event.elapsed_time() degrades to host-side sync timing —
// dominated by host/sim overhead, hence inaccurate as a kernel-perf metric.
//
// This tool instead uses the same primitive CUTLASS/sycl-tla's GPU_Clock uses:
// SYCL device-event profiling (command_start / command_end timestamps). We run
// the prebuilt sgl launcher on our OWN profiling+in-order queue and bracket the
// launch with ext_oneapi_submit_barrier() events, reading the ns-resolution
// device timestamps between them. (The bf16 sgl kernel is a raw q.submit that
// discards its event, so the EventManager path GPU_Clock relies on never sees
// it; barrier bracketing is the event-accurate equivalent.)
//
// Standalone on purpose: it links the prebuilt per-tile .so files directly, so
// it needs no rebuild of sgl-kernel-xpu, and each library runs in its own
// process (the sgl and vllm SYCL runtimes cannot coexist in one process on the
// sim — the documented "dangling queue" abort).
//
// Layout: NT — B is [E, N, K], computes A[M,K] @ B[N,K]^T -> O[M,N].
// Pure bf16 GEMM (fuse_act=false) so FLOPs == 2*M*N*K, comparable to vllm.

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cute/tensor.hpp>
#include <string>
#include <sycl/sycl.hpp>
#include <vector>

// cute/print.hpp does `#define printf sycl::...::printf` for device-side
// formatted print; undo it so host std::printf / std::fprintf work normally.
#ifdef printf
#undef printf
#endif

namespace MoE {
// Mirror of ActivationType in moe_mainloop.hpp (only SILU needed for pure GEMM).
enum class ActivationType { SILU = 0, GELU = 1, SWIGLU_GPT_OSS = 2, RELU2 = 3 };
}  // namespace MoE

using namespace cute;

// Prebuilt launcher, declared exactly as src/sycl/kernels/moe/xe20/GroupGemm.hpp
// declares it; the definitions live in the linked GroupGemmSIMD_inst_*.so files.
template <typename Tile, typename SGLayout, MoE::ActivationType ActType, bool FuseAct, bool WithBias>
void Xe20MoEGEMMLauncher(
    sycl::queue q,
    const void* activations,
    const void* weights,
    const void* scales,
    const void* bias,
    void* outputs,
    const int gemm_n,
    const int gemm_k,
    const int* num_rows_per_expert_device,
    const int num_experts,
    int* workspace,
    float gemm1_alpha,
    float gemm1_limit,
    int ld_b);

// ---- Tile / subgroup-layout aliases (match GroupGemm.hpp) --------------------
using Tile_8_64_32 = Shape<_8, _64, _32>;
using Tile_16_64_32 = Shape<_16, _64, _32>;
using Tile_32_64_32 = Shape<_32, _64, _32>;
using Tile_128_64_32 = Shape<_128, _64, _32>;
using Tile_128_128_32 = Shape<_128, _128, _32>;
using Tile_256_256_32 = Shape<_256, _256, _32>;
using SG_1_4_1 = Layout<Shape<_1, _4, _1>, Stride<_4, _1, _0>>;
using SG_4_2_1 = Layout<Shape<_4, _2, _1>, Stride<_2, _1, _0>>;
using SG_8_4_1 = Layout<Shape<_8, _4, _1>, Stride<_4, _1, _0>>;

using bf16 = sycl::ext::oneapi::bfloat16;

// A callable that launches the correct instantiated tile for a given launch.
using LaunchFn = void (*)(sycl::queue&, const bf16*, const bf16*, bf16*, const int*, int, int, int, int*, int);

template <typename Tile, typename SG>
static void launch_tile(
    sycl::queue& q, const bf16* A, const bf16* B, bf16* D, const int* rows, int E, int N, int K, int* ws, int ld_b) {
  Xe20MoEGEMMLauncher<Tile, SG, MoE::ActivationType::SILU, /*FuseAct=*/false, /*WithBias=*/false>(
      q, A, B, /*scales=*/nullptr, /*bias=*/nullptr, D, N, K, rows, E, ws, 1.702f, 7.0f, ld_b);
}

// Tile selection for pure-bf16 unfused GEMM, mirroring the dispatch in
// moe_grouped_mm_nt_xe20() for the benchmark's shapes (both are "small_weight"
// with K > 256, so the small_weight / non-narrow branches apply).
static int g_force_tile = -1;  // -1 = auto, 0..4 = force specific tile

static LaunchFn pick_launch(int avg_m, int gemm_n, int gemm_k) {
  if (g_force_tile >= 0) {
    switch (g_force_tile) {
      case 0:
        return &launch_tile<Tile_8_64_32, SG_1_4_1>;
      case 1:
        return &launch_tile<Tile_16_64_32, SG_1_4_1>;
      case 2:
        return &launch_tile<Tile_32_64_32, SG_1_4_1>;
      case 3:
        return &launch_tile<Tile_128_128_32, SG_4_2_1>;
      case 4:
        return &launch_tile<Tile_256_256_32, SG_8_4_1>;
      case 5:
        return &launch_tile<Tile_128_64_32, SG_4_2_1>;
    }
  }
  const int64_t SMALL = int64_t(4096) * 4096;
  bool small_weight = (int64_t)gemm_k * gemm_n <= SMALL;
  bool narrow_k = gemm_k <= 256;
  if (avg_m <= 8) return &launch_tile<Tile_8_64_32, SG_1_4_1>;
  if (avg_m <= 128 && small_weight) {
    if (avg_m >= 64 && gemm_k <= 1024) return &launch_tile<Tile_256_256_32, SG_8_4_1>;
    return &launch_tile<Tile_128_128_32, SG_4_2_1>;
  }
  if (narrow_k) return &launch_tile<Tile_128_128_32, SG_4_2_1>;
  return &launch_tile<Tile_256_256_32, SG_8_4_1>;
}

struct Config {
  const char* label;
  int n, k;
};

int main(int argc, char** argv) {
  // Defaults mirror PERF_grouped_gemm_xe35.md.
  int E = 8;
  std::vector<int> avg_ms = {1, 4, 8, 32, 64, 128};
  int iters = 3, warmup = 1;
  std::vector<Config> configs = {{"gemm1_gate_up", 1536, 2048}, {"gemm2_down", 2048, 768}};

  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    auto next = [&]() { return std::atoi(argv[++i]); };
    if (a == "--experts")
      E = next();
    else if (a == "--iters")
      iters = next();
    else if (a == "--warmup")
      warmup = next();
    else if (a == "--force-tile")
      g_force_tile = next();
    else if (a == "--avg-m") {
      avg_ms.clear();
      while (i + 1 < argc && argv[i + 1][0] != '-')
        avg_ms.push_back(std::atoi(argv[++i]));
    }
  }

  sycl::queue q{
      sycl::gpu_selector_v,
      sycl::property_list{sycl::property::queue::enable_profiling{}, sycl::property::queue::in_order{}}};
  std::fprintf(
      stderr,
      "[sgl-ctimer] device: %s  profiling=%d\n",
      q.get_device().get_info<sycl::info::device::name>().c_str(),
      q.has_property<sycl::property::queue::enable_profiling>());

  // JSON-ish output for the merge step; human table goes to stderr.
  std::printf("{\"which\":\"sgl\",\"experts\":%d,\"results\":{", E);
  bool first_label = true;

  for (const auto& c : configs) {
    std::fprintf(stderr, "\n%s: N=%d K=%d\n", c.label, c.n, c.k);
    std::fprintf(stderr, "%6s %8s %13s %10s\n", "avg_m", "total_m", "TFLOP/s", "ms");
    if (!first_label) std::printf(",");
    first_label = false;
    std::printf("\"%s\":{", c.label);

    bool first_m = true;
    for (int avg_m : avg_ms) {
      int total_m = E * avg_m;
      // Uniform grouped GEMM: every expert gets avg_m rows.
      bf16* A = sycl::malloc_device<bf16>((size_t)total_m * c.k, q);
      bf16* B = sycl::malloc_device<bf16>((size_t)E * c.n * c.k, q);  // NT: [E, N, K]
      bf16* D = sycl::malloc_device<bf16>((size_t)total_m * c.n, q);
      int* rows = sycl::malloc_device<int>(E, q);
      int* ws = sycl::malloc_device<int>(1, q);
      q.fill(A, bf16(0.01f), (size_t)total_m * c.k);
      q.fill(B, bf16(0.01f), (size_t)E * c.n * c.k);
      std::vector<int> h(E, avg_m);
      q.memcpy(rows, h.data(), E * sizeof(int));
      q.wait();

      LaunchFn fn = pick_launch(avg_m, c.n, c.k);
      int ld_b = c.k;  // B is [E, N, K] row-major -> stride(N-dim) = K

      for (int w = 0; w < warmup; ++w)
        fn(q, A, B, D, rows, E, c.n, c.k, ws, ld_b);
      q.wait();

      double best = 1e30, tot = 0;
      for (int it = 0; it < iters; ++it) {
        auto b0 = q.ext_oneapi_submit_barrier();
        fn(q, A, B, D, rows, E, c.n, c.k, ws, ld_b);
        auto b1 = q.ext_oneapi_submit_barrier();
        q.wait();
        auto end0 = b0.get_profiling_info<sycl::info::event_profiling::command_end>();
        auto start1 = b1.get_profiling_info<sycl::info::event_profiling::command_start>();
        double ms = double(start1 - end0) * 1e-6;
        tot += ms;
        if (ms < best) best = ms;
      }
      double avg = tot / iters;
      double tflops = (2.0 * total_m * c.n * c.k) * 1e-12 / (avg * 1e-3);
      std::fprintf(stderr, "%6d %8d %13.3f %10.4f\n", avg_m, total_m, tflops, avg);

      if (!first_m) std::printf(",");
      first_m = false;
      // seconds/iter, matching the python script's stored units.
      std::printf("\"%d\":{\"avg_s\":%.9g,\"best_s\":%.9g}", avg_m, avg * 1e-3, best * 1e-3);
      std::fflush(stdout);

      sycl::free(A, q);
      sycl::free(B, q);
      sycl::free(D, q);
      sycl::free(rows, q);
      sycl::free(ws, q);
    }
    std::printf("}");
  }
  std::printf("}}\n");
  return 0;
}

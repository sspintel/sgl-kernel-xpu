// One-time C++ device-clock benchmark for vllm-xpu-kernels grouped GEMM
// (cutlass_grouped_gemm_xe3, bf16). Companion to bench_sgl_ctimer.cpp — same
// SYCL device-event timing (command_start / command_end), same shapes, so the
// TFLOP/s numbers are directly comparable.
//
// See bench_sgl_ctimer.cpp for the rationale (torch.xpu.Event is inaccurate on
// the CRI sim because the torch XPU queue lacks enable_profiling). Here we call
// the prebuilt gpu::cutlass_kernel::grouped_gemm::kernel_functor<policy> on our
// own profiling+in-order queue and bracket it with barrier events.
//
// Layout: NN — B is [E, K, N], computes A[M,K] @ B[K,N] -> O[M,N].
// FLOPs == 2*M*N*K, identical to the sgl NT path.
//
// MUST be run in its own process, never concurrently with the sgl binary: both
// drive the same simulated device, and the two SYCL runtimes cannot coexist.

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <sycl/sycl.hpp>
#include <vector>

// Prebuilt kernel entry points (definitions in libgrouped_gemm_xe_3.so).
// The policy types are only needed as names for mangling, so forward-declare
// them as incomplete structs.
namespace gpu {
namespace cutlass_kernel {
namespace grouped_gemm {
struct moe_bf16_policy;
struct moe_bf16_256x128_policy;
struct moe_bf16_128x256_policy;
struct moe_bf16_128x128_policy;
struct moe_bf16_mid_policy;
struct moe_bf16_decode_policy;
struct moe_bf16_decode_k64_policy;

template <class P>
void kernel_functor(sycl::queue&, void*, void*, void*, void*, void*, void*, void*, int64_t, int64_t, int64_t);

#define EXTERN_POLICY(P)                  \
  extern template void kernel_functor<P>( \
      sycl::queue&, void*, void*, void*, void*, void*, void*, void*, int64_t, int64_t, int64_t);
EXTERN_POLICY(moe_bf16_policy)
EXTERN_POLICY(moe_bf16_256x128_policy)
EXTERN_POLICY(moe_bf16_128x256_policy)
EXTERN_POLICY(moe_bf16_128x128_policy)
EXTERN_POLICY(moe_bf16_mid_policy)
EXTERN_POLICY(moe_bf16_decode_policy)
EXTERN_POLICY(moe_bf16_decode_k64_policy)
#undef EXTERN_POLICY
}  // namespace grouped_gemm
}  // namespace cutlass_kernel
}  // namespace gpu

using namespace gpu::cutlass_kernel::grouped_gemm;
using bf16 = sycl::ext::oneapi::bfloat16;

using LaunchFn = void (*)(sycl::queue&, const bf16*, const bf16*, bf16*, const int*, int, int, int);

template <class Policy>
static void launch_policy(sycl::queue& q, const bf16* A, const bf16* B, bf16* D, const int* rows, int N, int K, int E) {
  kernel_functor<Policy>(q, (void*)A, nullptr, (void*)B, nullptr, nullptr, (void*)D, (void*)rows, N, K, E);
}

// Prefill tile pick, replicating grouped_gemm::pick_prefill_tile for BF16
// (kMinUtil = 0.90, kCores = 32). Only 256x256 vs 128x256 are chosen.
static LaunchFn pick_prefill_bf16(int64_t M_total, int64_t N, int64_t groups) {
  if (const char* env = std::getenv("XE3_GG_FORCE_TILE")) {
    switch (std::atoi(env)) {
      case 1:
        return &launch_policy<moe_bf16_256x128_policy>;
      case 2:
        return &launch_policy<moe_bf16_128x256_policy>;
      case 3:
        return &launch_policy<moe_bf16_128x128_policy>;
      default:
        return &launch_policy<moe_bf16_policy>;
    }
  }
  const int kCores = 32;
  const double kMinUtil = 0.90;
  int64_t M_g = M_total / (groups > 0 ? groups : 1);
  auto cdiv = [](int64_t a, int64_t b) { return (a + b - 1) / b; };
  int64_t tiles_256 = groups * cdiv(M_g > 0 ? M_g : 1, 256) * cdiv(N, 256);
  int64_t waves_256 = cdiv(tiles_256, kCores);
  double util_256 = double(tiles_256) / double(waves_256 * kCores);
  return util_256 >= kMinUtil ? &launch_policy<moe_bf16_policy> : &launch_policy<moe_bf16_128x256_policy>;
}

// Full bf16 dispatch, mirroring grouped_gemm_func().
static LaunchFn pick_launch(int avg_m, int64_t N, int64_t K, int64_t E) {
  if (avg_m > 32) return pick_prefill_bf16(E * avg_m, N, E);
  if (avg_m > 4) return &launch_policy<moe_bf16_mid_policy>;
  if (K >= 1024) return &launch_policy<moe_bf16_decode_k64_policy>;
  return &launch_policy<moe_bf16_decode_policy>;
}

struct Config {
  const char* label;
  int n, k;
};

int main(int argc, char** argv) {
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
      "[vllm-ctimer] device: %s  profiling=%d\n",
      q.get_device().get_info<sycl::info::device::name>().c_str(),
      q.has_property<sycl::property::queue::enable_profiling>());

  std::printf("{\"which\":\"vllm\",\"experts\":%d,\"results\":{", E);
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
      bf16* A = sycl::malloc_device<bf16>((size_t)total_m * c.k, q);
      bf16* B = sycl::malloc_device<bf16>((size_t)E * c.k * c.n, q);  // NN: [E, K, N]
      bf16* D = sycl::malloc_device<bf16>((size_t)total_m * c.n, q);
      int* rows = sycl::malloc_device<int>(E, q);
      q.fill(A, bf16(0.01f), (size_t)total_m * c.k);
      q.fill(B, bf16(0.01f), (size_t)E * c.k * c.n);
      std::vector<int> h(E, avg_m);
      q.memcpy(rows, h.data(), E * sizeof(int));
      q.wait();

      LaunchFn fn = pick_launch(avg_m, c.n, c.k, E);

      for (int w = 0; w < warmup; ++w)
        fn(q, A, B, D, rows, c.n, c.k, E);
      q.wait();

      double best = 1e30, tot = 0;
      for (int it = 0; it < iters; ++it) {
        auto b0 = q.ext_oneapi_submit_barrier();
        fn(q, A, B, D, rows, c.n, c.k, E);
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
      std::printf("\"%d\":{\"avg_s\":%.9g,\"best_s\":%.9g}", avg_m, avg * 1e-3, best * 1e-3);
      std::fflush(stdout);

      sycl::free(A, q);
      sycl::free(B, q);
      sycl::free(D, q);
      sycl::free(rows, q);
    }
    std::printf("}");
  }
  std::printf("}}\n");
  return 0;
}

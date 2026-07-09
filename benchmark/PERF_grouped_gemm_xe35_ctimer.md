# Grouped GEMM performance (C++ device-clock): sgl-kernel-xpu vs vllm-xpu-kernels (Xe3)

Same comparison as [PERF_grouped_gemm_xe35.md](PERF_grouped_gemm_xe35.md), but timed
with a **C++ SYCL device clock** instead of `torch.xpu.Event`, because the latter is
inaccurate on the CRI functional simulator.

## Why a C++ timer

`torch.xpu.Event(enable_timing=True).elapsed_time()` cannot read device timestamps
here: PyTorch's XPU stream queue is created **without**
`sycl::property::queue::enable_profiling` (verified: `has_property<enable_profiling>()
== 0`, although the CRI device advertises `aspect::queue_profiling == 1`). With no
device-side profiling, its elapsed-time degrades to a **host-side queue sync**, which
on the simulator is dominated by host/sim overhead — not kernel time.

CUTLASS/sycl-tla's own timer (`GPU_Clock` → `SYCLTimer` → `EventManager`) avoids this
by reading SYCL event profiling timestamps
(`sycl::info::event_profiling::command_start` / `command_end`, nanosecond device
clock). This benchmark uses that exact primitive:

- Create a `sycl::queue` with `enable_profiling` **on PyTorch's own SYCL
  context/device** is not needed here — the tools are standalone and create a fresh
  profiling+in-order queue on the CRI device directly.
- Bracket each kernel launch with `queue.ext_oneapi_submit_barrier()` events and read
  `command_start(after) − command_end(before)`. (The sgl bf16 kernel is a raw
  `q.submit` that discards its event, so CUTLASS's `EventManager` never sees it;
  barrier bracketing is the device-accurate equivalent of `GPU_Clock`.)

The two libraries' kernels each take a `sycl::queue` argument, so the standalone tools
link the **prebuilt** kernel `.so` files and run them on the profiling queue — no
rebuild of either project is required.

## Layout / FLOPs (unchanged from the original doc)

| | sgl-kernel-xpu | vllm-xpu-kernels |
|---|---|---|
| Op | `Xe20MoEGEMMLauncher<...>` (behind `moe_grouped_mm_nt_xe35`) | `grouped_gemm::kernel_functor<moe_bf16_*_policy>` (xe3) |
| Layout | NT — B is `[E, N, K]`, `A[M,K] @ B[N,K]ᵀ` | NN — B is `[E, K, N]`, `A[M,K] @ B[K,N]` |
| Math | pure bf16 GEMM, `fuse_act=false`, no scales | pure bf16 GEMM |

Both do `2·M·N·K` FLOPs, so TFLOP/s is directly comparable. Tile/policy selection in
each tool mirrors the production dispatch logic (sgl: `moe_grouped_mm_nt_xe20()`
avg_m branches; vllm: `grouped_gemm_func()` prefill/mid/decode + `pick_prefill_tile`).

## Environment

- Device: `Intel(R) Graphics [0x674c]` — CRI functional simulator
- Compiler: `icpx` (Intel oneAPI DPC++ 2026.2)
- sgl-kernel-xpu: branch `rebase_up_0.2_2` @ `fe33d4b`
- vllm-xpu-kernels: `libgrouped_gemm_xe_3.so` (prebuilt)
- Shapes: Qwen3-30B-A3B MoE, **E=8** (sgl requires n_experts % 8 == 0)
  - `gemm1_gate_up`: N=1536, K=2048
  - `gemm2_down`:    N=2048, K=768
- `total_m = E · avg_m`

## Reproduce

```bash
cd benchmark/ctimer
./run_ctimer_bench.sh all          # build + run sgl, then vllm, then report
# or step by step:
./run_ctimer_bench.sh build
./run_ctimer_bench.sh sgl          # writes out/sgl.json
./run_ctimer_bench.sh vllm         # writes out/vllm.json
./run_ctimer_bench.sh report       # prints the comparison table
```

Env overrides: `EXPERTS`, `ITERS`, `WARMUP`, `AVG_M` (e.g. `AVG_M="1 8 64 128"`).

> **The sgl and vllm binaries are always run one at a time, never concurrently.**
> They drive the same simulated device and the two SYCL runtimes cannot coexist in a
> process; overlapping them would contend for the simulator and corrupt timings. The
> driver's `all` target runs sgl to completion before starting vllm.

## Results (E=8, bf16, fuse_act=false, C++ SYCL device clock)

`vllm/sgl` = vllm latency ÷ sgl latency (higher ⇒ sgl faster).

<!-- Filled by run_ctimer_bench.sh report; paste out/report.txt tables below. -->

### gemm1_gate_up — N=1536, K=2048

| avg_m | total_m | sgl TFLOP/s | sgl ms | vllm TFLOP/s | vllm ms | vllm/sgl |
|------:|--------:|------------:|-------:|-------------:|--------:|---------:|
| _pending_ | | | | | | |

### gemm2_down — N=2048, K=768

| avg_m | total_m | sgl TFLOP/s | sgl ms | vllm TFLOP/s | vllm ms | vllm/sgl |
|------:|--------:|------------:|-------:|-------------:|--------:|---------:|
| _pending_ | | | | | | |

## Notes / caveats

- **Simulated device clock**, not real silicon. The device-event timestamps are the
  simulator's modeled kernel time — accurate as a *relative* metric between the two
  kernels, but re-run on real Xe3 HW for absolute numbers.
- The C++ device-clock latencies are far more stable than the `torch.xpu.Event`
  numbers (per-iteration spread < 1% on the sim vs. large host-noise in the original).
- Only pure bf16 unfused GEMM is covered; fused activation / quantized paths (fp8 /
  mxfp8 / mxfp4 / int4) and production E=128 are future work.

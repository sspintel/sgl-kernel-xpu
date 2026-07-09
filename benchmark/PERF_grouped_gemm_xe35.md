# Grouped GEMM performance: sgl-kernel-xpu vs vllm-xpu-kernels (Xe3)

Comparison of the two MoE grouped-GEMM implementations on Intel Xe3:

| | sgl-kernel-xpu | vllm-xpu-kernels |
|---|---|---|
| Op | `torch.ops.sgl_kernel.moe_grouped_mm_nt_xe35` | `torch.ops._xpu_C.cutlass_grouped_gemm_interface` (xe3 path) |
| Backend | SYCL kernel (`src/sycl/xe35/GroupGemm.cpp` → `kernels/moe/xe20/GroupGemm.hpp`) | CUTLASS (`cutlass_grouped_gemm_xe3`) |
| Layout | NT — B is `[E, N, K]`, computes `A[M,K] @ B[N,K]ᵀ` | NN — B is `[E, K, N]`, computes `A[M,K] @ B[K,N]` |
| Fusion | optional fused activation (silu/gelu/swiglu) | pure GEMM |
| Constraint | `n_experts` must be a multiple of 8 | — |

To keep the math identical, the comparison runs **pure bf16 GEMM** (sgl `fuse_act=False`, vllm no scales) on the same M/N/K/E, so both perform `2·M·N·K` FLOPs.

## Environment

- Device: `Intel(R) Graphics [0x674c]` — **CRI functional simulator** (`torch.ops._xpu_C.is_cri(0) == True`)
- torch: `2.13.0a0+git5d13517`, conda env `hym`
- sgl-kernel-xpu: branch `rebase_up_0.2_2` @ `fe33d4b` ("enable Xe35 group gemm")
- vllm-xpu-kernels: branch `main` @ `2cc8aa3` ("Remove oneDNN submodule")
- Date: 2026-06-30

## Methodology

- Benchmark script: [bench_grouped_gemm_compare.py](bench_grouped_gemm_compare.py)
- **Device-side latency** via `torch.xpu.Event(enable_timing=True)` + `start.elapsed_time(end)` — the PyTorch equivalent of CUTLASS `GPU_Clock` / `syclEventElapsedTime`. On the simulator this reports simulated *device kernel time*, not host wall-clock (host wall-clock per call is 1–5 min and is meaningless as a perf metric).
- `iterations=1`, `warmup=0` (each sim call is extremely expensive; see caveats).
- Shapes: Qwen3-30B-A3B MoE (E=128 in production; **E=8 here** to satisfy the sgl multiple-of-8 constraint and keep sim time tractable).
  - `gemm1_gate_up`: N=1536, K=2048
  - `gemm2_down`:    N=2048, K=768
- `total_m = E · avg_m`.

### Reproduce

The two libraries **cannot be imported in the same process** (SYCL dangling-queue abort), so each runs separately and results are merged from JSON:

```bash
conda run -n hym python benchmark/bench_grouped_gemm_compare.py --which sgl  --experts 8 \
    --avg-m 1 4 8 32 64 128 --iterations 100 --out sgl.json
conda run -n hym python benchmark/bench_grouped_gemm_compare.py --which vllm --experts 8 \
    --avg-m 1 4 8 32 64 128 --iterations 100 --out vllm.json
conda run -n hym python benchmark/bench_grouped_gemm_compare.py --which merge --experts 8 \
    --avg-m 1 4 8 32 64 128 --sgl-json sgl.json --vllm-json vllm.json
```

## Results (E=8, bf16, fuse_act=False, simulated device clock)

`vllm/sgl` = vllm latency ÷ sgl latency (higher ⇒ sgl faster).

### gemm1_gate_up — N=1536, K=2048

| avg_m | total_m | sgl TFLOP/s | sgl ms | vllm TFLOP/s | vllm ms | vllm/sgl |
|------:|--------:|------------:|-------:|-------------:|--------:|---------:|
| 1     | 8       | 1.549       | 0.0325 | 0.311        | 0.1619  | 4.98×    |
| 4     | 32      | 6.116       | 0.0329 | 1.630        | 0.1235  | 3.75×    |
| 8     | 64      | 11.989      | 0.0336 | 3.117        | 0.1292  | 3.85×    |
| 32    | 256     | 50.200      | 0.0321 | 14.118       | 0.1141  | 3.56×    |
| 64    | 512     | 91.382      | 0.0352 | 26.530       | 0.1214  | 3.44×    |
| 128   | 1024    | 163.100     | 0.0395 | 52.484       | 0.1227  | 3.11×    |

### gemm2_down — N=2048, K=768

| avg_m | total_m | sgl TFLOP/s | sgl ms | vllm TFLOP/s | vllm ms | vllm/sgl |
|------:|--------:|------------:|-------:|-------------:|--------:|---------:|
| 1     | 8       | 1.269       | 0.0198 | 0.228        | 0.1102  | 5.56×    |
| 4     | 32      | 5.054       | 0.0199 | 0.807        | 0.1247  | 6.26×    |
| 8     | 64      | 9.741       | 0.0207 | 1.776        | 0.1133  | 5.48×    |
| 32    | 256     | 40.433      | 0.0199 | 7.967        | 0.1011  | 5.08×    |
| 64    | 512     | 75.203      | 0.0214 | 14.180       | 0.1136  | 5.30×    |
| 128   | 1024    | 115.730     | 0.0278 | 28.215       | 0.1142  | 4.10×    |

## Observations

- **sgl `moe_grouped_mm_nt_xe35` (SYCL) is consistently faster than vllm `cutlass_grouped_gemm_xe3`** across the whole range: ~3.1–5.0× on gemm1, ~4.1–6.3× on gemm2.
- **Both kernels have near-flat device latency vs M** in this band (sgl gemm1 0.032→0.040 ms; vllm gemm1 0.114→0.123 ms), i.e. still grid/overhead-dominated rather than compute-saturated, so TFLOP/s rises almost linearly with `total_m`. Neither has plateaued at total_m=1024, so peak throughput is higher still.
- **Peak observed**: sgl 163 TFLOP/s (gemm1) / 116 TFLOP/s (gemm2) vs vllm 52 / 28 TFLOP/s.
- The gap narrows slightly as M grows (vllm scales marginally better), but sgl retains a large lead throughout.

## Correctness

- sgl xe35 verified directly against an fp32 grouped-GEMM reference: max abs diff ≈ 2e-4 (E=8, bf16).
- vllm xe3 correctness covered by its own CI (`tests/fused_moe/test_grouped_gemm_xe3.py`).

## Caveats / gotchas

1. **Simulated device clock**, not real silicon — treat ratios as directional; re-run on real Xe3 HW with `--iterations 100` to confirm absolute numbers. Single-iteration measurement at tens-of-µs latencies carries overhead noise.
2. **Cannot mix the two libraries in one process** — importing both triggers a SYCL "dangling queue" assertion. Run each `--which` in its own process and `merge` from JSON.
3. **Use `torch.xpu.current_stream().synchronize()`**, not `torch.xpu.synchronize()` — the latter trips the same dangling-queue abort on the vllm path on this simulator.
4. **`n_experts` must be a multiple of 8** for sgl xe35; this fixed E=8 here.
5. Each sim call is very slow (host wall-clock 1–5 min, up to ~25 min at total_m=1024); the script saves results to JSON incrementally per data point and supports `--config <label>` to resume a partial sweep.

## TODO / not yet measured

- Fused-activation path (sgl fuses silu/gelu/swiglu; vllm would need a separate activation kernel — expected to widen sgl's advantage for gemm1_gate_up).
- Production E=128.
- Larger `total_m` to find where each kernel saturates.
- Quantized paths (fp8 / mxfp8 / mxfp4 / int4).
- Real Xe3 hardware numbers.

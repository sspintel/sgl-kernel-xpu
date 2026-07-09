"""Merge sgl / vllm C++ device-clock benchmark JSON into a comparison table.

Mirrors the merge step of benchmark/bench_grouped_gemm_compare.py, but the
latencies come from the standalone C++ tools (SYCL device-event profiling)
instead of torch.xpu.Event, so they are accurate on the CRI simulator.
"""

import argparse
import json

GEMM_CONFIGS = [("gemm1_gate_up", 1536, 2048), ("gemm2_down", 2048, 768)]


def tflops(total_m, n, k, elapsed_s):
    return (2.0 * total_m * n * k) * 1e-12 / elapsed_s


def get_s(d, label, avg_m):
    e = d["results"].get(label, {}).get(str(avg_m))
    if e is None:
        return None
    # C++ tool stores {"avg_s":..., "best_s":...}; accept a bare float too.
    return e["avg_s"] if isinstance(e, dict) else e


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sgl-json", required=True)
    ap.add_argument("--vllm-json", required=True)
    ap.add_argument("--avg-m", type=int, nargs="+", default=[1, 4, 8, 32, 64, 128])
    args = ap.parse_args()

    with open(args.sgl_json) as f:
        sgl = json.load(f)
    with open(args.vllm_json) as f:
        vllm = json.load(f)
    E = sgl["experts"]

    print(f"Grouped GEMM comparison (C++ SYCL device clock)  E={E}, bf16")
    print(
        "sgl: moe_grouped_mm_nt_xe35 (SYCL, NT)   vs   vllm: cutlass_grouped_gemm_xe3 (CUTLASS, NN)"
    )
    print("=" * 86)
    for label, n, k in GEMM_CONFIGS:
        print(f"\n{label}: N={n} K={k}")
        print(
            f"{'avg_m':>6} {'total_m':>8} {'sgl TFLOP/s':>13} {'sgl ms':>10} "
            f"{'vllm TFLOP/s':>14} {'vllm ms':>10} {'vllm/sgl':>10}"
        )
        print("-" * 86)
        for avg_m in args.avg_m:
            total_m = E * avg_m
            s_t = get_s(sgl, label, avg_m)
            v_t = get_s(vllm, label, avg_m)
            s_tf = tflops(total_m, n, k, s_t) if s_t else float("nan")
            v_tf = tflops(total_m, n, k, v_t) if v_t else float("nan")
            ratio = (v_t / s_t) if (s_t and v_t) else float("nan")
            print(
                f"{avg_m:>6} {total_m:>8} {s_tf:>13.3f} "
                f"{(s_t*1000 if s_t else float('nan')):>10.4f} {v_tf:>14.3f} "
                f"{(v_t*1000 if v_t else float('nan')):>10.4f} {ratio:>10.3f}"
            )


if __name__ == "__main__":
    main()

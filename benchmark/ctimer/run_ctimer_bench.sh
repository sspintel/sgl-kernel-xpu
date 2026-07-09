#!/usr/bin/env bash
# One-time C++ device-clock grouped-GEMM benchmark driver.
#
# Builds and runs bench_sgl_ctimer / bench_vllm_ctimer, which time the prebuilt
# sgl and vllm grouped-GEMM kernels with SYCL device-event profiling (the same
# command_start/command_end timestamps CUTLASS's GPU_Clock reads) instead of
# torch.xpu.Event, which is inaccurate on the CRI simulator.
#
# The two binaries are run STRICTLY SEQUENTIALLY (never concurrently): they
# drive the same simulated device and the sgl/vllm SYCL runtimes cannot coexist,
# so overlapping them would distort timings.
#
# Usage:
#   ./run_ctimer_bench.sh [build|sgl|vllm|report|all]   (default: all)
# Env overrides:
#   EXPERTS (8)  ITERS (3)  WARMUP (1)  AVG_M ("1 4 8 32 64 128")
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SGL_ROOT="$(cd "$HERE/../.." && pwd)"
VLLM_ROOT="/home/sdp/hengyume/applications.ai.gpu.vllm-xpu-kernels"
CUTLASS_SRC="$SGL_ROOT/build/_deps/repo-cutlass-sycl-src"
SGL_SO_DIR="$SGL_ROOT/build/src"
VLLM_SO="$VLLM_ROOT/build/temp/libgrouped_gemm_xe_3.so"

CXX="${CXX:-icpx}"
CONDA_ENV="${CONDA_ENV:-sglang}"
TORCHLIB="$(conda run -n "$CONDA_ENV" python -c 'import torch,os;print(os.path.dirname(torch.__file__)+"/lib")')"

EXPERTS="${EXPERTS:-8}"
ITERS="${ITERS:-3}"
WARMUP="${WARMUP:-1}"
AVG_M="${AVG_M:-1 4 8 32 64 128}"

OUT="$HERE/out"
mkdir -p "$OUT"

# The sgl launcher symbols live in per-tile .so files; the five we dispatch to
# for the benchmark shapes (pure bf16, fuse_act=false, no bias):
SGL_TILES=(8__64__32 16__64__32 32__64__32 128__64__32 128__128__32 256__256__32)
sgl_so_args() {
  local args=()
  for t in "${SGL_TILES[@]}"; do
    args+=("$SGL_SO_DIR/libsgl-ops-sycl-GroupGemmSIMD_inst_xe35__${t}_a0_ffalse_bfalse.so")
  done
  printf '%s\n' "${args[@]}"
}

build_sgl() {
  echo "[build] bench_sgl_ctimer"
  mapfile -t sos < <(sgl_so_args)
  "$CXX" -fsycl -std=c++17 -O2 -DCUTLASS_ENABLE_SYCL -DSYCL_INTEL_TARGET=35 \
    -I"$CUTLASS_SRC/include" -I"$SGL_ROOT/src" \
    "$HERE/bench_sgl_ctimer.cpp" "${sos[@]}" \
    -L"$TORCHLIB" -lc10 -ltorch_cpu \
    -Wl,-rpath,"$TORCHLIB" -Wl,-rpath,"$SGL_SO_DIR" \
    -o "$HERE/bench_sgl_ctimer"
}

build_vllm() {
  echo "[build] bench_vllm_ctimer"
  "$CXX" -fsycl -std=c++17 -O2 \
    "$HERE/bench_vllm_ctimer.cpp" "$VLLM_SO" \
    -L"$TORCHLIB" -lc10 -ltorch_cpu \
    -Wl,-rpath,"$TORCHLIB" -Wl,-rpath,"$(dirname "$VLLM_SO")" \
    -o "$HERE/bench_vllm_ctimer"
}

run_one() {  # $1=sgl|vllm
  local which="$1"
  echo "[run] $which  (E=$EXPERTS iters=$ITERS warmup=$WARMUP avg_m='$AVG_M')"
  LD_LIBRARY_PATH="$TORCHLIB:${LD_LIBRARY_PATH:-}" \
    "$HERE/bench_${which}_ctimer" \
      --experts "$EXPERTS" --iters "$ITERS" --warmup "$WARMUP" --avg-m $AVG_M \
      > "$OUT/${which}.json"
  echo "[run] wrote $OUT/${which}.json"
}

report() {
  conda run -n "$CONDA_ENV" python "$HERE/report_ctimer.py" \
    --sgl-json "$OUT/sgl.json" --vllm-json "$OUT/vllm.json" \
    --avg-m $AVG_M | tee "$OUT/report.txt"
}

cmd="${1:-all}"
case "$cmd" in
  build) build_sgl; build_vllm ;;
  sgl)   build_sgl; run_one sgl ;;
  vllm)  build_vllm; run_one vllm ;;
  report) report ;;
  all)
    build_sgl; build_vllm
    run_one sgl       # sequential: sgl fully finishes ...
    run_one vllm      # ... before vllm starts
    report
    ;;
  *) echo "usage: $0 [build|sgl|vllm|report|all]"; exit 1 ;;
esac

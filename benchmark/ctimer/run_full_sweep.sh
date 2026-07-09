#!/usr/bin/env bash
# Full performance sweep: GRF=512 with K=32 vs GRF=512 with K=64 (auto)
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SGL_ROOT="$(cd "$HERE/../.." && pwd)"
SGL_SO_DIR="$SGL_ROOT/build/src"
TORCHLIB="/home/sdp/miniforge3/envs/sglang/lib/python3.12/site-packages/torch/lib"
OUT="$HERE/out"
mkdir -p "$OUT"

export LD_LIBRARY_PATH="$TORCHLIB:$SGL_SO_DIR:${LD_LIBRARY_PATH:-}"

EXPERTS=8
ITERS=3
WARMUP=1
AVG_M="1 32 128"

echo "=== Full Performance Sweep ==="
echo "E=$EXPERTS, iters=$ITERS, warmup=$WARMUP, avg_m=$AVG_M"
echo ""

# Run 1: Auto dispatch (GRF=512 + K=64 where dispatch selects it)
echo "--- Run 1: AUTO (GRF=512 + K=64 auto) ---"
"$HERE/bench_sgl_ctimer" --experts $EXPERTS --iters $ITERS --warmup $WARMUP --avg-m $AVG_M \
  > "$OUT/sgl_auto.json" 2>"$OUT/sgl_auto.log"
echo "Done. Results in $OUT/sgl_auto.json"
cat "$OUT/sgl_auto.log"
echo ""

# Run 2: Force K=32 tiles (128x128x32 for medium M, 256x256x32 for large M)
# force-tile 3 = Tile_128_128_32 (good for M=32 with small_weight)
echo "--- Run 2: FORCE K=32 (128x128x32) ---"
"$HERE/bench_sgl_ctimer" --experts $EXPERTS --iters $ITERS --warmup $WARMUP --avg-m $AVG_M --force-tile 3 \
  > "$OUT/sgl_k32_128.json" 2>"$OUT/sgl_k32_128.log"
echo "Done. Results in $OUT/sgl_k32_128.json"
cat "$OUT/sgl_k32_128.log"
echo ""

# Run 3: Force 256x256x32 for comparison
echo "--- Run 3: FORCE K=32 (256x256x32) ---"
"$HERE/bench_sgl_ctimer" --experts $EXPERTS --iters $ITERS --warmup $WARMUP --avg-m $AVG_M --force-tile 4 \
  > "$OUT/sgl_k32_256.json" 2>"$OUT/sgl_k32_256.log"
echo "Done. Results in $OUT/sgl_k32_256.json"
cat "$OUT/sgl_k32_256.log"
echo ""

# Run 4: Force K=64 (128x128x64) explicitly
echo "--- Run 4: FORCE K=64 (128x128x64) ---"
"$HERE/bench_sgl_ctimer" --experts $EXPERTS --iters $ITERS --warmup $WARMUP --avg-m $AVG_M --force-tile 6 \
  > "$OUT/sgl_k64_128.json" 2>"$OUT/sgl_k64_128.log"
echo "Done. Results in $OUT/sgl_k64_128.json"
cat "$OUT/sgl_k64_128.log"
echo ""

# Run 5: Force K=64 (256x256x64) explicitly
echo "--- Run 5: FORCE K=64 (256x256x64) ---"
"$HERE/bench_sgl_ctimer" --experts $EXPERTS --iters $ITERS --warmup $WARMUP --avg-m $AVG_M --force-tile 7 \
  > "$OUT/sgl_k64_256.json" 2>"$OUT/sgl_k64_256.log"
echo "Done. Results in $OUT/sgl_k64_256.json"
cat "$OUT/sgl_k64_256.log"
echo ""

echo "=== All runs complete ==="
echo ""
echo "Baseline (from prior commit, GRF=256 K=32):"
cat "$OUT/sgl.json" 2>/dev/null || echo "(not available)"
echo ""
echo "New results:"
for f in "$OUT"/sgl_auto.json "$OUT"/sgl_k32_128.json "$OUT"/sgl_k32_256.json "$OUT"/sgl_k64_128.json "$OUT"/sgl_k64_256.json; do
  echo "$(basename $f):"
  cat "$f" 2>/dev/null || echo "  (failed/missing)"
  echo ""
done

#!/usr/bin/env bash
# SGMV grouped-vs-per-token timing under 5 GCC flag sets, 5 runs each, plus what the compiler did to the inner loops of common/lora-sgmv.h.
# Usage: scripts/bench-lora-sgmv-flags.sh [OUT_DIR]   (run on an otherwise idle machine: the numbers are microsecond timings)
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUT="${1:-$(mktemp -d)}"
CXX="${CXX:-g++}"
declare -A F=( [O2]="-O2" [O3]="-O3" [O3_avx2_fma]="-O3 -mavx2 -mfma" [O3_native]="-O3 -march=native" [O3_novec]="-O3 -fno-tree-vectorize" [O3_fastmath_avx2]="-O3 -ffast-math -mavx2 -mfma" )
$CXX --version | head -1
for k in O2 O3 O3_avx2_fma O3_native O3_novec O3_fastmath_avx2; do
  $CXX ${F[$k]} -std=c++17 -I"$ROOT/common" "$ROOT/tests/bench-lora-sgmv-flags.cpp" -o "$OUT/bench_$k"
  echo "== ${F[$k]}"
  for i in 1 2 3 4 5; do "$OUT/bench_$k"; done
done
echo "== larger working sets (-O3 -mavx2 -mfma): shape arguments T NA R D"
for cfg in "64 32 16 4096" "256 32 64 4096" "16 32 64 4096"; do
  echo "-- $cfg"; for i in 1 2 3 4 5; do "$OUT/bench_O3_avx2_fma" $cfg; done
done
echo "== what GCC did to the inner loops (-O3 -mavx2 -mfma): vectorised loops reported for common/lora-sgmv.h"
$CXX -O3 -mavx2 -mfma -std=c++17 -I"$ROOT/common" -fopt-info-vec-optimized -c "$ROOT/tests/bench-lora-sgmv-flags.cpp" -o /dev/null 2>&1 | grep "lora-sgmv.h" | sort -u || true
echo "== instruction mix of the two kernels (noinline wrappers), -O3 -mavx2 -mfma"
cat > "$OUT/wrap.cpp" <<'EOW'
#include "lora-sgmv.h"
__attribute__((noinline)) void w_naive(const float * x, float * y, const int * ids, int n_tokens, int n_in, int n_out, const lora_sgmv_adapter * ad, int na) { lora_sgmv_apply_naive(x, y, ids, n_tokens, n_in, n_out, ad, na); }
__attribute__((noinline)) void w_grouped(const float * x, float * y, const lora_sgmv_plan & p, int n_in, int n_out, const lora_sgmv_adapter * ad) { lora_sgmv_apply_grouped(x, y, p, n_in, n_out, ad); }
EOW
$CXX -O3 -mavx2 -mfma -std=c++17 -I"$ROOT/common" -S "$OUT/wrap.cpp" -o "$OUT/wrap.s"
for fn in w_naive w_grouped; do
  body=$(awk -v f="$fn" '$0 ~ "^_Z[0-9]+"f"[A-Za-z0-9_]*:" {on=1} on {print} on && /\.cfi_endproc/ {exit}' "$OUT/wrap.s")
  printf "%-10s packed multiplies (vmulps): %s   chained scalar adds (vaddss): %s   packed adds (vaddps/vfmadd*ps): %s\n" "$fn" \
    "$(grep -cE 'vmulps' <<<"$body")" "$(grep -cE 'vaddss' <<<"$body")" "$(grep -cE 'vaddps|vfmadd[0-9]+ps' <<<"$body")"
done

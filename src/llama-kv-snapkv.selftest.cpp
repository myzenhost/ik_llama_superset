// Standalone selftest runner for the SnapKV/H2O CPU core (S69 #22 slice-2a).
// NOT part of the library build (has its own main) — compile on demand:
//   g++ -std=c++17 -I. llama-kv-snapkv.cpp llama-kv-snapkv.selftest.cpp -o snapkv_selftest
// Verifies pool1d / topk_idx / snapkv_keep_plan / H2OAccumulator (8 host cases). rc=0 = PASS.
// Re-verified PASS 2026-06-17 (portable mingw64 g++).
#include "llama-kv-snapkv.h"
#include <cstdio>
int main() {
    int r = justician_kv::_selftest();
    printf("SNAPKV_SELFTEST rc=%d (%s)\n", r, r == 0 ? "PASS" : "FAIL");
    return r;
}

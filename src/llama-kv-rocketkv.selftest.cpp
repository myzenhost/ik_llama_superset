// standalone CPU runner for justician_kv::_rocketkv_selftest (RocketKV stage-2 HSA). No ggml/CUDA.
#include "llama-kv-rocketkv.h"
#include <cstdio>
int main() {
    int fails = justician_kv::_rocketkv_selftest();
    if (fails == 0) { std::printf("[rocketkv-selftest] PASS — 0 failures (GATEs A bound-soundness / B recall / C budget / D headdim-reduction / E noop)\n"); return 0; }
    std::printf("[rocketkv-selftest] FAIL — %d gate(s) failed\n", fails);
    return 1;
}

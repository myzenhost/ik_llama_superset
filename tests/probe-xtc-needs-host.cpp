// RESEARCH-DONE: common/sampling.cpp:2376-2391 (the existing host-fallback list), src/llama-sampler.cpp:2246-2277 (chain XTC, no vocab handle), ExLlamaV2 sampler.py:277-302 (MIT, behaviour only)
// Probe: does an XTC request fall back to the host sampler when backend sampling is on? (row S69-EXLLAMAV2-PICK-XTC-EXEMPT-EOG-NEWLINE-2026-09-29-01)
// The backend chain's XTC (llama_sampler_init_xtc, src/llama-sampler.cpp) has no vocab and cannot exempt end-of-generation or
// newline tokens; the host path (llama_sample_xtc) does. common_sampler_needs_host(params) decides which one serves a request, so
// it must be true when XTC is active and false when it is not (controls below).
// Not in CMake on purpose -- links the static archives directly. Build (mingw, from the fork root, build-static):
//   g++ -O1 -std=c++17 -fopenmp -I include -I ggml/include -I common -I src -I vendor tests/probe-xtc-needs-host.cpp -o probe-needs-host.exe
//       -Wl,--start-group build-static/common/libllama-common.a build-static/common/libllama-common-base.a build-static/src/libllama.a
//       build-static/ggml/src/ggml.a build-static/ggml/src/ggml-cpu.a build-static/ggml/src/ggml-base.a
//       build-static/vendor/cpp-httplib/libcpp-httplib.a build-static/vendor/hash/libvendor-hash.a -Wl,--end-group
//       -fopenmp -lstdc++ -lwinhttp -lws2_32 -lcrypt32 -lbcrypt -lole32 -lshlwapi -ladvapi32 -lshell32 -luser32
// Expected after the change: "4 checks, 0 failed".
#include "sampling.h"

#include <algorithm>
#include <cstdio>

static int g_checks = 0, g_fails = 0;
#define CHECK(cond, what) do { ++g_checks; if (!(cond)) { ++g_fails; std::printf("FAIL: %s\n", what); } } while (0)

static bool has_xtc(const common_params_sampling & p) {
    return std::find(p.samplers_sequence.begin(), p.samplers_sequence.end(), llama_sampler_type::XTC) != p.samplers_sequence.end();
}

int main() {
    common_params_sampling base;
    // make sure XTC is in the sequence for the active cases
    if (!has_xtc(base)) {
        base.samplers_sequence.push_back(llama_sampler_type::XTC);
    }

    common_params_sampling on = base;
    on.xtc_probability = 0.5f;
    on.xtc_threshold   = 0.1f;
    CHECK(common_sampler_needs_host(on), "XTC active (p 0.5, threshold 0.1) -> host sampler");

    common_params_sampling p0 = base;
    p0.xtc_probability = 0.0f;
    p0.xtc_threshold   = 0.1f;
    CHECK(!common_sampler_needs_host(p0), "control: XTC probability 0 -> backend allowed");

    common_params_sampling thr = base;
    thr.xtc_probability = 0.5f;
    thr.xtc_threshold   = 0.9f; // a threshold above 0.5 means XTC never acts
    CHECK(!common_sampler_needs_host(thr), "control: XTC threshold above 0.5 (inactive) -> backend allowed");

    common_params_sampling noseq = on;
    noseq.samplers_sequence.erase(std::remove(noseq.samplers_sequence.begin(), noseq.samplers_sequence.end(), llama_sampler_type::XTC),
                                  noseq.samplers_sequence.end());
    CHECK(!common_sampler_needs_host(noseq), "control: XTC not in the sampler sequence -> backend allowed");

    std::printf("%d checks, %d failed\n", g_checks, g_fails);
    return g_fails ? 1 : 0;
}

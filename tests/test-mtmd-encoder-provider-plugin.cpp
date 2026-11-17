// Test plugin for tests/test-mtmd-encoder-provider.cpp: a provider shared library whose "model" string selects its
// behaviour. Output is a fixed linear mix of the mel so the test can recompute it independently.
//   model "ok"      : well-behaved
//   model "optout"  : init returns NULL
//   model "fail"    : encode returns 5
//   model "nan"     : encode writes a NaN at index 3 and returns 0 (a provider that must be refused)
//   model "short"   : encode returns 0 but leaves the output untouched (zeros from the caller) -> the caller compares
#include "mtmd-encoder-provider.h"

#include <cmath>
#include <cstring>

namespace {
struct handle_t { int mode; };

void * t_init(const char * model, const char *, const char *) {
    if (!model) return nullptr;
    if (!std::strcmp(model, "optout")) return nullptr;
    handle_t * h = new handle_t();
    h->mode = !std::strcmp(model, "fail") ? 1 : !std::strcmp(model, "nan") ? 2 : !std::strcmp(model, "short") ? 3 : 0;
    return h;
}

int t_encode(void * hp, const float * mel, int32_t n_step, int32_t n_mel, float * out, int64_t n_out) {
    const handle_t * h = (const handle_t *) hp;
    if (h->mode == 1) return 5;
    const int64_t n = (int64_t) n_step * n_mel;
    if (h->mode != 3) {
        for (int64_t i = 0; i < n_out; i++) {
            out[i] = 0.5f * mel[i % n] + 0.25f * mel[(i * 7 + 3) % n];
        }
    }
    if (h->mode == 2 && n_out > 3) {
        out[3] = (float) NAN;
    }
    return 0;
}

void t_free(void * hp) { delete (handle_t *) hp; }

const mtmd_encoder_provider_api g_api = { MTMD_ENCODER_PROVIDER_ABI, "test-plugin", t_init, t_encode, t_free };
} // namespace

extern "C"
#if defined(_WIN32)
__declspec(dllexport)
#else
__attribute__((visibility("default")))
#endif
const mtmd_encoder_provider_api * mtmd_encoder_provider_entry(void) {
    return &g_api;
}

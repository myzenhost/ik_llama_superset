// CUDA graph reuse under alternating graph shapes (row S69-SERVER-SPEC-PATH-NO-CUDA-GRAPHS-2026-10-18-01).
//
// A speculative decode alternates graphs of different shapes -- the n=1 decode, the n=2..4 verify batch and the
// MTP draft graph -- and every one of them is rebuilt in the SAME compute-meta buffer, so its first node is the same
// ggml_tensor pointer each time. The CUDA backend used that pointer alone as the graph-cache key, so all shapes
// shared one entry, every call looked like a property change, and after four consecutive changes the entry was
// switched off for the rest of the process (disable_due_to_too_many_updates, sticky).
// RESEARCH-DONE: upstream llama.cpp master ggml-cuda.cu (sha256 68f09269...7777) replaced that switch with a warmup
// state machine; DISABLE-REVERSIBLE: GGML_CUDA_GRAPH_LEGACY=1 keeps the old policy reachable for A/B.
//
// This reproduces that: two shapes (n=1 and n=4) rebuilt alternately in one fixed meta buffer, 20 rounds each,
// with the result of every call checked against a host reference. It reads the backend's own counters
// (GGML_CUDA_GRAPH_STATS, printed when the backend is freed) and requires at least 30 of the 40 calls to be
// served by a graph launch.
//
// Fail-before arm: GGML_CUDA_GRAPH_LEGACY=1 restores the old key and the sticky switch in the same binary; there
// the launch count collapses to the first few calls and this test exits 1.
//
// Exit: 0 pass, 1 fail, 77 skip (no CUDA device, or the backend could not be created).
// Needs a GPU: built under GGML_CUDA, run by hand in the heavy lane like the other CUDA tests.

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cuda.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

static std::string g_log;

static void log_cb(ggml_log_level level, const char * text, void * user) {
    (void) level; (void) user;
    g_log += text;
    fputs(text, stderr);
}

static unsigned long long stat_of(const std::string & log, const char * name) {
    const std::string key = std::string(" ") + name + " ";
    const size_t at = log.rfind("cuda graph stats:");
    if (at == std::string::npos) return ~0ull;
    const size_t p = log.find(key, at);
    if (p == std::string::npos) return ~0ull;
    return strtoull(log.c_str() + p + key.size(), nullptr, 10);
}

static void set_env(const char * k, const char * v) {
#ifdef _WIN32
    _putenv_s(k, v);
#else
    setenv(k, v, 1);
#endif
}

int main(int argc, char ** argv) {
    (void) argc; (void) argv;
    const bool legacy = getenv("GGML_CUDA_GRAPH_LEGACY") != nullptr && atoi(getenv("GGML_CUDA_GRAPH_LEGACY")) != 0;
    set_env("GGML_CUDA_GRAPH_STATS", "1");

    if (ggml_backend_cuda_get_device_count() < 1) {
        printf("SKIP: no CUDA device\n");
        return 77;
    }
    ggml_backend_cuda_log_set_callback(log_cb, nullptr);
    ggml_backend_t backend = ggml_backend_cuda_init(0, NULL, NULL);
    if (backend == nullptr) {
        printf("SKIP: CUDA backend could not be created\n");
        return 77;
    }

    const int K = 64;
    const int shapes[2] = { 1, 4 };
    const int rounds = 20;

    // persistent weight, in its own context and buffer
    ggml_init_params wp = { ggml_tensor_overhead() * 4, nullptr, true };
    ggml_context * wctx = ggml_init(wp);
    ggml_tensor * w = ggml_new_tensor_2d(wctx, GGML_TYPE_F32, K, 1);
    ggml_backend_buffer_t wbuf = ggml_backend_alloc_ctx_tensors(wctx, backend);
    std::vector<float> wh(K);
    for (int i = 0; i < K; ++i) wh[i] = 0.5f + 0.01f * (float) i;
    ggml_backend_tensor_set(w, wh.data(), 0, sizeof(float) * K);

    // ONE meta buffer, reused for every rebuild: the first node lands at the same address for both shapes,
    // as it does for the server's decode / verify / draft graphs
    const size_t meta_size = ggml_tensor_overhead() * 32 + ggml_graph_overhead();
    std::vector<uint8_t> meta(meta_size);

    ggml_gallocr_t galloc[2] = {
        ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend)),
        ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend)),
    };

    int bad = 0;
    const void * first_node[2] = { nullptr, nullptr };

    for (int r = 0; r < rounds * 2; ++r) {  // PROGRESS-OK: 40 small graph computes
        const int s = r % 2;
        const int n = shapes[s];

        ggml_init_params ip = { meta_size, meta.data(), true };
        ggml_context * ctx = ggml_init(ip);
        ggml_tensor * x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, K, n);
        ggml_set_input(x);
        ggml_tensor * m   = ggml_mul(ctx, x, w);        // nodes[0]
        ggml_tensor * a   = ggml_add(ctx, m, x);
        ggml_tensor * out = ggml_sum_rows(ctx, a);      // [1, n]
        ggml_set_output(out);
        ggml_cgraph * gf = ggml_new_graph(ctx);
        ggml_build_forward_expand(gf, out);

        if (!ggml_gallocr_alloc_graph(galloc[s], gf)) {
            printf("FAIL: allocation failed (round %d)\n", r);
            return 1;
        }
        first_node[s] = ggml_graph_node(gf, 0);

        std::vector<float> xh((size_t) K * n);
        for (int j = 0; j < n; ++j)
            for (int i = 0; i < K; ++i)
                xh[(size_t) j * K + i] = 0.25f * (float) ((i + 3 * j + r) % 11) - 1.0f;
        ggml_backend_tensor_set(x, xh.data(), 0, sizeof(float) * xh.size());

        if (ggml_backend_graph_compute(backend, gf) != GGML_STATUS_SUCCESS) {
            printf("FAIL: compute failed (round %d)\n", r);
            return 1;
        }

        std::vector<float> got(n);
        ggml_backend_tensor_get(out, got.data(), 0, sizeof(float) * n);
        for (int j = 0; j < n; ++j) {
            double ref = 0.0;
            for (int i = 0; i < K; ++i) {
                const double v = xh[(size_t) j * K + i];
                ref += v * wh[i] + v;
            }
            if (std::fabs(got[j] - ref) > 1e-3 * (1.0 + std::fabs(ref))) {
                if (bad < 5) printf("  mismatch round %d shape n=%d row %d: got %.6f ref %.6f\n", r, n, j, got[j], ref);
                ++bad;
            }
        }
        ggml_free(ctx);
    }

    const bool same_first = first_node[0] == first_node[1];

    ggml_gallocr_free(galloc[0]);
    ggml_gallocr_free(galloc[1]);
    ggml_backend_buffer_free(wbuf);
    ggml_free(wctx);
    ggml_backend_free(backend);   // prints the counters

    const unsigned long long launches = stat_of(g_log, "launches");
    const unsigned long long computes = stat_of(g_log, "computes");
    const unsigned long long direct   = stat_of(g_log, "direct");

    printf("policy %s | first node shared by both shapes: %s | computes %llu launches %llu direct %llu | mismatches %d\n",
           legacy ? "LEGACY" : "warmup", same_first ? "yes" : "no", computes, launches, direct, bad);

    int fail = 0;
    if (launches == ~0ull) {
        printf("  FAIL: no 'cuda graph stats' line was logged (counters missing from this build?)\n");
        ++fail;
    } else {
        if (computes != (unsigned long long) (rounds * 2)) { printf("  FAIL: expected %d computes\n", rounds * 2); ++fail; }
        if (launches < 30) { printf("  FAIL: %llu of %d calls launched a CUDA graph (need >= 30)\n", launches, rounds * 2); ++fail; }
    }
    if (!same_first) { printf("  FAIL: the two shapes did not share a first-node pointer, so this does not reproduce the collision\n"); ++fail; }
    if (bad) { printf("  FAIL: %d output mismatches\n", bad); ++fail; }

    printf("%s\n", fail ? "FAIL" : "PASS");
    return fail ? 1 : 0;
}

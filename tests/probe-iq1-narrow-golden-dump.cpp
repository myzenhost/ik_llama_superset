// Deterministic dump of IQ1_XS / IQ1_XXS / IQ1_XXXS (and IQ1_S as a control) so two builds can be compared byte for byte.
//
// SOURCE-READ: build/work/ref-src/ggml/src/ggml-quants.c  (git archive of unslothai/llama.cpp
// iq1-narrow-grids-b10775 c86ed2699, MIT; quantize_row_iq1_narrow_impl / dequantize_row_iq1_xs). This file only CALLS that API.
//
// WHY. A port of a quantiser is only verified against a REFERENCE: this program is compiled twice from this one file,
//   (a) against the Unsloth tree -> the reference bytes,
//   (b) against this fork        -> the port's bytes,
// and the dumps are diffed. Inputs come from an integer xorshift generator (no libm, no transcendental), so both
// builds see bit-identical floats. -DIQ1_DUMP_FORK adds the fork's extra `user_data` argument to ggml_quantize_chunk.
//
// BUILD (reference, upstream layout):
//   g++ -O2 -I<unsloth>/ggml/include probe-iq1-narrow-golden-dump.cpp <libggml.a libggml-cpu.a libggml-base.a> -o dump_ref
// BUILD (fork): registered as a build-only target in tests/CMakeLists.txt (probe-iq1-narrow-golden-dump), -DIQ1_DUMP_FORK.
// RUN: dump <outdir>   -> <outdir>/<type>.q (quantised bytes), <type>.f32 (dequantised floats), <type>.txt (sizes + FNV-1a-64).

#include "ggml.h"
#include "ggml-cpu.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

static uint64_t g_s = 0x9E3779B97F4A7C15ull;
static uint32_t next_u32() {
    g_s ^= g_s << 13; g_s ^= g_s >> 7; g_s ^= g_s << 17;
    return (uint32_t) (g_s >> 32);
}
// integer-exact "approximately gaussian": sum of four 16-bit uniforms, centred, scaled by one float multiply
static float gauss_like() {
    int32_t v = 0;
    for (int i = 0; i < 4; ++i) v += (int32_t) (next_u32() >> 16);
    return (float) (v - 4*32768) * (0.12f / 131072.0f);
}
static float imat_like() {
    return 0.1f + (float) (next_u32() >> 16) * (1.9f / 65536.0f);
}

static uint64_t fnv1a(const void * p, size_t n) {
    const uint8_t * b = (const uint8_t *) p;
    uint64_t h = 0xcbf29ce484222325ull;
    for (size_t i = 0; i < n; ++i) { h ^= b[i]; h *= 0x100000001b3ull; }
    return h;
}

int main(int argc, char ** argv) {
    if (argc < 2) { fprintf(stderr, "usage: %s <outdir>\n", argv[0]); return 2; }
    const std::string out = argv[1];
    setvbuf(stdout, nullptr, _IONBF, 0);   // a crash must not eat the lines that already ran
    ggml_cpu_init();

    const int64_t K = 1024, R = 8;
    std::vector<float> w(K*R), im(K);
    for (auto & v : w)  v = gauss_like();
    // outlier columns and a flat row exercise the max_scale==0 and large-dynamic-range paths
    for (int64_t r = 0; r < R; r += 3) for (int64_t c = 0; c < K; c += 97) w[r*K + c] *= 6.0f;
    for (int64_t c = 0; c < K; ++c) w[(R-1)*K + c] = 0.0f;
    for (auto & v : im) v = imat_like();

    struct T { int id; const char * name; };
    const T types[] = { { 19, "iq1_s" }, { 64, "iq1_xs" }, { 65, "iq1_xxs" }, { 66, "iq1_xxxs" } };
    int rc = 0;
    for (const auto & t : types) {
        const ggml_type ty = (ggml_type) t.id;
        const ggml_type_traits * tr = ggml_get_type_traits(ty);
        if (!tr || !tr->type_name || strcmp(tr->type_name, t.name) != 0) {
            fprintf(stderr, "type id %d is %s here, expected %s\n", t.id, tr && tr->type_name ? tr->type_name : "<none>", t.name);
            rc = 1;
            continue;
        }
        printf("%-9s init\n", t.name);
        ggml_quantize_init(ty);
        printf("%-9s init done, quantizing\n", t.name);
        const size_t row_size = ggml_row_size(ty, K);
        std::vector<uint8_t> q(row_size*R, 0);
#ifdef IQ1_DUMP_FORK
        const size_t got = ggml_quantize_chunk(ty, w.data(), q.data(), 0, R, K, im.data(), nullptr);
#else
        const size_t got = ggml_quantize_chunk(ty, w.data(), q.data(), 0, R, K, im.data());
#endif
        std::vector<float> dq(K*R);
        for (int64_t r = 0; r < R; ++r) tr->to_float(q.data() + r*row_size, dq.data() + r*K, K);

        FILE * f = fopen((out + "/" + t.name + ".q").c_str(), "wb");
        if (!f) { fprintf(stderr, "cannot write into '%s' (does the directory exist?)\n", out.c_str()); return 3; }
        fwrite(q.data(), 1, q.size(), f); fclose(f);
        f = fopen((out + "/" + t.name + ".f32").c_str(), "wb");
        if (!f) { fprintf(stderr, "cannot write %s.f32\n", t.name); return 3; }
        fwrite(dq.data(), 4, dq.size(), f); fclose(f);
        f = fopen((out + "/" + t.name + ".txt").c_str(), "w");
        if (!f) { fprintf(stderr, "cannot write %s.txt\n", t.name); return 3; }
        fprintf(f, "%s type_size=%zu row_size=%zu chunk_ret=%zu q_fnv=%016llx f32_fnv=%016llx\n", t.name, ggml_type_size(ty), row_size, got,
                (unsigned long long) fnv1a(q.data(), q.size()), (unsigned long long) fnv1a(dq.data(), dq.size()*4));
        fclose(f);
        printf("%-9s type_size=%zu row_size=%zu chunk_ret=%zu q_fnv=%016llx f32_fnv=%016llx\n", t.name, ggml_type_size(ty), row_size, got,
               (unsigned long long) fnv1a(q.data(), q.size()), (unsigned long long) fnv1a(dq.data(), dq.size()*4));
    }
    return rc;
}

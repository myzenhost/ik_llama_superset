// Differential fixture for the base/cpu split of the fork's quantize/dequantize implementations.
//
// Row S69-GGML-BASE-DLL-CANNOT-LINK-CIRCULAR-ACROSS-A-DLL-BOUNDARY-2026-09-15-01, half 2: the .to_float and
// .from_float_ref members of ggml.c's type_traits[] are being extracted out of ggml/src/iqk/iqk_quantize.cpp
// (ggml-cpu, compiled /arch:AVX2) into an arch-neutral ggml-base source. "Intrinsic-free" says no arch decision is
// needed to MOVE a function; it does not say the function is still correct after the move (repack_q8_KV's generic
// arm was intrinsic-free and had never compiled in its life). This test is the differential check.
//
// For every ggml_type that has a to_float and/or from_float_ref in the type_traits table, and for three fixed-seed
// row profiles (gaussian, heavy-tailed, sparse with outliers):
//   * from_float_ref quantizes the row (or an R-row interleaved group for the _r4/_r8/_r16 types) into a buffer,
//   * to_float dequantizes that buffer,
//   * an FNV-1a/64 digest of the QUANTIZED BYTES and another of the DEQUANTIZED FLOAT BYTES are printed per type.
// Run it before and after the move and diff the outputs: every line must be byte-identical.
//
// Guard canaries follow both buffers, so an overrun is reported as such instead of surfacing as an unrelated
// heap fault later (the failure mode of dequantize_row_mxfp4_r8, see test-rowint-dequant-bounds.cpp).
//
// Usage:  test-iqk-base-dequant [--only NAME[,NAME...]] [--skip NAME[,NAME...]] [--list]
// Output: one "HASH <name> ..." line per type; "SUMMARY ..." last. Exit 0 unless a guard was hit.

#include "ggml.h"
#include "ggml-cpu.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <set>
#include <string>
#include <vector>

static uint64_t fnv1a64(const void * data, size_t n, uint64_t h = 0xcbf29ce484222325ULL) {
    const uint8_t * p = (const uint8_t *) data;
    for (size_t i = 0; i < n; ++i) {
        h ^= p[i];
        h *= 0x100000001b3ULL;
    }
    return h;
}

// xorshift64* -- fixed seed, platform independent
struct rng_t {
    uint64_t s;
    explicit rng_t(uint64_t seed) : s(seed ? seed : 0x9E3779B97F4A7C15ULL) {}
    uint64_t next() {
        s ^= s >> 12; s ^= s << 25; s ^= s >> 27;
        return s * 0x2545F4914F6CDD1DULL;
    }
    double uniform() { return ((next() >> 11) + 0.5) * (1.0 / 9007199254740992.0); } // (0,1)
    double gauss() {
        const double u1 = uniform(), u2 = uniform();
        return std::sqrt(-2.0 * std::log(u1)) * std::cos(6.283185307179586 * u2);
    }
};

static void fill_profile(std::vector<float> & v, int profile, uint64_t seed) {
    rng_t r(seed);
    for (size_t i = 0; i < v.size(); ++i) {
        double x;
        if (profile == 0) {            // gaussian, sigma 0.05 (weight-like)
            x = 0.05 * r.gauss();
        } else if (profile == 1) {     // laplace-ish heavy tail
            const double u = r.uniform() - 0.5;
            x = -0.04 * (u < 0 ? -1.0 : 1.0) * std::log(1.0 - 2.0 * std::fabs(u));
        } else {                       // sparse: mostly small, ~2% large outliers, some exact zeros
            const double u = r.uniform();
            if (u < 0.05)      x = 0.0;
            else if (u < 0.07) x = 0.8 * r.gauss();
            else               x = 0.02 * r.gauss();
        }
        v[i] = (float) x;
    }
}

static int interleave_of(const std::string & name) {
    auto ends = [&](const char * suf) {
        const size_t n = strlen(suf);
        return name.size() > n && name.compare(name.size() - n, n, suf) == 0;
    };
    if (ends("_r16") || ends("_R16")) return 16;
    if (ends("_r8")  || ends("_R8"))  return 8;
    if (ends("_r4")  || ends("_R4"))  return 4;
    return 1;
}

static std::set<std::string> split_csv(const char * s) {
    std::set<std::string> out;
    std::string cur;
    for (const char * p = s; ; ++p) {
        if (*p == ',' || *p == 0) { if (!cur.empty()) out.insert(cur); cur.clear(); if (!*p) break; }
        else cur += *p;
    }
    return out;
}

int main(int argc, char ** argv) {
    std::set<std::string> only, skip;
    bool list_only = false;
    for (int i = 1; i < argc; ++i) {
        if (!strcmp(argv[i], "--only") && i + 1 < argc) only = split_csv(argv[++i]);
        else if (!strcmp(argv[i], "--skip") && i + 1 < argc) skip = split_csv(argv[++i]);
        else if (!strcmp(argv[i], "--list")) list_only = true;
    }

    ggml_cpu_init();

    const int64_t n_per_row = 1024; // multiple of every block size in the table (32, 128, 256)
    const uint32_t kCanary = 0xDEADBEEFu;
    int n_types = 0, n_q = 0, n_d = 0, n_skipped = 0, n_guard = 0;

    for (int ti = 0; ti < GGML_TYPE_COUNT; ++ti) {
        const ggml_type type = (ggml_type) ti;
        const char * cname = ggml_type_name(type);
        if (!cname) continue;
        const std::string name = cname;
        const auto * tr = ggml_get_type_traits(type);
        if (!tr || (!tr->to_float && !tr->from_float_ref)) continue;
        if (!only.empty() && !only.count(name)) continue;
        if (skip.count(name)) { printf("SKIP %s (--skip)\n", cname); ++n_skipped; continue; }
        // Trellis (KT) from_float_ref aborts BY DESIGN without AVX2/AVX512 (iqk_quantize.cpp iqk_kt_can_quantize);
        // on aarch64 that ended the run at iq2_kt with every later type unhashed (qemu-user lane, engq-c-1019).
        if ((type == GGML_TYPE_IQ1_KT || type == GGML_TYPE_IQ2_KT || type == GGML_TYPE_IQ3_KT || type == GGML_TYPE_IQ4_KT) &&
                tr->from_float_ref && !ggml_cpu_has_avx2() && !ggml_cpu_has_avx512()) {
            printf("SKIP %s (trellis quantization needs AVX2/AVX512 on the CPU)\n", cname);
            ++n_skipped;
            continue;
        }
        if (list_only) {
            printf("TYPE %s to_float=%d from_float_ref=%d\n", cname, tr->to_float != nullptr, tr->from_float_ref != nullptr);
            continue;
        }
        const int64_t blck = ggml_blck_size(type);
        if (blck <= 0 || n_per_row % blck != 0) {
            printf("SKIP %s (blck_size %lld does not divide %lld)\n", cname, (long long) blck, (long long) n_per_row);
            ++n_skipped;
            continue;
        }
        const int R = interleave_of(name);
        const int64_t k = (int64_t) R * n_per_row;
        const size_t qbytes = (size_t) R * ggml_row_size(type, n_per_row);

        printf("RUN %s\n", cname);
        fflush(stdout);
        ggml_quantize_init(type);

        uint64_t hq = 0xcbf29ce484222325ULL, hy = 0xcbf29ce484222325ULL;
        bool have_q = false, have_y = false, guard_ok = true;

        for (int profile = 0; profile < 3; ++profile) {
            std::vector<float> src((size_t) k);
            fill_profile(src, profile, 0x1234567ULL + 977ULL * (uint64_t) ti + 31ULL * (uint64_t) profile);

            const size_t kGuard = 64;
            std::vector<uint8_t> q(qbytes + kGuard * sizeof(uint32_t), 0);
            for (size_t g = 0; g < kGuard; ++g) memcpy(q.data() + qbytes + g * 4, &kCanary, 4);

            if (tr->from_float_ref) {
                tr->from_float_ref(src.data(), q.data(), k);
                have_q = true;
                hq = fnv1a64(q.data(), qbytes, hq);
                for (size_t g = 0; g < kGuard; ++g) {
                    uint32_t v; memcpy(&v, q.data() + qbytes + g * 4, 4);
                    guard_ok &= (v == kCanary);
                }
            }
            if (tr->to_float && tr->from_float_ref) {
                std::vector<float> out((size_t) k + kGuard, 0.f);
                for (size_t g = 0; g < kGuard; ++g) memcpy(&out[(size_t) k + g], &kCanary, 4);
                tr->to_float(q.data(), out.data(), k);
                have_y = true;
                hy = fnv1a64(out.data(), (size_t) k * sizeof(float), hy);
                for (size_t g = 0; g < kGuard; ++g) {
                    uint32_t v; memcpy(&v, &out[(size_t) k + g], 4);
                    guard_ok &= (v == kCanary);
                }
            }
        }
        ++n_types;
        n_q += have_q;
        n_d += have_y;
        if (!guard_ok) ++n_guard;
        printf("HASH %-14s k=%lld qbytes=%zu to_float=%d from_float_ref=%d q=%s y=%s guard=%s\n", cname, (long long) k, qbytes,
               tr->to_float != nullptr, tr->from_float_ref != nullptr,
               have_q ? (std::string("0x") + [&]{ char b[32]; snprintf(b, sizeof b, "%016llx", (unsigned long long) hq); return std::string(b); }()).c_str() : "-",
               have_y ? (std::string("0x") + [&]{ char b[32]; snprintf(b, sizeof b, "%016llx", (unsigned long long) hy); return std::string(b); }()).c_str() : "-",
               guard_ok ? "ok" : "OVERRUN");
        fflush(stdout);
    }

    printf("SUMMARY types=%d quantized=%d dequantized=%d skipped=%d guard_failures=%d\n", n_types, n_q, n_d, n_skipped, n_guard);
    return n_guard ? 1 : 0;
}

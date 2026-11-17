// Quantize fudge factors: upstream ik_llama's per-type scale multipliers, ggml_{set,get}_quantize_fudge_factor.
//
// The fork used to hard-code these multipliers inside each quantiser. Porting the settable table must not change a
// single quantised byte at the defaults, so this test hashes the output of every type whose default is not 1.0.
//
// MODE --hashes : print one FNV-1a hash per type and nothing else (this mode uses no fudge API, so the SAME file
//                 printed the baseline on the pre-port tree; the hashes are recorded in the commit message).
// default mode  : also checks (a) every getter default equals upstream's value, (b) setting a factor CHANGES the
//                 bytes (the knob is wired, not decorative), (c) restoring the default gives back the default bytes.

#include "ggml.h"
#include "ggml-cpu.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <string>
#include <vector>

struct fudge_case { ggml_type type; float upstream_default; };

// Every type upstream gives a non-1.0 default (ik_llama ggml.c fudge_factors[], fetched 2026-09-26), then every
// type whose upstream quantiser reads the factor at default 1.0.
static const fudge_case k_cases[] = {
    { GGML_TYPE_IQ2_XS,  1.05f   }, { GGML_TYPE_IQ3_XXS, 1.0125f }, { GGML_TYPE_IQ3_S,  1.033f },
    { GGML_TYPE_IQ2_S,   0.9875f }, { GGML_TYPE_IQ1_S,   1.125f  }, { GGML_TYPE_IQ1_M,  1.085f },
    { GGML_TYPE_IQ2_K,   1.03f   }, { GGML_TYPE_IQ2_KL,  1.025f  }, { GGML_TYPE_IQ3_K,  1.01f  },
    { GGML_TYPE_IQ4_KSS, 1.01f   }, { GGML_TYPE_IQ1_KT,  1.07f   },
    // default 1.0 upstream, but upstream's quantiser still reads the factor -- the knob must be live for these too
    { GGML_TYPE_IQ3_KS,  1.0f    }, { GGML_TYPE_IQ4_K,   1.0f    }, { GGML_TYPE_IQ5_K,  1.0f   },
    { GGML_TYPE_IQ6_K,   1.0f    }, { GGML_TYPE_IQ4_KS,  1.0f    }, { GGML_TYPE_IQ5_KS, 1.0f   },
    { GGML_TYPE_IQ2_KT,  1.0f    }, { GGML_TYPE_IQ3_KT,  1.0f    }, { GGML_TYPE_IQ4_KT, 1.0f   },
};

static uint64_t fnv1a(const uint8_t * p, size_t n) {
    uint64_t h = 1469598103934665603ull;
    for (size_t i = 0; i < n; i++) { h ^= p[i]; h *= 1099511628211ull; }
    return h;
}

static uint64_t quant_hash(ggml_type t) {
    const int64_t K = 1024, M = 16;
    std::mt19937 rng(20260926);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    std::vector<float> w((size_t) (K * M));
    for (auto & v : w) v = nd(rng);
    std::vector<float> imat((size_t) K, 1.0f);   // flat: IQ2_XS / IQ1_S require one
    std::vector<uint8_t> q(ggml_row_size(t, K) * M);
    ggml_quantize_chunk(t, w.data(), q.data(), 0, M, K, imat.data(), nullptr);
    return fnv1a(q.data(), q.size());
}

int main(int argc, char ** argv) {
    ggml_cpu_init();
    const bool hashes_only = argc > 1 && std::string(argv[1]) == "--hashes";
    for (const auto & c : k_cases) {
        printf("%-8s %016llx\n", ggml_type_name(c.type), (unsigned long long) quant_hash(c.type));
    }
    if (hashes_only) return 0;
#ifdef GGML_HAS_QUANTIZE_FUDGE_FACTOR
    int fail = 0;
    for (const auto & c : k_cases) {
        const float got = ggml_get_quantize_fudge_factor(c.type);
        const bool def_ok = got == c.upstream_default;
        const uint64_t h0 = quant_hash(c.type);
        ggml_set_quantize_fudge_factor(c.type, got * 1.1f);
        const uint64_t h1 = quant_hash(c.type);
        ggml_set_quantize_fudge_factor(c.type, got);
        const uint64_t h2 = quant_hash(c.type);
        const bool wired = h1 != h0, restored = h2 == h0;
        printf("  %-8s default %.4f (upstream %.4f) %s | x1.1 changes bytes: %s | restore gives default bytes: %s\n",
               ggml_type_name(c.type), got, c.upstream_default, def_ok ? "ok" : "MISMATCH",
               wired ? "yes" : "NO -- the factor is not read", restored ? "yes" : "NO");
        fail += !def_ok + !wired + !restored;
    }
    // CONTROL: a type with default 1.0 must report exactly 1.0
    const float q4 = ggml_get_quantize_fudge_factor(GGML_TYPE_Q4_0);
    printf("  CONTROL q4_0 default %.4f (must be 1.0) %s\n", q4, q4 == 1.0f ? "ok" : "FAIL");
    fail += q4 != 1.0f;
    printf("%s -- %d failure(s)\n", fail ? "FAIL" : "PASS", fail);
    return fail ? 1 : 0;
#else
    printf("fudge-factor API not present in this build -- only hashes printed\n");
    return 1;
#endif
}

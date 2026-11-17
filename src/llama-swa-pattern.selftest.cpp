// Selftest: set_swa_pattern(N) must agree, layer for layer, with the modular formulas this tree
// already uses in its built graph builders. Nothing here is a performance claim.
//
// WHY THIS FILE EXISTS. This tree holds THREE representations of "is layer il sliding":
//   1. hparams.is_swa_impl[il]                     -- read by hparams.is_swa(), by src/models/*,
//                                                     by llama-kv-cache-iswa.cpp, and by the RoPE
//                                                     accessors added in fork 25a80af6c
//   2. hparams.n_swa_pattern + a modular formula   -- src/graphs/build_llama.cpp:44 computes
//                                                     `il % n_swa_pattern < (n_swa_pattern - 1)`
//   3. a hardcoded local constant                  -- src/graphs/build_cohere2.cpp:26 declares its
//                                                     own `sliding_window_pattern = 4` and applies
//                                                     the same formula
//
// Three architectures -- COHERE2, GEMMA3 and LLAMA4 -- set a pattern via (2) and never populate
// (1). That is not a live defect today, because their BUILT graph builders read (2) or (3). It is
// a latent one, and it blocks the current work: the moment src/models/ or the memory family enters
// the build, those files read (1) and get "no layer is sliding" for all three architectures.
//
// The fix is to make the loaders populate (1) as well. THAT FIX IS ONLY SAFE IF (1) AND (2) AGREE
// FOR EVERY LAYER -- otherwise it would change behaviour rather than merely record it. This file
// proves the equivalence instead of assuming it, and it checks BOTH directions: agreement where
// agreement is claimed, and disagreement where the semantics genuinely differ (dense_first), so a
// test that always passes is distinguishable from one that works.
//
// Build:  g++ -fsyntax-only -std=c++17 -I src -I include -I ggml/include -I ggml/src -I common \
//             src/llama-swa-pattern.selftest.cpp
// Run:    compile it to an executable and run; it returns non-zero on any failure.

#include "llama-hparams.h"

#include <cstdio>
#include <cstdint>

static int g_checks = 0;
static int g_failed = 0;

static void ck(const char * label, bool ok) {
    g_checks++;
    if (!ok) {
        g_failed++;
    }
    printf("  %-4s %s\n", ok ? "ok" : "FAIL", label);
}

// The formula src/graphs/build_llama.cpp:44 and src/graphs/build_cohere2.cpp:31 both use.
static bool formula_sliding(uint32_t il, uint32_t n_pattern) {
    return n_pattern == 0 || (il % n_pattern < (n_pattern - 1));
}

int main() {
    // ---- 1. EQUIVALENCE, the property the fix depends on -------------------------------------
    // For every pattern the three architectures use (COHERE2 4, GEMMA3 6, LLAMA4 4) and every
    // layer count in range, set_swa_pattern must reproduce the formula exactly.
    const uint32_t patterns[] = {1, 2, 3, 4, 5, 6, 8};
    const uint32_t layer_counts[] = {1, 4, 6, 26, 32, 34, 48, 62, 80};

    for (uint32_t p : patterns) {
        for (uint32_t nl : layer_counts) {
            llama_hparams hp;
            hp.n_layer = nl;
            hp.set_swa_pattern(p);
            bool all_match = true;
            for (uint32_t il = 0; il < nl; ++il) {
                if (hp.is_swa(il) != formula_sliding(il, p)) {
                    all_match = false;
                    printf("       MISMATCH pattern=%u n_layer=%u il=%u : is_swa=%d formula=%d\n",
                           p, nl, il, (int) hp.is_swa(il), (int) formula_sliding(il, p));
                }
            }
            char buf[128];
            snprintf(buf, sizeof(buf),
                     "set_swa_pattern(%u) == build_llama formula, n_layer=%u", p, nl);
            ck(buf, all_match);
        }
    }

    // ---- 2. THE THREE REAL CASES, spelled out ------------------------------------------------
    // COHERE2: pattern 4 -> layers 0,1,2 sliding, layer 3 dense, repeating. This is exactly what
    // src/graphs/build_cohere2.cpp:31 computes from its own local constant.
    {
        llama_hparams hp;
        hp.n_layer = 32;
        hp.set_swa_pattern(4);
        ck("COHERE2 pattern 4: layer 0 sliding", hp.is_swa(0));
        ck("COHERE2 pattern 4: layer 1 sliding", hp.is_swa(1));
        ck("COHERE2 pattern 4: layer 2 sliding", hp.is_swa(2));
        ck("COHERE2 pattern 4: layer 3 DENSE", !hp.is_swa(3));
        ck("COHERE2 pattern 4: layer 7 DENSE", !hp.is_swa(7));
        ck("COHERE2 pattern 4: layer 31 DENSE", !hp.is_swa(31));
    }
    // GEMMA3: pattern 6 -> five sliding then one dense.
    {
        llama_hparams hp;
        hp.n_layer = 62;
        hp.set_swa_pattern(6);
        ck("GEMMA3 pattern 6: layer 4 sliding", hp.is_swa(4));
        ck("GEMMA3 pattern 6: layer 5 DENSE", !hp.is_swa(5));
        ck("GEMMA3 pattern 6: layer 11 DENSE", !hp.is_swa(11));
    }
    // LLAMA4: pattern 4 -> "3 chunked - 1 full", the fork's own comment.
    {
        llama_hparams hp;
        hp.n_layer = 48;
        hp.set_swa_pattern(4);
        ck("LLAMA4 pattern 4: layer 2 chunked", hp.is_swa(2));
        ck("LLAMA4 pattern 4: layer 3 FULL", !hp.is_swa(3));
    }

    // ---- 3. NEGATIVE CONTROLS -- a test that cannot fail proves nothing ----------------------
    // dense_first INVERTS the phase. If this agreed with the default formula, the test above
    // would be insensitive to which variant was called.
    {
        llama_hparams hp;
        hp.n_layer = 8;
        hp.set_swa_pattern(4, /* dense_first = */ true);
        bool differs = false;
        for (uint32_t il = 0; il < 8; ++il) {
            if (hp.is_swa(il) != formula_sliding(il, 4)) {
                differs = true;
            }
        }
        ck("NEGATIVE CONTROL dense_first genuinely DIFFERS from the default", differs);
        ck("NEGATIVE CONTROL dense_first: layer 0 is dense", !hp.is_swa(0));
    }
    // Out-of-range must be false, matching the fork's stated convention at llama-hparams.h:836-841
    // (return false rather than upstream's GGML_ABORT), NOT crash.
    {
        llama_hparams hp;
        hp.n_layer = 4;
        hp.set_swa_pattern(4);
        ck("out-of-range il returns false, does not abort", !hp.is_swa(9999));
    }
    // A pattern of 1 means NO layer is sliding -- il % 1 < 0 is never true. Guards against an
    // off-by-one that would silently make every layer sliding.
    {
        llama_hparams hp;
        hp.n_layer = 8;
        hp.set_swa_pattern(1);
        bool none = true;
        for (uint32_t il = 0; il < 8; ++il) {
            if (hp.is_swa(il)) {
                none = false;
            }
        }
        ck("pattern 1 marks NO layer sliding", none);
    }
    // A pattern of 0 means EVERY layer is sliding.
    {
        llama_hparams hp;
        hp.n_layer = 8;
        hp.set_swa_pattern(0);
        bool all = true;
        for (uint32_t il = 0; il < 8; ++il) {
            if (!hp.is_swa(il)) {
                all = false;
            }
        }
        ck("pattern 0 marks EVERY layer sliding", all);
    }

    printf("\n%d checks, %d failed\n", g_checks, g_failed);
    printf("SELFTEST %s\n", g_failed == 0 ? "PASS" : "FAIL");
    return g_failed == 0 ? 0 : 1;
}

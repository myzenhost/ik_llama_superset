// Per-layer K/V cache type policy (src/llama-kv-layer-types.h) and the Boundary V default
// (src/llama-turbo-boundary-v.h). Both are pure, so they are tested without a model; the end-to-end check
// (the memory-module cache honouring -ctv-first/-ctv-last) is the perplexity A/B in the commit message.

#include "../src/llama-kv-layer-types.h"
#include "../src/llama-turbo-boundary-v.h"

#include <cstdio>

static int g_fail = 0;

#define CHECK(cond) do { if (!(cond)) { std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); g_fail++; } } while (0)

int main() {
    // --- llama_kv_layer_types_apply: first/last ranges, base type unchanged elsewhere
    {
        const llama_kv_layer_types p = { GGML_TYPE_F16, GGML_TYPE_F16, GGML_TYPE_Q8_0, GGML_TYPE_Q8_0,
                                         0, 0, 2, 2, 32 };
        int n_override = 0;
        for (int il = 0; il < 32; ++il) {
            ggml_type k = GGML_TYPE_Q8_0, v = GGML_TYPE_TURBO2_0;
            llama_kv_layer_types_apply(p, il, k, v);
            CHECK(k == GGML_TYPE_Q8_0);
            const bool edge = il < 2 || il >= 30;
            CHECK(v == (edge ? GGML_TYPE_Q8_0 : GGML_TYPE_TURBO2_0));
            n_override += v != GGML_TYPE_TURBO2_0;
        }
        CHECK(n_override == 4);
    }
    // an override equal to the base type, or with a non-positive count, does nothing (legacy-cache rule)
    {
        const llama_kv_layer_types p = { GGML_TYPE_Q8_0, GGML_TYPE_F16, GGML_TYPE_TURBO2_0, GGML_TYPE_Q8_0,
                                         4, 0, 3, -1, 32 };
        for (int il = 0; il < 32; ++il) {
            ggml_type k = GGML_TYPE_Q8_0, v = GGML_TYPE_TURBO2_0;
            llama_kv_layer_types_apply(p, il, k, v);
            CHECK(k == GGML_TYPE_Q8_0);
            CHECK(v == GGML_TYPE_TURBO2_0);
        }
    }
    // K overrides apply independently of V
    {
        const llama_kv_layer_types p = { GGML_TYPE_F16, GGML_TYPE_Q8_0, GGML_TYPE_F16, GGML_TYPE_F16,
                                         1, 3, 0, 0, 10 };
        ggml_type k = GGML_TYPE_Q4_0, v = GGML_TYPE_Q4_0;
        llama_kv_layer_types_apply(p, 0, k, v); CHECK(k == GGML_TYPE_F16);  CHECK(v == GGML_TYPE_Q4_0);
        k = GGML_TYPE_Q4_0; llama_kv_layer_types_apply(p, 5, k, v); CHECK(k == GGML_TYPE_Q4_0);
        k = GGML_TYPE_Q4_0; llama_kv_layer_types_apply(p, 7, k, v); CHECK(k == GGML_TYPE_Q8_0);
    }

    // --- Boundary V (opt-in)
    const ggml_type U = GGML_TYPE_F16; // "unset" first/last type, as llama_context_default_params leaves it
    const char * ON = "1";
    {   // not asked for -> nothing changes
        CHECK(!llama_turbo_boundary_v_resolve(GGML_TYPE_TURBO2_0, U, U, -1, -1, 32, nullptr, nullptr).applied);
    }
    {   // turbo2 V, opted in, nothing chosen -> first 2 + last 2 at q8_0
        auto r = llama_turbo_boundary_v_resolve(GGML_TYPE_TURBO2_0, U, U, -1, -1, 32, ON, nullptr);
        CHECK(r.applied);
        CHECK(r.type_v_first == GGML_TYPE_Q8_0 && r.n_v_first == 2);
        CHECK(r.type_v_last  == GGML_TYPE_Q8_0 && r.n_v_last  == 2);
    }
    {   // other V types are left alone
        const ggml_type others[] = { GGML_TYPE_TURBO3_0, GGML_TYPE_TURBO4_0, GGML_TYPE_Q8_0, GGML_TYPE_Q4_0, GGML_TYPE_F16 };
        for (ggml_type t : others) {
            auto r = llama_turbo_boundary_v_resolve(t, U, U, -1, -1, 32, ON, nullptr);
            CHECK(!r.applied);
            CHECK(r.n_v_first == -1 && r.n_v_last == -1);
        }
    }
    {   // an explicit -ctv-first or -ctv-last wins, untouched
        auto r = llama_turbo_boundary_v_resolve(GGML_TYPE_TURBO2_0, GGML_TYPE_F16, U, 4, -1, 32, ON, nullptr);
        CHECK(!r.applied);
        CHECK(r.type_v_first == GGML_TYPE_F16 && r.n_v_first == 4 && r.n_v_last == -1);
        auto r2 = llama_turbo_boundary_v_resolve(GGML_TYPE_TURBO2_0, U, GGML_TYPE_TURBO3_0, -1, 1, 32, ON, nullptr);
        CHECK(!r2.applied && r2.type_v_last == GGML_TYPE_TURBO3_0 && r2.n_v_last == 1);
    }
    {   // opt-in through either variable; "0", "" or anything else leaves the cache uniform
        CHECK( llama_turbo_boundary_v_resolve(GGML_TYPE_TURBO2_0, U, U, -1, -1, 32, "1", nullptr).applied);
        CHECK( llama_turbo_boundary_v_resolve(GGML_TYPE_TURBO2_0, U, U, -1, -1, 32, nullptr, "1").applied);
        CHECK(!llama_turbo_boundary_v_resolve(GGML_TYPE_TURBO2_0, U, U, -1, -1, 32, "0", nullptr).applied);
        CHECK(!llama_turbo_boundary_v_resolve(GGML_TYPE_TURBO2_0, U, U, -1, -1, 32, "", nullptr).applied);
    }
    {   // tiny models: below 8 layers the boundary would be half the model -> not applied
        CHECK(!llama_turbo_boundary_v_resolve(GGML_TYPE_TURBO2_0, U, U, -1, -1, 7, ON, nullptr).applied);
        CHECK( llama_turbo_boundary_v_resolve(GGML_TYPE_TURBO2_0, U, U, -1, -1, 8, ON, nullptr).applied);
    }

    // --- coverage WARN (row S69-ENGREV-APHRODITE-KVLAYER-COVERAGE-WARN-2026-10-27-01)
    // hybrid model, attention on every 4th layer (il%4==3): -ctk-first 2 reaches layers 0,1 which hold no KV, so the attention cache
    // changes nothing. Simulate the cache constructor: count the layers of the cache that the policy changed.
    {
        const llama_kv_layer_types p = { GGML_TYPE_Q8_0, GGML_TYPE_F16, GGML_TYPE_F16, GGML_TYPE_F16,
                                         2, 0, 0, 0, 32 };
        int64_t n_k = 0, n_v = 0;
        for (int il = 0; il < 32; ++il) {
            if (il % 4 != 3) continue; // recurrent layer: no KV
            ggml_type k = GGML_TYPE_F16, v = GGML_TYPE_F16;
            llama_kv_layer_types_apply(p, il, k, v);
            n_k += k != GGML_TYPE_F16;
            n_v += v != GGML_TYPE_F16;
        }
        CHECK(n_k == 0 && n_v == 0);
        const std::string w = llama_kv_layer_types_uncovered(p, GGML_TYPE_F16, GGML_TYPE_F16, n_k, n_v, true);
        CHECK(!w.empty());                                  // BEFORE: no such function and no log line; AFTER: one WARN text
        CHECK(w.find("-ctk-first 2") != std::string::npos); // names the flag and the requested count
        CHECK(w.find("-ctv") == std::string::npos);         // V has no override requested: not named
    }
    {   // control: dense model, the same flag reaches layers 0 and 1 -> no WARN
        const llama_kv_layer_types p = { GGML_TYPE_Q8_0, GGML_TYPE_F16, GGML_TYPE_F16, GGML_TYPE_F16,
                                         2, 0, 0, 0, 32 };
        int64_t n_k = 0, n_v = 0;
        for (int il = 0; il < 32; ++il) {
            ggml_type k = GGML_TYPE_F16, v = GGML_TYPE_F16;
            llama_kv_layer_types_apply(p, il, k, v);
            n_k += k != GGML_TYPE_F16;
            n_v += v != GGML_TYPE_F16;
        }
        CHECK(n_k == 2);
        CHECK(llama_kv_layer_types_uncovered(p, GGML_TYPE_F16, GGML_TYPE_F16, n_k, n_v, true).empty());
    }
    {   // no override requested (type == base, or count <= 0): never a WARN, even with zero layers changed
        const llama_kv_layer_types p = { GGML_TYPE_F16, GGML_TYPE_F16, GGML_TYPE_F16, GGML_TYPE_F16, 4, 4, 0, -1, 32 };
        CHECK(llama_kv_layer_types_uncovered(p, GGML_TYPE_F16, GGML_TYPE_F16, 0, 0, true).empty());
    }
    {   // MLA (no V tensor): a V override cannot cover anything and is not reported
        const llama_kv_layer_types p = { GGML_TYPE_F16, GGML_TYPE_F16, GGML_TYPE_Q8_0, GGML_TYPE_F16, 0, 0, 2, 0, 32 };
        CHECK(llama_kv_layer_types_uncovered(p, GGML_TYPE_F16, GGML_TYPE_F16, 0, 0, false).empty());
        CHECK(!llama_kv_layer_types_uncovered(p, GGML_TYPE_F16, GGML_TYPE_F16, 0, 0, true).empty());
    }

    if (g_fail) {
        std::fprintf(stderr, "test-kv-layer-types: %d failure(s)\n", g_fail);
        return 1;
    }
    std::printf("test-kv-layer-types: PASS\n");
    return 0;
}

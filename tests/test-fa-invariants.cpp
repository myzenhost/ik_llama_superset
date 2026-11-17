// test-fa-invariants.cpp -- flash-attention invariants taken from the OpenVINO intel_gpu / CM kernels
// (row S69-F2-TEST-FA-INVARIANTS-POISON-ALLMASKED-SINK-SPLIT-2026-10-06-01).
//
// Runs ggml_flash_attn_ext on a ggml backend (default CPU; pass a backend name such as CUDA0 as argv[1] to run the
// same cases on a GPU, which is how the parent runs the DEVICE-PENDING half) and compares with a double-precision
// reference that is DEFINED here:
//   * a KV row whose mask entry is -inf takes no part in the result, whatever its K and V bytes hold (even NaN);
//   * a query row with no visible KV row and no sink returns zeros, never NaN;
//   * a sink adds exp(sink - max) to the denominator and nothing to the numerator.
//
// Cases, each with a partner that must behave the OTHER way so the case is known to be sensitive:
//   T1 poisoned unused KV rows (NaN in K and V beyond n_valid, mask -inf)   partner: same rows made visible -> NaN
//   T2 one query row fully masked among live rows                            partner: finite sentinel mask -> not zero
//   T3 sink far above the logits (result ~ 0) and a mid-size sink           partner: reference with the sink counted twice
//   T4 padded mask rows (n_tokens padded to GGML_KQ_MASK_PAD) are never read for the result of the real rows
// Exit code 0 = every control behaved as stated and every invariant held; failures are printed with the case name.

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <random>
#include <string>
#include <vector>

static int g_fail = 0;
#define CHECK(c, ...) do { if (!(c)) { std::fprintf(stderr, "FAIL %s:%d: %s -- ", __FILE__, __LINE__, #c); std::fprintf(stderr, __VA_ARGS__); std::fprintf(stderr, "\n"); ++g_fail; } } while (0)

static const int HS = 64;    // head size (K and V)
static const int NH = 4;     // query heads
static const int NHKV = 2;   // kv heads (GQA factor 2)
static const int NB = 5;     // query tokens
static const int KV = 96;    // KV rows handed to the kernel (valid rows + padding)

struct fa_case {
    std::vector<float>    q;       // [NH][NB][HS]
    std::vector<uint16_t> k, v;    // [NHKV][KV][HS] f16
    std::vector<uint16_t> mask;    // [pad(NB)][KV]   f16 additive
    std::vector<float>    sinks;   // [NH] (empty = none)
};

static uint16_t f16(float x) { return ggml_fp32_to_fp16(x); }
static float    f32(uint16_t x) { return ggml_fp16_to_fp32(x); }

static fa_case make_case(int n_valid, uint32_t seed) {
    std::mt19937 rng(seed);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    fa_case c;
    c.q.resize((size_t) NH * NB * HS);
    for (auto & x : c.q) { x = nd(rng); }
    c.k.resize((size_t) NHKV * KV * HS);
    c.v.resize((size_t) NHKV * KV * HS);
    for (auto & x : c.k) { x = f16(nd(rng)); }
    for (auto & x : c.v) { x = f16(nd(rng)); }
    const int npad = GGML_PAD(NB, GGML_KQ_MASK_PAD);
    c.mask.assign((size_t) npad * KV, f16(0.0f));
    for (int t = 0; t < npad; ++t) {
        for (int j = n_valid; j < KV; ++j) {
            c.mask[(size_t) t * KV + j] = f16(-INFINITY);
        }
    }
    return c;
}

// double-precision reference with the contract above
static std::vector<float> reference(const fa_case & c, float scale) {
    std::vector<float> out((size_t) NB * NH * HS, 0.0f);
    for (int t = 0; t < NB; ++t) {
        for (int h = 0; h < NH; ++h) {
            const int hk = h / (NH / NHKV);
            const float * q = &c.q[((size_t) h * NB + t) * HS];
            std::vector<double> s(KV, -INFINITY);
            double m = -INFINITY;
            for (int j = 0; j < KV; ++j) {
                const float mj = f32(c.mask[(size_t) t * KV + j]);
                if (std::isinf(mj) && mj < 0) { continue; }
                const uint16_t * k = &c.k[((size_t) hk * KV + j) * HS];
                double d = 0;
                for (int i = 0; i < HS; ++i) { d += (double) q[i] * (double) f32(k[i]); }
                s[j] = d * scale + mj;
                m = std::max(m, s[j]);
            }
            double sink = -INFINITY;
            if (!c.sinks.empty()) { sink = c.sinks[h]; m = std::max(m, sink); }
            if (std::isinf(m)) { continue; } // nothing visible, no sink: zeros
            double den = std::isinf(sink) ? 0.0 : std::exp(sink - m);
            std::vector<double> acc(HS, 0.0);
            for (int j = 0; j < KV; ++j) {
                if (std::isinf(s[j])) { continue; }
                const double p = std::exp(s[j] - m);
                den += p;
                const uint16_t * v = &c.v[((size_t) hk * KV + j) * HS];
                for (int i = 0; i < HS; ++i) { acc[i] += p * (double) f32(v[i]); }
            }
            for (int i = 0; i < HS; ++i) { out[((size_t) t * NH + h) * HS + i] = (float) (acc[i] / den); }
        }
    }
    return out;
}

static std::vector<float> run_backend(ggml_backend_t be, const fa_case & c, float scale, bool * ok) {
    *ok = false;
    ggml_init_params ip = { ggml_tensor_overhead() * 16 + ggml_graph_overhead(), nullptr, true };
    ggml_context * ctx = ggml_init(ip);
    ggml_tensor * q = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, HS, NB, NH, 1);
    ggml_tensor * k = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, HS, KV, NHKV, 1);
    ggml_tensor * v = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, HS, KV, NHKV, 1);
    ggml_tensor * m = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, KV, GGML_PAD(NB, GGML_KQ_MASK_PAD), 1, 1);
    ggml_tensor * s = c.sinks.empty() ? nullptr : ggml_new_tensor_1d(ctx, GGML_TYPE_F32, NH);
    ggml_tensor * out = ggml_flash_attn_ext(ctx, q, k, v, m, scale, 0.0f, 0.0f);
    ggml_flash_attn_ext_set_prec(out, GGML_PREC_F32);
    if (s) { ggml_flash_attn_ext_add_sinks(out, s); }
    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, out);
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, be);
    if (!buf) { ggml_free(ctx); return {}; }
    ggml_backend_tensor_set(q, c.q.data(), 0, c.q.size() * sizeof(float));
    ggml_backend_tensor_set(k, c.k.data(), 0, c.k.size() * sizeof(uint16_t));
    ggml_backend_tensor_set(v, c.v.data(), 0, c.v.size() * sizeof(uint16_t));
    ggml_backend_tensor_set(m, c.mask.data(), 0, c.mask.size() * sizeof(uint16_t));
    if (s) { ggml_backend_tensor_set(s, c.sinks.data(), 0, c.sinks.size() * sizeof(float)); }
    const ggml_status st = ggml_backend_graph_compute(be, gf);
    std::vector<float> res((size_t) NB * NH * HS);
    if (st == GGML_STATUS_SUCCESS) {
        ggml_backend_tensor_get(out, res.data(), 0, res.size() * sizeof(float));
        *ok = true;
    }
    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
    return res;
}

static bool any_nonfinite(const std::vector<float> & x) {
    for (float f : x) { if (!std::isfinite(f)) { return true; } }
    return false;
}

static double max_abs_diff(const std::vector<float> & a, const std::vector<float> & b) {
    double d = 0;
    for (size_t i = 0; i < a.size(); ++i) { d = std::max(d, (double) std::fabs(a[i] - b[i])); }
    return d;
}

// where the largest difference sits, and how many outputs are non-finite: a failing invariant must say WHICH query row
// and head moved, because that is what separates "the kernel read the poison" from "the harness reads the wrong layout"
static void describe_diff(const char * what, const std::vector<float> & r, const std::vector<float> & ref) {
    double worst = 0; int wt = -1, wh = -1; int n_nonfinite = 0, n_over = 0;
    for (int t = 0; t < NB; ++t) {
        for (int h = 0; h < NH; ++h) {
            double d = 0;
            for (int i = 0; i < HS; ++i) {
                const size_t idx = ((size_t) t * NH + h) * HS + i;
                if (!std::isfinite(r[idx])) { ++n_nonfinite; continue; }
                d = std::max(d, (double) std::fabs(r[idx] - ref[idx]));
            }
            if (d > 2e-3) { ++n_over; }
            if (d > worst) { worst = d; wt = t; wh = h; }
        }
    }
    std::printf("  [diag %s] (query row, head) pairs over tol: %d of %d, worst %.4g at t=%d h=%d, non-finite outputs: %d\n",
                what, n_over, NB * NH, worst, wt, wh, n_nonfinite);
}

int main(int argc, char ** argv) {
    const std::string name = argc > 1 ? argv[1] : "CPU";
    // argv[2] == "generic": force ggml's generic FA kernel instead of iqk (same switch the CPU backend documents)
    if (argc > 2 && std::string(argv[2]) == "generic") {
#if defined(_WIN32)
        _putenv_s("GGML_NO_IQK_FA", "1");
#else
        setenv("GGML_NO_IQK_FA", "1", 1);
#endif
    }
    std::printf("fa path: %s\n", (argc > 2 && std::string(argv[2]) == "generic") ? "generic (GGML_NO_IQK_FA=1)" : "default (iqk when it accepts the shape)");
    ggml_backend_t be = nullptr;
    if (name == "CPU") {
        be = ggml_backend_cpu_init();
        ggml_backend_cpu_set_n_threads(be, 1);
    } else {
        be = ggml_backend_init_by_name(name.c_str(), nullptr);
    }
    if (!be) { std::fprintf(stderr, "backend %s not available\n", name.c_str()); return 2; }
    std::printf("backend: %s\n", ggml_backend_name(be));
    const float scale = 1.0f / std::sqrt((float) HS);
    // the generic kernel rounds P to f16 before the V product: measured 2.05e-3 / 2.36e-3 on the clean case and the mid sink
    // (temp/_f2g_1011_fa_runb.txt), so its bound is 5e-3; iqk accumulates in f32 and measured 3e-7
    const bool generic_path = argc > 2 && std::string(argv[2]) == "generic";
    const double tol = generic_path ? 5e-3 : 2e-3;
    // argv[3] selects the group: "all" (default), "finite" (T0, finite stale-data variants, T2-T4) or "nan" (T0 and every
    // NaN-poison check). Measured 2026-10-11 (temp/_f2g_1011_fa_run3b.txt): the generic kernel skips -inf keys and contains
    // NaN poison in masked rows; the iqk kernel ADDS the mask (iqk_fa_templates.h:883-884), so K=NaN gives a finite wrong
    // result (0.9547) and V=NaN gives all-NaN (0*NaN). The "nan" group therefore fails on the default CPU path and CMake
    // registers it WILL_FAIL (row S69-F2-IQK-FA-MASKED-KEYS-NOT-SKIPPED-NAN-KV-2026-10-11-01); fixing iqk turns it red and
    // the WILL_FAIL is then removed. Every assertion stays a real CHECK.
    const std::string group = argc > 3 ? argv[3] : "all";
    const bool run_nan = group == "all" || group == "nan";
    const bool run_finite = group == "all" || group == "finite";

    // ---- T1 poisoned unused rows -------------------------------------------------------------------------------
    {
        const int n_valid = 70;
        fa_case clean = make_case(n_valid, 11);
        const std::vector<float> ref = reference(clean, scale);
        fa_case pois = clean;
        const uint16_t nan16 = f16(std::numeric_limits<float>::quiet_NaN());
        for (int hk = 0; hk < NHKV; ++hk) {
            for (int j = n_valid; j < KV; ++j) {
                for (int i = 0; i < HS; ++i) {
                    pois.k[((size_t) hk * KV + j) * HS + i] = nan16;
                    pois.v[((size_t) hk * KV + j) * HS + i] = nan16;
                }
            }
        }
        // T0: the same data WITHOUT poison must match the reference, otherwise a T1 difference is the harness or the layout
        bool ok0 = false;
        const std::vector<float> r0 = run_backend(be, clean, scale, &ok0);
        CHECK(ok0, "T0 backend declined the op");
        if (ok0) {
            CHECK(max_abs_diff(r0, ref) < tol, "T0 clean case differs from the reference by %g (harness/layout error, T1-T4 are meaningless)", max_abs_diff(r0, ref));
            describe_diff("T0 clean vs reference", r0, ref);
        }
        bool ok = false;
        const std::vector<float> r = run_backend(be, pois, scale, &ok);
        CHECK(ok, "T1 backend declined the op");
        if (ok) {
            describe_diff("T1 poisoned vs reference", r, ref);
            if (run_nan) {
                CHECK(!any_nonfinite(r), "T1 NaN from masked, poisoned KV rows reached the output");
                CHECK(max_abs_diff(r, ref) < tol, "T1 result differs from the clean reference by %g", max_abs_diff(r, ref));
            }
        }
        // partner: the same poison made VISIBLE must reach the output, otherwise T1 proves nothing
        fa_case vis = pois;
        std::fill(vis.mask.begin(), vis.mask.end(), f16(0.0f));
        bool ok2 = false;
        const std::vector<float> r2 = run_backend(be, vis, scale, &ok2);
        if (ok2) { describe_diff("T1 partner (poison visible) vs reference", r2, ref); }
        // visible poison must have SOME effect (NaN, or a result far from the clean one), else the poison never reached the kernel
        CHECK(ok2 && (any_nonfinite(r2) || max_abs_diff(r2, ref) > 10 * tol), "T1 partner: visible NaN rows had no effect on the output, the poison never reached the kernel");
        // Which half of the poison leaks, and does a FINITE stale value leak too? A real KV cache never holds NaN in
        // its padding but always holds stale finite data, so the finite variants are the ones that matter in service
        // and the NaN variants say whether the kernel merely survives or actually contains the poison.
        struct variant { const char * name; float kval, vval; bool k, v; };
        const variant variants[] = {
            { "K NaN only",    std::numeric_limits<float>::quiet_NaN(), 0.0f, true,  false },
            { "V NaN only",    0.0f, std::numeric_limits<float>::quiet_NaN(), false, true  },
            { "K+V +6e4",      6.0e4f, 6.0e4f, true, true },
            { "K+V zero",      0.0f, 0.0f, true, true },
        };
        for (const variant & var : variants) {
            const bool is_nan_variant = std::string(var.name).find("NaN") != std::string::npos;
            if (is_nan_variant ? !run_nan : !run_finite) { continue; }
            fa_case pv = clean;
            for (int hk = 0; hk < NHKV; ++hk) {
                for (int j = n_valid; j < KV; ++j) {
                    for (int i = 0; i < HS; ++i) {
                        if (var.k) { pv.k[((size_t) hk * KV + j) * HS + i] = f16(var.kval); }
                        if (var.v) { pv.v[((size_t) hk * KV + j) * HS + i] = f16(var.vval); }
                    }
                }
            }
            bool okv = false;
            const std::vector<float> rv = run_backend(be, pv, scale, &okv);
            CHECK(okv, "T1 variant '%s': backend declined the op", var.name);
            if (okv) {
                std::printf("  [T1 variant '%s'] non-finite outputs: %d, max |diff| over finite outputs vs clean reference: %.4g\n",
                            var.name, (int) std::count_if(rv.begin(), rv.end(), [](float f) { return !std::isfinite(f); }), max_abs_diff(rv, ref));
                CHECK(!any_nonfinite(rv) && max_abs_diff(rv, ref) < 10 * tol, "T1 variant '%s': masked rows with this content changed the output by %g", var.name, max_abs_diff(rv, ref));
            }
        }
        std::printf("T1 poisoned unused KV rows\n");
    }
    if (!run_finite) {
        ggml_backend_free(be);
        if (g_fail) { std::fprintf(stderr, "test-fa-invariants[%s]: %d failure(s)\n", group.c_str(), g_fail); return 1; }
        std::printf("test-fa-invariants[%s]: held\n", group.c_str());
        return 0;
    }

    // ---- T2 fully masked query row ------------------------------------------------------------------------------
    {
        fa_case c = make_case(70, 12);
        const int dead = 3;
        for (int j = 0; j < KV; ++j) { c.mask[(size_t) dead * KV + j] = f16(-INFINITY); }
        const std::vector<float> ref = reference(c, scale);
        bool ok = false;
        const std::vector<float> r = run_backend(be, c, scale, &ok);
        CHECK(ok, "T2 backend declined the op");
        if (ok) {
            // live rows must be unaffected, the dead row must be finite (zeros by the contract above)
            bool live_bad = false, dead_nonfinite = false, dead_zero = true;
            for (int t = 0; t < NB; ++t) {
                for (int h = 0; h < NH; ++h) {
                    for (int i = 0; i < HS; ++i) {
                        const size_t idx = ((size_t) t * NH + h) * HS + i;
                        if (t == dead) {
                            if (!std::isfinite(r[idx])) { dead_nonfinite = true; }
                            if (r[idx] != 0.0f) { dead_zero = false; }
                        } else if (!std::isfinite(r[idx]) || std::fabs(r[idx] - ref[idx]) > tol) {
                            live_bad = true;
                        }
                    }
                }
            }
            CHECK(!live_bad, "T2 a fully masked row disturbed the live rows");
            CHECK(!dead_nonfinite, "T2 fully masked row produced NaN/Inf");
            CHECK(dead_zero, "T2 fully masked row is finite but not zero (contract: zeros)");
        }
        // partner: a FINITE sentinel instead of -inf makes the dead row a plain average over masked rows, not zeros
        fa_case sent = c;
        for (int j = 0; j < KV; ++j) { sent.mask[(size_t) dead * KV + j] = f16(-60000.0f); }
        bool ok2 = false;
        const std::vector<float> r2 = run_backend(be, sent, scale, &ok2);
        bool nonzero = false;
        if (ok2) {
            for (int h = 0; h < NH && !nonzero; ++h) {
                for (int i = 0; i < HS; ++i) { if (r2[((size_t) dead * NH + h) * HS + i] != 0.0f) { nonzero = true; break; } }
            }
        }
        CHECK(ok2 && nonzero, "T2 partner: a finite -60000 sentinel should give a non-zero average (control for the sentinel rule)");
        std::printf("T2 fully masked row\n");
    }

    // ---- T3 sinks ---------------------------------------------------------------------------------------------
    {
        fa_case c = make_case(70, 13);
        c.sinks.assign(NH, 0.0f);
        for (int h = 0; h < NH; ++h) { c.sinks[h] = 0.5f * h; }              // mid-size sinks
        const std::vector<float> ref_mid = reference(c, scale);
        bool ok = false;
        const std::vector<float> r = run_backend(be, c, scale, &ok);
        CHECK(ok, "T3 backend declined the op with sinks");
        if (ok) {
            CHECK(!any_nonfinite(r), "T3 mid sink: non-finite output");
            CHECK(max_abs_diff(r, ref_mid) < tol, "T3 mid sink: differs from reference by %g", max_abs_diff(r, ref_mid));
        }
        fa_case big = c;
        big.sinks.assign(NH, 30.0f);                                          // far above any logit: output ~ 0
        const std::vector<float> ref_big = reference(big, scale);
        bool ok2 = false;
        const std::vector<float> r2 = run_backend(be, big, scale, &ok2);
        CHECK(ok2, "T3 backend declined the op with a big sink");
        if (ok2) {
            CHECK(!any_nonfinite(r2), "T3 big sink: non-finite output");
            CHECK(max_abs_diff(r2, ref_big) < tol, "T3 big sink: differs from reference by %g", max_abs_diff(r2, ref_big));
            double mx = 0; for (float f : r2) { mx = std::max(mx, (double) std::fabs(f)); }
            CHECK(mx < 1e-3, "T3 big sink should drive the output to ~0, max |out| = %g", mx);
        }
        // partner: an implementation that counts the sink twice (a split-KV kernel that adds it in every split would
        // do exactly that with two splits) must be distinguishable from the correct one by this case
        fa_case twice = c;
        for (auto & x : twice.sinks) { x = x + (float) std::log(2.0); }       // exp(sink)*2 == sink counted twice
        const std::vector<float> ref_twice = reference(twice, scale);
        CHECK(max_abs_diff(ref_twice, ref_mid) > 0.01, "T3 partner: counting the sink twice is not distinguishable (%g)", max_abs_diff(ref_twice, ref_mid));
        std::printf("T3 sinks\n");
    }

    // ---- T4 padded mask rows ------------------------------------------------------------------------------------
    {
        fa_case c = make_case(70, 14);
        const int npad = GGML_PAD(NB, GGML_KQ_MASK_PAD);
        const std::vector<float> ref = reference(c, scale);
        fa_case junk = c;
        for (int t = NB; t < npad; ++t) {
            for (int j = 0; j < KV; ++j) { junk.mask[(size_t) t * KV + j] = f16(std::numeric_limits<float>::quiet_NaN()); }
        }
        bool ok = false;
        const std::vector<float> r = run_backend(be, junk, scale, &ok);
        CHECK(ok, "T4 backend declined the op");
        if (ok) {
            CHECK(!any_nonfinite(r), "T4 NaN in the padded mask rows leaked into the real rows");
            CHECK(max_abs_diff(r, ref) < tol, "T4 padded mask rows changed the result by %g", max_abs_diff(r, ref));
        }
        std::printf("T4 padded mask rows\n");
    }

    ggml_backend_free(be);
    if (g_fail) { std::fprintf(stderr, "test-fa-invariants: %d failure(s)\n", g_fail); return 1; }
    std::printf("test-fa-invariants: all invariants held\n");
    return 0;
}

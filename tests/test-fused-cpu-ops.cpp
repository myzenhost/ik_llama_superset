// Correctness harness for the CPU kernels of this fork's FUSED_* ops.
//
// GATE-CLAIM-OK: "CPU" throughout this file names a BACKEND SOURCE DIRECTORY and which of the
// backends contains an op's implementation. Nothing here claims a device is unavailable or
// contended -- this test runs its own graphs on the host and touches no accelerator.
//
// WHY THIS EXISTS. GGML_OP_FUSED_RMS_NORM and GGML_OP_FUSED_MUL_UNARY were implemented for CUDA
// and Vulkan and never for the host backend (measured: 32 of 134 ops have no host kernel, row
// S69-GGML-CPU-MISSING-32-OF-134-OPS-2026-09-14-01). Host kernels were added 2026-09-14. A model
// then ran end to end and produced DEGENERATE output -- the same token twelve times -- which is
// the precise situation where "it no longer crashes" says nothing about whether the arithmetic is
// right. A crash is loud; wrong numerics are silent.
//
// WHAT IT COMPARES, and why this is a real oracle rather than a restatement. Each fused op is
// checked against the SAME COMPUTATION EXPRESSED WITH ggml's ORDINARY OPS -- rms_norm then mul,
// silu then mul -- built in the same graph, on the same inputs, executed by the same backend.
// Those ordinary ops are long-standing and independently exercised, so a disagreement localises to
// the fused kernel. This does NOT test the fused op against itself, and it does not depend on any
// model, tokenizer or reference output file.
//
// CONTROLS, because a comparison that cannot fail proves nothing:
//   * a NEGATIVE control deliberately perturbs one input and asserts the check FAILS -- if the
//     comparator cannot report a difference, every PASS below is meaningless.
//   * each test asserts the constructor actually produced a FUSED node; these constructors
//     decompose themselves for shapes they do not fuse, and a decomposed node would silently make
//     the test compare the reference against itself.
//   * inputs are non-trivial (mixed signs, values far from zero), so a kernel returning its input
//     unchanged, or zero, would be caught.
//
// TOLERANCE: the fused and unfused forms are not required to be bit-identical -- they multiply in a
// different association, which differs in floating point. The bound is relative and tight enough
// that a wrong formula (operand swapped, gate taken from the wrong half of a row, missing epsilon)
// cannot pass it.

#include "ggml.h"
#include "ggml-cpu.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

static int g_failures = 0;

static void fill(std::vector<float> & v, std::mt19937 & rng, float lo = -3.0f, float hi = 3.0f) {
    std::uniform_real_distribution<float> d(lo, hi);
    for (auto & x : v) {
        x = d(rng);
    }
}

// Returns the largest relative difference. Reported rather than thresholded inside, so the number
// appears in the output whether the check passes or fails.
static double max_rel_diff(const float * a, const float * b, size_t n) {
    double worst = 0.0;
    for (size_t i = 0; i < n; i++) {
        const double denom = std::max(1e-4, (double) std::max(std::fabs(a[i]), std::fabs(b[i])));
        const double rel   = std::fabs((double) a[i] - (double) b[i]) / denom;
        if (rel > worst) {
            worst = rel;
        }
    }
    return worst;
}

static bool check(const char * name, const float * got, const float * ref, size_t n, double tol) {
    // A NaN is called out separately: max_rel_diff would propagate it, but a NaN here is a
    // different defect from numerical drift and should not be reported as a large difference.
    for (size_t i = 0; i < n; i++) {
        if (std::isnan(got[i]) || std::isinf(got[i])) {
            printf("  FAIL %-34s produced %s at element %zu\n", name,
                   std::isnan(got[i]) ? "NaN" : "Inf", i);
            g_failures++;
            return false;
        }
    }
    const double d = max_rel_diff(got, ref, n);
    const bool ok = d <= tol;
    printf("  %-4s %-34s max rel diff = %.3e (tol %.1e)\n", ok ? "ok" : "FAIL", name, d, tol);
    if (!ok) {
        g_failures++;
    }
    return ok;
}

struct graph_runner {
    ggml_context * ctx = nullptr;
    std::vector<uint8_t> buf;

    explicit graph_runner(size_t mem_mb = 256) : buf(mem_mb*1024*1024) {
        ggml_init_params p = { buf.size(), buf.data(), false };
        ctx = ggml_init(p);
    }
    ~graph_runner() { if (ctx) ggml_free(ctx); }

    void run(std::vector<ggml_tensor *> outs, int n_threads = 2) {
        ggml_cgraph * gf = ggml_new_graph(ctx);
        for (auto * t : outs) {
            ggml_build_forward_expand(gf, t);
        }
        ggml_graph_compute_with_ctx(ctx, gf, n_threads);
    }
};

// ---------------------------------------------------------------------------------------------
// FUSED_RMS_NORM:  fused(a, b, eps)  ==  rms_norm(a, eps) * b
// b is ONE ROW of width ne0 -- the only shape the constructor produces a fused node for; anything
// else it decomposes itself (ggml_fused_rms_norm_impl, ggml/src/ggml.c).
static void test_fused_rms_norm(std::mt19937 & rng, int64_t ne0, int64_t ne1) {
    graph_runner g;
    const float eps = 1e-5f;

    ggml_tensor * a = ggml_new_tensor_2d(g.ctx, GGML_TYPE_F32, ne0, ne1);
    ggml_tensor * b = ggml_new_tensor_1d(g.ctx, GGML_TYPE_F32, ne0);

    std::vector<float> va(ne0*ne1), vb(ne0);
    fill(va, rng);
    fill(vb, rng, 0.25f, 2.0f);   // weights: positive and away from zero, as a norm weight is
    memcpy(a->data, va.data(), va.size()*sizeof(float));
    memcpy(b->data, vb.data(), vb.size()*sizeof(float));

    ggml_tensor * fused = ggml_fused_rms_norm(g.ctx, a, b, eps);
    ggml_tensor * ref   = ggml_mul(g.ctx, ggml_rms_norm(g.ctx, a, eps), b);

    g.run({fused, ref});

    // If the constructor decomposed instead of fusing, this would compare the reference against
    // itself and pass regardless of what the kernel does.
    if (fused->op != GGML_OP_FUSED_RMS_NORM) {
        printf("  FAIL %-34s constructor did not produce a fused node (op = %s)\n",
               "fused_rms_norm shape check", ggml_op_name(fused->op));
        g_failures++;
        return;
    }

    char name[64];
    snprintf(name, sizeof(name), "fused_rms_norm %lldx%lld", (long long) ne0, (long long) ne1);
    check(name, (const float *) fused->data, (const float *) ref->data, (size_t) (ne0*ne1), 2e-5);
}

// ---------------------------------------------------------------------------------------------
// FUSED_MUL_UNARY shape 1: a and b the same shape -> unary(a) * b
static void test_fused_mul_unary_same(std::mt19937 & rng, ggml_unary_op op, const char * opname,
                                      int64_t ne0, int64_t ne1) {
    graph_runner g;

    ggml_tensor * a = ggml_new_tensor_2d(g.ctx, GGML_TYPE_F32, ne0, ne1);
    ggml_tensor * b = ggml_new_tensor_2d(g.ctx, GGML_TYPE_F32, ne0, ne1);

    std::vector<float> va(ne0*ne1), vb(ne0*ne1);
    fill(va, rng);
    fill(vb, rng);
    memcpy(a->data, va.data(), va.size()*sizeof(float));
    memcpy(b->data, vb.data(), vb.size()*sizeof(float));

    ggml_tensor * fused = ggml_fused_mul_unary(g.ctx, a, b, op);

    ggml_tensor * un = nullptr;
    switch (op) {
        case GGML_UNARY_OP_SILU: un = ggml_silu(g.ctx, a); break;
        case GGML_UNARY_OP_RELU: un = ggml_relu(g.ctx, a); break;
        case GGML_UNARY_OP_GELU: un = ggml_gelu(g.ctx, a); break;
        default: printf("  FAIL unhandled op in the test itself\n"); g_failures++; return;
    }
    ggml_tensor * ref = ggml_mul(g.ctx, un, b);

    g.run({fused, ref});

    if (fused->op != GGML_OP_FUSED_MUL_UNARY) {
        printf("  FAIL %-34s constructor did not produce a fused node (op = %s)\n",
               opname, ggml_op_name(fused->op));
        g_failures++;
        return;
    }

    char name[64];
    snprintf(name, sizeof(name), "fused_mul_%s same-shape", opname);
    // One bound for every op routed here. GELU is deliberately NOT routed here: its ggml reference
    // is an f16 table (GGML_GELU_FP16, vec.h:46) and cannot meet this bound for reasons that have
    // nothing to do with the kernel, so it gets its own test against the exact expression below.
    // Loosening the bound here to accommodate it would have weakened SILU and RELU too.
    GGML_ASSERT(op != GGML_UNARY_OP_GELU);
    check(name, (const float *) fused->data, (const float *) ref->data, (size_t) (ne0*ne1), 2e-5);
}

// ---------------------------------------------------------------------------------------------
// GELU needs a DIFFERENT ORACLE, and the reason is measured rather than assumed.
//
// ggml/src/ggml-cpu/vec.h:46 defines GGML_GELU_FP16 UNCONDITIONALLY. So ggml_vec_gelu_f32
// (vec.h:988) does not evaluate gelu: it rounds x to f16, indexes a 65536-entry table, and
// returns an f16 result. That is the path ggml_gelu() takes, so using it as the reference bounds
// this test at f16 precision -- and the input rounding is amplified wherever gelu is steep
// relative to its own magnitude.
//
// The fused kernel, by contrast, evaluates 0.5*x*(1+tanh(sqrt(2/pi)*x*(1+0.044715*x*x))) in f32 --
// the SAME expression as ggml_gelu_f32 (vec.h:969), which is also the expression the table itself
// is built from (ggml-cpu.c:3932). So "fused vs table" measures the TABLE, not the kernel.
//
// This test therefore computes the exact expression HERE, in f64, and checks:
//   1. fused vs exact            -- the real check on the kernel, at the same 2e-5 bound as SILU
//   2. ggml's gelu op vs exact   -- how coarse the table reference actually is, MEASURED
//   3. an assertion that (2) is far larger than (1), which is what localises any remaining
//      "fused vs ggml op" difference to the reference rather than to the kernel
// If the kernel had a wrong constant, a swapped operand or a missing term, check 1 fails. This is
// a tighter test than the one it replaces, not a looser one.
static void test_fused_mul_gelu_exact(std::mt19937 & rng, int64_t ne0, int64_t ne1) {
    graph_runner g;

    ggml_tensor * a = ggml_new_tensor_2d(g.ctx, GGML_TYPE_F32, ne0, ne1);
    ggml_tensor * b = ggml_new_tensor_2d(g.ctx, GGML_TYPE_F32, ne0, ne1);

    const size_t n = (size_t) (ne0*ne1);
    std::vector<float> va(n), vb(n);
    fill(va, rng);
    fill(vb, rng);
    memcpy(a->data, va.data(), n*sizeof(float));
    memcpy(b->data, vb.data(), n*sizeof(float));

    ggml_tensor * fused    = ggml_fused_mul_unary(g.ctx, a, b, GGML_UNARY_OP_GELU);
    ggml_tensor * ggml_ref = ggml_mul(g.ctx, ggml_gelu(g.ctx, a), b);

    g.run({fused, ggml_ref});

    if (fused->op != GGML_OP_FUSED_MUL_UNARY) {
        printf("  FAIL %-34s constructor did not produce a fused node (op = %s)\n",
               "fused_mul_gelu", ggml_op_name(fused->op));
        g_failures++;
        return;
    }

    // The exact expression, evaluated independently of ggml entirely.
    std::vector<float> exact(n);
    for (size_t i = 0; i < n; i++) {
        const double x = (double) va[i];
        const double t = std::tanh(0.79788456080286535587989211986876 * x * (1.0 + 0.044715*x*x));
        exact[i] = (float) (0.5*x*(1.0 + t) * (double) vb[i]);
    }

    const float * f = (const float *) fused->data;
    const float * r = (const float *) ggml_ref->data;

    const double d_fused_exact = max_rel_diff(f, exact.data(), n);
    const double d_ggml_exact  = max_rel_diff(r, exact.data(), n);
    const double d_fused_ggml  = max_rel_diff(f, r, n);

    check("fused_mul_gelu vs EXACT", f, exact.data(), n, 2e-5);

    printf("       ggml's own gelu op vs EXACT        = %.3e   <- the f16 table's error, measured\n",
           d_ggml_exact);
    printf("       fused vs ggml's gelu op            = %.3e\n", d_fused_ggml);

    // Localisation. If the table were NOT the coarser of the two, the story above would be wrong
    // and the fused-vs-ggml difference would need a different explanation.
    const bool localised = d_ggml_exact > 10.0*d_fused_exact;
    printf("  %-4s %-34s table error is %.0fx the kernel's\n",
           localised ? "ok" : "FAIL", "LOCALISATION (table, not kernel)",
           d_fused_exact > 0.0 ? d_ggml_exact/d_fused_exact : 0.0);
    if (!localised) {
        printf("       the reference is NOT demonstrably coarser than the kernel -- the\n");
        printf("       fused-vs-ggml difference is unexplained and must not be dismissed\n");
        g_failures++;
    }
}

// ---------------------------------------------------------------------------------------------
// GATED_GROUP_RMS_NORM (2026-10-16, row S69-MODULAR-MAX-PICK-FUSED-GATED-GROUP-RMSNORM-2026-10-15-01):
//     fused(y, z, w, eps, n_group) == rms_norm_per_group(silu(z) * y) * w
// Mamba-2 RMSNormGated with norm_before_gate=False (Modular MAX gated_group_rmsnorm.mojo:24-33,
// HF MambaRMSNormGated). Three oracles, each able to fail on its own:
//   1. CHAIN-CLASS  the exact op chain src/models/mamba-base.cpp builds (cont(z), swiglu_split,
//                   reshape to groups, rms_norm, mul by the {gs, n_group} weight). For gs a
//                   multiple of 8 the kernel is written to be BIT-IDENTICAL to it, and that is
//                   asserted, not just a tolerance.
//   2. CHAIN-LEGACY n_group == 1 only: the chain src/graphs/build_granite_hybrid.cpp builds
//                   (mul(y, silu(cont(z))), fused_rms_norm), also bit-identical.
//   3. EXACT        the formula evaluated here in double, independent of ggml.
// Controls: the node must really be the fused op; the OTHER gate order (norm, then gate) must be
// far from the result, so a test that could not tell the two orders apart cannot pass; and the
// result must not depend on the thread count. The gate is optionally a row-strided VIEW of a wider
// tensor, the way the graphs take z straight out of the fused in-projection.
static void test_gated_group_rms_norm(std::mt19937 & rng, int64_t d, int n_group, int64_t n_tok, int64_t n_seq,
                                      bool strided_gate) {
    graph_runner g;
    const float   eps = 1e-5f;
    const int64_t gs  = d / n_group;
    const int64_t pad = strided_gate ? 2*d + 3 : 0;   // gate = first d floats of each (d + pad) row

    ggml_tensor * y  = ggml_new_tensor_3d(g.ctx, GGML_TYPE_F32, d, n_tok, n_seq);
    ggml_tensor * zx = ggml_new_tensor_3d(g.ctx, GGML_TYPE_F32, d + pad, n_tok, n_seq);
    ggml_tensor * z  = strided_gate ? ggml_view_3d(g.ctx, zx, d, n_tok, n_seq, zx->nb[1], zx->nb[2], 0) : zx;
    ggml_tensor * w  = ggml_new_tensor_2d(g.ctx, GGML_TYPE_F32, gs, n_group);   // the model tensor's shape

    std::vector<float> vy(ggml_nelements(y)), vzx(ggml_nelements(zx)), vw(ggml_nelements(w));
    fill(vy, rng);
    fill(vzx, rng);
    fill(vw, rng, 0.25f, 2.0f);
    memcpy(y->data,  vy.data(),  vy.size()*sizeof(float));
    memcpy(zx->data, vzx.data(), vzx.size()*sizeof(float));
    memcpy(w->data,  vw.data(),  vw.size()*sizeof(float));

    ggml_tensor * fused = ggml_gated_group_rms_norm(g.ctx, y, z, w, eps, n_group);

    // 1. the class-path chain, op for op
    ggml_tensor * t_cls = ggml_swiglu_split(g.ctx, ggml_cont(g.ctx, z), y);
    t_cls = ggml_reshape_4d(g.ctx, t_cls, gs, n_group, n_tok, n_seq);
    ggml_tensor * ref_cls = ggml_mul(g.ctx, ggml_rms_norm(g.ctx, t_cls, eps), w);

    // 2. the legacy granite-hybrid chain (one group only)
    ggml_tensor * ref_leg = nullptr;
    if (n_group == 1) {
        ggml_tensor * t_leg = ggml_mul(g.ctx, y, ggml_silu(g.ctx, ggml_cont(g.ctx, z)));
        ref_leg = ggml_fused_rms_norm(g.ctx, t_leg, ggml_reshape_1d(g.ctx, w, d), eps);
    }

    std::vector<ggml_tensor *> outs = {fused, ref_cls};
    if (ref_leg) {
        outs.push_back(ref_leg);
    }
    g.run(outs);

    char name[96];
    snprintf(name, sizeof(name), "gated_group_rms_norm d=%lld g=%d %lldx%lld%s",
             (long long) d, n_group, (long long) n_tok, (long long) n_seq, strided_gate ? " strided-z" : "");

    if (fused->op != GGML_OP_GATED_GROUP_RMS_NORM) {
        printf("  FAIL %-34s constructor did not produce the fused node (op = %s)\n", name, ggml_op_name(fused->op));
        g_failures++;
        return;
    }

    const size_t n = (size_t) ggml_nelements(y);
    const float * f = (const float *) fused->data;

    // 3. exact, in double, and the WRONG-ORDER control (norm first, gate after)
    std::vector<float> exact(n), wrong_order(n);
    for (int64_t s = 0; s < n_seq; s++) {
        for (int64_t r = 0; r < n_tok; r++) {
            const float * yr = vy.data()  + (s*n_tok + r)*d;
            const float * zr = vzx.data() + (s*n_tok + r)*(d + pad);
            float * er = exact.data()       + (s*n_tok + r)*d;
            float * wr = wrong_order.data() + (s*n_tok + r)*d;
            for (int gi = 0; gi < n_group; gi++) {
                double m2 = 0.0, m2y = 0.0;
                for (int64_t j = 0; j < gs; j++) {
                    const int64_t c = gi*gs + j;
                    const double t = (double) yr[c] * ((double) zr[c] / (1.0 + std::exp(-(double) zr[c])));
                    m2  += t*t;
                    m2y += (double) yr[c] * (double) yr[c];
                }
                const double sc  = 1.0/std::sqrt(m2 /gs + eps);
                const double scy = 1.0/std::sqrt(m2y/gs + eps);
                for (int64_t j = 0; j < gs; j++) {
                    const int64_t c = gi*gs + j;
                    const double sz = (double) zr[c] / (1.0 + std::exp(-(double) zr[c]));
                    er[c] = (float) ((double) yr[c] * sz * sc * (double) vw[c]);
                    wr[c] = (float) ((double) yr[c] * scy * (double) vw[c] * sz);
                }
            }
        }
    }

    char nm[128];
    snprintf(nm, sizeof(nm), "%s vs EXACT", name);
    check(nm, f, exact.data(), n, 2e-5);
    snprintf(nm, sizeof(nm), "%s vs CHAIN-CLASS", name);
    check(nm, f, (const float *) ref_cls->data, n, 2e-6);

    // bit identity with the chain it replaces, where the vector split is the same (gs % 8 == 0)
    auto n_bit_diff = [&](const float * a, const float * b) {
        size_t k = 0;
        for (size_t i = 0; i < n; i++) {
            k += memcmp(&a[i], &b[i], sizeof(float)) != 0;
        }
        return k;
    };
    if (gs % 8 == 0) {
        const size_t k = n_bit_diff(f, (const float *) ref_cls->data);
        printf("  %-4s %-34s %zu of %zu elements differ in bits\n", k == 0 ? "ok" : "FAIL", "  bit-identical to CHAIN-CLASS", k, n);
        if (k != 0) {
            g_failures++;
        }
    }
    if (ref_leg) {
        snprintf(nm, sizeof(nm), "%s vs CHAIN-LEGACY", name);
        check(nm, f, (const float *) ref_leg->data, n, 2e-6);
        if (gs % 8 == 0) {
            const size_t k = n_bit_diff(f, (const float *) ref_leg->data);
            printf("  %-4s %-34s %zu of %zu elements differ in bits\n", k == 0 ? "ok" : "FAIL", "  bit-identical to CHAIN-LEGACY", k, n);
            if (k != 0) {
                g_failures++;
            }
        }
    }

    // ORDER CONTROL: norm-then-gate must NOT match
    const double d_wrong = max_rel_diff(f, wrong_order.data(), n);
    const bool order_ok = d_wrong > 1e-2;
    printf("  %-4s %-34s norm_before_gate=True differs by %.3e\n", order_ok ? "ok" : "FAIL", "  ORDER CONTROL", d_wrong);
    if (!order_ok) {
        g_failures++;
    }

    // THREAD INVARIANCE: 1 and 5 threads, same bits as the 2-thread run above
    std::vector<float> f2(f, f + n);
    for (int nth : {1, 5}) {
        graph_runner g2;
        ggml_tensor * y2  = ggml_new_tensor_3d(g2.ctx, GGML_TYPE_F32, d, n_tok, n_seq);
        ggml_tensor * zx2 = ggml_new_tensor_3d(g2.ctx, GGML_TYPE_F32, d + pad, n_tok, n_seq);
        ggml_tensor * z2  = strided_gate ? ggml_view_3d(g2.ctx, zx2, d, n_tok, n_seq, zx2->nb[1], zx2->nb[2], 0) : zx2;
        ggml_tensor * w2  = ggml_new_tensor_2d(g2.ctx, GGML_TYPE_F32, gs, n_group);
        memcpy(y2->data,  vy.data(),  vy.size()*sizeof(float));
        memcpy(zx2->data, vzx.data(), vzx.size()*sizeof(float));
        memcpy(w2->data,  vw.data(),  vw.size()*sizeof(float));
        ggml_tensor * o2 = ggml_gated_group_rms_norm(g2.ctx, y2, z2, w2, eps, n_group);
        g2.run({o2}, nth);
        const size_t k = n_bit_diff((const float *) o2->data, f2.data());
        printf("  %-4s %-34s %d thread(s): %zu elements differ from the 2-thread run\n", k == 0 ? "ok" : "FAIL", "  THREAD INVARIANCE", nth, k);
        if (k != 0) {
            g_failures++;
        }
    }
}

// ---------------------------------------------------------------------------------------------
// NEGATIVE CONTROL. Compare a correct result against a DELIBERATELY WRONG one and require the
// comparator to report a difference. Without this, a comparator that always returned 0 would make
// every test above pass silently.
static void test_negative_control(std::mt19937 & rng) {
    const size_t n = 256;
    std::vector<float> x(n), y(n);
    fill(x, rng);
    y = x;
    y[n/2] += 1.0f;   // one element, far outside any tolerance

    const double d_same = max_rel_diff(x.data(), x.data(), n);
    const double d_diff = max_rel_diff(x.data(), y.data(), n);

    const bool ok = (d_same == 0.0) && (d_diff > 1e-2);
    printf("  %-4s %-34s identical = %.3e, perturbed = %.3e\n",
           ok ? "ok" : "FAIL", "NEGATIVE CONTROL (comparator)", d_same, d_diff);
    if (!ok) {
        printf("       the comparator cannot tell these apart -- every PASS above is meaningless\n");
        g_failures++;
    }
}

int main(void) {
    std::mt19937 rng(1234);

    printf("Fused-op kernels checked against the same maths written with ordinary ggml ops.\n");
    printf("A pass means the fused kernel agrees with rms_norm/silu/relu/gelu + mul on identical\n");
    printf("inputs (and, for GATED_GROUP_RMS_NORM, with an exact double reference as well).\n");
    printf("It does NOT mean the model is correct, and it does not cover the other 30 ops\n");
    printf("this fork has no host kernel for.\n\n");

    test_negative_control(rng);

    printf("\nFUSED_RMS_NORM\n");
    test_fused_rms_norm(rng, 64, 8);
    test_fused_rms_norm(rng, 576, 5);      // not a multiple of any vector width
    test_fused_rms_norm(rng, 4096, 2);

    printf("\nFUSED_MUL_UNARY (same-shape)\n");
    test_fused_mul_unary_same(rng, GGML_UNARY_OP_SILU, "silu", 128, 6);
    test_fused_mul_unary_same(rng, GGML_UNARY_OP_RELU, "relu", 128, 6);
    test_fused_mul_unary_same(rng, GGML_UNARY_OP_SILU, "silu", 577, 3);

    printf("\nFUSED_MUL_UNARY (gelu, against the exact expression -- see the note above the test)\n");
    test_fused_mul_gelu_exact(rng, 128, 6);
    test_fused_mul_gelu_exact(rng, 577, 3);

    printf("\nGATED_GROUP_RMS_NORM (Mamba-2 gated norm, norm_before_gate=False)\n");
    test_gated_group_rms_norm(rng,  512, 2, 7, 1, false);   // tiny mamba2, prefill
    test_gated_group_rms_norm(rng,  512, 2, 1, 3, true);    // decode, 3 sequences, z a strided view
    test_gated_group_rms_norm(rng, 1536, 1, 5, 1, true);    // granite-4.0-h-350m width, one group (legacy chain too)
    test_gated_group_rms_norm(rng, 7680, 8, 1, 1, true);    // Nemotron-H decode row (the reference's target shape)
    test_gated_group_rms_norm(rng,   36, 3, 4, 2, false);   // 12-wide groups: scalar tails, tolerance only
    test_gated_group_rms_norm(rng,   44, 1, 3, 1, true);    // one 44-wide group: SIMD body + scalar tail, legacy chain

    printf("\n%s -- %d failure(s)\n", g_failures == 0 ? "PASS" : "FAIL", g_failures);
    return g_failures == 0 ? 0 : 1;
}

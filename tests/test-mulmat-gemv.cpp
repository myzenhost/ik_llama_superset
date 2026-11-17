// Does MUL_MAT against a quantised weight give the same answer for one column as for many?
//
// GATE-CLAIM-OK: "CPU" below names a BACKEND SOURCE DIRECTORY / build configuration. No device is
// claimed unavailable; this test computes its own graphs on the host and touches no accelerator.
//
// WHY THIS EXISTS. A per-node scan of a real decode (tests/test-logits-sanity.cpp with
// LLAMA_NAN_SCAN=1, SmolLM2-135M-Instruct-Q8_0) showed every tensor healthy through the prompt
// pass -- inp_embd within +/-0.5, attn_norm within +/-0.44, Qcur within +/-10 -- and then the
// first non-finite value appearing at:
//     prompt pass (5 columns): node #748 'ffn_gate-29' MUL_MAT, 528 NaN + 970 Inf of 1536,
//                              from a src1 'node_747' whose own values were within +/-0.38
//     generation (1 column)  : node #2  'Qcur-0'      MUL_MAT, 576 NaN of 576,
//                              from a src1 'attn_norm-0' whose own values were within +/-0.04
// Both failures are MUL_MAT of a q8_0 weight against a f32 activation of ONE COLUMN. The prompt
// pass survives layers 0-28 because those run five columns; layer 29 is narrowed to the single
// output row and fails there. Every generation step is one column, so every generation step is
// affected -- which is why the model emitted the same token repeatedly.
//
// That is a HYPOTHESIS drawn from two observations inside one model. This file tests it directly,
// away from any model, tokenizer or KV cache.
//
// THE ORACLE, and why it needs no dequantisation reference. The same weight matrix multiplied by
// the same activation vector must give the same answer whether that vector is presented alone or
// as one column of a wider batch. So the test computes BOTH and compares column 0 to column 0.
// Identical inputs, identical maths, one difference: the number of columns, which is what selects
// between the matrix and vector code paths. A disagreement localises to that choice and cannot be
// explained by quantisation error, since both sides use the same quantised weights.
//
// A SECOND, INDEPENDENT ORACLE is also computed: the dequantised weights multiplied out in double
// precision. Without it, a defect that corrupted BOTH widths equally would pass the first check.
//
// CONTROLS:
//   * NEGATIVE control on the comparator -- it must report a deliberate perturbation.
//   * the wide case is checked for finiteness in its own right, so "they agree" cannot be
//     satisfied by both sides being NaN (NaN != NaN would fail, but Inf == Inf would not).
//   * f32 weights are tested alongside q8_0, which separates "narrow columns" from "quantised
//     weights" as the operative factor.

#include "ggml.h"
#include "ggml-cpu.h"

// The types swept numerically at one column. Declared at file scope because the divergence
// report below asserts against it: any type whose two traits tables disagree MUST appear here.
static const ggml_type g_sweep[] = {
    GGML_TYPE_Q4_0, GGML_TYPE_Q4_1, GGML_TYPE_Q5_0, GGML_TYPE_Q5_1, GGML_TYPE_Q8_0,
    GGML_TYPE_Q2_K, GGML_TYPE_Q3_K, GGML_TYPE_Q4_K, GGML_TYPE_Q5_K, GGML_TYPE_Q6_K,
    GGML_TYPE_IQ4_NL, GGML_TYPE_IQ4_XS,
    // This fork's own quant types. mxfp4 and nvfp4 were added because the divergence report
    // below flagged them as the only quantised types whose two traits tables disagree while
    // nothing verified them numerically -- nvfp4 most sharply, base=f32 against cpu=q8_0.
    // q1_0 and q2_0 are the fork's other unique vec_dot_*_q8_0 types (see the de-duplication note
    // at ggml/src/ggml-cpu/quants.c:1322). MEASURED once they were listed here: both are declared
    // in the enum -- renumbered to 400 and 401 at ggml.h:511-512 because upstream's 41/42 are
    // this fork's Q1_0_G128/Q2_0_G128 in files on disk -- and NEITHER has a row in the traits
    // table, so ggml_type_name returns NULL for them. They are kept in this list deliberately:
    // they skip with a named reason every run, which is a standing, visible statement that two
    // declared types are unbacked, rather than a silence nobody would notice.
    // A type this build cannot quantise prints its own SKIPPED line rather than passing silently.
    GGML_TYPE_MXFP4, GGML_TYPE_NVFP4, GGML_TYPE_Q1_0, GGML_TYPE_Q2_0,
    // NVFP4_KS (ik 30903876b): NVFP4 blocks + one fp32 scale per row, iqk kernels (1-8 cols) and the Q8_0_R8/R16 repack (>=32 cols).
    GGML_TYPE_NVFP4_KS,
    // Q6_0 ADDED 2026-09-26: the fork's own 6-bit type, which llama-quantize uses to bump attn_v
    // for IQ4_XS models. A SmolLM2-360M IQ4_XS requant scored PPL ~5e4 on BOTH arms of the iqk-GEMM
    // A/B (the Q8_0 source scores 7.25), and Q6_0 was the only type in that file neither verified
    // here nor present in a healthy file (the IQ3_S requant, PPL 12.3, carries Q5_0 there instead).
    GGML_TYPE_Q6_0,
    // TQ1_0 / TQ2_0 ADDED 2026-10-16 (row S69-MERGE-FEATURE-TQ2-0-WIP-1016-01). Mainline's ternary types carry no iqk kernel (MulMat::prepare has
    // no case for them and iqk_gemm_type_b answers -1), so they run on the scalar ggml_vec_dot_tq*_q8_K through type_traits_cpu in ggml-cpu.c.
    // The feature/tq2_0-wip port loaded, dequantised bit-exactly and still decoded wrong text; nothing in this sweep covered the ternary types,
    // so a wrong activation type or a mis-dispatch into iqk could not have been seen here. K=512 is a whole number of their 256-wide blocks;
    // the sweep runs one column AND a wide batch. MEASURED 2026-10-16: giving TQ2_0 the wrong vec_dot_type (Q8_0 for Q8_K) makes this sweep
    // report tq2_0 wide-vs-ref 3.2e+34 and FAIL twice; the unmodified tree reports 3.7e-3 (see the findings file S69_ESERVING_MERGE_2026-10-16.md).
    GGML_TYPE_TQ1_0, GGML_TYPE_TQ2_0,
    // ---- THE 21 IQ*_K TYPES, ADDED 2026-09-15 -------------------------------------------------
    // These are the types this fork exists for, and until today NOTHING verified their CPU
    // arithmetic. test-iqk-golden covers their quantize/dequantize -- their bytes -- and stops
    // there; this sweep's own SCOPE line read "14 weight types (f32, f16 and 12 quantised)" and
    // named none of them.
    //
    // They were not merely unlisted: they were INVISIBLE TO THE DIVERGENCE REPORT BELOW, which
    // skips any type whose cpu->vec_dot is NULL on the stated grounds that "a null vec_dot means
    // no kernel is reached and the declaration cannot mislead one". That was true and became
    // false. The per-backend merge left every one of these without a type_traits_cpu row, so the
    // report correctly skipped 21 types while an IQ2_K model could not run on CPU at all --
    // measured: it aborted in get_rows, then segfaulted on the NULL vec_dot. Both are repaired
    // (fork 0344eab57), so these types now HAVE a cpu vec_dot and the report will judge them.
    //
    // Listing them here is the other half of that repair. The repair proved the path EXECUTES; it
    // proved nothing about the numbers, and a quant kernel that is 99% right corrupts a model
    // silently rather than failing. A type this build cannot quantise prints its own SKIPPED line,
    // so anything unreachable says so by name instead of passing quietly.
    GGML_TYPE_IQ2_K,  GGML_TYPE_IQ2_KS, GGML_TYPE_IQ2_KL, GGML_TYPE_IQ2_KT,
    GGML_TYPE_IQ3_K,  GGML_TYPE_IQ3_KS, GGML_TYPE_IQ3_KT,
    GGML_TYPE_IQ4_K,  GGML_TYPE_IQ4_KS, GGML_TYPE_IQ4_KSS, GGML_TYPE_IQ4_KT,
    GGML_TYPE_IQ5_K,  GGML_TYPE_IQ5_KS,
    GGML_TYPE_IQ6_K,  GGML_TYPE_IQ1_KT,
    // TQ2_T is NOT listed: it has no .from_float (the trellis encoder is CUDA-only), so this harness
    // cannot quantise to it -- MEASURED 2026-11-12, listing it printed SKIPPED and the divergence
    // report still failed. It is verified by test-tq-trellis instead; see g_verified_elsewhere.
    //
    // THE _R4 / _R8 REPACKED VARIANTS ARE DELIBERATELY NOT HERE, and the reason is a row-layout
    // one, not a guess about crashes. They are not row-independent: the from_float_ref for an _r4
    // type reads k as FOUR rows of k/4 and writes four rows' worth of payload, so the buffer it
    // actually needs is 4*ggml_row_size(type, k/4), which is LARGER than ggml_row_size(type, k).
    // tests/test-iqk-golden.cpp measures the gap directly:
    //     iq4_ks_r4  wrote 2192;  ggml_row_size(t,4096)=2180;  4*ggml_row_size(t,1024)=2192
    //     iq5_ks_r4  wrote 2704;  ggml_row_size(t,4096)=2692;  4*ggml_row_size(t,1024)=2704
    // and concludes that "ggml_row_size(type, N) is the wrong question for these types, not a
    // wrong answer". This harness sizes and strides every row by ggml_row_size, so for these types
    // it would both under-allocate and mis-stride. Covering them needs a 4-row-aware allocation,
    // which is a real addition to this file rather than a list entry.
    //
    // ⚠ THAT EVIDENCE IS FROM test-iqk-golden, NOT FROM HERE. An earlier version of this comment
    // ended "MEASURED: it segfaults", asserting these types were the cause of a segfault this
    // sweep was hitting. THEY WERE NOT. Removing them changed nothing -- the very next run
    // crashed in the same place. The actual cause was a NULL .from_float for the ACTIVATION type
    // Q8_0_X4, reached from iq2_kt and found at rip=0x0 under gdb, which is fixed in
    // ggml-cpu.c. The exclusion above survives that correction because it never depended on the
    // segfault; the word MEASURED did, and it had not been earned here.
    // -------------------------------------------------------------------------------------------
};

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <random>
#include <vector>
#include <string>
#include <cctype>

static int g_failures = 0;

// RUN CONFIGURATION, set once in main() from argv. The default seed is the one this file has always
// used. It does NOT reproduce logs from before the per-test generators (test_rng): the same base seed
// now feeds every test a stream keyed by its identity, so all data, and every number, moved once.
static uint32_t g_seed = 20260914;

// Which block of main() is running. Part of a verdict's IDENTITY below: the same type at the same shape
// runs in more than one block (q8_0 K=512 M=1536 wide=8 runs in the localisation block AND the sweep).
static const char * g_section = "";

// Every numeric verdict of a run, keyed by WHAT was tested (section, type, shape, metric) and never by
// WHEN it ran. --isolation-check runs the suite twice and compares these key by key; a key whose value
// moves between two runs that differ only by an unrelated inserted type is a test whose data depends
// on its position in the run (row S69-GEMV-Q6K-R4-RATIO-1080-AFTER-SWEEP-RNG-SHIFT-2026-10-17-01).
struct metric { std::string key; double value; };
static std::vector<metric> g_metrics;
static void record(const std::string & key, double v) { g_metrics.push_back({ key, v }); }

// Types that actually reached a numeric verdict. Appended ONLY at the end of test_width, after
// the comparisons have run -- never on the strength of appearing in g_sweep. The divergence
// report below asserts against THIS list, because "listed in the sweep" and "verified" are not
// the same thing: nvfp4 is listed and is skipped (this build cannot quantise to it), and a
// coverage claim honouring the list rather than the outcome would have called it covered.
static std::vector<ggml_type> g_verified;

// DISTINCT types that reached a numeric verdict. g_verified is appended once per test_width CALL
// and the same type is run at several shapes, so g_verified.size() counts VERDICTS, not types --
// it printed "33 weight types listed ... of which 38 reached a numeric verdict" before this
// existed. The membership test in the divergence report reads g_verified directly and is correct
// with duplicates; only a COUNT needs to de-duplicate.
static size_t n_verified_types(void) {
    std::vector<ggml_type> seen;
    for (size_t i = 0; i < g_verified.size(); i++) {
        bool dup = false;
        for (size_t j = 0; j < seen.size(); j++) {
            if (seen[j] == g_verified[i]) { dup = true; break; }
        }
        if (!dup) seen.push_back(g_verified[i]);
    }
    return seen.size();
}


// ggml_type_name returns NULL for an enum value that has no traits entry, and printf("%s", NULL)
// is undefined behaviour that happened to render "(null)" here. MEASURED: GGML_TYPE_Q1_0 (=400)
// and GGML_TYPE_Q2_0 (=401), renumbered by this fork at ggml.h:511-512, are declared in the enum
// and have NO row in the traits table at all -- so the skip line named neither the type nor the
// problem. This makes an unnamed type identify itself.
static const char * type_label(ggml_type t) {
    const char * n = ggml_type_name(t);
    if (n) {
        return n;
    }
    static char buf[48];
    snprintf(buf, sizeof(buf), "type#%d(no traits)", (int) t);
    return buf;
}

static void fill(std::vector<float> & v, std::mt19937 & rng, float lo, float hi) {
    std::uniform_real_distribution<float> d(lo, hi);
    for (auto & x : v) x = d(rng);
}

// ONE GENERATOR PER TEST, seeded from the base seed and the test's identity (section, type, shape).
// Until 2026-10-17 the whole run drew from a single std::mt19937 passed by reference, so every test's
// data was a function of how many numbers every EARLIER test had drawn. Adding TQ1_0/TQ2_0 to g_sweep
// therefore re-drew the _rN section's weights, and q6_k_r4's width-1 ratio went 1.009 -> 1.080 with no
// kernel change (row S69-GEMV-Q6K-R4-RATIO-1080-AFTER-SWEEP-RNG-SHIFT-2026-10-17-01). Now a test's
// numbers change only when its own key or the base seed does; --isolation-check enforces that.
// std::seed_seq's generate() and std::mt19937 are fully specified by the standard, so a key gives the
// same stream on every conforming library (uniform_real_distribution is not -- that was true before).
static std::mt19937 test_rng(const char * section, int type, int64_t K, int64_t M, int64_t NW) {
    std::vector<uint32_t> words = { g_seed, (uint32_t) type, (uint32_t) K, (uint32_t) M, (uint32_t) NW };
    for (const char * p = section; *p; p++) words.push_back((uint32_t) (unsigned char) *p);
    std::seed_seq ss(words.begin(), words.end());
    return std::mt19937(ss);
}

static int64_t count_non_finite(const float * p, int64_t n) {
    int64_t c = 0;
    for (int64_t i = 0; i < n; i++) {
        if (std::isnan(p[i]) || std::isinf(p[i])) c++;
    }
    return c;
}

static double max_rel(const float * a, const float * b, int64_t n) {
    double worst = 0.0;
    for (int64_t i = 0; i < n; i++) {
        if (std::isnan(a[i]) || std::isnan(b[i])) return INFINITY;
        const double den = std::max(1e-3, (double) std::max(std::fabs(a[i]), std::fabs(b[i])));
        const double r = std::fabs((double) a[i] - (double) b[i]) / den;
        if (r > worst) worst = r;
    }
    return worst;
}

static double vec_max_abs(const float * p, int64_t n) {
    double m = 0.0;
    for (int64_t i = 0; i < n; i++) {
        if (std::isnan(p[i]) || std::isinf(p[i])) continue;
        m = std::max(m, (double) std::fabs(p[i]));
    }
    return m;
}

// Error relative to the LARGEST element of the reference, which is the right normalisation for
// comparing whole vectors. A per-element relative metric is not: q8_0 bounds ABSOLUTE error per
// 32-element block (one step = block_max/127), so elements near zero have unbounded relative
// error by construction. MEASURED: the first version of this file used a per-element metric with
// a 1e-3 denominator floor and reported a correct q8_0 round-trip as 39% wrong, because a weight
// of ~3.9e-4 divided by that floor is 0.39. It then blamed the kernel.
static double rel_to_max(const float * a, const float * b, int64_t n) {
    double worst = 0.0;
    for (int64_t i = 0; i < n; i++) {
        if (std::isnan(a[i]) || std::isnan(b[i])) return INFINITY;
        worst = std::max(worst, (double) std::fabs((double) a[i] - (double) b[i]));
    }
    const double scale = std::max(vec_max_abs(b, n), 1e-12);
    return worst / scale;
}

// RMS error of a against b, relative to b's RMS. The _rN rule below is judged on this, not on rel_to_max:
// see rn_judge for the measured spread of the MAX statistic.
static double rel_rms(const float * a, const float * b, int64_t n) {
    double se = 0.0, sr = 0.0;
    for (int64_t i = 0; i < n; i++) {
        if (std::isnan(a[i]) || std::isnan(b[i])) return INFINITY;
        const double d = (double) a[i] - (double) b[i];
        se += d * d;
        sr += (double) b[i] * (double) b[i];
    }
    return std::sqrt(se / std::max(sr, 1e-24));
}

struct runner {
    ggml_context * ctx = nullptr;
    std::vector<uint8_t> buf;
    explicit runner(size_t mb) : buf(mb*1024*1024) {
        ggml_init_params p = { buf.size(), buf.data(), false };
        ctx = ggml_init(p);
    }
    ~runner() { if (ctx) ggml_free(ctx); }
    void run(std::vector<ggml_tensor *> outs) {
        ggml_cgraph * gf = ggml_new_graph(ctx);
        for (auto * t : outs) ggml_build_forward_expand(gf, t);
        ggml_graph_compute_with_ctx(ctx, gf, 2);
    }
};

// K = rows of the weight (n_embd), M = columns (output features), NW = width of the wide batch.
static void test_width(ggml_type wtype, int64_t K, int64_t M, int64_t NW) {
    // This test's OWN generator, keyed by WHAT it tests (see test_rng) -- never by how many draws ran
    // before it, so inserting, removing or skipping any other test cannot change this one's data.
    std::mt19937 rng = test_rng(g_section, (int) wtype, K, M, NW);
    runner g(512);

    // PRECONDITION. A type with no dequantiser cannot be given a reference, so it is skipped
    // rather than guessed at. Reported, not silent: a type that vanishes from the output without
    // a line is indistinguishable from one that passed.
    {
        const ggml_type_traits * tr = ggml_get_type_traits(wtype);
        if (!tr || (wtype != GGML_TYPE_F32 && tr->to_float == NULL)) {
            printf("  ---- %-7s SKIPPED: no to_float in this type's traits, so no reference can\n"
                   "       be built for it. Not a verdict about the kernel.\n",
                   type_label(wtype));
            return;
        }
        // A type with no .from_float in the base traits cannot be produced from f32 by the
        // generic path, and ggml_quantize_chunk has no case for it either -- it falls to
        // `default: assert(false)`, which is a no-op under NDEBUG, leaving result = 0 and
        // tripping GGML_ASSERT(result == nrows * row_size) at ggml.c:30167. That ABORTS the
        // process, so it must be caught here rather than survived.
        // MEASURED: nvfp4 hit exactly that. Its base traits entry (ggml.c:1538) carries only
        // to_float and from_float_ref -- no from_float, no vec_dot, no vec_dot_type -- while
        // mxfp4, which quantises fine, carries all three. Checking from_float is therefore the
        // in-tree criterion and not a guess. Reported, never silent: a type that vanished from
        // the output without a line would be indistinguishable from one that passed.
        if (wtype != GGML_TYPE_F32 && wtype != GGML_TYPE_F16 && tr->from_float == NULL) {
            printf("  ---- %-7s SKIPPED: no .from_float in its traits, so this build cannot\n"
                   "       quantise to it (ggml_quantize_chunk would abort). NOT a verdict about\n"
                   "       the kernel -- and it means nothing verifies this type numerically.\n",
                   type_label(wtype));
            return;
        }
    }

    // PRECONDITION. Block-quantised types require the row length to be a whole number of blocks;
    // the K-quants use 256. Passing K=576 (= 2*256 + 64) produces nonsense rather than an error:
    // the first run of this file did exactly that and reported a q6_K round-trip error of 5307 on
    // weights bounded by 0.1. That is a misuse of the format by the test, not a defect in ggml,
    // and reporting it as one would have been a fabricated finding.
    {
        const int64_t blk = ggml_blck_size(wtype);
        if (blk > 1 && (K % blk) != 0) {
            printf("  ---- %-7s SKIPPED: K=%lld is not a multiple of this type's block size %lld\n",
                   type_label(wtype), (long long) K, (long long) blk);
            return;
        }
    }

    // ---- weights ----------------------------------------------------------------------------
    std::vector<float> Wf((size_t) (K*M));
    fill(Wf, rng, -0.1f, 0.1f);          // weight-like magnitudes

    ggml_tensor * W = ggml_new_tensor_2d(g.ctx, wtype, K, M);
    if (wtype == GGML_TYPE_F32) {
        memcpy(W->data, Wf.data(), Wf.size()*sizeof(float));
    } else if (wtype == GGML_TYPE_F16) {
        // f16 is not a block-quantised format, so ggml_quantize_chunk is not its entry point.
        ggml_fp32_to_fp16_row(Wf.data(), (ggml_fp16_t *) W->data, K*M);
    } else {
        // The trailing quantize_user_data* is this fork's own extra parameter; upstream's
        // ggml_quantize_chunk ends at the imatrix argument.
        ggml_quantize_chunk(wtype, Wf.data(), W->data, 0, M, K, nullptr, nullptr);
    }

    // ---- activations: one shared vector, presented twice ---------------------------------------
    std::vector<float> xv((size_t) K);
    fill(xv, rng, -0.5f, 0.5f);          // activation-like magnitudes, as measured in the model

    ggml_tensor * x_wide = ggml_new_tensor_2d(g.ctx, GGML_TYPE_F32, K, NW);
    ggml_tensor * x_one  = ggml_new_tensor_2d(g.ctx, GGML_TYPE_F32, K, 1);

    std::vector<float> wide((size_t) (K*NW));
    for (int64_t c = 0; c < NW; c++) {
        for (int64_t r = 0; r < K; r++) {
            // column 0 is the vector under test; the others are different, so the wide case is a
            // genuine multi-column computation and not NW copies of one column.
            wide[c*K + r] = (c == 0) ? xv[r] : xv[(r + c*37) % K] * 0.5f;
        }
    }
    memcpy(x_wide->data, wide.data(), wide.size()*sizeof(float));
    memcpy(x_one->data,  xv.data(),   xv.size()*sizeof(float));

    ggml_tensor * y_wide = ggml_mul_mat(g.ctx, W, x_wide);   // [M, NW]
    ggml_tensor * y_one  = ggml_mul_mat(g.ctx, W, x_one);    // [M, 1]

    g.run({y_wide, y_one});

    const float * pw = (const float *) y_wide->data;   // column 0 is the first M floats
    const float * po = (const float *) y_one->data;

    // ---- independent reference: dequantised weights, accumulated in double ---------------------
    std::vector<float> Wdq((size_t) (K*M));
    if (wtype == GGML_TYPE_F32) {
        Wdq = Wf;
    } else {
        const ggml_type_traits * tr = ggml_get_type_traits(wtype);
        // ROW BY ROW, and the old whole-tensor call was a REAL DEFECT rather than a style point.
        // It read the M*K quantised values as ONE row:  tr->to_float(W->data, Wdq.data(), K*M).
        // That is only valid when .row_meta_size == 0. This fork has 10 types where it is 2 or 4:
        // their row layout is [row meta][blocks], so an M-row tensor carries M meta headers
        // INTERLEAVED, and a single to_float over K*M elements applies row 0s meta to the whole
        // tensor and reads every later rows blocks at an offset short by (row_meta_size * r).
        //
        // MEASURED, and the split is total: with the whole-tensor call, all 5 types whose
        // row_meta_size is 0 (iq2_k, iq3_k, iq4_k, iq5_k, iq6_k) passed the round-trip control,
        // and all 10 whose row_meta_size is 2 or 4 (iq2_ks, iq2_kl, iq2_kt, iq3_ks, iq3_kt,
        // iq4_ks, iq4_kss, iq4_kt, iq5_ks, iq1_kt) failed it at relative RMS 1.13 to 1.27 --
        // error as large as the signal, i.e. uncorrelated, which is the signature of reading
        // structure at the wrong offset and NOT of a coarse quantiser. Controls: q4_K, a type
        // that has passed here since the file was written, has row_meta_size 0; an invented type
        // has no entry at all.
        //
        // The quantise side never had this bug: ggml_quantize_chunk above is passed nrows = M and
        // lays the per-row meta out correctly. Only the reference was wrong -- which is exactly
        // what the control below exists to say, and it said it.
        //
        // ggml_row_size() already includes row_meta_size (ggml.c: row_meta_size + type_size*ne/
        // blck_size), so this stride is right for every type and this is not a special case for
        // the ten. The five that already passed must produce byte-identical numbers.
        const size_t row_bytes = ggml_row_size(wtype, K);
        for (int64_t r = 0; r < M; ++r) {
            tr->to_float((const char *) W->data + (size_t) r*row_bytes, Wdq.data() + (size_t) r*K, K);
        }
    }
    // CONTROL ON THE REFERENCE ITSELF, before any verdict is drawn from it. The dequantised
    // weights must agree with the originals to roughly q8_0's precision. Without this check a
    // faulty dequantisation would be reported as a faulty kernel -- and the first run of this
    // test did exactly that, reporting the WIDE path (which demonstrably works for 29 layers of
    // a real model) as disagreeing with the reference by 145%.
    //
    // ITS VALUE IS NOW ALWAYS REPORTED, not only when it trips, and that is a real change in what
    // this file can tell you. A control that speaks only on failure hides its MARGIN, and the
    // margin is what turns it from a gate into EVIDENCE: it says how close the dequantised
    // weights are to the ORIGINAL floats, which is a reference wholly independent of both the
    // dequantiser and the kernel, because this harness generated those floats.
    //
    // That independence is what breaks a tie. iq4_kss and iq5_ks disagree with the
    // dequantiser-derived reference by ~0.21 while agreeing with themselves at every shape. Two
    // decoders disagreeing says nothing about which is wrong -- but if the dequantiser also
    // round-trips CLOSE to the original weights, then it agrees with an independent third party
    // and the kernel is the outlier. Printing dq_err is what lets anyone read that off the run
    // instead of inferring it from the silence of a control that did not fire.
    // HOISTED so the row line can report it. A threshold that is a function of bpw is
    // undecidable from the output unless the output says what bpw it read -- and the three KT
    // types were rejected by a bpw-keyed bound whose input was never printed. Reporting the
    // INPUT of a gate alongside its verdict is what lets the gate itself be audited.
    const double bpw = (double) ggml_type_size(wtype) * 8.0 /
                       (double) std::max<int64_t>(1, ggml_blck_size(wtype));
    double dq_err = -1.0;
    {
        // q8_0 stores one f16 scale per 32 values and an int8 per value, so the round-trip error
        // of any element is at most half a step, and a step is at most (global max)/127. Checked
        // as an ABSOLUTE bound against that, which is what the format actually guarantees.
        // TYPE-AGNOSTIC bound: RMS error relative to the RMS of the weights. Every working
        // quantiser in this family sits well under 15% by this measure (q4_0, the coarsest tried
        // here, measures about 3%); a reference that is genuinely wrong is of order 100%.
        //
        // This replaced a per-element ABSOLUTE bound, and the reason is measured: an absolute
        // bound needs each type's own block step, and the q8_0 step (gmax/127 = 7.874e-04) marked
        // q4_0's round-trip of 1.251e-02 as a control failure. q4_0 is coarser by design -- 4 bits
        // against 8 -- so that bound was measuring the constant it was given, not the quantiser.
        double se = 0.0, sw = 0.0;
        bool all_finite = true;
        for (int64_t i = 0; i < K*M; i++) {
            if (!std::isfinite(Wdq[i])) { all_finite = false; break; }
            const double d = (double) Wdq[i] - (double) Wf[i];
            se += d*d;
            sw += (double) Wf[i] * (double) Wf[i];
        }
        dq_err = all_finite ? std::sqrt(se / std::max(sw, 1e-30)) : INFINITY;
        // The bound has to scale with the format's precision, or it measures the constant it was
        // given rather than the quantiser. Bits per weight is read from the type itself, so no
        // table of magic numbers is needed and a new type is covered automatically.
        // MEASURED: q2_K (2.625 bpw here) round-trips at 0.2333 relative RMS on uniform random
        // weights, which is correct for 2-bit and was rejected by a flat 0.15 -- the sweep then
        // returned no verdict for the one type most likely to be interesting.
        // ------------------------------------------------------------------- WHERE THIS BOUND IS
        // THE BOUND IS NOW PLACED BETWEEN THE TWO MEASURED POPULATIONS, not at a number chosen to
        // look strict. Both populations were measured in this tree, which is what makes this a
        // calibration rather than a preference:
        //
        //   BROKEN reference, measured:  1.13 - 1.27 relative RMS. That is what the row-metadata
        //       bug produced on all 10 types with row_meta_size 2 or 4 -- error as large as the
        //       signal, i.e. the decode was uncorrelated with the data. A reference that is truly
        //       wrong lands here, which is exactly what this control's own comment says above:
        //       "a reference that is genuinely wrong is of order 100%".
        //   WORKING but coarse, measured: 0.0002 (f16) to 0.4436 (iq1_kt at 1.75 bpw), a smooth
        //       curve in bits per weight across all 31 types the sweep reaches.
        //
        // There is a factor of 2.5 between 0.4436 and 1.13 with NOTHING in it. A bound at 0.7 sits
        // in that empty gap and separates the two populations on evidence from both sides.
        //
        // THE OLD BOUND (bpw < 3 ? 0.35 : 0.15) WAS A TWO-STEP FUNCTION OF bpw, AND IT WAS WRONG
        // IN BOTH DIRECTIONS AT ONCE -- which is the reason it is being replaced rather than
        // nudged. It rejected iq1_kt (0.4436), iq2_kt (0.3505) and iq3_kt (0.1984), while ADMITTING
        // iq4_kt at 0.1411 -- a type whose round-trip is 75% worse than iq4_kss at the IDENTICAL
        // 4.000 bpw. A gate that fails the 0.198 case and passes the 0.141 case, when the 0.141
        // case is the one that deviates most from its own bpw-matched sibling, is not measuring
        // what it claims to.
        //
        // AND THE THREE REJECTIONS WERE AN ARTIFACT OF THIS FILE'S OWN INPUT DISTRIBUTION.
        // tests/test-quant-roundtrip-distribution.cpp measured the same round-trip under uniform,
        // Gaussian and Laplace weights, pairing each KT type with the non-KT type nearest it in
        // bpw. The TRELLIS/round ratio falls monotonically in all three pairs:
        //       iq2_kt / iq2_ks    1.39x -> 1.01x -> 0.90x
        //       iq3_kt / iq3_ks    1.55x -> 1.11x -> 1.08x
        //       iq4_kt / iq4_kss   1.73x -> 0.97x -> 0.83x
        // Under Laplace -- tails heavier than a Gaussian, which is the shape real transformer
        // weights have -- two of the three KT types are BETTER than their round sibling. KT is
        // trellis-quantised and its codebook is fitted to a weight-like distribution; this file
        // fills weights with std::uniform_real_distribution, which is that codebook's worst case
        // and neutral for a round quantiser. The control was measuring the harness.
        //
        // THE FILL IS DELIBERATELY LEFT UNIFORM. Changing it would move every number in this file
        // at once, including the kernel comparisons that are its actual product, and the tolerance
        // above it was calibrated on uniform draws. The distribution dependence is recorded and
        // measured in its own test instead, which is where it can be read without disturbing this
        // one. Row S69-KT-TYPES-FAIL-THE-DEQUANT-ROUND-TRIP-CONTROL-2026-09-15-01.
        //
        // WHAT IS GIVEN UP, said plainly rather than left for someone to discover: a bound at 0.7
        // no longer flags a type that is merely well off the bpw curve. That signal is not lost --
        // bpw is now printed on every row, so the curve is readable directly from the output, and
        // iq4_kt's 0.1411 against iq4_kss's 0.0807 at the same bpw is visible there. It is no
        // longer treated as grounds to withhold a kernel verdict, because being coarse and being
        // WRONG are different things and only the second one invalidates a reference.
        const double dq_tol = (wtype == GGML_TYPE_F32) ? 1e-6 : 0.7;
        if (!(dq_err <= dq_tol)) {
            printf("  ---- %-7s DEQUANTISATION CONTROL FAILED: relative RMS round-trip %.4g > %.2g"
                   " (bpw=%.3f)%s\n",
                   type_label(wtype), dq_err, dq_tol, bpw,
                   all_finite ? "." : " (non-finite values present).");
            printf("       The reference is not trustworthy for this type, so no verdict about\n"
                   "       the kernel is reported for it. Fix the reference first.\n");
            g_failures++;
            return;
        }
    }

    std::vector<float> ref((size_t) M);
    for (int64_t m = 0; m < M; m++) {
        double acc = 0.0;
        const float * wrow = Wdq.data() + m*K;
        for (int64_t k = 0; k < K; k++) acc += (double) wrow[k] * (double) xv[k];
        ref[m] = (float) acc;
    }

    const int64_t bad_wide = count_non_finite(pw, M);
    const int64_t bad_one  = count_non_finite(po, M);
    const double d_one_wide = rel_to_max(po, pw, M);
    const double d_wide_ref = rel_to_max(pw, ref.data(), M);
    const double d_one_ref  = rel_to_max(po, ref.data(), M);

    // q8_0 carries roughly 0.4% per-element error; accumulated over K terms with cancellation the
    // bound is loose on purpose. A code-path divergence is orders of magnitude larger than this.
    // CALIBRATION, measured rather than assumed. The f32 arm of the first run reported
    // one-vs-wide = 8.7e-05 and wide-vs-ref = 1.2e-04 against a 1e-05 bound. Those are ordinary
    // f32 accumulation differences over K=576 terms with cancellation -- the wide and narrow
    // paths sum in a different order, and the reference sums in double -- not a defect. The
    // bound is set above that measured floor and far below the failure being looked for, which
    // is a relative difference of 1.0 (one side zero or non-finite).
    const double tol_ref  = (wtype == GGML_TYPE_F32) ? 1e-3 : 5e-2;
    const double tol_self = 1e-3;   // same weights on both sides; only the summation order differs

    {
        char key[160];
        snprintf(key, sizeof(key), "%s|%s|K%lld|M%lld|NW%lld|", g_section, type_label(wtype),
                 (long long) K, (long long) M, (long long) NW);
        record(std::string(key) + "one-vs-wide", d_one_wide);
        record(std::string(key) + "wide-vs-ref", d_wide_ref);
        record(std::string(key) + "one-vs-ref",  d_one_ref);
        record(std::string(key) + "dq-vs-orig",  dq_err);
    }

    printf("  %-7s K=%-5lld M=%-5lld wide=%-3lld | non-finite: wide=%-5lld one=%-5lld | "
           "one-vs-wide=%-11.4g wide-vs-ref=%-11.4g one-vs-ref=%-11.4g dq-vs-orig=%-9.4g bpw=%-6.3f\n",
           type_label(wtype), (long long) K, (long long) M, (long long) NW,
           (long long) bad_wide, (long long) bad_one, d_one_wide, d_wide_ref, d_one_ref, dq_err, bpw);

    bool ok = true;
    if (bad_wide) {
        printf("       FAIL the WIDE case is itself non-finite (%lld of %lld)\n",
               (long long) bad_wide, (long long) M);
        ok = false;
    }
    if (bad_one) {
        printf("       FAIL the SINGLE-COLUMN case is non-finite (%lld of %lld)\n",
               (long long) bad_one, (long long) M);
        ok = false;
    }
    if (d_one_wide > tol_self) {
        printf("       FAIL one column disagrees with column 0 of the wide batch (%.4g > %.1g)\n",
               d_one_wide, tol_self);
        ok = false;
    }
    if (d_wide_ref > tol_ref) {
        printf("       FAIL the wide result disagrees with the double-precision reference\n");
        ok = false;
    }
    if (d_one_ref > tol_ref) {
        printf("       FAIL the single-column result disagrees with the double-precision reference\n");
        ok = false;
    }
    if (!ok) g_failures++;

    // Reached only if every precondition passed and the comparisons actually ran.
    g_verified.push_back(wtype);
}

// ---------------------------------------------------------------------------------------------
// TRAITS-TABLE DIVERGENCE REPORT, and the coverage invariant it enforces.
//
// This tree carries TWO type-traits tables. ggml_get_type_traits (ggml.c) and
// ggml_get_type_traits_cpu (ggml-cpu.c) both declare a vec_dot_type, and they do not agree: the
// token GGML_TYPE_Q8_2_X4 appears on 29 lines of ggml.c and on ZERO lines of ggml-cpu.c. The
// q8_0 defect fixed in d2f7ab197 came from a kernel written against the first table while being
// driven by the second -- ggml_compute_forward_mul_mat reads type_traits_cpu (ggml-cpu.c:1274).
//
// WHY THE TWO TABLES DIVERGE AT ALL, and one cause is NOT a deliberate override -- read this
// before "fixing" either table to make the list below shorter. MEASURED in
// build-static/build.ninja:
//
//     ggml.c.obj        -O3 ... -fopenmp                      <- NO -mavx2
//     ggml-cpu.c.obj    -O3 ... -mavx -mavx2 -fopenmp
//     iqk_quantize.cpp  -O3 ... -mavx -mavx2 -fopenmp
//
// Several entries select .vec_dot_type under "#if defined __AVX2__". ggml.c therefore compiles
// the #else arm and ggml-cpu.c the #if arm, BY CONSTRUCTION. A base-vs-cpu difference on those
// types is EXPECTED and says nothing about correctness. Copying one table's value into the other
// imports a choice made for an arch the other is not built for -- that is precisely how three
// consecutive attempts at 15 of these values were wrong in one session, each attempt trusting a
// more authoritative-looking source than the last while asking the wrong question.
//
// The authority for .vec_dot_type is NEITHER table: it is iqk_set_kernels_*(), which returns
// false for any activation type it has no kernel for. That is now enforced at RUNTIME -- the
// vec_dot wrappers in iqk_quantize.cpp abort instead of returning with *s unwritten -- so a wrong
// pairing announces itself during this sweep rather than being inferred from this report.
//
// THE INVARIANT BELOW IS NOT WEAKENED BY ANY OF THAT. It never claimed divergence was a fault; it
// demands that a divergent type be COVERED NUMERICALLY, which is a separate requirement and still
// unmet for 14 types. The CPU table also overrides the base one deliberately,
// for many types. A test that failed on any disagreement would fail on day one, be silenced, and
// protect nothing. The load-bearing invariant is weaker and actually holds:
//
//     IF the two tables disagree about a type, that type MUST be verified numerically here.
//
// That is what the q8_0 defect violated -- the tables disagreed about Q8_0 and nothing checked
// the arithmetic. The report prints every divergence so the picture is visible, and fails only
// when a divergent, quantised type is absent from the sweep.
// ---- ROW-INTERLEAVED (_rN) TYPES ---------------------------------------------------------------
// ADDED 2026-09-26. The divergence report below failed on exactly 11 types -- q4_0_r8, q5_0_r4,
// q8_0_r8, q4_k_r4, iq1_s_r4, iq4_nl_r4, iq4_xs_r8, iq1_m_r4, q6_0_r4, iq2_bn_r4, mxfp4_r8 -- all
// row-interleaved, so none of them can go through test_width: its reference dequantises row by row
// on ggml_row_size strides, which is wrong for a layout that interleaves 4/8/16 rows (see the note
// in g_sweep). This check never dequantises the _rN tensor at all. It judges the whole MUL_MAT
// output, which is the thing a model consumes:
//   same weights quantised as the _rN type AND as its base type (flat imatrix on BOTH sides -- a
//   one-sided imatrix produced a false "iq2_xxs_r4 16.6% worse" on 2026-09-26), same activations,
//   one column and an 8-column batch, against an exact double-precision product of the f32 weights.
// PASS: finite, and the _rN error vs exact is no more than RN_ACC_SLACK worse than the base type's at
// the same width. Equality with the base is NOT required: q4/q5/q6_k_r4 and iq2_bn_r4 round
// activations differently (~7.8e-3 apart on identical bytes) while being exactly as accurate. The
// base type is the oracle, and the base types are verified by the sweep above.
// Method ported from tests/probe-interleaved-vs-base-mulmat.cpp (53 checks, 0 failed, 2026-09-26),
// brought INTO this binary so that "verified" is earned where the invariant is asserted, not borrowed
// from a test that may not have run.
static const double RN_ACC_SLACK = 1.05;
static int g_rn_judged = 0;   // _rN types are ENUMERATED, not listed in g_sweep -- the SCOPE line counts them apart

static ggml_type rn_base_type(const char * name) {
    std::string nm = name;
    const size_t us = nm.rfind('_');
    if (us == std::string::npos || us + 2 > nm.size() || nm[us + 1] != 'r' || !isdigit((unsigned char) nm[us + 2])) {
        return GGML_TYPE_COUNT;
    }
    std::string base = nm.substr(0, us);
    for (int pass = 0; pass < 2; pass++) {
        for (int t = 0; t < GGML_TYPE_COUNT; t++) {
            const char * tn = ggml_type_name((ggml_type) t);
            if (tn && base == tn) return (ggml_type) t;
        }
        if (!base.empty() && base.back() == 'k') base.back() = 'K';   // q4_k_r4 -> q4_K
        else break;
    }
    return GGML_TYPE_COUNT;
}

// y = W x through a real CPU ggml_mul_mat. W is M rows of K, x is K x ncols.
static std::vector<float> rn_mul_mat(ggml_type wt, const std::vector<float> & Wf, const std::vector<float> & xf,
                                     int64_t K, int64_t M, int64_t ncols) {
    runner g(256);
    ggml_tensor * W = ggml_new_tensor_2d(g.ctx, wt, K, M);
    ggml_tensor * x = ggml_new_tensor_2d(g.ctx, GGML_TYPE_F32, K, ncols);
    ggml_quantize_init(wt);
    std::vector<float> imat((size_t) K, 1.0f);
    ggml_quantize_chunk(wt, Wf.data(), W->data, 0, M, K, imat.data(), nullptr);
    memcpy(x->data, xf.data(), xf.size() * sizeof(float));
    ggml_tensor * y = ggml_mul_mat(g.ctx, W, x);
    g.run({ y });
    std::vector<float> out((size_t) (M * ncols));
    memcpy(out.data(), y->data, out.size() * sizeof(float));
    return out;
}

// true when `cand` is no more than RN_ACC_SLACK worse than `base` against exact, at width 1 AND NW.
static bool rn_judge(ggml_type cand, ggml_type base, const std::vector<float> & Wf, const std::vector<float> & xw,
                     const std::vector<float> & ref_w, int64_t K, int64_t M, int64_t NW, bool quiet,
                     double * worst_ratio) {
    bool ok = true;
    double worst = 0.0;
    const int64_t widths[2] = { 1, NW };
    for (int64_t nc : widths) {
        std::vector<float> x(xw.begin(), xw.begin() + (size_t) (K * nc));
        std::vector<float> ref(ref_w.begin(), ref_w.begin() + (size_t) (M * nc));
        const std::vector<float> yc = rn_mul_mat(cand, Wf, x, K, M, nc);
        const std::vector<float> yb = rn_mul_mat(base, Wf, x, K, M, nc);
        const int64_t nf = count_non_finite(yc.data(), (int64_t) yc.size());
        // Judged on the RMS error. MEASURED 2026-10-17 (24 seeds, this test's shapes): the MAX-error ratio of
        // q6_k_r4 vs q6_K spans 0.85..1.21 at width 1 with the kernels unchanged -- a single extreme order
        // statistic on two different quantisations moves with the draw -- so a 5% rule on it failed 7-12 of 24
        // seeds. The max-error ratio is still printed and recorded (mx), but it does not decide.
        // The RMS ratio over the same 24 seeds: q6_k_r4 w1 0.983..1.023, every _rN row at or under 1.023, all 24
        // seeds pass; the q4_0-vs-q8_0 negative control still reads 10.9..11.9 under it (it must FAIL, and does).
        const double ec = rel_rms(yc.data(), ref.data(), (int64_t) ref.size());
        const double eb = rel_rms(yb.data(), ref.data(), (int64_t) ref.size());
        const double ratio = ec / std::max(eb, 1e-12);
        const double mx = rel_to_max(yc.data(), ref.data(), (int64_t) ref.size()) /
                          std::max(rel_to_max(yb.data(), ref.data(), (int64_t) ref.size()), 1e-12);
        const bool w_ok = nf == 0 && std::isfinite(ec) && ec <= RN_ACC_SLACK * eb;
        worst = std::max(worst, ratio);
        ok = ok && w_ok;
        {
            char key[160];
            snprintf(key, sizeof(key), "%s|%s~%s|K%lld|M%lld|width%lld|", g_section, type_label(cand), type_label(base),
                     (long long) K, (long long) M, (long long) nc);
            record(std::string(key) + "err", ec);
            record(std::string(key) + "base-err", eb);
            record(std::string(key) + "ratio", ratio);
            record(std::string(key) + "max-ratio", mx);
        }
        if (!quiet) {
            printf("  %-9s vs %-7s width %-2lld  err vs exact %.3e  base %.3e  ratio %.3f  non-finite %lld  %s  (max-err ratio %.3f)\n",
                   type_label(cand), type_label(base), (long long) nc, ec, eb, ratio, (long long) nf, w_ok ? "ok" : "FAIL", mx);
        }
    }
    if (worst_ratio) *worst_ratio = worst;
    return ok;
}

static void test_interleaved(int64_t K, int64_t M, int64_t NW) {
    // One generator for the section, keyed by its shape: every _rN type below is judged on the SAME
    // weights and activations, and none of them can be moved by anything that ran before this section.
    std::mt19937 rng = test_rng(g_section, -1, K, M, NW);
    printf("\nROW-INTERLEAVED (_rN) TYPES -- whole-MUL_MAT accuracy against the base type, K=%lld M=%lld\n",
           (long long) K, (long long) M);
    std::vector<float> Wf((size_t) (K * M)), xw((size_t) (K * NW));
    fill(Wf, rng, -0.1f, 0.1f);
    fill(xw, rng, -0.5f, 0.5f);
    std::vector<float> ref((size_t) (M * NW));
    for (int64_t c = 0; c < NW; c++) {
        for (int64_t r = 0; r < M; r++) {
            double s = 0.0;
            for (int64_t i = 0; i < K; i++) s += (double) Wf[(size_t) (r * K + i)] * xw[(size_t) (c * K + i)];
            ref[(size_t) (c * M + r)] = (float) s;
        }
    }

    // NEGATIVE CONTROL: q4_0 judged against a q8_0 "base" must FAIL -- it is far less accurate. If this
    // passes, the rule cannot see a bad kernel and every _rN verdict below would be meaningless.
    {
        double ratio = 0.0;
        const bool passed = rn_judge(GGML_TYPE_Q4_0, GGML_TYPE_Q8_0, Wf, xw, ref, K, M, NW, true, &ratio);
        printf("  %-4s NEGATIVE CONTROL q4_0 judged against q8_0: worst ratio %.2f (must FAIL the %.2f rule)\n",
               passed ? "FAIL" : "ok", ratio, RN_ACC_SLACK);
        if (passed) {
            printf("       the accuracy rule cannot tell a worse kernel from a good one; _rN verdicts withheld\n");
            g_failures++;
            return;
        }
    }

    int n_run = 0, n_fail = 0, n_skip = 0;
    for (int t = 0; t < GGML_TYPE_COUNT; t++) {
        const ggml_type ty = (ggml_type) t;
        const char * tn = ggml_type_name(ty);
        if (!tn) continue;
        const ggml_type base = rn_base_type(tn);
        if (base == GGML_TYPE_COUNT) continue;                        // not an _rN type
        const ggml_type_traits_cpu * cpu = ggml_get_type_traits_cpu(ty);
        if (!cpu || cpu->vec_dot == NULL) continue;                   // no CPU kernel: out of scope, as in the report
        const ggml_type_traits * bt = ggml_get_type_traits(base);
        if (!bt || !bt->to_float || base == GGML_TYPE_Q8_K) {
            printf("  ---- %-9s SKIPPED: base %s is an activation type or has no to_float\n", tn, type_label(base));
            n_skip++;
            continue;
        }
        const int64_t bk = ggml_blck_size(ty), bb = ggml_blck_size(base);
        if ((bk > 1 && K % bk) || (bb > 1 && K % bb)) {
            printf("  ---- %-9s SKIPPED: K=%lld is not a multiple of its block size\n", tn, (long long) K);
            n_skip++;
            continue;
        }
        n_run++;
        if (rn_judge(ty, base, Wf, xw, ref, K, M, NW, false, nullptr)) {
            g_verified.push_back(ty);
        } else {
            n_fail++;
            g_failures++;
        }
    }
    g_rn_judged = n_run;
    printf("  %d _rN type(s) judged, %d failed, %d skipped with a reason\n", n_run, n_fail, n_skip);
    if (n_run == 0) {
        printf("  FAIL no _rN type was judged -- this section inspected nothing\n");
        g_failures++;
    }
}

// Types this harness cannot quantise (no CPU .from_float) whose MUL_MAT numerics a sibling test
// pins. Each entry must name that test; the report prints it, so it is a visible statement and not
// a silent waiver. TQ2_T (Gyro trellis, ik df93e4cc9 #2812): base q8_0 vs cpu q8_2_x4 because it is
// the one trellis type with a dense iqk GEMM; tests/test-tq-trellis.cpp runs ggml_mul_mat on CPU
// against a double-precision reference (rel err <= 2e-2 asserted, measured worst 1.098e-2).
static const struct { ggml_type type; const char * test; } g_verified_elsewhere[] = {
    { GGML_TYPE_TQ2_T, "test-tq-trellis" },
};

static void traits_divergence_report(void) {
    printf("TRAITS-TABLE DIVERGENCE (ggml.c vs ggml-cpu.c) -- every type with a CPU vec_dot,\n"
           "judged against what was actually VERIFIED above, not against what was scheduled\n");

    int n_div = 0, n_uncovered = 0, n_checked = 0;
    for (int t = 0; t < GGML_TYPE_COUNT; t++) {
        const ggml_type ty = (ggml_type) t;
        const ggml_type_traits     * base = ggml_get_type_traits(ty);
        const ggml_type_traits_cpu * cpu  = ggml_get_type_traits_cpu(ty);
        if (!base || !cpu || !base->type_name) {
            continue;
        }
        // Only types the CPU backend can actually multiply are in scope: a null vec_dot means no
        // kernel is reached and the declaration cannot mislead one.
        if (cpu->vec_dot == NULL) {
            continue;
        }
        n_checked++;
        if (base->vec_dot_type == cpu->vec_dot_type) {
            continue;
        }
        n_div++;

        // VERIFIED, not merely listed. g_verified is appended at the END of test_width, so a
        // type that was listed in g_sweep and then skipped does NOT count here.
        bool verified = false;
        for (size_t i = 0; i < g_verified.size(); i++) {
            if (g_verified[i] == ty) { verified = true; break; }
        }
        const char * elsewhere = nullptr;
        if (!verified) {
            for (const auto & e : g_verified_elsewhere) {
                if (e.type == ty) { elsewhere = e.test; break; }
            }
        }
        printf("  %-9s base=%-12s cpu=%-12s  %s%s\n",
               base->type_name,
               ggml_type_name(base->vec_dot_type),
               ggml_type_name(cpu->vec_dot_type),
               verified ? "verified numerically above" : (elsewhere ? "verified numerically in " : "NOT VERIFIED"),
               elsewhere ? elsewhere : "");
        if (!verified && !elsewhere && base->is_quantized) {
            n_uncovered++;
        }
    }

    printf("  %d type(s) have a CPU vec_dot; %d disagree between the tables; %d of those are\n"
           "  quantised and NOT numerically covered here.\n", n_checked, n_div, n_uncovered);

    // CONTROL: if nothing was checked the report proves nothing, and a clean result would be
    // indistinguishable from a broken loop.
    if (n_checked == 0) {
        printf("  FAIL no type had a CPU vec_dot -- this report inspected nothing, so its\n"
               "       silence is not a result.\n");
        g_failures++;
        return;
    }
    if (n_uncovered) {
        printf("  FAIL %d quantised type(s) disagree between the two tables and are not verified\n"
               "       numerically. That is the exact condition that let the q8_0 defect ship:\n"
               "       the tables disagreed and nothing checked the arithmetic. Add them to\n"
               "       g_sweep, or establish why the disagreement is safe for them.\n", n_uncovered);
        g_failures++;
    } else {
        printf("  ok   every divergent quantised type is numerically verified below\n");
    }
    printf("\n");
}

// One whole run of every check below, over the given class-sweep list. Returns its failure count.
// Global per-run state is reset here so --isolation-check can call it twice in one process.
static int run_suite(const std::vector<ggml_type> & sweep) {
    g_failures = 0;
    g_verified.clear();
    g_rn_judged = 0;
    g_metrics.clear();
    g_section = "";

    printf("MUL_MAT: one activation column vs the same column inside a wider batch.\n");
    printf("seed %u\n", (unsigned) g_seed);
    printf("Same weights, same vector; the only difference is the number of columns.\n\n");

    // NEGATIVE CONTROL on the comparator.
    {
        std::vector<float> a(512), b(512);
        std::mt19937 rng = test_rng("comparator", -1, 512, 0, 0);
        fill(a, rng, -1.0f, 1.0f);
        b = a;
        const double same = max_rel(a.data(), b.data(), 512);
        b[100] += 1.0f;
        const double diff = max_rel(a.data(), b.data(), 512);
        const bool ok = same == 0.0 && diff > 1e-2;
        printf("  %-4s NEGATIVE CONTROL (comparator)  identical=%.3e perturbed=%.3e\n",
               ok ? "ok" : "FAIL", same, diff);
        if (!ok) {
            printf("       the comparator cannot report a difference; nothing below means anything\n");
            g_failures++;
        }
    }

    // f32 weights first: if these pass and q8_0 fails at the same shapes, the quantised path is
    // implicated rather than narrow columns as such.
    printf("\nF32 weights (control family -- no quantised kernel involved)\n");
    g_section = "f32";
    test_width(GGML_TYPE_F32, 576, 576,  8);
    test_width(GGML_TYPE_F32, 576, 1536, 8);

    // LOCALISATION. If every quantised type fails at one column, the fault is in the shared
    // single-column dispatch or in the activation quantisation that feeds it. If only some fail,
    // it is in those types' own kernels. f16 is included because it takes the quantised-style
    // path (src1 is converted) without being a block-quantised format.
    printf("\nOther weight types at the same shape -- does this affect one type or all of them?\n");
    g_section = "local";
    test_width(GGML_TYPE_F16,  576, 1536, 8);
    test_width(GGML_TYPE_Q4_0, 576, 1536, 8);
    // K-quants need K divisible by 256, so they get K=512 rather than the model's 576.
    test_width(GGML_TYPE_Q4_K, 512, 1536, 8);
    test_width(GGML_TYPE_Q6_K, 512, 1536, 8);
    test_width(GGML_TYPE_Q8_0, 512, 1536, 8);   // same K as the K-quants, for comparability

    printf("\nQ8_0 weights (the shapes the model actually failed at)\n");
    g_section = "q8shape";
    test_width(GGML_TYPE_Q8_0, 576, 576,  8);   // blk.N.attn_q.weight  -> Qcur
    test_width(GGML_TYPE_Q8_0, 576, 192,  8);   // blk.N.attn_k.weight  -> Kcur
    test_width(GGML_TYPE_Q8_0, 576, 1536, 8);   // blk.N.ffn_gate.weight -> ffn_gate
    test_width(GGML_TYPE_Q8_0, 576, 1536, 5);   // the prompt width that WORKED in the model
    test_width(GGML_TYPE_Q8_0, 576, 1536, 2);   // narrowest batch that is still not a vector

    // ------------------------------------------------------------------------------------------
    // CLASS SWEEP.
    //
    // INVENT-LADDER-OK: nothing here is declared impossible or unreachable -- this BUILDS the
    // missing capability rather than conceding it. (1) existing fix: none, no test in this tree
    // covered single-column quantised mul_mat, which is why the defect survived; (2) prior art:
    // the in-tree harnesses were read -- tests/test-backend-ops.cpp compares backends against each
    // other, so a host-only fault reproduced on both sides of its comparison and could not be seen
    // by it; (3) adapt: the existing test_width above is reused unchanged, driven over a type
    // table; (4) invention was not needed because (3) succeeded. The falsifiable test IS this
    // sweep, and its controls are the types already measured correct.
    //
    // WHY A SWEEP AND NOT A READ OF THE TABLES. The q8_0 defect was one kernel handing
    // iqk_mul_mat the wrong activation type, and it originated in two type-traits tables in this
    // tree disagreeing (ggml.c:1019 gives Q8_0 a vec_dot_type of Q8_2_X4; ggml-cpu.c:275 gives
    // Q8_0). Nine further ggml.c entries name Q8_2_X4. Fixing one kernel says nothing about the
    // rest, and reading the tables would report what is WRITTEN rather than which kernel is
    // actually reached at run time. The arithmetic answers that; the tables only describe it.
    //
    // K=512 divides both 32 and 256, so every block-quantised type below is validly shaped.
    printf("\nCLASS SWEEP -- every quantisable type at ONE column, K=512, M=1536\n");
    printf("(the width that exposed the q8_0 defect; a type this build cannot quantise prints a\n");
    printf(" SKIPPED line naming the reason, so it cannot be mistaken for a pass)\n");
    g_section = "sweep";
    for (size_t i = 0; i < sweep.size(); i++) {
        test_width(sweep[i], 512, 1536, 8);
    }

    // TARGETED PROBE for the two types that disagree with the double-precision reference by 0.26
    // and 0.21 while agreeing with THEMSELVES across batch widths (one-vs-wide = 0). Reading the
    // kernels did not settle it: the scale convention, the grid family and the merge were all
    // checked and all matched a PASSING sibling. The question left is narrow and shape answers it.
    //
    //   CONSTANT across K and width  -> a value/grid fault, which does not care about shape.
    //   GROWS with K                 -> an indexing or stride fault.
    //   VARIES with width            -> a kernel selected per nrc_y, not the decoder.
    //
    // iq4_ks runs the identical shapes as the CONTROL. It shares the scale convention
    // d*((scales&254)-127), the .vec_dot_type, the .nrows, the row_meta_size and the grid family,
    // and it PASSES at 0.003653 -- so any shape effect IT also shows belongs to this harness
    // rather than to the two suspects, and the comparison is worthless without it.
    printf("\nTARGETED PROBE -- iq4_kss / iq5_ks against the passing iq4_ks control\n");
    const ggml_type probe[] = { GGML_TYPE_IQ4_KS, GGML_TYPE_IQ4_KSS, GGML_TYPE_IQ5_KS };
    g_section = "probe";
    for (size_t i = 0; i < sizeof(probe)/sizeof(probe[0]); i++) {
        test_width(probe[i],  256, 1536, 8);
        test_width(probe[i], 1024, 1536, 8);
        test_width(probe[i],  512, 1536, 1);
        test_width(probe[i],  512, 1536, 2);
    }

    // Row-interleaved types: whole-MUL_MAT check against their base type (see test_interleaved).
    // M=1536 is a multiple of every interleave factor (4/8/16); K=512 divides 32 and 256.
    g_section = "rN";
    test_interleaved(512, 1536, 8);

    // Runs LAST, so it judges what was actually verified rather than what was scheduled.
    printf("\n");
    traits_divergence_report();

    printf("\n%s -- %d failure(s)\n", g_failures == 0 ? "PASS" : "FAIL", g_failures);
    // THE TYPE COUNTS ARE COMPUTED, NEVER WRITTEN DOWN. This line said "14 weight types" and
    // was still saying 14 after the sweep grew to cover this fork's IQ*_K family. That is not
    // cosmetic: this sentence is quoted as the authority on what the harness does NOT cover,
    // so a SCOPE line that understates its own reach is worse than no SCOPE line at all.
    //
    // Three numbers, because they are three different facts. LISTED is what was scheduled.
    // VERIFIED is what actually reached a numeric verdict -- g_verified is appended only at
    // the end of test_width, after the comparisons run. A type that SKIPPED, or whose
    // dequantisation control failed, is listed and NOT verified, and collapsing the two is
    // exactly how "listed" comes to be read as "covered".
    printf("SCOPE: %zu weight types listed (f32, f16 and %zu in the class sweep) plus %d row-interleaved\n"
           "_rN types found by enumeration and judged against their base type; %zu distinct types\n"
           "reached a numeric verdict this run -- the rest print their own reason above.\n"
           "K in {512, 576}, M in {192, 576, 1536}, batch widths 1/2/5/8, host backend only,\n"
           "nrc == 1. A PASS says those widths agree with each other and with a double-\n"
           "precision reference at those shapes. It does NOT certify MUL_MAT generally.\n"
           "NOT COVERED: nrc == 2 (the ARM MATMUL_INT8 path), non-contiguous or permuted src,\n"
           "batch dimensions ne02/ne03 > 1, any backend but the host, and every type whose\n"
           "quantiser this build does not provide (each prints its own SKIPPED line above).\n",
           2 + sweep.size(), sweep.size(),
           g_rn_judged, n_verified_types());
    return g_failures;
}

static ggml_type type_by_name(const char * name) {
    for (int t = 0; t < GGML_TYPE_COUNT; t++) {
        const char * tn = ggml_type_name((ggml_type) t);
        if (tn && strcmp(tn, name) == 0) return (ggml_type) t;
    }
    return GGML_TYPE_COUNT;
}

// Key-by-key comparison of two runs' verdicts. Only keys present in BOTH are compared (the inserted
// type's own keys exist in one run only). Bitwise: these are deterministic CPU computations, and an
// isolated test must reproduce its numbers exactly, not approximately.
static int count_moved(const std::vector<metric> & a, const std::vector<metric> & b, int * n_compared, bool print) {
    int moved = 0, compared = 0;
    for (size_t i = 0; i < a.size(); i++) {
        bool first = true;   // first occurrence of a key in a
        for (size_t j = 0; j < i; j++) { if (a[j].key == a[i].key) { first = false; break; } }
        if (!first) continue;
        for (size_t j = 0; j < b.size(); j++) {
            if (b[j].key != a[i].key) continue;
            compared++;
            const double x = a[i].value, y = b[j].value;
            const bool same = (x == y) || (std::isnan(x) && std::isnan(y));
            if (!same) {
                moved++;
                if (print && moved <= 40) {
                    printf("  MOVED %-58s %.6e -> %.6e\n", a[i].key.c_str(), x, y);
                }
            }
            break;
        }
    }
    if (n_compared) *n_compared = compared;
    return moved;
}

int main(int argc, char ** argv) {
    // UNBUFFERED STDOUT, AND IT IS A CORRECTNESS FIX FOR THIS HARNESS, NOT A CONVENIENCE.
    // This program exists to localise a fault to a TYPE, and it does that by being the last thing
    // it printed. Redirected to a file, stdout is block-buffered, so on a crash the unflushed tail
    // is lost and the last visible line names a type several entries EARLIER than the one that
    // actually died. That happened here: the run ended showing a PARTIAL row line -- a line
    // printed by ONE printf, so the truncation was itself the evidence that the buffer had been
    // cut mid-content -- and the type it named was innocent. A wrong type was diagnosed twice
    // before the buffering was accounted for. Unbuffered costs a syscall per line in a test that
    // runs for seconds; the alternative costs a wrong answer, which is the one thing this file
    // must not produce.
    setvbuf(stdout, NULL, _IONBF, 0);

    // --seed N            base seed (default 20260914, the historic one)
    // --sweep-prepend T   insert type T (by ggml name) at the FRONT of the class sweep -- a dummy entry
    //                     for measuring whether one test's data depends on what ran before it
    // --isolation-check   run the suite twice, plain and with --sweep-prepend (default q4_0), and FAIL
    //                     if any verdict present in both runs changed. Exit code is that verdict alone.
    bool isolation = false;
    const char * prepend = nullptr;
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--seed") == 0 && i + 1 < argc) {
            g_seed = (uint32_t) strtoul(argv[++i], nullptr, 10);
        } else if (strcmp(argv[i], "--sweep-prepend") == 0 && i + 1 < argc) {
            prepend = argv[++i];
        } else if (strcmp(argv[i], "--isolation-check") == 0) {
            isolation = true;
        } else {
            fprintf(stderr, "usage: %s [--seed N] [--sweep-prepend TYPE] [--isolation-check]\n", argv[0]);
            return 2;
        }
    }

    const std::vector<ggml_type> plain(g_sweep, g_sweep + sizeof(g_sweep)/sizeof(g_sweep[0]));
    std::vector<ggml_type> extra = plain;
    if (prepend || isolation) {
        const char * nm = prepend ? prepend : "q4_0";
        const ggml_type t = type_by_name(nm);
        if (t == GGML_TYPE_COUNT) {
            fprintf(stderr, "unknown type '%s'\n", nm);
            return 2;
        }
        extra.insert(extra.begin(), t);
    }

    if (!isolation) {
        return run_suite(prepend ? extra : plain) == 0 ? 0 : 1;
    }

    const int f_plain = run_suite(plain);
    const std::vector<metric> m_plain = g_metrics;
    printf("\n==================== ISOLATION RUN 2: class sweep with %s inserted at the front ====================\n\n",
           type_label(extra[0]));
    const int f_extra = run_suite(extra);
    const std::vector<metric> m_extra = g_metrics;

    printf("\nISOLATION CHECK (seed %u): does inserting %s into the class sweep move any OTHER verdict?\n",
           (unsigned) g_seed, type_label(extra[0]));
    // CONTROL on the comparison: a copy with one value perturbed must count exactly one move, or a
    // clean result below could be a comparison that cannot see a difference.
    bool ctl_ok = false;
    if (!m_plain.empty()) {
        std::vector<metric> pert = m_plain;
        pert.front().value = pert.front().value * 1.5 + 1.0;   // the first entry is always a first occurrence
        int nc = 0;
        ctl_ok = count_moved(m_plain, pert, &nc, false) == 1;
    }
    printf("  %-4s CONTROL comparison sees a single perturbed verdict\n", ctl_ok ? "ok" : "FAIL");
    int n_compared = 0;
    const int moved = count_moved(m_plain, m_extra, &n_compared, true);
    printf("  %d verdict(s) compared, %d moved (suite failures: plain %d, with insert %d -- reported, not judged here)\n",
           n_compared, moved, f_plain, f_extra);
    const bool ok = ctl_ok && n_compared > 0 && moved == 0;
    if (n_compared == 0) printf("  FAIL nothing was compared -- this check inspected nothing\n");
    if (moved) printf("  FAIL a test's data depends on its position in the run: inserting an unrelated type moved %d verdict(s)\n", moved);
    printf("\nISOLATION %s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}

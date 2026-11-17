// IQ1_XS / IQ1_XXS / IQ1_XXXS (Unsloth narrow-grid IQ1 types, enum ids 64-66).
//
// SOURCE-READ: unslothai/llama.cpp branch iq1-narrow-grids-b10775 commit c86ed2699 (MIT).
//
// WHAT THIS PROVES, and what it does not (rule: presence is not correctness for a quant).
//   1. LAYOUT   block bytes 46 / 42 / 38 per 256 weights (1.4375 / 1.3125 / 1.1875 bpw), enum ids 64/65/66,
//               type-traits rows on BOTH the ggml.c and the CPU side, ftype -> type mapping.
//   2. ENCODE   ggml_quantize_chunk with an imatrix returns exactly nrows*row_size bytes, the row passes
//               ggml_validate_row_data, and every dequantised weight is finite.
//   3. DECODE   dequantize_row_<t> (the library) equals an INDEPENDENT decoder written here from the layout
//               spec with raw byte offsets (bit-for-bit). A NEGATIVE CONTROL decoder with one bit field
//               swapped must DISAGREE, otherwise the equality above would be vacuous.
//   4. AGREE    the vec_dot kernel and dequantize_row decode the same weights: a one-hot q8_K activation is
//               lossless, so dot(w, e_i) must return the decoded weight i (precedent: test-iqk-decode-agreement).
//   5. MULMAT   the CPU backend CLAIMS and COMPUTES MUL_MAT, MUL_MAT_ID, FUSED_UP_GATE and MOE_UP_GATE on these
//               types (an unwired type used to be a GGML_ABORT in this fork, not a slow path) and agrees with
//               the same op on the dequantised weights.
// It says nothing about GPU backends (there is no CUDA/Vulkan path for these types in this tree) and nothing
// about model quality: that is the head-to-head in temp/research/findings/S69_UNSL_IQ1_2026-10-01.md.

#define GGML_COMMON_DECL_C
#define GGML_COMMON_IMPL_C
#include "ggml-common.h"

#include "ggml.h"
#include "ggml-cpu.h"
#include "ggml-backend.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

static int g_fail = 0;
#define CHECK(cond, ...) do { if (!(cond)) { ++g_fail; printf("FAIL %s:%d  ", __FILE__, __LINE__); printf(__VA_ARGS__); printf("\n"); } } while (0)

struct NarrowInfo {
    ggml_type type;
    ggml_ftype ftype;
    const char * name;
    int id;
    size_t block_bytes;
    double bpw;
    const uint64_t * grid;
    int ngrid;
};

static const NarrowInfo k_types[] = {
    { GGML_TYPE_IQ1_XS,   GGML_FTYPE_MOSTLY_IQ1_XS,   "iq1_xs",   64, 46, 1.4375, iq1_xs_grid,   NGRID_IQ1XS   },
    { GGML_TYPE_IQ1_XXS,  GGML_FTYPE_MOSTLY_IQ1_XXS,  "iq1_xxs",  65, 42, 1.3125, iq1_xxs_grid,  NGRID_IQ1XXS  },
    { GGML_TYPE_IQ1_XXXS, GGML_FTYPE_MOSTLY_IQ1_XXXS, "iq1_xxxs", 66, 38, 1.1875, iq1_xxxs_grid, NGRID_IQ1XXXS },
};

static float fp16_to_f32(const uint8_t * p) {
    uint16_t h; memcpy(&h, p, 2);
    return ggml_fp16_to_fp32((ggml_fp16_t) h);
}

// Independent decoder, written from the layout spec in ggml-common.h using raw byte offsets only.
// `swap_fields` is the negative control: it deliberately swaps two index-high fields.
static void ref_decode_row(const NarrowInfo & t, const uint8_t * row, float * y, int64_t k, bool swap_fields) {
    const size_t bb = t.block_bytes;
    for (int64_t i = 0; i < k/256; ++i) {
        const uint8_t * b = row + i*bb;
        const float d = fp16_to_f32(b);
        const uint8_t * qs = b + 2;          // 32 bytes in every type
        const uint8_t * qh = b + 2 + 32;     // xs: 8 bytes, xxs: 8 bytes, xxxs: none
        const uint8_t * sc = (t.type == GGML_TYPE_IQ1_XS) ? b + 2 + 32 + 8 : b + 2 + 32;   // xs: 4 bytes after qh; xxxs: 4 bytes after qs
        for (int ib = 0; ib < 8; ++ib) {
            int ls, neg;
            if (t.type == GGML_TYPE_IQ1_XXS) {
                ls  = (qh[ib] >> 4) & 7;
                neg = (qh[ib] >> 7) & 1;
            } else {
                const int nib = (sc[ib/2] >> (4*(ib & 1))) & 0xf;
                ls  = nib & 7;
                neg = (nib >> 3) & 1;
            }
            const float dl    = d * (float)(2*ls + 1);
            const float delta = neg ? -0.125f : 0.125f;
            for (int l = 0; l < 4; ++l) {
                int idx = qs[4*ib + l];
                int hi_slot = swap_fields ? (l ^ 1) : l;
                if (t.type == GGML_TYPE_IQ1_XS)  idx |= ((qh[ib] >> (2*hi_slot)) & 3) << 8;
                if (t.type == GGML_TYPE_IQ1_XXS) idx |= ((qh[ib] >> hi_slot) & 1) << 8;
                const uint64_t g = t.grid[idx];
                for (int j = 0; j < 8; ++j) {
                    const int8_t gv = (int8_t) ((g >> (8*j)) & 0xff);
                    y[(int64_t)i*256 + ib*32 + l*8 + j] = dl * ((float) gv + delta);
                }
            }
        }
    }
}

static void test_layout() {
    printf("[layout]\n");
    for (const auto & t : k_types) {
        CHECK((int) t.type == t.id, "%s enum id %d != %d", t.name, (int) t.type, t.id);
        const ggml_type_traits * tr = ggml_get_type_traits(t.type);
        CHECK(tr && tr->type_name && strcmp(tr->type_name, t.name) == 0, "%s type_name", t.name);
        CHECK(ggml_blck_size(t.type) == 256, "%s blck_size %d", t.name, (int) ggml_blck_size(t.type));
        CHECK(ggml_type_size(t.type) == t.block_bytes, "%s type_size %zu want %zu", t.name, ggml_type_size(t.type), t.block_bytes);
        CHECK(ggml_is_quantized(t.type), "%s is_quantized", t.name);
        const double bpw = 8.0 * (double) ggml_type_size(t.type) / (double) ggml_blck_size(t.type);
        CHECK(std::fabs(bpw - t.bpw) < 1e-12, "%s bpw %.6f want %.6f", t.name, bpw, t.bpw);
        CHECK(tr && tr->to_float && tr->from_float && tr->from_float_ref, "%s ggml.c traits incomplete", t.name);
        const ggml_type_traits_cpu * cpu = ggml_get_type_traits_cpu(t.type);
        CHECK(cpu && cpu->vec_dot && cpu->vec_dot_type == GGML_TYPE_Q8_K, "%s CPU traits (vec_dot / vec_dot_type)", t.name);
        CHECK(ggml_ftype_to_ggml_type(t.ftype) == t.type, "%s ftype->type", t.name);
        CHECK(ggml_quantize_requires_imatrix(t.type), "%s must require an imatrix (the search is only defined against one)", t.name);
        printf("  %-9s id=%d  %zu B/256  %.4f bpw  ok\n", t.name, t.id, t.block_bytes, bpw);
    }
}

static std::vector<float> make_weights(int64_t n, uint32_t seed) {
    std::mt19937 rng(seed);
    std::normal_distribution<float> N(0.0f, 0.05f);
    std::vector<float> w(n);
    for (auto & v : w) v = N(rng);
    return w;
}

static std::vector<float> make_imatrix(int64_t n, uint32_t seed) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> U(0.1f, 2.0f);
    std::vector<float> w(n);
    for (auto & v : w) v = U(rng);
    return w;
}

static void test_encode_decode() {
    printf("[encode/decode]\n");
    const int64_t K = 1024, R = 16;
    const auto w  = make_weights(K*R, 11);
    const auto im = make_imatrix(K, 12);
    double prev_err = 0;
    for (const auto & t : k_types) {
        const size_t row_size = ggml_row_size(t.type, K);
        std::vector<uint8_t> q(row_size*R, 0xA5);
        const size_t got = ggml_quantize_chunk(t.type, w.data(), q.data(), 0, R, K, im.data(), nullptr);
        CHECK(got == row_size*R, "%s quantize_chunk returned %zu want %zu", t.name, got, row_size*R);
        CHECK(ggml_validate_row_data(t.type, q.data(), q.size()), "%s ggml_validate_row_data rejected its own output", t.name);

        std::vector<float> lib(K*R), ref(K*R), neg(K*R);
        const ggml_type_traits * tr = ggml_get_type_traits(t.type);
        for (int64_t r = 0; r < R; ++r) {
            tr->to_float(q.data() + r*row_size, lib.data() + r*K, K);
            ref_decode_row(t, q.data() + r*row_size, ref.data() + r*K, K, false);
            ref_decode_row(t, q.data() + r*row_size, neg.data() + r*K, K, true);
        }
        int64_t n_nonfinite = 0, n_mismatch = 0, n_neg_diff = 0;
        double num = 0, den = 0;
        for (int64_t i = 0; i < K*R; ++i) {
            if (!std::isfinite(lib[i])) ++n_nonfinite;
            if (memcmp(&lib[i], &ref[i], 4) != 0) ++n_mismatch;
            if (memcmp(&lib[i], &neg[i], 4) != 0) ++n_neg_diff;
            const double e = (double) lib[i] - w[i];
            num += e*e; den += (double) w[i]*w[i];
        }
        CHECK(n_nonfinite == 0, "%s %lld non-finite weights", t.name, (long long) n_nonfinite);
        CHECK(n_mismatch == 0, "%s library decode != independent decode at %lld of %lld weights", t.name, (long long) n_mismatch, (long long) (K*R));
        if (t.type != GGML_TYPE_IQ1_XXXS) {
            // XXXS has no index-high field to swap, so the control only bites for XS / XXS
            CHECK(n_neg_diff > 0, "%s negative control (swapped index-high fields) did NOT change the decode: equality above is vacuous", t.name);
        }
        const double nrmse = std::sqrt(num/den);
        printf("  %-9s bytes=%zu  imatrix-quantised nrmse=%.4f  lib==ref %s  negctl-diff=%lld\n", t.name, got, nrmse,
               n_mismatch == 0 ? "bit-exact" : "MISMATCH", (long long) n_neg_diff);
        CHECK(nrmse < 0.95, "%s nrmse %.4f is not better than guessing zero by a clear margin", t.name, nrmse);
        // a smaller codebook can only widen the error; allow 2% slack for the greedy search
        if (prev_err > 0) CHECK(nrmse >= prev_err*0.98, "%s nrmse %.4f below the wider-grid type's %.4f", t.name, nrmse, prev_err);
        prev_err = nrmse;
    }
}

static void test_dot_agrees_with_dequant() {
    printf("[vec_dot vs dequantize_row, one-hot q8_K activation]\n");
    const int64_t K = 256;
    const auto w  = make_weights(K, 21);
    const auto im = make_imatrix(K, 22);
    const auto * q8 = ggml_get_type_traits_cpu(GGML_TYPE_Q8_K);
    for (const auto & t : k_types) {
        std::vector<uint8_t> q(ggml_row_size(t.type, K));
        ggml_quantize_chunk(t.type, w.data(), q.data(), 0, 1, K, im.data(), nullptr);
        std::vector<float> dq(K);
        ggml_get_type_traits(t.type)->to_float(q.data(), dq.data(), K);
        const auto * cpu = ggml_get_type_traits_cpu(t.type);
        std::vector<uint8_t> xq(ggml_row_size(GGML_TYPE_Q8_K, K));
        double worst = 0, amax = 0;
        for (int64_t i = 0; i < K; ++i) amax = std::fmax(amax, std::fabs(dq[i]));
        for (int64_t i = 0; i < K; ++i) {
            std::vector<float> e(K, 0.0f); e[i] = 1.0f;
            q8->from_float(e.data(), xq.data(), K);
            float s = NAN;
            cpu->vec_dot(K, &s, 0, q.data(), 0, xq.data(), 0, 1);
            worst = std::fmax(worst, std::fabs((double) s - dq[i]));
        }
        printf("  %-9s worst |dot(w,e_i) - dequant[i]| = %.3e  (max |w| = %.3e)\n", t.name, worst, amax);
        CHECK(worst <= 1e-5*amax + 1e-9, "%s vec_dot decodes different weights than dequantize_row (worst %.3e)", t.name, worst);
    }
}

// GATE-CLAIM-OK: test code; nothing is declared unavailable here.
// the scalar reference kernels (ggml-quants.c) and the AVX2 kernels (ggml-cpu/arch/x86/quants.c) are separate code
typedef void (*vec_dot_fn)(int, float *, size_t, const void *, size_t, const void *, size_t, int);
extern "C" {
void ggml_vec_dot_iq1_xs_q8_K(int, float *, size_t, const void *, size_t, const void *, size_t, int);
void ggml_vec_dot_iq1_xxs_q8_K(int, float *, size_t, const void *, size_t, const void *, size_t, int);
void ggml_vec_dot_iq1_xxxs_q8_K(int, float *, size_t, const void *, size_t, const void *, size_t, int);
void ggml_vec_dot_iq1_xs_q8_K_avx2(int, float *, size_t, const void *, size_t, const void *, size_t, int);
void ggml_vec_dot_iq1_xxs_q8_K_avx2(int, float *, size_t, const void *, size_t, const void *, size_t, int);
void ggml_vec_dot_iq1_xxxs_q8_K_avx2(int, float *, size_t, const void *, size_t, const void *, size_t, int);
}

// DENSE dot products: one-hot probes cannot see a bug that needs two non-zero activations (a wrong sub-block scale applied to the
// wrong half, a sign taken from the neighbouring sub-block). Random q8_K activations, three independent answers must agree:
// the registered kernel, the scalar reference, and a double-precision dot of the two DEQUANTISED vectors.
static void test_dense_dot() {
    printf("[dense dot: registered kernel vs scalar reference vs double(dequant)]\n");
    const int64_t K = 1024;
    const struct { ggml_type t; vec_dot_fn scalar; vec_dot_fn avx2; } k[] = {
        { GGML_TYPE_IQ1_XS,   ggml_vec_dot_iq1_xs_q8_K,   ggml_vec_dot_iq1_xs_q8_K_avx2   },
        { GGML_TYPE_IQ1_XXS,  ggml_vec_dot_iq1_xxs_q8_K,  ggml_vec_dot_iq1_xxs_q8_K_avx2  },
        { GGML_TYPE_IQ1_XXXS, ggml_vec_dot_iq1_xxxs_q8_K, ggml_vec_dot_iq1_xxxs_q8_K_avx2 },
    };
    const auto * q8t = ggml_get_type_traits_cpu(GGML_TYPE_Q8_K);
    std::mt19937 rng(77);
    std::uniform_real_distribution<float> U(-1.0f, 1.0f);
    for (const auto & e : k) {
        const auto w  = make_weights(K*64, 91);
        const auto im = make_imatrix(K, 92);
        const size_t rs = ggml_row_size(e.t, K);
        std::vector<uint8_t> q(rs*64);
        ggml_quantize_chunk(e.t, w.data(), q.data(), 0, 64, K, im.data(), nullptr);
        const auto * cpu = ggml_get_type_traits_cpu(e.t);
        double worst_scalar = 0, worst_avx = 0, worst_reg = 0;
        for (int r = 0; r < 64; ++r) {
            std::vector<float> x(K);
            for (auto & v : x) v = U(rng);
            std::vector<uint8_t> xq(ggml_row_size(GGML_TYPE_Q8_K, K));
            q8t->from_float(x.data(), xq.data(), K);
            std::vector<float> xd(K), wd(K);
            // q8_K has no to_float in this tree: dequantise by its layout (float d, int8 qs[256], per 256-block)
            for (int64_t b = 0; b < K/256; ++b) {
                const block_q8_K * bq = (const block_q8_K *) xq.data() + b;
                for (int j = 0; j < 256; ++j) xd[b*256 + j] = bq->d * (float) bq->qs[j];
            }
            ggml_get_type_traits(e.t)->to_float(q.data() + r*rs, wd.data(), K);
            double ref = 0, mag = 0;
            for (int64_t i = 0; i < K; ++i) { ref += (double) wd[i]*xd[i]; mag += std::fabs((double) wd[i]*xd[i]); }
            float s_sc = NAN, s_av = NAN, s_reg = NAN;
            e.scalar(K, &s_sc, 0, q.data() + r*rs, 0, xq.data(), 0, 1);
            e.avx2  (K, &s_av, 0, q.data() + r*rs, 0, xq.data(), 0, 1);
            cpu->vec_dot(K, &s_reg, 0, q.data() + r*rs, 0, xq.data(), 0, 1);
            worst_scalar = std::fmax(worst_scalar, std::fabs(s_sc - ref)/mag);
            worst_avx    = std::fmax(worst_avx,    std::fabs(s_av - ref)/mag);
            worst_reg    = std::fmax(worst_reg,    std::fabs(s_reg - ref)/mag);
        }
        printf("  %-9s worst |dot-ref|/sum|w*x|:  scalar=%.2e  avx2-entry=%.2e  registered=%.2e\n", ggml_type_name(e.t), worst_scalar, worst_avx, worst_reg);
        CHECK(worst_scalar < 1e-5, "%s scalar dot disagrees with double(dequant): %.3e", ggml_type_name(e.t), worst_scalar);
        CHECK(worst_avx < 1e-5,    "%s AVX2 dot disagrees with double(dequant): %.3e", ggml_type_name(e.t), worst_avx);
        CHECK(worst_reg < 1e-5,    "%s registered dot disagrees with double(dequant): %.3e", ggml_type_name(e.t), worst_reg);
    }
}

static double rel_rms(const std::vector<double> & ref, const float * y, size_t n) {
    double num = 0, den = 0;
    for (size_t i = 0; i < n; ++i) { const double d = (double) y[i] - ref[i]; num += d*d; den += ref[i]*ref[i]; }
    return den > 0 ? std::sqrt(num/den) : std::sqrt(num);
}

static std::vector<float> dequant_tensor(const NarrowInfo & t, const uint8_t * data, int64_t K, int64_t rows) {
    std::vector<float> out(K*rows);
    const size_t rs = ggml_row_size(t.type, K);
    for (int64_t r = 0; r < rows; ++r) ggml_get_type_traits(t.type)->to_float(data + r*rs, out.data() + r*K, K);
    return out;
}

static void quantize_tensor(const NarrowInfo & t, const std::vector<float> & w, uint8_t * dst, int64_t K, int64_t rows, const std::vector<float> & im) {
    const size_t got = ggml_quantize_chunk(t.type, w.data(), dst, 0, rows, K, im.data(), nullptr);
    CHECK(got == ggml_row_size(t.type, K)*rows, "%s quantize size", t.name);
}

static void test_graph_ops() {
    printf("[CPU backend: MUL_MAT / MUL_MAT_ID / FUSED_UP_GATE / MOE_UP_GATE]\n");
    const int64_t K = 512, M = 64, E = 4, T = 5, NU = 2;
    ggml_backend_t be = ggml_backend_cpu_init();
    CHECK(be != nullptr, "no host backend");
    if (!be) return;
    ggml_backend_cpu_set_n_threads(be, 4);
    const auto im = make_imatrix(K, 31);

    for (const auto & t : k_types) {
        // ---- MUL_MAT, N = 1 / 7 / 40 columns
        for (int64_t N : {1, 7, 40}) {
            ggml_init_params ip = { 64u*1024*1024, nullptr, false };
            ggml_context * ctx = ggml_init(ip);
            const auto wf = make_weights(K*M, 41);
            ggml_tensor * W = ggml_new_tensor_2d(ctx, t.type, K, M);
            quantize_tensor(t, wf, (uint8_t *) W->data, K, M, im);
            const auto wd = dequant_tensor(t, (const uint8_t *) W->data, K, M);
            ggml_tensor * X = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, K, N);
            std::mt19937 rng(42); std::uniform_real_distribution<float> U(-1.0f, 1.0f);
            for (int64_t i = 0; i < K*N; ++i) ((float *) X->data)[i] = U(rng);
            ggml_tensor * Y = ggml_mul_mat(ctx, W, X);
            CHECK(ggml_backend_supports_op(be, Y), "%s MUL_MAT N=%lld not claimed by the host backend", t.name, (long long) N);
            ggml_cgraph * gf = ggml_new_graph(ctx);
            ggml_build_forward_expand(gf, Y);
            CHECK(ggml_backend_graph_compute(be, gf) == GGML_STATUS_SUCCESS, "%s MUL_MAT N=%lld compute", t.name, (long long) N);
            std::vector<double> ref(M*N);
            for (int64_t j = 0; j < N; ++j) for (int64_t i = 0; i < M; ++i) {
                double s = 0; for (int64_t k = 0; k < K; ++k) s += (double) wd[i*K + k] * ((const float *) X->data)[j*K + k];
                ref[j*M + i] = s;
            }
            const double e = rel_rms(ref, (const float *) Y->data, M*N);
            printf("  %-9s MUL_MAT       N=%-2lld rel-rms vs double(dequant) = %.5f\n", t.name, (long long) N, e);
            CHECK(std::isfinite(e) && e < 0.02, "%s MUL_MAT N=%lld rel-rms %.5f", t.name, (long long) N, e);
            ggml_free(ctx);
        }
        // ---- MUL_MAT_ID / MOE_UP_GATE / FUSED_UP_GATE
        {
            ggml_init_params ip = { 128u*1024*1024, nullptr, false };
            ggml_context * ctx = ggml_init(ip);
            const auto wf = make_weights(K*M*E, 51);
            const auto wf2 = make_weights(K*M*E, 52);
            ggml_tensor * A  = ggml_new_tensor_3d(ctx, t.type, K, M, E);
            ggml_tensor * A2 = ggml_new_tensor_3d(ctx, t.type, K, M, E);
            quantize_tensor(t, wf,  (uint8_t *) A->data,  K, M*E, im);
            quantize_tensor(t, wf2, (uint8_t *) A2->data, K, M*E, im);
            // F32 / Q8_0 copies of the dequantised weights: the same op on them is the semantics-free reference
            ggml_tensor * Af  = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, K, M, E);
            ggml_tensor * A8  = ggml_new_tensor_3d(ctx, GGML_TYPE_Q8_0, K, M, E);
            ggml_tensor * A28 = ggml_new_tensor_3d(ctx, GGML_TYPE_Q8_0, K, M, E);
            {
                const auto d1 = dequant_tensor(t, (const uint8_t *) A->data,  K, M*E);
                const auto d2 = dequant_tensor(t, (const uint8_t *) A2->data, K, M*E);
                memcpy(Af->data, d1.data(), d1.size()*sizeof(float));
                ggml_quantize_chunk(GGML_TYPE_Q8_0, d1.data(), A8->data,  0, M*E, K, nullptr, nullptr);
                ggml_quantize_chunk(GGML_TYPE_Q8_0, d2.data(), A28->data, 0, M*E, K, nullptr, nullptr);
            }
            ggml_tensor * B   = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, K, 1, T);
            std::mt19937 rng(53); std::uniform_real_distribution<float> U(-1.0f, 1.0f);
            for (int64_t i = 0; i < K*T; ++i) ((float *) B->data)[i] = U(rng);
            ggml_tensor * ids = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, NU, T);
            for (int64_t tt = 0; tt < T; ++tt) for (int64_t u = 0; u < NU; ++u) ((int32_t *) ids->data)[tt*NU + u] = (int32_t) ((tt + 2*u) % E);

            struct Case { const char * name; ggml_tensor * y; ggml_tensor * yref; };
            std::vector<Case> cases;
            cases.push_back({ "MUL_MAT_ID",   ggml_mul_mat_id(ctx, A, B, ids), ggml_mul_mat_id(ctx, Af, B, ids) });
            cases.push_back({ "MOE_UP_GATE",  ggml_moe_up_gate(ctx, A, A2, B, ids, GGML_UNARY_OP_SILU), ggml_moe_up_gate(ctx, A8, A28, B, ids, GGML_UNARY_OP_SILU) });
            // dense FUSED_UP_GATE over expert 0's 2D slice
            ggml_tensor * U0  = ggml_view_2d(ctx, A,   K, M, A->nb[1],   0);
            ggml_tensor * G0  = ggml_view_2d(ctx, A2,  K, M, A2->nb[1],  0);
            ggml_tensor * U08 = ggml_view_2d(ctx, A8,  K, M, A8->nb[1],  0);
            ggml_tensor * G08 = ggml_view_2d(ctx, A28, K, M, A28->nb[1], 0);
            ggml_tensor * X1  = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, K, 1);
            memcpy(X1->data, B->data, K*sizeof(float));
            cases.push_back({ "FUSED_UP_GATE", ggml_fused_up_gate(ctx, U0, G0, X1, GGML_UNARY_OP_SILU), ggml_fused_up_gate(ctx, U08, G08, X1, GGML_UNARY_OP_SILU) });

            for (auto & c : cases) {
                const bool claimed = ggml_backend_supports_op(be, c.y);
                CHECK(claimed, "%s %s not claimed by the host backend", t.name, c.name);
                if (!claimed) continue;
                ggml_cgraph * gf = ggml_new_graph(ctx);
                ggml_build_forward_expand(gf, c.y);
                ggml_build_forward_expand(gf, c.yref);
                const auto st = ggml_backend_graph_compute(be, gf);
                CHECK(st == GGML_STATUS_SUCCESS, "%s %s compute status %d", t.name, c.name, (int) st);
                const size_t n = ggml_nelements(c.y);
                std::vector<double> ref(n);
                for (size_t i = 0; i < n; ++i) ref[i] = ((const float *) c.yref->data)[i];
                const double e = rel_rms(ref, (const float *) c.y->data, n);
                // tolerance: both sides quantise the activation; the Q8_0 weight copy adds ~0.5%
                printf("  %-9s %-13s rel-rms vs same-op-on-dequantised-weights = %.5f\n", t.name, c.name, e);
                CHECK(std::isfinite(e) && e < 0.03, "%s %s rel-rms %.5f", t.name, c.name, e);
            }
            ggml_free(ctx);
        }
    }
    ggml_backend_free(be);
}

// ---- REFERENCE DIFFERENTIAL (golden hashes)
// The same deterministic input as tests/probe-iq1-narrow-golden-dump.cpp (integer xorshift, no libm) was quantised by the
// UNSLOTH reference build (unslothai/llama.cpp iq1-narrow-grids-b10775 c86ed2699, ggml built standalone) on 2026-10-01 and
// the FNV-1a-64 of the quantised bytes and of the dequantised floats recorded below. Whole dumps were also diffed byte for
// byte at that time (8 files, 0 differing bytes, IQ1_S included as the control that our IQ1_S == theirs).
static uint64_t g_s = 0x9E3779B97F4A7C15ull;
static uint32_t next_u32() { g_s ^= g_s << 13; g_s ^= g_s >> 7; g_s ^= g_s << 17; return (uint32_t) (g_s >> 32); }
static float gauss_like() {
    int32_t v = 0;
    for (int i = 0; i < 4; ++i) v += (int32_t) (next_u32() >> 16);
    return (float) (v - 4*32768) * (0.12f / 131072.0f);
}
static float imat_like() { return 0.1f + (float) (next_u32() >> 16) * (1.9f / 65536.0f); }
static uint64_t fnv1a(const void * p, size_t n) {
    const uint8_t * b = (const uint8_t *) p;
    uint64_t h = 0xcbf29ce484222325ull;
    for (size_t i = 0; i < n; ++i) { h ^= b[i]; h *= 0x100000001b3ull; }
    return h;
}

static void test_golden_vs_unsloth_reference() {
    printf("[reference differential vs Unsloth build, FNV-1a-64]\n");
    const int64_t K = 1024, R = 8;
    g_s = 0x9E3779B97F4A7C15ull;
    std::vector<float> w(K*R), im(K);
    for (auto & v : w) v = gauss_like();
    for (int64_t r = 0; r < R; r += 3) for (int64_t c = 0; c < K; c += 97) w[r*K + c] *= 6.0f;
    for (int64_t c = 0; c < K; ++c) w[(R-1)*K + c] = 0.0f;
    for (auto & v : im) v = imat_like();
    struct G { ggml_type t; const char * n; uint64_t q; uint64_t f; };
    const G gold[] = {
        { GGML_TYPE_IQ1_S,    "iq1_s (control)", 0x39829d86b2cb46cfull, 0xe6ff7cc849d5a2ceull },
        { GGML_TYPE_IQ1_XS,   "iq1_xs",          0x103d829b36d6b631ull, 0xe30beb453124be39ull },
        { GGML_TYPE_IQ1_XXS,  "iq1_xxs",         0xae9bab9f7609d14eull, 0x4eea74b26a987153ull },
        { GGML_TYPE_IQ1_XXXS, "iq1_xxxs",        0xca423dffebc9571eull, 0x1a159ffb0f6b677aull },
    };
    for (const auto & g : gold) {
        ggml_quantize_init(g.t);
        const size_t rs = ggml_row_size(g.t, K);
        std::vector<uint8_t> q(rs*R, 0);
        ggml_quantize_chunk(g.t, w.data(), q.data(), 0, R, K, im.data(), nullptr);
        std::vector<float> dq(K*R);
        for (int64_t r = 0; r < R; ++r) ggml_get_type_traits(g.t)->to_float(q.data() + r*rs, dq.data() + r*K, K);
        const uint64_t hq = fnv1a(q.data(), q.size()), hf = fnv1a(dq.data(), dq.size()*4);
        printf("  %-16s q_fnv=%016llx %s  f32_fnv=%016llx %s\n", g.n, (unsigned long long) hq, hq == g.q ? "== reference" : "DIFFERS",
               (unsigned long long) hf, hf == g.f ? "== reference" : "DIFFERS");
        CHECK(hq == g.q, "%s quantised bytes differ from the Unsloth reference", g.n);
        CHECK(hf == g.f, "%s dequantised floats differ from the Unsloth reference", g.n);
    }
}

int main() {
    setvbuf(stdout, nullptr, _IONBF, 0);   // if a kernel crashes, the lines that already ran must survive
    ggml_cpu_init();
    for (const auto & t : k_types) ggml_quantize_init(t.type);
    test_golden_vs_unsloth_reference();
    test_layout();
    test_encode_decode();
    test_dot_agrees_with_dequant();
    test_dense_dot();
    test_graph_ops();
    ggml_quantize_free();
    if (g_fail) { printf("\nFAILED: %d check(s)\n", g_fail); return 1; }
    printf("\nall IQ1 narrow-grid checks passed\n");
    return 0;
}

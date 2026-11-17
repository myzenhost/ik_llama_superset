// NVFP4_KS (ik_llama.cpp 30903876b, ggml type 168): 64-wide NVFP4 blocks (UE4M3 scale per 16, E2M1 nibbles) plus ONE fp32
// scale at the head of every ROW. Also the opt-in load path that turns a mainline NVFP4 GGUF (type 40 + per-tensor
// `.scale`) into NVFP4_KS tensors (LLAMA_NVFP4_IMPORT_KS=1; OFF by default in this fork, which runs type 40 natively).
//
// Part A (ggml only, CPU):
//   A1 type traits and row geometry;       A2 quantize -> dequantize round trip, and against the fork's own type 40;
//   A3 the scale-fold identity the loader relies on: dequant(KS row {s, blocks}) == s * dequant(type 40 blocks);
//   A4 ggml_mul_mat on the CPU backend (iqk kernels: 1..8 columns, and >= 32 columns through the Q8_0_R8/R16 repack)
//      against a double-precision reference built from the dequantized weights.
// Part B (loader): a synthetic llama-arch GGUF with a type-40 weight, an expert-stacked type-40 weight (one scale per expert)
//   and `.scale` / `.input_scale` tensors.
//   B1 default (env unset): the loader leaves type 40 and the scale tensors alone (the fork's native path is unchanged);
//   B2 LLAMA_NVFP4_IMPORT_KS=1: weights become NVFP4_KS, `.scale` and `.input_scale` are consumed (both `weights` and
//      `weights_map`), and load_data_for() yields rows that decode to scale * (type 40 decode).
// SOURCE-READ: ik_llama.cpp 30903876b (MIT): ggml/src/iqk/iqk_quantize.cpp, iqk_gemm_legacy_quants.cpp, src/llama-model-loader.cpp.
#include "../src/llama-model-loader.h"

#include "ggml.h"
#include "gguf.h"
#include "ggml-cpu.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

static int g_fail = 0;
#define CHECK(cond, ...) do { if (!(cond)) { g_fail++; printf("FAIL  %s:%d  %s  --  ", __FILE__, __LINE__, #cond); printf(__VA_ARGS__); printf("\n"); } else { printf("ok    %s\n", #cond); } } while (0)

static uint64_t g_rng = 0x9E3779B97F4A7C15ull;
static uint32_t rnd() { g_rng = g_rng*6364136223846793005ull + 1442695040888963407ull; return (uint32_t)(g_rng >> 33); }
static float    frnd() { return (rnd() & 0xFFFFFF)/(float)0x1000000; }
static float    gauss() { float u = std::max(frnd(), 1e-7f), v = frnd(); return std::sqrt(-2.f*std::log(u))*std::cos(6.2831853f*v); }

static void set_env(const char * k, const char * v) {
#ifdef _WIN32
    _putenv_s(k, v ? v : "");
#else
    if (v) setenv(k, v, 1); else unsetenv(k);
#endif
}

static std::vector<float> dequant(ggml_type t, const void * row, int64_t n) {
    std::vector<float> out(n);
    auto tr = ggml_get_type_traits(t);
    tr->to_float(row, out.data(), n);
    return out;
}

// ---------------------------------------------------------------------------------------------------------------------------
static void part_a() {
    // A1
    CHECK(strcmp(ggml_type_name(GGML_TYPE_NVFP4_KS), "nvfp4_ks") == 0, "type name");
    CHECK(ggml_blck_size(GGML_TYPE_NVFP4_KS) == 64, "block size");
    CHECK(ggml_is_quantized(GGML_TYPE_NVFP4_KS), "is_quantized");
    CHECK(ggml_row_size(GGML_TYPE_NVFP4_KS, 256) == 4 + 4*36, "row size = fp32 row scale + 4 blocks of 36 B, got %zu", ggml_row_size(GGML_TYPE_NVFP4_KS, 256));
    CHECK((int) GGML_TYPE_NVFP4_KS == 168 && (int) GGML_FTYPE_MOSTLY_NVFP4_KS == 156, "ik ids kept (168 / 156)");
    CHECK((int) GGML_TYPE_NVFP4 == 40, "the fork's mainline NVFP4 keeps id 40");

    // A2 / A3
    const int64_t k = 256, nrows = 8;
    std::vector<float> x(k*nrows);
    for (auto & v : x) v = gauss()*0.05f;
    const size_t rs = ggml_row_size(GGML_TYPE_NVFP4_KS, k);
    std::vector<uint8_t> q(rs*nrows);
    size_t wrote = ggml_quantize_chunk(GGML_TYPE_NVFP4_KS, x.data(), q.data(), 0, nrows, k, nullptr);
    CHECK(wrote == rs*nrows, "quantize_chunk wrote %zu, want %zu", wrote, rs*nrows);

    // the fork's own type 40 on the same data
    const size_t rs40 = ggml_row_size(GGML_TYPE_NVFP4, k);
    std::vector<uint8_t> q40(rs40*nrows);
    ggml_quantize_chunk(GGML_TYPE_NVFP4, x.data(), q40.data(), 0, nrows, k, nullptr);

    double e_ks = 0, e_40 = 0, nrm = 0;
    for (int64_t r = 0; r < nrows; ++r) {
        auto d = dequant(GGML_TYPE_NVFP4_KS, q.data() + r*rs, k);
        auto d40 = dequant(GGML_TYPE_NVFP4, q40.data() + r*rs40, k);
        for (int64_t i = 0; i < k; ++i) {
            const double xv = x[r*k + i];
            e_ks += (d[i] - xv)*(d[i] - xv);
            e_40 += (d40[i] - xv)*(d40[i] - xv);
            nrm += xv*xv;
        }
    }
    const double rel_ks = std::sqrt(e_ks/nrm), rel_40 = std::sqrt(e_40/nrm);
    printf("info  round trip rel RMSE: nvfp4_ks %.4f, nvfp4 (type 40, fork) %.4f\n", rel_ks, rel_40);
    CHECK(rel_ks < 0.14, "nvfp4_ks round trip rel RMSE %.4f", rel_ks);
    CHECK(rel_ks < 1.5*rel_40 + 0.01, "nvfp4_ks within 1.5x of the type 40 codec (%.4f vs %.4f)", rel_ks, rel_40);

    // a zero row: scale 0, every value 0, no NaN
    {
        std::vector<float> z(k, 0.f);
        std::vector<uint8_t> zq(rs);
        ggml_quantize_chunk(GGML_TYPE_NVFP4_KS, z.data(), zq.data(), 0, 1, k, nullptr);
        auto d = dequant(GGML_TYPE_NVFP4_KS, zq.data(), k);
        bool ok = true; for (float v : d) ok = ok && v == 0.f;
        CHECK(ok, "zero row decodes to zeros");
    }

    // A3: scale fold identity
    {
        double worst = 0;
        for (int64_t r = 0; r < nrows; ++r) {
            const float s = 0.37f + r;
            std::vector<uint8_t> ks(rs);
            memcpy(ks.data(), &s, sizeof(float));
            memcpy(ks.data() + 4, q40.data() + r*rs40, rs40);
            auto a = dequant(GGML_TYPE_NVFP4_KS, ks.data(), k);
            auto b = dequant(GGML_TYPE_NVFP4, q40.data() + r*rs40, k);
            for (int64_t i = 0; i < k; ++i) {
                worst = std::max(worst, (double) std::fabs(a[i] - s*b[i])/(std::fabs(s*b[i]) + 1e-12));
            }
        }
        CHECK(worst < 1e-5, "KS row {s, type-40 blocks} == s * type-40 decode, worst rel diff %.3g", worst);
    }

    // A4: mul_mat on the CPU backend vs a double reference
    // gate b1112a round 4: test-backend-ops MUL_MAT(nvfp4_ks, m=1, n=64, k=256) FAILED on CUDA0 and the run died with heap
    // corruption (0xC0000374). Cover single-row / non-multiple-of-16 / multiple-of-16 weights x repack (>= 32 columns) x 1 and 4 threads.
    for (int64_t mkn : { 1, 16, 24, 64 }) for (int64_t kk : { 256, 512 }) for (int nth : { 1, 4 }) for (int64_t n : { 1, 2, 3, 8, 16, 31, 32, 33, 64 }) {
        const int64_t m = mkn;
        std::vector<float> w(kk*m);
        for (auto & v : w) v = gauss()*0.1f;
        const size_t rsk = ggml_row_size(GGML_TYPE_NVFP4_KS, kk);
        std::vector<uint8_t> wq(rsk*m);
        ggml_quantize_chunk(GGML_TYPE_NVFP4_KS, w.data(), wq.data(), 0, m, kk, nullptr);
        std::vector<float> y(kk*n);
        for (auto & v : y) v = gauss();

        ggml_init_params ip = { 64*1024*1024, nullptr, false };
        ggml_context * ctx = ggml_init(ip);
        ggml_tensor * W = ggml_new_tensor_2d(ctx, GGML_TYPE_NVFP4_KS, kk, m);
        ggml_tensor * Y = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, kk, n);
        memcpy(W->data, wq.data(), wq.size());
        memcpy(Y->data, y.data(), y.size()*sizeof(float));
        ggml_tensor * R = ggml_mul_mat(ctx, W, Y);
        ggml_cgraph * gf = ggml_new_graph(ctx);
        ggml_build_forward_expand(gf, R);
        ggml_graph_compute_with_ctx(ctx, gf, nth);

        double worst = 0;
        for (int64_t j = 0; j < n; ++j) {
            for (int64_t i = 0; i < m; ++i) {
                auto d = dequant(GGML_TYPE_NVFP4_KS, wq.data() + i*rsk, kk);
                double ref = 0, nw = 0, ny = 0;
                for (int64_t t = 0; t < kk; ++t) { ref += (double) d[t]*y[j*kk + t]; nw += (double) d[t]*d[t]; ny += (double) y[j*kk + t]*y[j*kk + t]; }
                const double got = ((const float *) R->data)[j*m + i];
                worst = std::max(worst, std::fabs(got - ref)/(std::sqrt(nw*ny) + 1e-12));
            }
        }
        ggml_free(ctx);
        printf("info  mul_mat m=%lld k=%lld nth=%d n=%lld worst |got-ref|/(|w||y|) = %.4g\n", (long long) m, (long long) kk, nth, (long long) n, worst);
        CHECK(worst < 0.03, "mul_mat nvfp4_ks x f32, m=%lld k=%lld nth=%d n=%lld: %.4g", (long long) m, (long long) kk, nth, (long long) n, worst);
    }

    // A5: CPU graph ops on a row-scaled tensor (gate b1112a, test-backend-ops CPY / GET_ROWS):
    //   A5a contiguous same-type ggml_cpy must carry the per-row fp32 scales (dup_same_cont copied blocks only);
    //   A5b ggml_get_rows(nvfp4_ks) -> f32 must not abort (missing case in the get_rows switch) and must equal the row decode;
    //   A5c ggml_add(nvfp4_ks, f32) -> nvfp4_ks (add_q_f32) must not abort and must be close to decode + y.
    for (int64_t kk : { 64, 128, 192, 256 }) {
        const int64_t m = 24;
        std::vector<float> w(kk*m);
        for (auto & v : w) v = gauss()*0.1f;
        const size_t rsk = ggml_row_size(GGML_TYPE_NVFP4_KS, kk);
        std::vector<uint8_t> wq(rsk*m);
        ggml_quantize_chunk(GGML_TYPE_NVFP4_KS, w.data(), wq.data(), 0, m, kk, nullptr);

        for (int nth : { 1, 4 }) {
            ggml_init_params ip = { 64*1024*1024, nullptr, false };
            ggml_context * ctx = ggml_init(ip);
            ggml_tensor * W   = ggml_new_tensor_2d(ctx, GGML_TYPE_NVFP4_KS, kk, m);
            ggml_tensor * Dst = ggml_new_tensor_2d(ctx, GGML_TYPE_NVFP4_KS, kk, m);
            memcpy(W->data, wq.data(), wq.size());
            memset(Dst->data, 0xA5, wq.size());
            ggml_tensor * Cp = ggml_cpy(ctx, W, Dst);
            ggml_tensor * Idx = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 5);
            const int32_t rows[5] = { 3, 0, 23, 7, 7 };
            memcpy(Idx->data, rows, sizeof(rows));
            ggml_tensor * Gr = ggml_get_rows(ctx, W, Idx);
            ggml_tensor * Y  = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, kk, m);
            std::vector<float> yv(kk*m);
            for (auto & v : yv) v = gauss()*0.05f;
            memcpy(Y->data, yv.data(), yv.size()*sizeof(float));
            ggml_tensor * Ad = ggml_add(ctx, W, Y);
            ggml_cgraph * gf = ggml_new_graph(ctx);
            ggml_build_forward_expand(gf, Cp);
            ggml_build_forward_expand(gf, Gr);
            ggml_build_forward_expand(gf, Ad);
            ggml_graph_compute_with_ctx(ctx, gf, nth);

            CHECK(memcmp(Cp->data, wq.data(), wq.size()) == 0, "A5a cpy nvfp4_ks -> nvfp4_ks bytes identical, k=%lld nth=%d", (long long) kk, nth);

            double gworst = 0;
            for (int j = 0; j < 5; ++j) {
                auto d = dequant(GGML_TYPE_NVFP4_KS, wq.data() + rows[j]*rsk, kk);
                for (int64_t i = 0; i < kk; ++i) gworst = std::max(gworst, (double) std::fabs(((const float *) Gr->data)[j*kk + i] - d[i]));
            }
            CHECK(gworst == 0.0, "A5b get_rows nvfp4_ks == row decode, k=%lld nth=%d worst %.3g", (long long) kk, nth, gworst);

            CHECK(Ad->type == GGML_TYPE_NVFP4_KS, "A5c add result type");
            double e = 0, nrm = 0;
            for (int64_t r = 0; r < m; ++r) {
                auto d  = dequant(GGML_TYPE_NVFP4_KS, wq.data() + r*rsk, kk);
                auto a  = dequant(GGML_TYPE_NVFP4_KS, (const uint8_t *) Ad->data + r*rsk, kk);
                for (int64_t i = 0; i < kk; ++i) { const double ref = d[i] + yv[r*kk + i]; e += (a[i] - ref)*(a[i] - ref); nrm += ref*ref; }
            }
            CHECK(std::sqrt(e/nrm) < 0.16, "A5c add nvfp4_ks + f32 within quantization error, k=%lld nth=%d rel %.4f", (long long) kk, nth, std::sqrt(e/nrm));
            ggml_free(ctx);
        }
    }
}

// ---------------------------------------------------------------------------------------------------------------------------
static const char * kPath = "test-nvfp4-ks.tmp.gguf";
static const int64_t NE0 = 128, NE1 = 4, NEXP = 3;
static const float   kScale   = 0.37f;
static const float   kEx[NEXP] = { 0.25f, 0.5f, 2.0f };

static void fill_nvfp4(uint8_t * p, size_t nrows, int64_t ne0) {
    const size_t rs = ggml_row_size(GGML_TYPE_NVFP4, ne0);
    for (size_t r = 0; r < nrows; ++r) {
        uint8_t * row = p + r*rs;
        for (int64_t b = 0; b < ne0/64; ++b) {
            uint8_t * blk = row + b*36;
            for (int s = 0; s < 4; ++s) blk[s] = 0x28 + (rnd() % 0x20);  // finite UE4M3 scales, never 0x7F
            for (int j = 0; j < 32; ++j) blk[4 + j] = rnd() & 0xFF;
        }
    }
}

static void write_gguf() {
    ggml_init_params ip = { 4*1024*1024, nullptr, false };
    ggml_context * ctx = ggml_init(ip);
    gguf_context * g = gguf_init_empty();
    gguf_set_val_str(g, "general.architecture", "llama");

    ggml_tensor * a = ggml_new_tensor_2d(ctx, GGML_TYPE_NVFP4, NE0, NE1);
    ggml_set_name(a, "blk.0.ffn_down.weight");
    fill_nvfp4((uint8_t *) a->data, NE1, NE0);
    gguf_add_tensor(g, a);

    ggml_tensor * as = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1);
    ggml_set_name(as, "blk.0.ffn_down.scale");
    ((float *) as->data)[0] = kScale;
    gguf_add_tensor(g, as);

    ggml_tensor * ai = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1);
    ggml_set_name(ai, "blk.0.ffn_down.input_scale");
    ((float *) ai->data)[0] = 5.0f;
    gguf_add_tensor(g, ai);

    ggml_tensor * e = ggml_new_tensor_3d(ctx, GGML_TYPE_NVFP4, NE0, NE1, NEXP);
    ggml_set_name(e, "blk.0.ffn_down_exps.weight");
    fill_nvfp4((uint8_t *) e->data, NE1*NEXP, NE0);
    gguf_add_tensor(g, e);

    ggml_tensor * es = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, NEXP);
    ggml_set_name(es, "blk.0.ffn_down_exps.scale");
    for (int i = 0; i < NEXP; ++i) ((float *) es->data)[i] = kEx[i];
    gguf_add_tensor(g, es);

    gguf_write_to_file(g, kPath, false);
    gguf_free(g);
    ggml_free(ctx);
}

static std::vector<uint8_t> read_tensor_bytes(const char * name, size_t nbytes) {
    gguf_init_params gp = { /*no_alloc*/ true, nullptr };
    gguf_context * g = gguf_init_from_file(kPath, gp);
    const int64_t id = gguf_find_tensor(g, name);
    const size_t off = gguf_get_data_offset(g) + gguf_get_tensor_offset(g, id);
    gguf_free(g);
    std::vector<uint8_t> out(nbytes);
    FILE * f = fopen(kPath, "rb");
    fseek(f, (long) off, SEEK_SET);
    size_t got = fread(out.data(), 1, nbytes, f);
    fclose(f);
    if (got != nbytes) out.clear();
    return out;
}

static void part_b() {
    write_gguf();
    const size_t rs40 = ggml_row_size(GGML_TYPE_NVFP4, NE0);
    const auto raw_a = read_tensor_bytes("blk.0.ffn_down.weight", rs40*NE1);
    const auto raw_e = read_tensor_bytes("blk.0.ffn_down_exps.weight", rs40*NE1*NEXP);
    CHECK(raw_a.size() == rs40*NE1 && raw_e.size() == rs40*NE1*NEXP, "read the file's tensor bytes back");

    // B1: default, conversion OFF
    set_env("LLAMA_NVFP4_IMPORT_KS", nullptr);
    {
        llama_model_loader ml(kPath, 0, /*use_mmap*/ true, false, false, false, false, false, false, nullptr, nullptr);
        const auto * w = ml.get_weight("blk.0.ffn_down.weight");
        CHECK(w && w->tensor->type == GGML_TYPE_NVFP4, "default: weight stays type 40");
        CHECK(w && w->nvfp4_scale.empty(), "default: no import scale attached");
        CHECK(ml.get_weight("blk.0.ffn_down.scale") != nullptr, "default: .scale tensor still present");
        CHECK(ml.get_weight("blk.0.ffn_down.input_scale") != nullptr, "default: .input_scale tensor still present");
        CHECK(ml.n_tensors == 5, "default: 5 tensors, got %d", (int) ml.n_tensors);
    }

    // B2: conversion ON
    set_env("LLAMA_NVFP4_IMPORT_KS", "1");
    {
        llama_model_loader ml(kPath, 0, /*use_mmap*/ true, false, false, false, false, false, false, nullptr, nullptr);
        const auto * w  = ml.get_weight("blk.0.ffn_down.weight");
        const auto * we = ml.get_weight("blk.0.ffn_down_exps.weight");
        CHECK(w && w->tensor->type == GGML_TYPE_NVFP4_KS, "import: weight is NVFP4_KS");
        CHECK(we && we->tensor->type == GGML_TYPE_NVFP4_KS, "import: expert weight is NVFP4_KS");
        CHECK(w && w->nvfp4_scale.size() == 1 && std::fabs(w->nvfp4_scale[0] - kScale) < 1e-7f, "import: dense scale folded");
        CHECK(we && we->nvfp4_scale.size() == NEXP && we->nvfp4_scale[0] == kEx[0] && we->nvfp4_scale[2] == kEx[2], "import: one scale per expert");
        CHECK(ml.get_weight("blk.0.ffn_down.scale") == nullptr, "import: .scale consumed (weights_map)");
        CHECK(ml.get_weight("blk.0.ffn_down.input_scale") == nullptr, "import: .input_scale dropped (weights_map)");
        CHECK(ml.get_weight("blk.0.ffn_down_exps.scale") == nullptr, "import: expert .scale consumed");
        CHECK(ml.n_tensors == 2, "import: 2 tensors left, got %d", (int) ml.n_tensors);
        bool vec_has_scale = false;
        for (const auto & x : ml.weights) { vec_has_scale = vec_has_scale || strstr(x.tensor->name, ".scale") != nullptr; }
        CHECK(!vec_has_scale, "import: scale tensors are gone from `weights` too");

        // the data path: rows come out as {fp32 scale, blocks} and decode to scale * type-40 decode
        auto check_tensor = [&](const char * name, int64_t ne2, const std::vector<uint8_t> & raw, const float * scales) {
            ggml_init_params ip = { 1*1024*1024, nullptr, false };
            ggml_context * c2 = ggml_init(ip);
            ggml_tensor * cur = ne2 > 1 ? ggml_new_tensor_3d(c2, GGML_TYPE_NVFP4_KS, NE0, NE1, ne2) : ggml_new_tensor_2d(c2, GGML_TYPE_NVFP4_KS, NE0, NE1);
            ggml_set_name(cur, name);
            ml.load_data_for(cur);
            double worst = 0;
            const size_t rsks = ggml_row_size(GGML_TYPE_NVFP4_KS, NE0);
            for (int64_t r = 0; r < NE1*ne2; ++r) {
                const float s = scales[ne2 > 1 ? r/NE1 : 0];
                auto a = dequant(GGML_TYPE_NVFP4_KS, (const uint8_t *) cur->data + r*rsks, NE0);
                auto b = dequant(GGML_TYPE_NVFP4, raw.data() + r*rs40, NE0);
                for (int64_t i = 0; i < NE0; ++i) {
                    worst = std::max(worst, (double) std::fabs(a[i] - s*b[i])/(std::fabs(s*b[i]) + 1e-12));
                }
            }
            ggml_free(c2);
            CHECK(worst < 1e-5, "%s: loaded rows decode to scale x type-40 decode, worst %.3g", name, worst);
        };
        const float one[1] = { kScale };
        check_tensor("blk.0.ffn_down.weight", 1, raw_a, one);
        check_tensor("blk.0.ffn_down_exps.weight", NEXP, raw_e, kEx);
    }
    // B3: activation precision. The checkpoint's .input_scale is 5.0; the expert tensor has none.
    const int precs[3]       = { LLAMA_NVFP4_PRECISION_AUTO, LLAMA_NVFP4_PRECISION_INT8, LLAMA_NVFP4_PRECISION_W4A4 };
    const float want_dense[3] = { 5.0f, 0.0f, -1.0f };
    const float want_exps[3]  = { 0.0f, 0.0f, -1.0f };
    for (int k = 0; k < 3; ++k) {
        llama_model_loader ml(kPath, 0, /*use_mmap*/ true, false, false, false, false, false, false, nullptr, nullptr,
                std::vector<std::string>{}, nullptr, precs[k]);
        const auto * w  = ml.get_weight("blk.0.ffn_down.weight");
        const auto * we = ml.get_weight("blk.0.ffn_down_exps.weight");
        CHECK(w && ggml_get_input_scale(w->tensor) == want_dense[k], "precision %d: dense input scale %g, want %g",
                precs[k], w ? ggml_get_input_scale(w->tensor) : -99.0, want_dense[k]);
        CHECK(we && ggml_get_input_scale(we->tensor) == want_exps[k], "precision %d: expert input scale %g, want %g",
                precs[k], we ? ggml_get_input_scale(we->tensor) : -99.0, want_exps[k]);
    }
    set_env("LLAMA_NVFP4_IMPORT_KS", nullptr);
    remove(kPath);
}

// Part C (ik_llama b71fe5368): per-tensor activation (input) scale carried in the last op_params slot of NVFP4_KS tensors.
static void part_c() {
    ggml_init_params ip = { 16u*1024*1024, nullptr, true };
    ggml_context * ctx = ggml_init(ip);
    ggml_tensor * w = ggml_new_tensor_2d(ctx, GGML_TYPE_NVFP4_KS, 128, 4);
    CHECK(ggml_get_input_scale(w) == 0.0f, "fresh tensor has no input scale");
    ggml_set_input_scale(w, 0.375f);
    CHECK(ggml_get_input_scale(w) == 0.375f, "set/get round trip");
    ggml_tensor * v = ggml_view_2d(ctx, w, 128, 2, w->nb[1], 0);
    CHECK(ggml_get_input_scale(v) == 0.375f, "view inherits scale");
    ggml_tensor * d = ggml_dup_tensor(ctx, w);
    CHECK(ggml_get_input_scale(d) == 0.375f, "dup_tensor copies scale");
    ggml_tensor * f = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 128, 4);
    ggml_tensor * fv = ggml_view_2d(ctx, f, 128, 2, f->nb[1], 0);
    CHECK(ggml_get_input_scale(fv) == 0.0f, "non-KS view carries no scale");
    ggml_free(ctx);
}

// Part D (ik_llama c7ea52b33): MXFP4 row -> NVFP4_KS row repack. Exact when every nonzero block's E8M0 exponent is within the UE4M3 range.
static void part_d() {
    const int64_t n = 256;                                  // 8 MXFP4 blocks = 4 NVFP4 blocks
    const size_t rs_m = ggml_row_size(GGML_TYPE_MXFP4, n), rs_k = ggml_row_size(GGML_TYPE_NVFP4_KS, n);
    auto tm = ggml_get_type_traits(GGML_TYPE_MXFP4);
    std::vector<float> x(n);
    std::vector<uint8_t> m(rs_m), k(rs_k);
    // D1: per-block magnitudes spread over 2^0..2^7 (well inside the range): exact, dequantization identical.
    for (int64_t i = 0; i < n; ++i) x[i] = gauss()*std::ldexp(1.0f, (int)((i/32)%8));
    tm->from_float_ref(x.data(), m.data(), n);
    bool ex = ggml_mxfp4_to_nvfp4_ks(m.data(), k.data(), (int)n);
    std::vector<float> a = dequant(GGML_TYPE_MXFP4, m.data(), n), b = dequant(GGML_TYPE_NVFP4_KS, k.data(), n);
    double md = 0;
    for (int64_t i = 0; i < n; ++i) md = std::max(md, (double)std::fabs(a[i] - b[i]));
    CHECK(ex, "D1: in-range row reports exact");
    CHECK(md == 0.0, "D1: dequantized values identical to the MXFP4 source");
    // D2: an all-zero row.
    std::fill(x.begin(), x.end(), 0.0f);
    tm->from_float_ref(x.data(), m.data(), n);
    ex = ggml_mxfp4_to_nvfp4_ks(m.data(), k.data(), (int)n);
    b = dequant(GGML_TYPE_NVFP4_KS, k.data(), n);
    md = 0;
    for (int64_t i = 0; i < n; ++i) md = std::max(md, (double)std::fabs(b[i]));
    CHECK(ex && md == 0.0, "D2: zero row exact and zero");
    // D3: one block 2^-40 below the others (outside UE4M3 range): reported inexact, no crash, large blocks still exact.
    for (int64_t i = 0; i < n; ++i) x[i] = gauss()*((i/32) == 3 ? std::ldexp(1.0f, -40) : 1.0f);
    tm->from_float_ref(x.data(), m.data(), n);
    ex = ggml_mxfp4_to_nvfp4_ks(m.data(), k.data(), (int)n);
    a = dequant(GGML_TYPE_MXFP4, m.data(), n); b = dequant(GGML_TYPE_NVFP4_KS, k.data(), n);
    md = 0;
    for (int64_t i = 0; i < n; ++i) if ((i/32) != 3) md = std::max(md, (double)std::fabs(a[i] - b[i]));
    CHECK(!ex, "D3: out-of-range block reports inexact");
    CHECK(md == 0.0, "D3: in-range blocks still exact");
}

int main() {
    setvbuf(stdout, NULL, _IONBF, 0);
    ggml_cpu_init();
    part_a();
    part_c();
    part_d();
    part_b();
    printf("%s (%d failure%s)\n", g_fail ? "FAILED" : "PASSED", g_fail, g_fail == 1 ? "" : "s");
    return g_fail ? 1 : 0;
}

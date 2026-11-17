// DeepSeek-V4 dense-mask removal (ik #2779 805d1f61c, #2801 bb272bf56): CPU checks of the ops the chain added.
//  C. ggml_indexer_mask_to_index(mask, topk, n) (#2801): the ascending top-k columns whose mask entry is 0 -- against
//     a host list and against ggml_mask_to_index(ggml_indexer_mask(mask, topk)).
//
// WHAT IT CHECKS (CPU backend, host references built from the same input arrays):
//  A. ggml_visible_mask(n_visible, n_kv, F16|F32): row i is 0 in its first n_visible[i] columns, -inf after.
//  B. ggml_mask_to_index_topk(mask, topk, n_visible, n): row i = ascending zero columns of mask row i, then
//     ne00 + c for every top-k column c < n_visible[i] (ascending), -1 padded -- against a host list AND against the
//     dense route it replaces, ggml_mask_to_index(concat(mask, ggml_indexer_mask(visible_mask(n_visible), topk))).
//  D. FLASH_ATTN_EXT with NO mask and src[5] = that index list (the graph #2779 builds) against FLASH_ATTN_EXT with the
//     equivalent dense mask, on the iqk indexed kernel and on the generic kernel, D=512 with V aliasing K (DSV4 layout)
//     and D=128 with separate V. The generic kernel is selected per node through the existing n_swa slot value
//     GGML_FLASH_ATTN_EXT_IQK_DISABLED (DISABLE-REVERSIBLE: per-node graph flag, this comparison only; the iqk node
//     beside it runs the default path).
// test-backend-ops compares a backend against the CPU backend, so on a CPU-only build it cannot see a wrong CPU kernel;
// this is the independent check. On the base commit (before the chain) A and B do not compile (the ops do not exist)
// and D attends every K row (the CPU kernels ignored src[5] without a mask).
// CONTROLS: the dense-mask FA must differ from an unmasked FA (else D could pass with the index list ignored), and the
// list comparison must flag a deliberately corrupted copy. Exit 0 pass, 1 mismatch, 2 setup/control failure.
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <random>
#include <vector>

static const int64_t KV_RAW  = 256;  // raw (window) columns of the mask
static const int64_t N_EXTRA = 256;  // compressed rows behind the raw rows
static const int64_t NTOPK   = 64;
static const int64_t N_IDX   = 256;  // max_row_size of the index list
static const int64_t NQ      = 32;
static const int64_t NROWS   = GGML_PAD(NQ, GGML_KQ_MASK_PAD);

struct inputs {
    std::vector<ggml_fp16_t> mask;   // [KV_RAW, NROWS]
    std::vector<int32_t>     topk;   // [NTOPK, NROWS]
    std::vector<int32_t>     nvis;   // [1, NROWS]
};

static inputs make_inputs(uint32_t seed) {
    std::mt19937 rng(seed);
    inputs in;
    in.mask.resize(KV_RAW*NROWS);
    in.topk.resize(NTOPK*NROWS);
    in.nvis.resize(NROWS);
    const ggml_fp16_t h0 = ggml_fp32_to_fp16(0.0f), hinf = ggml_fp32_to_fp16(-INFINITY);
    for (int64_t r = 0; r < NROWS; ++r) {
        for (int64_t j = 0; j < KV_RAW; ++j) {
            in.mask[r*KV_RAW + j] = (rng() % 10) < 3 ? h0 : hinf;
        }
        std::vector<int32_t> perm(N_EXTRA);
        for (int64_t j = 0; j < N_EXTRA; ++j) perm[j] = (int32_t) j;
        std::shuffle(perm.begin(), perm.end(), rng);
        for (int64_t j = 0; j < NTOPK; ++j) in.topk[r*NTOPK + j] = perm[j];
        in.nvis[r] = (int32_t) (rng() % (N_EXTRA + 1));
    }
    in.nvis[0] = 0;                    // no compressed row visible
    in.nvis[1] = (int32_t) N_EXTRA;    // all visible
    return in;
}

static std::vector<int32_t> host_topk_list(const inputs & in, int64_t r) {
    std::vector<int32_t> y(N_IDX, -1);
    int64_t idx = 0;
    for (int64_t j = 0; j < KV_RAW && idx < N_IDX; ++j) {
        if (ggml_fp16_to_fp32(in.mask[r*KV_RAW + j]) == 0.0f) y[idx++] = (int32_t) j;
    }
    std::vector<int32_t> c;
    for (int64_t j = 0; j < NTOPK; ++j) {
        if (in.topk[r*NTOPK + j] < in.nvis[r]) c.push_back(in.topk[r*NTOPK + j]);
    }
    std::sort(c.begin(), c.end());
    for (int32_t v : c) {
        if (idx < N_IDX) y[idx++] = (int32_t) (KV_RAW + v);
    }
    return y;
}

static std::vector<int32_t> host_indexer_list(const inputs & in, int64_t r, int64_t n) {
    std::vector<int32_t> c;
    for (int64_t j = 0; j < NTOPK; ++j) {
        const int32_t t = in.topk[r*NTOPK + j];
        if (ggml_fp16_to_fp32(in.mask[r*KV_RAW + t]) == 0.0f) c.push_back(t);
    }
    std::sort(c.begin(), c.end());
    std::vector<int32_t> y(n, -1);
    for (size_t j = 0; j < c.size() && (int64_t) j < n; ++j) y[j] = c[j];
    return y;
}

static int count_list_mismatch(const std::vector<int32_t> & got, const std::vector<int32_t> & ref) {
    int bad = 0;
    for (size_t i = 0; i < ref.size(); ++i) bad += got[i] != ref[i];
    return bad;
}

struct graph_run {
    ggml_context * ctx = nullptr;
    ggml_backend_buffer_t buf = nullptr;
    ~graph_run() {
        if (buf) ggml_backend_buffer_free(buf);
        if (ctx) ggml_free(ctx);
    }
};

static ggml_context * new_ctx() {
    ggml_init_params ip = { 64u * 1024 * 1024, nullptr, true };
    return ggml_init(ip);
}

static bool alloc(ggml_backend_t be, graph_run & g) {
    g.buf = ggml_backend_alloc_ctx_tensors(g.ctx, be);
    return g.buf != nullptr;
}

// ---- A
static int check_visible_mask(ggml_backend_t be, const inputs & in, ggml_type type) {
    graph_run g; g.ctx = new_ctx();
    ggml_tensor * nv  = ggml_new_tensor_2d(g.ctx, GGML_TYPE_I32, 1, NROWS);
    ggml_tensor * out = ggml_visible_mask(g.ctx, nv, N_EXTRA, type);
    ggml_cgraph * gf = ggml_new_graph(g.ctx);
    ggml_build_forward_expand(gf, out);
    if (!alloc(be, g)) { printf("[A] SETUP FAILED: alloc\n"); return 2; }
    ggml_backend_tensor_set(nv, in.nvis.data(), 0, ggml_nbytes(nv));
    if (ggml_backend_graph_compute(be, gf) != GGML_STATUS_SUCCESS) { printf("[A] FAIL: compute\n"); return 1; }
    int bad = 0;
    std::vector<float> got(N_EXTRA*NROWS);
    if (type == GGML_TYPE_F16) {
        std::vector<ggml_fp16_t> h(N_EXTRA*NROWS);
        ggml_backend_tensor_get(out, h.data(), 0, ggml_nbytes(out));
        for (size_t i = 0; i < h.size(); ++i) got[i] = ggml_fp16_to_fp32(h[i]);
    } else {
        ggml_backend_tensor_get(out, got.data(), 0, ggml_nbytes(out));
    }
    for (int64_t r = 0; r < NROWS; ++r) {
        for (int64_t j = 0; j < N_EXTRA; ++j) {
            const float want = j < in.nvis[r] ? 0.0f : -INFINITY;
            bad += got[r*N_EXTRA + j] != want;
        }
    }
    printf("[A] visible_mask %s: %d mismatches of %lld\n", ggml_type_name(type), bad, (long long) (N_EXTRA*NROWS));
    return bad ? 1 : 0;
}

// ---- B
static int check_mask_to_index_topk(ggml_backend_t be, const inputs & in) {
    graph_run g; g.ctx = new_ctx();
    ggml_tensor * m  = ggml_new_tensor_2d(g.ctx, GGML_TYPE_F16, KV_RAW, NROWS);
    ggml_tensor * tk = ggml_new_tensor_2d(g.ctx, GGML_TYPE_I32, NTOPK, NROWS);
    ggml_tensor * nv = ggml_new_tensor_2d(g.ctx, GGML_TYPE_I32, 1, NROWS);
    ggml_tensor * fused = ggml_mask_to_index_topk(g.ctx, m, tk, nv, N_IDX);
    ggml_tensor * dense = ggml_mask_to_index(g.ctx,
            ggml_concat(g.ctx, m, ggml_indexer_mask(g.ctx, ggml_visible_mask(g.ctx, nv, N_EXTRA, GGML_TYPE_F16), tk), 0), N_IDX);
    ggml_cgraph * gf = ggml_new_graph(g.ctx);
    ggml_build_forward_expand(gf, fused);
    ggml_build_forward_expand(gf, dense);
    if (!alloc(be, g)) { printf("[B] SETUP FAILED: alloc\n"); return 2; }
    ggml_backend_tensor_set(m,  in.mask.data(), 0, ggml_nbytes(m));
    ggml_backend_tensor_set(tk, in.topk.data(), 0, ggml_nbytes(tk));
    ggml_backend_tensor_set(nv, in.nvis.data(), 0, ggml_nbytes(nv));
    if (ggml_backend_graph_compute(be, gf) != GGML_STATUS_SUCCESS) { printf("[B] FAIL: compute\n"); return 1; }
    std::vector<int32_t> a(N_IDX*NROWS), b(N_IDX*NROWS);
    ggml_backend_tensor_get(fused, a.data(), 0, ggml_nbytes(fused));
    ggml_backend_tensor_get(dense, b.data(), 0, ggml_nbytes(dense));
    std::vector<int32_t> ref_all;
    for (int64_t r = 0; r < NROWS; ++r) {
        auto ref = host_topk_list(in, r);
        ref_all.insert(ref_all.end(), ref.begin(), ref.end());
    }
    const int bad_host = count_list_mismatch(a, ref_all), bad_dense = count_list_mismatch(a, b);
    std::vector<int32_t> c = a; c[5] = c[5] == -1 ? 7 : -1;   // control: a corrupted copy must be seen
    if (count_list_mismatch(c, ref_all) == 0) { printf("[B] CONTROL FAILED: corrupted list not detected\n"); return 2; }
    printf("[B] mask_to_index_topk: %d mismatches vs host, %d vs dense route (of %lld)\n", bad_host, bad_dense, (long long) a.size());
    return (bad_host || bad_dense) ? 1 : 0;
}

// ---- C
static int check_indexer_mask_to_index(ggml_backend_t be, const inputs & in) {
    graph_run g; g.ctx = new_ctx();
    const int64_t n = NTOPK;
    ggml_tensor * m  = ggml_new_tensor_2d(g.ctx, GGML_TYPE_F16, KV_RAW, NROWS);
    ggml_tensor * tk = ggml_new_tensor_2d(g.ctx, GGML_TYPE_I32, NTOPK, NROWS);
    ggml_tensor * fused = ggml_indexer_mask_to_index(g.ctx, m, tk, (int) n);
    ggml_tensor * dense = ggml_mask_to_index(g.ctx, ggml_indexer_mask(g.ctx, m, tk), (int) n);
    ggml_cgraph * gf = ggml_new_graph(g.ctx);
    ggml_build_forward_expand(gf, fused);
    ggml_build_forward_expand(gf, dense);
    if (!alloc(be, g)) { printf("[C] SETUP FAILED: alloc\n"); return 2; }
    ggml_backend_tensor_set(m,  in.mask.data(), 0, ggml_nbytes(m));
    ggml_backend_tensor_set(tk, in.topk.data(), 0, ggml_nbytes(tk));
    if (ggml_backend_graph_compute(be, gf) != GGML_STATUS_SUCCESS) { printf("[C] FAIL: compute\n"); return 1; }
    std::vector<int32_t> a(n*NROWS), b(n*NROWS), ref_all;
    ggml_backend_tensor_get(fused, a.data(), 0, ggml_nbytes(fused));
    ggml_backend_tensor_get(dense, b.data(), 0, ggml_nbytes(dense));
    for (int64_t r = 0; r < NROWS; ++r) {
        auto ref = host_indexer_list(in, r, n);
        ref_all.insert(ref_all.end(), ref.begin(), ref.end());
    }
    const int bad_host = count_list_mismatch(a, ref_all), bad_dense = count_list_mismatch(a, b);
    printf("[C] indexer_mask_to_index: %d mismatches vs host, %d vs dense route (of %lld)\n", bad_host, bad_dense, (long long) a.size());
    return (bad_host || bad_dense) ? 1 : 0;
}

// ---- D
static int check_fa(ggml_backend_t be, const inputs & in, int64_t D, bool v_is_k, uint32_t seed) {
    const int64_t NH = 4;
    const int64_t KV = KV_RAW + N_EXTRA;
    graph_run g; g.ctx = new_ctx();
    ggml_tensor * q  = ggml_new_tensor_3d(g.ctx, GGML_TYPE_F32, D, NQ, NH);
    ggml_tensor * k  = ggml_new_tensor_3d(g.ctx, GGML_TYPE_F16, D, KV, 1);
    ggml_tensor * v  = v_is_k ? k : ggml_new_tensor_3d(g.ctx, GGML_TYPE_F16, D, KV, 1);
    ggml_tensor * m  = ggml_new_tensor_2d(g.ctx, GGML_TYPE_F16, KV_RAW, NROWS);
    ggml_tensor * tk = ggml_new_tensor_2d(g.ctx, GGML_TYPE_I32, NTOPK, NROWS);
    ggml_tensor * nv = ggml_new_tensor_2d(g.ctx, GGML_TYPE_I32, 1, NROWS);
    const float scale = 1.0f/std::sqrt((float) D);

    ggml_tensor * dense_mask = ggml_concat(g.ctx, m,
            ggml_indexer_mask(g.ctx, ggml_visible_mask(g.ctx, nv, N_EXTRA, GGML_TYPE_F16), tk), 0);
    ggml_tensor * ref = ggml_flash_attn_ext(g.ctx, q, k, v, dense_mask, scale, 0.0f, 0.0f);
    ggml_flash_attn_ext_set_prec(ref, GGML_PREC_F32);
    ggml_flash_attn_ext_set_n_swa(ref, GGML_FLASH_ATTN_EXT_IQK_DISABLED);

    ggml_tensor * idx = ggml_mask_to_index_topk(g.ctx, m, tk, nv, N_IDX);
    ggml_tensor * sp_iqk = ggml_flash_attn_ext(g.ctx, q, k, v, nullptr, scale, 0.0f, 0.0f);
    ggml_flash_attn_ext_set_prec(sp_iqk, GGML_PREC_F32);
    sp_iqk->src[5] = idx;
    ggml_tensor * sp_gen = ggml_flash_attn_ext(g.ctx, q, k, v, nullptr, scale, 0.0f, 0.0f);
    ggml_flash_attn_ext_set_prec(sp_gen, GGML_PREC_F32);
    ggml_flash_attn_ext_set_n_swa(sp_gen, GGML_FLASH_ATTN_EXT_IQK_DISABLED);
    sp_gen->src[5] = idx;
    ggml_tensor * nomask = ggml_flash_attn_ext(g.ctx, q, k, v, nullptr, scale, 0.0f, 0.0f);   // control
    ggml_flash_attn_ext_set_prec(nomask, GGML_PREC_F32);

    ggml_cgraph * gf = ggml_new_graph(g.ctx);
    ggml_build_forward_expand(gf, ref);
    ggml_build_forward_expand(gf, sp_iqk);
    ggml_build_forward_expand(gf, sp_gen);
    ggml_build_forward_expand(gf, nomask);
    if (!alloc(be, g)) { printf("[D] SETUP FAILED: alloc\n"); return 2; }

    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> u(-1.0f, 1.0f);
    std::vector<float> hq(ggml_nelements(q));
    for (auto & x : hq) x = u(rng);
    std::vector<ggml_fp16_t> hk(ggml_nelements(k));
    for (auto & x : hk) x = ggml_fp32_to_fp16(u(rng));
    ggml_backend_tensor_set(q, hq.data(), 0, ggml_nbytes(q));
    ggml_backend_tensor_set(k, hk.data(), 0, ggml_nbytes(k));
    if (!v_is_k) {
        std::vector<ggml_fp16_t> hv(ggml_nelements(v));
        for (auto & x : hv) x = ggml_fp32_to_fp16(u(rng));
        ggml_backend_tensor_set(v, hv.data(), 0, ggml_nbytes(v));
    }
    ggml_backend_tensor_set(m,  in.mask.data(), 0, ggml_nbytes(m));
    ggml_backend_tensor_set(tk, in.topk.data(), 0, ggml_nbytes(tk));
    ggml_backend_tensor_set(nv, in.nvis.data(), 0, ggml_nbytes(nv));
    if (ggml_backend_graph_compute(be, gf) != GGML_STATUS_SUCCESS) { printf("[D] FAIL: compute\n"); return 1; }

    const int64_t n = ggml_nelements(ref);
    std::vector<float> r(n), a(n), b(n), c(n);
    ggml_backend_tensor_get(ref,    r.data(), 0, n*sizeof(float));
    ggml_backend_tensor_get(sp_iqk, a.data(), 0, n*sizeof(float));
    ggml_backend_tensor_get(sp_gen, b.data(), 0, n*sizeof(float));
    ggml_backend_tensor_get(nomask, c.data(), 0, n*sizeof(float));
    float da = 0, db = 0, dc = 0;
    bool finite = true;
    for (int64_t i = 0; i < n; ++i) {
        da = std::max(da, std::fabs(a[i] - r[i]));
        db = std::max(db, std::fabs(b[i] - r[i]));
        dc = std::max(dc, std::fabs(c[i] - r[i]));
        finite = finite && std::isfinite(a[i]) && std::isfinite(b[i]) && std::isfinite(r[i]);
    }
    printf("[D] FA D=%lld v_is_k=%d: max|iqk-dense|=%.3g max|generic-dense|=%.3g (control max|nomask-dense|=%.3g) finite=%d\n",
            (long long) D, (int) v_is_k, da, db, dc, (int) finite);
    if (!(dc > 1e-2f)) { printf("[D] CONTROL FAILED: the dense mask does not change the result\n"); return 2; }
    const float tol = 2e-3f;
    return (finite && da <= tol && db <= tol) ? 0 : 1;
}

int main() {
    ggml_backend_t be = ggml_backend_cpu_init();
    if (!be) { printf("SETUP FAILED: cpu backend\n"); return 2; }
    ggml_backend_cpu_set_n_threads(be, 4);
    const inputs in = make_inputs(1234);
    int rc = 0;
    auto acc = [&rc](int r) { if (r > rc) rc = r; };
    acc(check_visible_mask(be, in, GGML_TYPE_F16));
    acc(check_visible_mask(be, in, GGML_TYPE_F32));
    acc(check_mask_to_index_topk(be, in));
    acc(check_indexer_mask_to_index(be, in));
    acc(check_fa(be, in, 512, true, 7));
    acc(check_fa(be, in, 128, false, 11));
    ggml_backend_free(be);
    printf("%s\n", rc == 0 ? "PASS" : rc == 1 ? "FAIL" : "SETUP/CONTROL FAILURE");
    return rc;
}

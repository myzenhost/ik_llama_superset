#pragma once

// Block-diagonal Hessian (X^T X) sidecar for llama-imatrix, and its reader.
//
// Row S69-F1-PICK-IMATRIX-BLOCK-HESSIAN-OUTPUT-2026-10-06-01 (prerequisite of the GPTQ rounding row
// S69-ncnn-PICK-GPTQ-ERROR-COMPENSATED-ROUNDING-2026-09-29-01). The imatrix file keeps one value per input
// column per matrix (tools/imatrix/imatrix.cpp: e.values sized src1->ne[0] * n_mat): the DIAGONAL of
// sum_t x_t x_t^T. GPTQ's column-wise error propagation needs the off-diagonal terms. This header keeps the
// N x N diagonal blocks of that sum for every matrix, as an OPTIONAL extra (llama-imatrix --hessian-block N).
//
// SOURCE-READ (rule 1.87, no code copied):
//   - llmcompressor 0.10.0.2 (Apache-2.0) llmcompressor/modifiers/quantization/gptq/gptq_quantize.py:33-67
//     accumulate_hessian: H += (sqrt(2) X)(sqrt(2) X)^T over the flattened [tokens, K] layer input, with a
//     running sample count; :153-166 masks dead diagonal entries, adds damp * mean(diag H), then Cholesky.
//     Here the RAW sum sum_t x_t x_t^T and the exact token count are stored instead; GPTQ's damped Cholesky
//     is invariant to the positive factor (2 / count), so a consumer may rescale or not.
//   - tools/imatrix/imatrix.cpp:556-571 (dense) and :455-476 (MUL_MAT_ID) diagonal accumulation, whose
//     row addressing the collector reuses when it gathers the rows handed to imatrix_hessian_accumulate().
//
// FILE FORMAT, version 1 (GGUF; written by imatrix_hessian_save, read by imatrix_hessian_load):
//   KV  general.type                = "imatrix_hessian"
//   KV  imatrix_hessian.version     = u32 1
//   KV  imatrix_hessian.block_size  = u32 N
//   KV  imatrix_hessian.layout      = str (one-line description of the layout below)
//   KV  imatrix.datasets            = [str]  (same keys as the imatrix file it accompanies)
//   KV  imatrix.chunk_count         = u32
//   KV  imatrix.chunk_size          = u32
//   per matrix <name> (the weight name, e.g. blk.0.attn_q.weight):
//   T   <name>.hess_blocks  F32  ne = [N, K, n_mat]   (n_mat = 1 for a dense 2D weight, n_expert for MUL_MAT_ID)
//   T   <name>.hess_counts  I64  ne = [n_mat]         (tokens accumulated into each matrix)
//   Element (j, k, m) of hess_blocks = sum over tokens t routed to matrix m of x_t[k] * x_t[b*N + j], where
//   b = k / N and b*N + j < K; entries with b*N + j >= K (the tail of a last, partial block) are 0.
//   So row k of the stored [N, K] slab is row k of X^T X restricted to its own diagonal block: the K/N full
//   blocks are stacked in order, each as N consecutive rows of N floats, and a partial last block of
//   r = K mod N rows keeps only its first r columns. Every block is exactly symmetric. N may exceed K (one
//   block covering the whole matrix). The values are raw sums (NOT divided by the count), as in
//   <name>.in_sum2 of the imatrix file; the diagonal entries (j == k mod N) equal in_sum2 up to f32
//   summation order.
//
// Memory: K * N floats per matrix (N = 128, K = 4096: 2 MiB). Work: K * N / 2 multiply-adds per token per
// matrix (the upper triangle of each block is computed and mirrored).

#include "ggml.h"
#include "gguf.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <map>
#include <string>
#include <thread>
#include <vector>

inline constexpr const char *  IMATRIX_HESSIAN_TYPE          = "imatrix_hessian";
inline constexpr uint32_t      IMATRIX_HESSIAN_VERSION       = 1;
inline constexpr const char *  LLM_KV_IMATRIX_HESSIAN_VER    = "imatrix_hessian.version";
inline constexpr const char *  LLM_KV_IMATRIX_HESSIAN_BLOCK  = "imatrix_hessian.block_size";
inline constexpr const char *  LLM_KV_IMATRIX_HESSIAN_LAYOUT = "imatrix_hessian.layout";
inline constexpr const char *  IMATRIX_HESSIAN_SUFFIX_BLOCKS = ".hess_blocks";
inline constexpr const char *  IMATRIX_HESSIAN_SUFFIX_COUNTS = ".hess_counts";
inline constexpr const char *  IMATRIX_HESSIAN_LAYOUT_DESC   =
    "hess_blocks[j,k,m] = sum_t x_t[k]*x_t[(k/N)*N+j] (0 past K); raw sums; hess_counts[m] = tokens";
// --hessian-block bounds: below 2 there is no off-diagonal term; above 1024 a 4096-wide matrix already needs
// 16 MiB and a GPTQ block solve that large is outside what the GPTQ references use (block 128).
inline constexpr int32_t       IMATRIX_HESSIAN_BLOCK_MIN     = 2;
inline constexpr int32_t       IMATRIX_HESSIAN_BLOCK_MAX     = 1024;

struct imatrix_hessian_entry {
    int64_t              n_in  = 0; // K: input width of the matrix (src1->ne[0])
    int64_t              n_mat = 0; // 1 for a dense weight, n_expert for a MUL_MAT_ID weight
    std::vector<float>   blocks;    // n_mat * K * N floats, layout as in the file (see above)
    std::vector<int64_t> counts;    // n_mat token counts
};

struct imatrix_hessian_file {
    int32_t                                       block_size  = 0;
    uint32_t                                      version     = IMATRIX_HESSIAN_VERSION;
    int32_t                                       chunk_count = 0;
    int32_t                                       chunk_size  = 0;
    std::vector<std::string>                      datasets;
    std::map<std::string, imatrix_hessian_entry>  entries;
};

// Bytes of block storage one matrix needs.
inline uint64_t imatrix_hessian_entry_bytes(int64_t n_in, int64_t block, int64_t n_mat) {
    return (uint64_t) n_in * (uint64_t) block * (uint64_t) n_mat * sizeof(float);
}

// Sidecar path when none is given: <out without .gguf>.hessian.gguf.
inline std::string imatrix_hessian_default_path(const std::string & imatrix_out) {
    const std::string ext = ".gguf";
    std::string stem = imatrix_out.empty() ? std::string("imatrix") : imatrix_out;
    if (stem.size() > ext.size() && stem.compare(stem.size() - ext.size(), ext.size(), ext) == 0) {
        stem.resize(stem.size() - ext.size());
    }
    return stem + ".hessian" + ext;
}

// Allocate (zeroed) block storage for one matrix.
inline void imatrix_hessian_entry_init(imatrix_hessian_entry & e, int64_t n_in, int64_t block, int64_t n_mat) {
    e.n_in  = n_in;
    e.n_mat = n_mat;
    e.blocks.assign((size_t) (n_in * block * n_mat), 0.0f);
    e.counts.assign((size_t) n_mat, 0);
}

// Add sum_r rows[r] rows[r]^T, restricted to the N x N diagonal blocks, into acc (K * N floats: ONE matrix's
// slab, layout as in the file). rows[r] points at K contiguous floats. Each block's upper triangle is summed
// over all rows into a local buffer first (one partial per call, so the f32 error grows with the number of
// calls, not of tokens), then added to both mirrored positions, which keeps every block exactly symmetric.
// Blocks are independent, so they are split over n_threads.
inline void imatrix_hessian_accumulate(float * acc, int64_t n_in, int64_t block,
                                       const float * const * rows, int64_t n_rows, int n_threads) {
    if (n_rows <= 0 || n_in <= 0 || block <= 0) {
        return;
    }
    const int64_t n_blocks = (n_in + block - 1) / block;

    auto work = [&](int64_t ib0, int64_t ib_step) {
        std::vector<float> part;
        for (int64_t ib = ib0; ib < n_blocks; ib += ib_step) {
            const int64_t k0 = ib * block;
            const int64_t nb = std::min<int64_t>(block, n_in - k0);
            part.assign((size_t) (nb * nb), 0.0f);
            for (int64_t r = 0; r < n_rows; ++r) {
                const float * x = rows[r] + k0;
                for (int64_t i = 0; i < nb; ++i) {
                    const float xi = x[i];
                    if (xi == 0.0f) {
                        continue;
                    }
                    float * p = part.data() + i * nb;
                    for (int64_t j = i; j < nb; ++j) {
                        p[j] += xi * x[j];
                    }
                }
            }
            for (int64_t i = 0; i < nb; ++i) {
                float * arow = acc + (k0 + i) * block;
                arow[i] += part[i * nb + i];
                for (int64_t j = i + 1; j < nb; ++j) {
                    const float v = part[i * nb + j];
                    arow[j] += v;
                    acc[(k0 + j) * block + i] += v;
                }
            }
        }
    };

    // threads only pay off once a call carries real work (~4M multiply-adds)
    const double macs = (double) n_rows * (double) n_in * (double) block * 0.5;
    int nt = std::max(1, n_threads);
    if (macs < 4.0e6) {
        nt = 1;
    }
    nt = (int) std::min<int64_t>(nt, n_blocks);
    if (nt <= 1) {
        work(0, 1);
        return;
    }
    std::vector<std::thread> pool;
    pool.reserve(nt - 1);
    for (int t = 1; t < nt; ++t) {
        pool.emplace_back(work, (int64_t) t, (int64_t) nt);
    }
    work(0, nt);
    for (auto & th : pool) {
        th.join();
    }
}

// Write the sidecar: the header fields of f (its entries are ignored) and the given name -> entry map, written in
// name order. Tensor data is referenced in place (no copy of the blocks). Returns false on any failure.
inline bool imatrix_hessian_save(const std::string & fname, const imatrix_hessian_file & f,
                                 const std::map<std::string, imatrix_hessian_entry> & entries) {
    if (f.block_size < 1) {
        return false;
    }
    const int64_t N = f.block_size;

    struct ggml_init_params ip = {
        /* .mem_size   = */ (2 * entries.size() + 1) * ggml_tensor_overhead(),
        /* .mem_buffer = */ NULL,
        /* .no_alloc   = */ true,
    };
    struct ggml_context * ctx = ggml_init(ip);
    if (ctx == nullptr) {
        return false;
    }
    struct gguf_context * ctx_gguf = gguf_init_empty();

    gguf_set_val_str(ctx_gguf, "general.type", IMATRIX_HESSIAN_TYPE);
    gguf_set_val_u32(ctx_gguf, LLM_KV_IMATRIX_HESSIAN_VER, IMATRIX_HESSIAN_VERSION);
    gguf_set_val_u32(ctx_gguf, LLM_KV_IMATRIX_HESSIAN_BLOCK, (uint32_t) N);
    gguf_set_val_str(ctx_gguf, LLM_KV_IMATRIX_HESSIAN_LAYOUT, IMATRIX_HESSIAN_LAYOUT_DESC);
    {
        std::vector<const char *> ds;
        ds.reserve(f.datasets.size());
        for (const auto & d : f.datasets) {
            ds.push_back(d.c_str());
        }
        gguf_set_arr_str(ctx_gguf, "imatrix.datasets", ds.data(), ds.size());
    }
    gguf_set_val_u32(ctx_gguf, "imatrix.chunk_count", (uint32_t) std::max(0, f.chunk_count));
    gguf_set_val_u32(ctx_gguf, "imatrix.chunk_size",  (uint32_t) std::max(0, f.chunk_size));

    bool ok = true;
    for (const auto & [name, e] : entries) {
        if (e.n_in <= 0 || e.n_mat <= 0 ||
            e.blocks.size() != (size_t) (e.n_in * N * e.n_mat) || e.counts.size() != (size_t) e.n_mat) {
            fprintf(stderr, "%s: entry %s has inconsistent sizes, not written\n", __func__, name.c_str());
            ok = false;
            break;
        }
        const std::string tb = name + IMATRIX_HESSIAN_SUFFIX_BLOCKS;
        const std::string tc = name + IMATRIX_HESSIAN_SUFFIX_COUNTS;
        if (tb.size() >= GGML_MAX_NAME) {
            fprintf(stderr, "%s: tensor name %s is too long for GGUF (max %d), not written\n",
                    __func__, tb.c_str(), GGML_MAX_NAME - 1);
            ok = false;
            break;
        }
        struct ggml_tensor * hb = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, N, e.n_in, e.n_mat);
        struct ggml_tensor * hc = ggml_new_tensor_1d(ctx, GGML_TYPE_I64, e.n_mat);
        ggml_set_name(hb, tb.c_str());
        ggml_set_name(hc, tc.c_str());
        gguf_add_tensor(ctx_gguf, hb);
        gguf_add_tensor(ctx_gguf, hc);
        gguf_set_tensor_data(ctx_gguf, tb.c_str(), e.blocks.data());
        gguf_set_tensor_data(ctx_gguf, tc.c_str(), e.counts.data());
    }

    if (ok) {
        ok = gguf_write_to_file(ctx_gguf, fname.c_str(), false);
    }
    gguf_free(ctx_gguf);
    ggml_free(ctx);
    return ok;
}

inline bool imatrix_hessian_save(const std::string & fname, const imatrix_hessian_file & f) {
    return imatrix_hessian_save(fname, f, f.entries);
}

// Read a sidecar written by imatrix_hessian_save. Validates the type, version, block size and every tensor's
// type and shape; on failure returns false with a reason in *err (when given).
inline bool imatrix_hessian_load(const std::string & fname, imatrix_hessian_file & f, std::string * err = nullptr) {
    auto fail = [&](const std::string & why) {
        if (err) {
            *err = why;
        }
        return false;
    };
    struct ggml_context * ctx = nullptr;
    struct gguf_init_params gp = {
        /* .no_alloc = */ false,
        /* .ctx      = */ &ctx,
    };
    struct gguf_context * g = gguf_init_from_file(fname.c_str(), gp);
    if (g == nullptr) {
        return fail("not a readable GGUF file: " + fname);
    }
    struct closer {
        gguf_context * g; ggml_context * c;
        ~closer() { gguf_free(g); if (c) ggml_free(c); }
    } guard{g, ctx};

    const int64_t kt = gguf_find_key(g, "general.type");
    if (kt < 0 || gguf_get_kv_type(g, kt) != GGUF_TYPE_STRING ||
        std::string(gguf_get_val_str(g, kt)) != IMATRIX_HESSIAN_TYPE) {
        return fail("general.type is not imatrix_hessian");
    }
    const int64_t kv = gguf_find_key(g, LLM_KV_IMATRIX_HESSIAN_VER);
    if (kv < 0 || gguf_get_kv_type(g, kv) != GGUF_TYPE_UINT32) {
        return fail("missing imatrix_hessian.version");
    }
    f.version = gguf_get_val_u32(g, kv);
    if (f.version != IMATRIX_HESSIAN_VERSION) {
        return fail("unsupported imatrix_hessian.version " + std::to_string(f.version));
    }
    const int64_t kb = gguf_find_key(g, LLM_KV_IMATRIX_HESSIAN_BLOCK);
    if (kb < 0 || gguf_get_kv_type(g, kb) != GGUF_TYPE_UINT32 || gguf_get_val_u32(g, kb) < 1) {
        return fail("missing or zero imatrix_hessian.block_size");
    }
    f.block_size = (int32_t) gguf_get_val_u32(g, kb);
    const int64_t N = f.block_size;

    f.datasets.clear();
    if (const int64_t kd = gguf_find_key(g, "imatrix.datasets");
        kd >= 0 && gguf_get_kv_type(g, kd) == GGUF_TYPE_ARRAY && gguf_get_arr_type(g, kd) == GGUF_TYPE_STRING) {
        for (size_t i = 0; i < gguf_get_arr_n(g, kd); ++i) {
            f.datasets.emplace_back(gguf_get_arr_str(g, kd, i));
        }
    }
    if (const int64_t k = gguf_find_key(g, "imatrix.chunk_count"); k >= 0 && gguf_get_kv_type(g, k) == GGUF_TYPE_UINT32) {
        f.chunk_count = (int32_t) gguf_get_val_u32(g, k);
    }
    if (const int64_t k = gguf_find_key(g, "imatrix.chunk_size"); k >= 0 && gguf_get_kv_type(g, k) == GGUF_TYPE_UINT32) {
        f.chunk_size = (int32_t) gguf_get_val_u32(g, k);
    }

    f.entries.clear();
    const std::string sb = IMATRIX_HESSIAN_SUFFIX_BLOCKS;
    const std::string sc = IMATRIX_HESSIAN_SUFFIX_COUNTS;
    for (struct ggml_tensor * t = ggml_get_first_tensor(ctx); t; t = ggml_get_next_tensor(ctx, t)) {
        const std::string tn = ggml_get_name(t);
        if (tn.size() <= sb.size() || tn.compare(tn.size() - sb.size(), sb.size(), sb) != 0) {
            continue;
        }
        const std::string name = tn.substr(0, tn.size() - sb.size());
        if (t->type != GGML_TYPE_F32 || t->ne[0] != N || t->ne[3] != 1) {
            return fail("tensor " + tn + " is not F32 [block_size, K, n_mat]");
        }
        struct ggml_tensor * c = ggml_get_tensor(ctx, (name + sc).c_str());
        if (c == nullptr || c->type != GGML_TYPE_I64 || c->ne[0] != t->ne[2] || ggml_nelements(c) != t->ne[2]) {
            return fail("tensor " + name + sc + " is missing or not I64 [n_mat]");
        }
        auto & e = f.entries[name];
        e.n_in  = t->ne[1];
        e.n_mat = t->ne[2];
        const float   * pb = (const float   *) t->data;
        const int64_t * pc = (const int64_t *) c->data;
        e.blocks.assign(pb, pb + ggml_nelements(t));
        e.counts.assign(pc, pc + ggml_nelements(c));
    }
    return true;
}

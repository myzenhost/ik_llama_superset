// Ollama qwen2.5vl combined blob used as its own mmproj: structural and numeric check of the clip loader.
//
//   llama-test-ollama-qwen25vl-clip <ollama qwen25vl blob> [reference converter mmproj of the same model]
//   (or LLAMA_TEST_OLLAMA_QWEN25VL_BLOB / LLAMA_TEST_OLLAMA_QWEN25VL_REF; exit 77 when no blob is given or found)
//
// Ollama stores Qwen2.5-VL as ONE GGUF: general.architecture = qwen25vl, the text blocks, and the vision tower under v.*
// (converter-style block names, but v.merger.ln_q / v.merger.mlp.{0,2} / v.patch_embd_{0,1} instead of v.post_ln / mm.N /
// v.patch_embd.weight(.1)) with the vision hparams under qwen25vl.vision.*. Before the clip-side read the blob does not
// load as an mmproj at all (no clip.projector_type).
//
// Arms:
//   1. structural (no_alloc, CPU): the blob builds a vision clip context; projector qwen2.5vl_merger; hparams equal the
//      values in the blob's OWN header (n_embd, n_ff = ffn_up rows, n_head, n_layer, projection_dim = text n_embd,
//      patch, n_merge, eps, n_wa_pattern = fullatt_block_indexes[0]+1); the loader fetched every v.* + mm.* tensor;
//   2. LLAMA_OLLAMA_COMPAT=0 turns the read off (the blob is refused again);
//   3. with a reference mmproj: every hparam, mean/std, ffn_op, n_mmproj_embd and the tensor count match it;
//   4. with a reference mmproj: both are loaded with weights (CPU) and encode the same synthetic 224x224 image; the
//      embeddings must agree (cosine >= 0.99, the row's bound). Not bit-identical: Ollama quantized attn_v (Q4_K /
//      Q6_K) where the reference is F16. MEASURED 2026-10-20 on qwen2.5vl:7b: cosine 0.99757, worst token 0.97431, so
//      a per-token bound against the pristine reference would measure the quantization, not the mapping.
//   5. the mapping check without that noise: a copy of the reference whose 32 attn_v tensors are replaced by the
//      blob's own attn_v, dequantized to F16, is encoded the same way. Every other weight is the reference's, so any
//      misnamed, transposed or missing tensor in the blob read shows up here: cosine >= 0.999 and every token >= 0.99.
//      The copy is written under LLAMA_TEST_TMPDIR (else TEMP / TMP / .) and deleted afterwards.
//   6. flash attention on vs off, blob only, at 252x252: 324 patches, not a multiple of GGML_KQ_MASK_PAD (64 here), so
//      the window mask needs padding before ggml_flash_attn_ext (the server's default -fa auto path). Runs without a
//      reference mmproj too.
#include "clip.h"
#include "clip-impl.h"
#include "clip-model.h"
#include "ggml.h"
#include "gguf.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

static int g_fail = 0;
#define CHECK(cond, ...) do { if (!(cond)) { g_fail++; std::printf("FAIL %s:%d  ", __FILE__, __LINE__); std::printf(__VA_ARGS__); std::printf("\n"); } } while (0)

static int g_n_tensors_logged = -1;

static void probe_log(enum ggml_log_level level, const char * text, void * /*user_data*/) {
    const char * key = "skipping tensor data loading (";
    if (const char * p = std::strstr(text, key)) {
        g_n_tensors_logged = std::atoi(p + std::strlen(key));
    }
    if (level >= GGML_LOG_LEVEL_INFO) {
        std::fputs(text, stderr);
    }
}

struct probe_result {
    bool         ok = false;
    std::string  proj;
    clip_hparams hp{};
    int          n_mmproj_embd = 0;
    int          n_tensors = -1;
};

static clip_context_params make_params(bool no_alloc, clip_flash_attn_type fa = CLIP_FLASH_ATTN_TYPE_DISABLED) {
    clip_context_params p{};
    p.use_gpu          = false;
    p.device           = nullptr;
    p.flash_attn_type  = fa;
    p.image_min_tokens = -1;
    p.image_max_tokens = -1;
    p.warmup           = false;
    p.no_alloc         = no_alloc;
    return p;
}

static probe_result probe(const char * path) {
    probe_result res;
    g_n_tensors_logged = -1;
    clip_init_result r = clip_init(path, make_params(true));
    if (r.ctx_v) {
        res.ok            = true;
        res.proj          = PROJECTOR_TYPE_NAMES[clip_get_projector_type(r.ctx_v)];
        res.hp            = *clip_get_hparams(r.ctx_v);
        res.n_mmproj_embd = clip_n_mmproj_embd(r.ctx_v);
        res.n_tensors     = g_n_tensors_logged;
    }
    clip_free(r.ctx_v);
    clip_free(r.ctx_a);
    clip_free(r.ctx_gen_a);
    return res;
}

static void print_probe(const char * label, const probe_result & r) {
    std::printf("[%s] loaded=%d proj=%s image_size=%d patch_size=%d n_embd=%d n_ff=%d n_head=%d n_layer=%d "
                "projection_dim=%d n_merge=%d n_wa_pattern=%d eps=%g ffn_op=%d mean=(%g,%g,%g) std=(%g,%g,%g) "
                "n_mmproj_embd=%d n_tensors=%d\n",
                label, (int) r.ok, r.proj.c_str(), r.hp.image_size, r.hp.patch_size, r.hp.n_embd, r.hp.n_ff, r.hp.n_head,
                r.hp.n_layer, r.hp.projection_dim, r.hp.n_merge, r.hp.n_wa_pattern, (double) r.hp.eps, (int) r.hp.ffn_op,
                r.hp.image_mean[0], r.hp.image_mean[1], r.hp.image_mean[2], r.hp.image_std[0], r.hp.image_std[1], r.hp.image_std[2],
                r.n_mmproj_embd, r.n_tensors);
}

static void set_env(const char * k, const char * v) {
#ifdef _WIN32
    _putenv_s(k, v ? v : "");
#else
    if (v) setenv(k, v, 1); else unsetenv(k);
#endif
}

static bool file_exists(const char * path) {
    FILE * f = path ? std::fopen(path, "rb") : nullptr;
    if (f) std::fclose(f);
    return f != nullptr;
}

struct blob_expect {
    std::string arch;
    int patch_size = 0, n_embd = 0, n_ff = 0, n_head = 0, n_layer = 0, projection_dim = 0, n_merge = 0, n_wa = 0;
    int n_vis_tensors = 0;
    float eps = 0.0f;
};

static bool read_blob_expect(const char * path, blob_expect & e) {
    ggml_context * meta = nullptr;
    gguf_init_params ip = { /*no_alloc*/ true, &meta };
    gguf_context * g = gguf_init_from_file(path, ip);
    if (!g) return false;
    auto u32 = [&](const std::string & k) -> int {
        const int64_t id = gguf_find_key(g, k.c_str());
        return id < 0 ? -1 : (int) gguf_get_val_u32(g, id);
    };
    const int64_t aid = gguf_find_key(g, "general.architecture");
    e.arch = aid < 0 ? "" : gguf_get_val_str(g, aid);
    const std::string a = e.arch;
    e.patch_size     = u32(a + ".vision.patch_size");
    e.n_embd         = u32(a + ".vision.embedding_length");
    e.n_head         = u32(a + ".vision.attention.head_count");
    e.n_layer        = u32(a + ".vision.block_count");
    e.projection_dim = u32(a + ".embedding_length");
    e.n_merge        = u32(a + ".vision.spatial_merge_size");
    {
        const int64_t id = gguf_find_key(g, (a + ".vision.attention.layer_norm_epsilon").c_str());
        e.eps = id < 0 ? -1.0f : gguf_get_val_f32(g, id);
    }
    {
        const int64_t id = gguf_find_key(g, (a + ".vision.fullatt_block_indexes").c_str());
        e.n_wa = id < 0 || gguf_get_arr_n(g, id) == 0 ? 8 : ((const int32_t *) gguf_get_arr_data(g, id))[0] + 1;
    }
    if (ggml_tensor * up = ggml_get_tensor(meta, "v.blk.0.ffn_up.weight")) {
        e.n_ff = (int) up->ne[1];
    }
    for (int64_t i = 0; i < gguf_get_n_tensors(g); ++i) {
        const char * n = gguf_get_tensor_name(g, i);
        if (std::strncmp(n, "v.", 2) == 0 || std::strncmp(n, "mm.", 3) == 0) e.n_vis_tensors++;
    }
    gguf_free(g);
    ggml_free(meta);
    return true;
}

// encode one deterministic, already-normalized side x side RGB image on the CPU (224 -> 16x16 patches, 8x8 merged tokens)
static bool encode_synthetic(const char * path, int n_threads, std::vector<float> & out, int & n_tok, int side = 224,
                             clip_flash_attn_type fa = CLIP_FLASH_ATTN_TYPE_DISABLED) {
    clip_init_result r = clip_init(path, make_params(false, fa));
    if (!r.ctx_v) {
        return false;
    }
    const int W = side, H = side;
    clip_image_f32 img;
    img.set_size({ W, H }, false, false);
    std::vector<float> px((size_t) 3 * W * H);
    for (int y = 0; y < H; ++y) {
        for (int x = 0; x < W; ++x) {
            for (int c = 0; c < 3; ++c) {
                px[(size_t) 3 * (y * W + x) + c] =
                    0.8f * std::sin(0.071f * x + 0.37f * c) * std::cos(0.053f * y - 0.11f * c) + 0.1f * (float) ((x / 28 + y / 28 + c) % 3 - 1);
            }
        }
    }
    img.cpy_buf(px);
    n_tok = clip_n_output_tokens(r.ctx_v, &img);
    out.assign((size_t) n_tok * clip_n_mmproj_embd(r.ctx_v), 0.0f);
    const bool ok = clip_image_encode(r.ctx_v, n_threads, &img, out);
    clip_free(r.ctx_v);
    clip_free(r.ctx_a);
    clip_free(r.ctx_gen_a);
    return ok;
}

static int seek_abs(FILE * f, size_t offs) {
#ifdef _WIN32
    return _fseeki64(f, (long long) offs, SEEK_SET);
#else
    return fseeko(f, (off_t) offs, SEEK_SET);
#endif
}

// Write a copy of the reference mmproj whose v.blk.N.attn_v.weight tensors hold the blob's own attn_v, dequantized
// and stored in the reference's type (F16). Returns the number of tensors replaced, or -1 on any error.
static int write_ref_with_blob_attn_v(const char * ref, const char * blob, const std::string & out_path) {
    ggml_context * ctx_r = nullptr;
    gguf_init_params ipr = { /*no_alloc*/ false, &ctx_r };
    gguf_context * gr = gguf_init_from_file(ref, ipr);
    if (!gr) return -1;
    ggml_context * meta_b = nullptr;
    gguf_init_params ipb = { /*no_alloc*/ true, &meta_b };
    gguf_context * gb = gguf_init_from_file(blob, ipb);
    FILE * fb = gb ? std::fopen(blob, "rb") : nullptr;
    int n_rep = -1;
    if (gb && fb) {
        n_rep = 0;
        for (ggml_tensor * t = ggml_get_first_tensor(ctx_r); t; t = ggml_get_next_tensor(ctx_r, t)) {
            const std::string name = ggml_get_name(t);
            if (name.rfind("v.blk.", 0) != 0 || name.size() < 14 || name.compare(name.size() - 14, 14, ".attn_v.weight") != 0) {
                continue;
            }
            ggml_tensor * tb = ggml_get_tensor(meta_b, name.c_str());
            const int64_t id = gguf_find_tensor(gb, name.c_str());
            if (!tb || id < 0 || t->type != GGML_TYPE_F16 || ggml_nelements(tb) != ggml_nelements(t) || tb->ne[0] != t->ne[0]) {
                std::printf("[ref+blob attn_v] %s: missing or shape/type mismatch\n", name.c_str());
                n_rep = -1;
                break;
            }
            std::vector<uint8_t> raw(ggml_nbytes(tb));
            const size_t offs = gguf_get_data_offset(gb) + gguf_get_tensor_offset(gb, id);
            if (seek_abs(fb, offs) != 0 || std::fread(raw.data(), 1, raw.size(), fb) != raw.size()) {
                std::printf("[ref+blob attn_v] %s: read failed\n", name.c_str());
                n_rep = -1;
                break;
            }
            const int64_t nrow = ggml_nrows(tb), ncol = tb->ne[0];
            const auto * tr = ggml_get_type_traits(tb->type);
            std::vector<float> row(ncol);
            for (int64_t r = 0; r < nrow; ++r) {
                const uint8_t * src = raw.data() + r * tb->nb[1];
                if (tb->type == GGML_TYPE_F32) {
                    std::memcpy(row.data(), src, ncol * sizeof(float));
                } else if (tb->type == GGML_TYPE_F16) {
                    ggml_fp16_to_fp32_row((const ggml_fp16_t *) src, row.data(), ncol);
                } else {
                    tr->to_float(src, row.data(), ncol);
                }
                ggml_fp32_to_fp16_row(row.data(), (ggml_fp16_t *) ((char *) t->data + r * t->nb[1]), ncol);
            }
            n_rep++;
        }
    }
    if (n_rep > 0 && !gguf_write_to_file(gr, out_path.c_str(), false)) {
        std::printf("[ref+blob attn_v] writing %s failed\n", out_path.c_str());
        n_rep = -1;
    }
    if (fb) std::fclose(fb);
    if (gb) gguf_free(gb);
    if (meta_b) ggml_free(meta_b);
    gguf_free(gr);
    ggml_free(ctx_r);
    return n_rep;
}

struct embed_cmp {
    double cos = 0, worst = 1, max_abs = 0, ref_rms = 0;
    size_t nonfinite = 0;
    int d = 0;
};

static embed_cmp compare_embeddings(const std::vector<float> & eb, const std::vector<float> & er, int n_tok) {
    embed_cmp c;
    double dot = 0, nb = 0, nr = 0;
    for (size_t i = 0; i < eb.size(); ++i) {
        if (!std::isfinite(eb[i]) || !std::isfinite(er[i])) { c.nonfinite++; continue; }
        dot += (double) eb[i] * er[i];
        nb  += (double) eb[i] * eb[i];
        nr  += (double) er[i] * er[i];
        c.max_abs = std::fmax(c.max_abs, std::fabs((double) eb[i] - er[i]));
    }
    c.cos = dot / (std::sqrt(nb) * std::sqrt(nr) + 1e-30);
    c.ref_rms = std::sqrt(nr / (double) er.size());
    c.d = (int) (eb.size() / (size_t) n_tok);
    for (int t = 0; t < n_tok; ++t) {
        double tdot = 0, tnb = 0, tnr = 0;
        for (int j = 0; j < c.d; ++j) {
            const double x = eb[(size_t) t * c.d + j], y = er[(size_t) t * c.d + j];
            tdot += x * y; tnb += x * x; tnr += y * y;
        }
        c.worst = std::fmin(c.worst, tdot / (std::sqrt(tnb) * std::sqrt(tnr) + 1e-30));
    }
    return c;
}

int main(int argc, char ** argv) {
    const char * blob = argc > 1 ? argv[1] : std::getenv("LLAMA_TEST_OLLAMA_QWEN25VL_BLOB");
    const char * ref  = argc > 2 ? argv[2] : std::getenv("LLAMA_TEST_OLLAMA_QWEN25VL_REF");
    if (!file_exists(blob)) {
        std::printf("SKIP: no Ollama qwen25vl blob given (argv[1] or LLAMA_TEST_OLLAMA_QWEN25VL_BLOB)\n");
        return 77;
    }
    g_logger_state.log_callback = probe_log;
    g_logger_state.log_callback_user_data = nullptr;

    blob_expect e;
    if (!read_blob_expect(blob, e)) {
        std::printf("FAIL: cannot read the GGUF header of %s\n", blob);
        return 1;
    }
    if (e.arch != "qwen25vl") {
        std::printf("SKIP: %s is general.architecture=%s, not an Ollama qwen25vl blob\n", blob, e.arch.c_str());
        return 77;
    }
    std::printf("[expect from blob header] patch_size=%d n_embd=%d n_ff=%d n_head=%d n_layer=%d projection_dim=%d "
                "n_merge=%d n_wa_pattern=%d eps=%g v+mm tensors=%d\n",
                e.patch_size, e.n_embd, e.n_ff, e.n_head, e.n_layer, e.projection_dim, e.n_merge, e.n_wa, (double) e.eps,
                e.n_vis_tensors);

    // arm 1: the blob as its own mmproj
    const probe_result b = probe(blob);
    print_probe("blob", b);
    CHECK(b.ok, "the blob did not build a vision clip context");
    if (b.ok) {
        CHECK(b.proj == "qwen2.5vl_merger",              "projector %s, expected qwen2.5vl_merger", b.proj.c_str());
        CHECK(b.hp.patch_size     == e.patch_size,       "patch_size %d != %d", b.hp.patch_size, e.patch_size);
        CHECK(b.hp.n_embd         == e.n_embd,           "n_embd %d != %d", b.hp.n_embd, e.n_embd);
        CHECK(b.hp.n_ff           == e.n_ff,             "n_ff %d != %d", b.hp.n_ff, e.n_ff);
        CHECK(b.hp.n_head         == e.n_head,           "n_head %d != %d", b.hp.n_head, e.n_head);
        CHECK(b.hp.n_layer        == e.n_layer,          "n_layer %d != %d", b.hp.n_layer, e.n_layer);
        CHECK(b.hp.projection_dim == e.projection_dim,   "projection_dim %d != %d", b.hp.projection_dim, e.projection_dim);
        CHECK(b.hp.n_merge        == e.n_merge,          "n_merge %d != %d", b.hp.n_merge, e.n_merge);
        CHECK(b.hp.n_wa_pattern   == e.n_wa,             "n_wa_pattern %d != %d", b.hp.n_wa_pattern, e.n_wa);
        CHECK(std::fabs(b.hp.eps - e.eps) < 1e-12f,      "eps %g != %g", (double) b.hp.eps, (double) e.eps);
        CHECK(b.n_tensors         == e.n_vis_tensors,    "tensor count %d != %d (every v.* and mm.* tensor)", b.n_tensors, e.n_vis_tensors);
        CHECK(b.n_mmproj_embd     == e.projection_dim,   "n_mmproj_embd %d != %d", b.n_mmproj_embd, e.projection_dim);
    }

    // arm 2: the opt-out switch turns the read off
    set_env("LLAMA_OLLAMA_COMPAT", "0");
    const probe_result off = probe(blob);
    set_env("LLAMA_OLLAMA_COMPAT", nullptr);
    print_probe("blob LLAMA_OLLAMA_COMPAT=0", off);
    CHECK(!off.ok, "LLAMA_OLLAMA_COMPAT=0 should leave the blob unreadable as an mmproj");

    if (file_exists(ref)) {
        // arm 3: hparams against a converter-made mmproj of the same model
        const probe_result r = probe(ref);
        print_probe("reference mmproj", r);
        CHECK(r.ok, "the reference mmproj did not load");
        if (r.ok && b.ok) {
            CHECK(b.proj == r.proj,                                 "projector differs from the reference");
            CHECK(b.hp.image_size        == r.hp.image_size,        "image_size %d != reference %d", b.hp.image_size, r.hp.image_size);
            CHECK(b.hp.patch_size        == r.hp.patch_size,        "patch_size differs from the reference");
            CHECK(b.hp.n_embd            == r.hp.n_embd,            "n_embd differs from the reference");
            CHECK(b.hp.n_ff              == r.hp.n_ff,              "n_ff differs from the reference");
            CHECK(b.hp.n_head            == r.hp.n_head,            "n_head differs from the reference");
            CHECK(b.hp.n_layer           == r.hp.n_layer,           "n_layer differs from the reference");
            CHECK(b.hp.projection_dim    == r.hp.projection_dim,    "projection_dim differs from the reference");
            CHECK(b.hp.n_merge           == r.hp.n_merge,           "n_merge differs from the reference");
            CHECK(b.hp.n_wa_pattern      == r.hp.n_wa_pattern,      "n_wa_pattern differs from the reference");
            CHECK(b.hp.image_min_pixels  == r.hp.image_min_pixels,  "image_min_pixels differs from the reference");
            CHECK(b.hp.image_max_pixels  == r.hp.image_max_pixels,  "image_max_pixels differs from the reference");
            CHECK(b.hp.warmup_image_size == r.hp.warmup_image_size, "warmup_image_size differs from the reference");
            CHECK(b.hp.ffn_op            == r.hp.ffn_op,            "ffn_op differs from the reference");
            CHECK(std::fabs(b.hp.eps - r.hp.eps) < 1e-12f,          "eps differs from the reference");
            for (int i = 0; i < 3; ++i) {
                CHECK(std::fabs(b.hp.image_mean[i] - r.hp.image_mean[i]) < 1e-7f && std::fabs(b.hp.image_std[i] - r.hp.image_std[i]) < 1e-7f,
                      "mean/std[%d] differs from the reference", i);
            }
            CHECK(b.n_mmproj_embd == r.n_mmproj_embd,               "n_mmproj_embd differs from the reference");
            CHECK(b.n_tensors     == r.n_tensors,                   "tensor count %d != reference %d", b.n_tensors, r.n_tensors);
        }

        // arm 4: same synthetic image through both, weights loaded
        const int n_threads = std::getenv("LLAMA_TEST_THREADS") ? std::atoi(std::getenv("LLAMA_TEST_THREADS")) : 4;
        std::vector<float> eb, er;
        int tb = 0, tr = 0;
        const bool okb = encode_synthetic(blob, n_threads, eb, tb);
        const bool okr = encode_synthetic(ref,  n_threads, er, tr);
        CHECK(okb, "blob encode failed");
        CHECK(okr, "reference encode failed");
        if (okb && okr) {
            CHECK(tb == tr && eb.size() == er.size(), "token count %d vs %d, size %zu vs %zu", tb, tr, eb.size(), er.size());
            if (eb.size() == er.size() && !eb.empty()) {
                const embed_cmp c = compare_embeddings(eb, er, tb);
                std::printf("[embed vs reference] tokens=%d dim=%d cosine=%.6f worst_token_cosine=%.6f max_abs_diff=%.6g ref_rms=%.6g nonfinite=%zu\n",
                            tb, c.d, c.cos, c.worst, c.max_abs, c.ref_rms, c.nonfinite);
                CHECK(c.nonfinite == 0, "%zu non-finite values", c.nonfinite);
                CHECK(c.cos >= 0.99, "embedding cosine %.6f < 0.99", c.cos);
            }
        }

        // arm 5: the reference with the blob's own attn_v -- same weights on both sides, so the bound is tight
        if (okb) {
            const char * td = std::getenv("LLAMA_TEST_TMPDIR");
            if (!td) td = std::getenv("TEMP");
            if (!td) td = std::getenv("TMP");
            const std::string tmp = std::string(td ? td : ".") + "/test-ollama-qwen25vl-clip-ref-blob-attn-v.gguf";
            const int n_rep = write_ref_with_blob_attn_v(ref, blob, tmp);
            std::printf("[ref+blob attn_v] replaced %d attn_v tensors -> %s\n", n_rep, tmp.c_str());
            CHECK(n_rep == e.n_layer, "replaced %d attn_v tensors, expected one per layer (%d)", n_rep, e.n_layer);
            if (n_rep > 0) {
                std::vector<float> em;
                int tm = 0;
                const bool okm = encode_synthetic(tmp.c_str(), n_threads, em, tm);
                std::remove(tmp.c_str());
                CHECK(okm, "encode of the reference with the blob's attn_v failed");
                if (okm && tm == tb && em.size() == eb.size() && !eb.empty()) {
                    const embed_cmp c = compare_embeddings(eb, em, tb);
                    std::printf("[embed vs reference+blob attn_v] tokens=%d dim=%d cosine=%.6f worst_token_cosine=%.6f max_abs_diff=%.6g ref_rms=%.6g nonfinite=%zu\n",
                                tb, c.d, c.cos, c.worst, c.max_abs, c.ref_rms, c.nonfinite);
                    CHECK(c.nonfinite == 0, "%zu non-finite values", c.nonfinite);
                    CHECK(c.cos   >= 0.999, "embedding cosine %.6f < 0.999 with identical weights", c.cos);
                    CHECK(c.worst >= 0.99,  "worst token cosine %.6f < 0.99 with identical weights", c.worst);
                } else if (okm) {
                    CHECK(false, "token count %d vs %d, size %zu vs %zu", tb, tm, eb.size(), em.size());
                }
            }
        }
    } else {
        std::printf("[reference mmproj] not given; arms 3 and 4 skipped\n");
    }

    // arm 6: flash attention on the window-attention graph. 252x252 -> 18x18 = 324 patches, and 324 % 64 = 4, so the
    // n_pos x n_pos window mask is short of GGML_KQ_MASK_PAD (64 in this fork) and ggml_flash_attn_ext asserted before
    // clip padded it. This is the server's default path: -fa auto turns flash attention on at warmup.
    {
        const int n_threads = std::getenv("LLAMA_TEST_THREADS") ? std::atoi(std::getenv("LLAMA_TEST_THREADS")) : 4;
        const int side = 252;
        std::vector<float> e_off, e_on;
        int t_off = 0, t_on = 0;
        const bool ok_off = encode_synthetic(blob, n_threads, e_off, t_off, side, CLIP_FLASH_ATTN_TYPE_DISABLED);
        const bool ok_on  = encode_synthetic(blob, n_threads, e_on,  t_on,  side, CLIP_FLASH_ATTN_TYPE_ENABLED);
        CHECK(ok_off, "blob encode at %dx%d with flash attention off failed", side, side);
        CHECK(ok_on,  "blob encode at %dx%d with flash attention on failed", side, side);
        if (ok_off && ok_on && t_off == t_on && e_off.size() == e_on.size() && !e_off.empty()) {
            const embed_cmp c = compare_embeddings(e_on, e_off, t_off);
            std::printf("[flash attn on vs off, %dx%d] tokens=%d dim=%d cosine=%.6f worst_token_cosine=%.6f max_abs_diff=%.6g ref_rms=%.6g nonfinite=%zu\n",
                        side, side, t_off, c.d, c.cos, c.worst, c.max_abs, c.ref_rms, c.nonfinite);
            CHECK(c.nonfinite == 0, "%zu non-finite values with flash attention on", c.nonfinite);
            CHECK(c.cos   >= 0.999, "flash attention on vs off cosine %.6f < 0.999", c.cos);
            CHECK(c.worst >= 0.99,  "flash attention on vs off worst token cosine %.6f < 0.99", c.worst);
        } else if (ok_off && ok_on) {
            CHECK(false, "token count %d vs %d, size %zu vs %zu", t_off, t_on, e_off.size(), e_on.size());
        }
    }

    std::printf("\n[RESULT] %s (%d failed check%s)\n", g_fail == 0 ? "PASS" : "FAIL", g_fail, g_fail == 1 ? "" : "s");
    return g_fail == 0 ? 0 : 1;
}

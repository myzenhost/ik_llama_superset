// Ling 3.0 VL (BAILINGMOE3 + "ling3vl" projector) check on RANDOM-WEIGHT files. Row S69-DECL-OVERTURN-LING3VL-2026-10-17-01.
//
//   llama-test-ling3vl --plain <bailingmoe3 GGUF, no rope.dimension_sections> --vl <same + rope.dimension_sections>
//                      --mmproj <tiny ling3vl mmproj from make_tiny_ling3vl_mmproj.py> [--threads N]
//
// What it checks:
//   (a) rope dispatch from GGUF metadata: the plain file decodes with NORM rope, the VL file with MROPE
//       (llama_model_rope_type; mtmd picks M-RoPE decoder positions from the same call)
//   (b) text decode on both files: logits finite. Prints an FNV-1a hash of the plain file's logits so a build without this
//       change and a build with it can be compared for bit-identity (the text-only path must not move)
//   (c) the ling3vl projector loads (clip_init), encodes a 32x32 image, yields (32/patch/2)^2 tokens of the text model's width,
//       all finite
//   (d) those embeddings decode through the VL text model with 2-D M-RoPE positions (t, row, col, 0), logits finite
// It does NOT prove image understanding: every weight is random.
// SOURCE-READ: ik_llama 719f4aa5d (MIT) src/llama.cpp llama_rope_type, src/graphs/build_deepseek2.cpp, examples/mtmd/clip.cpp
#include "clip.h"
#include "clip-impl.h"
#include "clip-model.h"
#include "llama.h"
#include "ggml.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

static int g_fail = 0;
#define CHECK(cond, ...) do { if (!(cond)) { g_fail++; std::fprintf(stderr, "FAIL %s:%d  ", __FILE__, __LINE__); std::fprintf(stderr, __VA_ARGS__); std::fprintf(stderr, "\n"); } } while (0)

static uint64_t fnv1a(const void * p, size_t n) {
    uint64_t h = 1469598103934665603ULL;
    const auto * b = (const uint8_t *) p;
    for (size_t i = 0; i < n; ++i) {
        h ^= b[i];
        h *= 1099511628211ULL;
    }
    return h;
}

static const char * rope_name(int t) {
    switch (t) {
        case LLAMA_ROPE_TYPE_NONE:  return "NONE";
        case LLAMA_ROPE_TYPE_NORM:  return "NORM";
        case LLAMA_ROPE_TYPE_NEOX:  return "NEOX";
        case LLAMA_ROPE_TYPE_MROPE: return "MROPE";
        default:                    return "OTHER";
    }
}

static bool all_finite(const float * v, size_t n) {
    for (size_t i = 0; i < n; ++i) {
        if (!std::isfinite(v[i])) {
            return false;
        }
    }
    return true;
}

static llama_context * make_ctx(llama_model * model, int n_threads) {
    llama_context_params cp = llama_context_default_params();
    cp.n_ctx           = 128;
    cp.n_batch         = 64;
    cp.n_ubatch        = 64;
    cp.n_seq_max       = 1;
    cp.n_threads       = n_threads;
    cp.n_threads_batch = n_threads;
    return llama_init_from_model(model, cp);
}

// decode 8 fixed tokens at positions 0..7, every token's logits requested; returns all logits
static std::vector<float> decode_text(llama_context * ctx, const llama_model * model) {
    const int n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));
    const int n_tok = 8;
    llama_batch batch = llama_batch_init(n_tok, 0, 1);
    for (int i = 0; i < n_tok; ++i) {
        batch.token[i]     = (llama_token) ((i * 7 + 3) % n_vocab);
        batch.pos[i]       = i;
        batch.n_seq_id[i]  = 1;
        batch.seq_id[i][0] = 0;
        batch.logits[i]    = true;
    }
    batch.n_tokens = n_tok;
    std::vector<float> out;
    const int rc = llama_decode(ctx, batch);
    CHECK(rc == 0, "llama_decode (text) rc=%d", rc);
    if (rc == 0) {
        for (int i = 0; i < n_tok; ++i) {
            const float * l = llama_get_logits_ith(ctx, i);
            out.insert(out.end(), l, l + n_vocab);
        }
    }
    llama_batch_free(batch);
    return out;
}

int main(int argc, char ** argv) {
    std::string plain, vl, mmproj;
    int n_threads = 4;
    for (int i = 1; i < argc; ++i) {
        if      (!std::strcmp(argv[i], "--plain")   && i + 1 < argc) plain  = argv[++i];
        else if (!std::strcmp(argv[i], "--vl")      && i + 1 < argc) vl     = argv[++i];
        else if (!std::strcmp(argv[i], "--mmproj")  && i + 1 < argc) mmproj = argv[++i];
        else if (!std::strcmp(argv[i], "--threads") && i + 1 < argc) n_threads = std::atoi(argv[++i]);
    }
    if (plain.empty() || vl.empty() || mmproj.empty()) {
        std::fprintf(stderr, "usage: %s --plain GGUF --vl GGUF --mmproj GGUF [--threads N]\n", argv[0]);
        return 2;
    }

    llama_backend_init();
    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = 0;

    // ---- (a)+(b) plain file: NORM, text logits hash -------------------------------------------------------------------
    llama_model * m_plain = llama_model_load_from_file(plain.c_str(), mp);
    CHECK(m_plain != nullptr, "load %s", plain.c_str());
    uint64_t h_plain = 0;
    if (m_plain) {
        const int rt = (int) llama_model_rope_type(m_plain);
        std::printf("plain: rope_type=%s (%d)\n", rope_name(rt), rt);
        CHECK(rt == LLAMA_ROPE_TYPE_NORM, "plain file must decode with NORM rope, got %s", rope_name(rt));
        llama_context * ctx = make_ctx(m_plain, n_threads);
        CHECK(ctx != nullptr, "context for plain");
        if (ctx) {
            const std::vector<float> lg = decode_text(ctx, m_plain);
            CHECK(!lg.empty() && all_finite(lg.data(), lg.size()), "plain logits finite");
            h_plain = fnv1a(lg.data(), lg.size() * sizeof(float));
            std::printf("plain: TEXT_LOGITS_FNV=%016llx n=%zu l[0..3]=%.9g %.9g %.9g %.9g\n", (unsigned long long) h_plain, lg.size(),
                        lg.size() > 3 ? lg[0] : 0.f, lg.size() > 3 ? lg[1] : 0.f, lg.size() > 3 ? lg[2] : 0.f, lg.size() > 3 ? lg[3] : 0.f);
            llama_free(ctx);
        }
        llama_model_free(m_plain);
    }

    // ---- (a)+(b) VL file: MROPE, text decode ----------------------------------------------------------------------------
    llama_model * m_vl = llama_model_load_from_file(vl.c_str(), mp);
    CHECK(m_vl != nullptr, "load %s", vl.c_str());
    if (!m_vl) {
        std::printf("%s: %d failed check(s)\n", g_fail ? "FAIL" : "PASS", g_fail);
        return 1;
    }
    const int rt_vl = (int) llama_model_rope_type(m_vl);
    std::printf("vl: rope_type=%s (%d)\n", rope_name(rt_vl), rt_vl);
    CHECK(rt_vl == LLAMA_ROPE_TYPE_MROPE, "VL file (rope.dimension_sections present) must decode with MROPE, got %s", rope_name(rt_vl));
    const int n_embd = llama_model_n_embd(m_vl);
    {
        llama_context * ctx = make_ctx(m_vl, n_threads);
        CHECK(ctx != nullptr, "context for vl");
        if (ctx) {
            const std::vector<float> lg = decode_text(ctx, m_vl);
            CHECK(!lg.empty() && all_finite(lg.data(), lg.size()), "vl text logits finite");
            const uint64_t h = fnv1a(lg.data(), lg.size() * sizeof(float));
            std::printf("vl: TEXT_LOGITS_FNV=%016llx (%s the plain file's)\n", (unsigned long long) h, h == h_plain ? "EQUAL to" : "differs from");
            llama_free(ctx);
        }
    }

    // ---- (c) projector ----------------------------------------------------------------------------------------------------
    clip_context_params cparams{};
    cparams.use_gpu          = false;
    cparams.device           = nullptr;
    cparams.flash_attn_type  = CLIP_FLASH_ATTN_TYPE_DISABLED;
    cparams.image_min_tokens = -1;
    cparams.image_max_tokens = -1;
    cparams.warmup           = false;
    clip_ctx * ctx_v = nullptr;
    try {
        ctx_v = clip_init(mmproj.c_str(), cparams).ctx_v;
    } catch (const std::exception & e) {
        std::fprintf(stderr, "clip_init threw: %s\n", e.what());
    }
    CHECK(ctx_v != nullptr, "(c) the ling3vl projector must load: %s", mmproj.c_str());
    std::vector<float> img_embd;
    int n_img_tok = 0, nx = 0, ny = 0;
    if (ctx_v) {
        CHECK(clip_n_mmproj_embd(ctx_v) == n_embd, "mmproj width %d vs text n_embd %d", clip_n_mmproj_embd(ctx_v), n_embd);
        clip_image_f32_batch batch;
        clip_image_f32 img;
        img.set_size({32, 32}, false, false);
        std::vector<float> pix((size_t) 32 * 32 * 3);
        for (size_t i = 0; i < pix.size(); ++i) {
            pix[i] = (float) ((i * 37) % 255) / 127.5f - 1.0f;
        }
        img.cpy_buf(pix);
        batch.entries.push_back(img);
        n_img_tok = clip_n_output_tokens(ctx_v, &batch.entries[0]);
        nx = clip_n_output_tokens_x(ctx_v, &batch.entries[0]);
        ny = clip_n_output_tokens_y(ctx_v, &batch.entries[0]);
        std::printf("projector: n_tokens=%d grid=%dx%d width=%d\n", n_img_tok, nx, ny, clip_n_mmproj_embd(ctx_v));
        const int patch = clip_get_hparams(ctx_v)->patch_size;
        const int want  = (32 / patch / 2) * (32 / patch / 2);
        CHECK(n_img_tok == want && nx * ny == want, "token count %d (grid %dx%d), want %d", n_img_tok, nx, ny, want);
        img_embd.resize((size_t) n_img_tok * clip_n_mmproj_embd(ctx_v)); // the encoder writes into a caller-sized buffer (mtmd.cpp does the same)
        const bool ok = clip_image_batch_encode(ctx_v, n_threads, &batch, img_embd);
        CHECK(ok, "clip_image_batch_encode");
        CHECK(img_embd.size() == (size_t) n_img_tok * n_embd, "embedding size %zu want %d", img_embd.size(), n_img_tok * n_embd);
        CHECK(ok && all_finite(img_embd.data(), img_embd.size()), "image embeddings finite");
        if (ok && !img_embd.empty()) {
            std::printf("projector: EMBD_FNV=%016llx\n", (unsigned long long) fnv1a(img_embd.data(), img_embd.size() * sizeof(float)));
        }
        clip_free(ctx_v);
    }

    // ---- (d) image embeddings through the VL text model with 2-D M-RoPE positions ---------------------------------------
    if (!img_embd.empty() && rt_vl == LLAMA_ROPE_TYPE_MROPE && img_embd.size() == (size_t) n_img_tok * n_embd) {
        llama_context * ctx = make_ctx(m_vl, n_threads);
        CHECK(ctx != nullptr, "context for vl (image)");
        if (ctx) {
            llama_batch b = llama_batch_init(n_img_tok, n_embd, 1);
            std::memcpy(b.embd, img_embd.data(), img_embd.size() * sizeof(float));
            std::vector<llama_pos> pos((size_t) n_img_tok * 4);
            for (int i = 0; i < n_img_tok; ++i) {
                pos[i]                 = 0;          // t
                pos[i + n_img_tok]     = i / nx;     // row
                pos[i + n_img_tok * 2] = i % nx;     // col
                pos[i + n_img_tok * 3] = 0;
                b.n_seq_id[i]  = 1;
                b.seq_id[i][0] = 0;
                b.logits[i]    = i == n_img_tok - 1;
            }
            llama_pos * own_pos = b.pos;
            b.pos      = pos.data();
            b.n_tokens = n_img_tok;
            const int rc = llama_decode(ctx, b);
            CHECK(rc == 0, "(d) llama_decode of image embeddings rc=%d", rc);
            if (rc == 0) {
                const float * l = llama_get_logits_ith(ctx, n_img_tok - 1);
                const int n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(m_vl));
                CHECK(all_finite(l, n_vocab), "(d) logits after the image finite");
                std::printf("vl+image: LOGITS_FNV=%016llx\n", (unsigned long long) fnv1a(l, (size_t) n_vocab * sizeof(float)));
            }
            b.pos = own_pos;
            llama_batch_free(b);
            llama_free(ctx);
        }
    }

    llama_model_free(m_vl);
    llama_backend_free();
    std::printf("%s: %d failed check(s)\n", g_fail ? "FAIL" : "PASS", g_fail);
    return g_fail ? 1 : 0;
}

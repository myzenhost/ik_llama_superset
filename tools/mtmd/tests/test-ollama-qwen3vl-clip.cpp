// Ollama qwen3-vl combined blob used as its own mmproj: no_alloc structural check of the clip loader.
//
//   llama-test-ollama-qwen3vl-clip <ollama qwen3vl blob> [reference separate mmproj]
//   (or set LLAMA_TEST_OLLAMA_QWEN3VL_BLOB / LLAMA_TEST_OLLAMA_QWEN3VL_REF; exit 77 when no blob is given or found)
//
// Ollama stores qwen3-vl as ONE GGUF: general.architecture = qwen3vl, the text blocks, and the vision tower under v.*
// with Ollama's own names (v.blk.N.attn_q/k/v split, norm1/norm2, mlp.linear_fc1/2, v.merger.*, v.deepstack_merger.X.*,
// a Conv3D v.patch_embed.weight [16,16,2,out*in], F16 v.pos_embed.weight) and its vision hparams under qwen3vl.vision.*.
// The mtmd loader (clip.cpp) reads llama.cpp-converter mmproj files only, so before the clip-side Ollama read the blob
// does not load as an mmproj at all.
//
// What this checks, with no tensor data read (clip_context_params.no_alloc = true, no warmup, CPU only):
//   1. the blob builds a vision clip context;
//   2. projector type = qwen3vl_merger and the hparams equal the values derived from the BLOB's OWN metadata
//      (image_size = sqrt(n_pos) * patch_size, n_embd, n_ff, n_head, n_layer, projection_dim, n_merge, eps, mean/std);
//   3. the loader fetched exactly the tensor count the Ollama -> clip mapping implies:
//         (#v.* + #mm.* in the blob) - 4 * n_layer (attn_k/attn_v weight+bias folded into attn_qkv) + 1 (patch split);
//   4. clip_n_mmproj_embd = text n_embd * (1 + #deepstack layers);
//   5. LLAMA_OLLAMA_COMPAT=0 turns the read off (the blob is then refused again);
//   6. with a reference mmproj (a converter-made one of the same model), every number above matches it, and the weight
//      byte count differs only by the fused qkv being F32 here (n_layer * 3 * n_embd^2 * (4 - sizeof(ref qkv type))).
// It does NOT check numerics: no tensor bytes are read. The end-to-end image check is a separate run.
#include "clip.h"
// clip_image_f32 etc. are only forward-declared in clip.h; clip.cpp pairs clip.h with clip-impl.h, same here.
#include "clip-impl.h"
#include "clip-model.h" // clip_hparams + clip_get_hparams
#include "ggml.h"
#include "gguf.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

static int g_fail = 0;
#define CHECK(cond, ...) do { if (!(cond)) { g_fail++; std::printf("FAIL %s:%d  ", __FILE__, __LINE__); std::printf(__VA_ARGS__); std::printf("\n"); } } while (0)

static int g_n_tensors_logged = -1;

static void probe_log(enum ggml_log_level level, const char * text, void * /*user_data*/) {
    // load_tensors() reports how many tensors it fetched; with no_alloc it says so at DEBUG level
    const char * key = "skipping tensor data loading (";
    if (const char * p = std::strstr(text, key)) {
        g_n_tensors_logged = std::atoi(p + std::strlen(key));
    }
    if (level >= GGML_LOG_LEVEL_INFO) {
        std::fputs(text, stderr);
    }
}

struct probe_result {
    bool          ok = false;
    std::string   proj;
    clip_hparams  hp{};
    int           n_mmproj_embd = 0;
    int           n_tensors = -1;
    size_t        weight_bytes = 0;
};

static probe_result probe(const char * path) {
    probe_result res;
    g_n_tensors_logged = -1;

    clip_context_params p{};
    p.use_gpu          = false;
    p.device           = nullptr;
    p.flash_attn_type  = CLIP_FLASH_ATTN_TYPE_DISABLED;
    p.image_min_tokens = -1;
    p.image_max_tokens = -1;
    p.warmup           = false;
    p.no_alloc         = true;

    clip_init_result r = clip_init(path, p);
    if (r.ctx_v) {
        res.ok            = true;
        res.proj          = PROJECTOR_TYPE_NAMES[clip_get_projector_type(r.ctx_v)];
        res.hp            = *clip_get_hparams(r.ctx_v);
        res.n_mmproj_embd = clip_n_mmproj_embd(r.ctx_v);
        res.n_tensors     = g_n_tensors_logged;
        for (const auto & [dev, sz] : clip_get_mem_usage(r.ctx_v)) {
            (void) dev;
            res.weight_bytes += sz;
        }
    }
    clip_free(r.ctx_v);
    clip_free(r.ctx_a);
    clip_free(r.ctx_gen_a);
    return res;
}

static void print_probe(const char * label, const probe_result & r) {
    std::printf("[%s] loaded=%d proj=%s image_size=%d patch_size=%d n_embd=%d n_ff=%d n_head=%d n_layer=%d "
                "projection_dim=%d n_merge=%d eps=%g mean=(%g,%g,%g) std=(%g,%g,%g) n_mmproj_embd=%d n_tensors=%d weight_bytes=%zu\n",
                label, (int) r.ok, r.proj.c_str(), r.hp.image_size, r.hp.patch_size, r.hp.n_embd, r.hp.n_ff, r.hp.n_head,
                r.hp.n_layer, r.hp.projection_dim, r.hp.n_merge, (double) r.hp.eps,
                r.hp.image_mean[0], r.hp.image_mean[1], r.hp.image_mean[2], r.hp.image_std[0], r.hp.image_std[1], r.hp.image_std[2],
                r.n_mmproj_embd, r.n_tensors, r.weight_bytes);
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

// expected values read straight from the blob's own GGUF header (independent of clip.cpp)
struct blob_expect {
    std::string arch;
    int image_size = 0, patch_size = 0, n_embd = 0, n_ff = 0, n_head = 0, n_layer = 0, projection_dim = 0, n_merge = 0;
    int n_deepstack = 0, n_vis_tensors = 0;
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
        const int64_t id = gguf_find_key(g, (a + ".vision.deepstack_visual_indexes").c_str());
        e.n_deepstack = id < 0 ? 0 : (int) gguf_get_arr_n(g, id);
    }
    if (ggml_tensor * pe = ggml_get_tensor(meta, "v.pos_embed.weight")) {
        e.image_size = (int) std::lround(std::sqrt((double) pe->ne[1])) * e.patch_size;
    }
    if (ggml_tensor * fc1 = ggml_get_tensor(meta, "v.blk.0.mlp.linear_fc1.weight")) {
        e.n_ff = (int) fc1->ne[1];
    }
    for (int64_t i = 0; i < gguf_get_n_tensors(g); ++i) {
        const char * n = gguf_get_tensor_name(g, i);
        if (std::strncmp(n, "v.", 2) == 0 || std::strncmp(n, "mm.", 3) == 0) e.n_vis_tensors++;
    }
    gguf_free(g);
    ggml_free(meta);
    return true;
}

static ggml_type ref_qkv_type(const char * path) {
    gguf_init_params ip = { /*no_alloc*/ true, nullptr };
    gguf_context * g = gguf_init_from_file(path, ip);
    if (!g) return GGML_TYPE_COUNT;
    const int64_t id = gguf_find_tensor(g, "v.blk.0.attn_qkv.weight");
    const ggml_type t = id < 0 ? GGML_TYPE_COUNT : gguf_get_tensor_type(g, id);
    gguf_free(g);
    return t;
}

int main(int argc, char ** argv) {
    const char * blob = argc > 1 ? argv[1] : std::getenv("LLAMA_TEST_OLLAMA_QWEN3VL_BLOB");
    const char * ref  = argc > 2 ? argv[2] : std::getenv("LLAMA_TEST_OLLAMA_QWEN3VL_REF");
    if (!file_exists(blob)) {
        std::printf("SKIP: no Ollama qwen3vl blob given (argv[1] or LLAMA_TEST_OLLAMA_QWEN3VL_BLOB)\n");
        return 77;
    }
    g_logger_state.log_callback = probe_log;
    g_logger_state.log_callback_user_data = nullptr;

    blob_expect e;
    if (!read_blob_expect(blob, e)) {
        std::printf("FAIL: cannot read the GGUF header of %s\n", blob);
        return 1;
    }
    // qwen35 / qwen35moe (Ollama qwen3.5, qwen3.6) hold the same vision tower with an EMPTY deepstack list: the same
    // arms apply with n_deepstack = 0 (ollama-w2-1027; the qwen3.6:27b blob sha256-83c54730... is the real specimen)
    if (e.arch != "qwen3vl" && e.arch != "qwen3vlmoe" && e.arch != "qwen35" && e.arch != "qwen35moe") {
        std::printf("SKIP: %s is general.architecture=%s, not an Ollama qwen3vl / qwen35 blob\n", blob, e.arch.c_str());
        return 77;
    }
    const int exp_tensors = e.n_vis_tensors - 4 * e.n_layer + 1;
    std::printf("[expect from blob header] arch=%s image_size=%d patch_size=%d n_embd=%d n_ff=%d n_head=%d n_layer=%d "
                "projection_dim=%d n_merge=%d eps=%g n_deepstack=%d v+mm tensors=%d -> clip tensors=%d n_mmproj_embd=%d\n",
                e.arch.c_str(), e.image_size, e.patch_size, e.n_embd, e.n_ff, e.n_head, e.n_layer, e.projection_dim, e.n_merge,
                (double) e.eps, e.n_deepstack, e.n_vis_tensors, exp_tensors, e.projection_dim * (1 + e.n_deepstack));

    // arm 1: the blob as its own mmproj
    const probe_result b = probe(blob);
    print_probe("blob", b);
    CHECK(b.ok, "the blob did not build a vision clip context");
    if (b.ok) {
        CHECK(b.proj == "qwen3vl_merger",                 "projector %s, expected qwen3vl_merger", b.proj.c_str());
        CHECK(b.hp.image_size     == e.image_size,       "image_size %d != %d", b.hp.image_size, e.image_size);
        CHECK(b.hp.patch_size     == e.patch_size,       "patch_size %d != %d", b.hp.patch_size, e.patch_size);
        CHECK(b.hp.n_embd         == e.n_embd,           "n_embd %d != %d", b.hp.n_embd, e.n_embd);
        CHECK(b.hp.n_ff           == e.n_ff,             "n_ff %d != %d", b.hp.n_ff, e.n_ff);
        CHECK(b.hp.n_head         == e.n_head,           "n_head %d != %d", b.hp.n_head, e.n_head);
        CHECK(b.hp.n_layer        == e.n_layer,          "n_layer %d != %d", b.hp.n_layer, e.n_layer);
        CHECK(b.hp.projection_dim == e.projection_dim,   "projection_dim %d != %d", b.hp.projection_dim, e.projection_dim);
        CHECK(b.hp.n_merge        == e.n_merge,          "n_merge %d != %d", b.hp.n_merge, e.n_merge);
        // a blob without vision.attention.layer_norm_epsilon (the qwen3.6:27b blob has none) reads the loader default 1e-6
        // (clip.cpp:1492); e.eps is -1 for "key absent". MEASURED 2026-10-27: loaded eps=1e-06 on that blob.
        const float exp_eps = e.eps < 0.0f ? 1e-6f : e.eps;
        CHECK(std::fabs(b.hp.eps - exp_eps) < 1e-12f,    "eps %g != %g", (double) b.hp.eps, (double) exp_eps);
        CHECK(b.n_tensors         == exp_tensors,        "tensor count %d != %d", b.n_tensors, exp_tensors);
        CHECK(b.n_mmproj_embd     == e.projection_dim * (1 + e.n_deepstack),
              "n_mmproj_embd %d != %d", b.n_mmproj_embd, e.projection_dim * (1 + e.n_deepstack));
    }

    // arm 2: the opt-out switch turns the read off
    set_env("LLAMA_OLLAMA_COMPAT", "0");
    const probe_result off = probe(blob);
    set_env("LLAMA_OLLAMA_COMPAT", nullptr);
    print_probe("blob LLAMA_OLLAMA_COMPAT=0", off);
    CHECK(!off.ok, "LLAMA_OLLAMA_COMPAT=0 should leave the blob unreadable as an mmproj");

    // arm 3: a converter-made mmproj of the same model, through the ordinary path
    if (file_exists(ref)) {
        const probe_result r = probe(ref);
        print_probe("reference mmproj", r);
        CHECK(r.ok, "the reference mmproj did not load");
        if (r.ok && b.ok) {
            CHECK(b.proj == r.proj,                               "projector differs from the reference");
            CHECK(b.hp.image_size     == r.hp.image_size,         "image_size differs from the reference");
            CHECK(b.hp.patch_size     == r.hp.patch_size,         "patch_size differs from the reference");
            CHECK(b.hp.n_embd         == r.hp.n_embd,             "n_embd differs from the reference");
            CHECK(b.hp.n_ff           == r.hp.n_ff,               "n_ff differs from the reference");
            CHECK(b.hp.n_head         == r.hp.n_head,             "n_head differs from the reference");
            CHECK(b.hp.n_layer        == r.hp.n_layer,            "n_layer differs from the reference");
            CHECK(b.hp.projection_dim == r.hp.projection_dim,     "projection_dim differs from the reference");
            CHECK(b.hp.n_merge        == r.hp.n_merge,            "n_merge differs from the reference");
            CHECK(b.hp.image_min_pixels == r.hp.image_min_pixels, "image_min_pixels differs from the reference");
            CHECK(b.hp.image_max_pixels == r.hp.image_max_pixels, "image_max_pixels differs from the reference");
            CHECK(b.hp.warmup_image_size == r.hp.warmup_image_size, "warmup_image_size differs from the reference");
            CHECK(b.hp.ffn_op == r.hp.ffn_op,                     "ffn_op differs from the reference");
            for (int i = 0; i < 3; ++i) {
                CHECK(b.hp.image_mean[i] == r.hp.image_mean[i] && b.hp.image_std[i] == r.hp.image_std[i], "mean/std[%d] differs", i);
            }
            CHECK(b.n_mmproj_embd == r.n_mmproj_embd,             "n_mmproj_embd differs from the reference");
            CHECK(b.n_tensors     == r.n_tensors,                 "tensor count differs from the reference");
            const ggml_type qt = ref_qkv_type(ref);
            if (qt != GGML_TYPE_COUNT && !ggml_is_quantized(qt)) {
                const long long delta = (long long) b.weight_bytes - (long long) r.weight_bytes;
                const long long want  = (long long) r.hp.n_layer * 3LL * r.hp.n_embd * r.hp.n_embd *
                                        ((long long) sizeof(float) - (long long) ggml_type_size(qt));
                std::printf("[weights] blob - reference = %lld bytes, expected %lld (fused qkv F32 vs reference %s)\n",
                            delta, want, ggml_type_name(qt));
                CHECK(delta == want, "weight byte delta %lld != %lld", delta, want);
            }
        }
    } else {
        std::printf("[reference mmproj] not given; reference arm skipped\n");
    }

    // arm 4 (optional): a qwen3vl_merger mmproj with separate attn_q/k/v and no fused attn_qkv. The qwen3vl graph
    // has only the fused path, so such a file must be refused at load rather than dereference a null qkv_w at encode.
    const char * unfused = argc > 3 ? argv[3] : std::getenv("LLAMA_TEST_QWEN3VL_UNFUSED_MMPROJ");
    if (file_exists(unfused)) {
        const probe_result u = probe(unfused);
        print_probe("unfused qkv mmproj", u);
        CHECK(!u.ok, "a qwen3vl mmproj without fused attn_qkv was accepted (it faults at encode)");
    } else {
        std::printf("[unfused qkv mmproj] not given; arm skipped\n");
    }

    std::printf("\n[RESULT] %s (%d failed check%s)\n", g_fail == 0 ? "PASS" : "FAIL", g_fail, g_fail == 1 ? "" : "s");
    return g_fail == 0 ? 0 : 1;
}

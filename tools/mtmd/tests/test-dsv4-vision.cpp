// DeepSeek-V4 / V4.1 vision (deepseek4v / deepseek41 projectors) end-to-end check on a RANDOM-WEIGHT mmproj.
//
//   llama-test-dsv4-vision --mmproj <tiny mmproj from make_tiny_dsv4_mmproj.py> [--text <text gguf>] [--mmproj-peer <other variant>]
//
// What it proves (and what it does not):
//   (a) the projector loads                              -> mtmd_init_from_file() succeeds
//   (b) the graph builds and runs on CPU                 -> mtmd_encode_chunk() returns 0, embeddings are finite
//   (c) shapes / token counts match ik_llama's arithmetic -> n_tokens of the chunk == ik's clip_n_output_tokens() applied to the
//       size ik's solver picks (tests/dsv4-ik-reference.h holds ik's solver VERBATIM), and the encoder produced exactly that
//       many rows (clip_encode aborts otherwise)
//   (d) block layout                                     -> START / NEWLINE / END (and PAD for deepseek4v) rows sit where the
//       reference layout puts them; the grid rows are the aligner output
// It does NOT prove image understanding: the weights are random. No trained DeepSeek-V4.x vision checkpoint exists locally.
//
// SOURCE-READ: ik_llama.cpp 60c74d9c4 / 248e77729 examples/mtmd/clip.cpp, mtmd.cpp (MIT)
#include "mtmd.h"
#include "mtmd-helper.h"
#include "llama.h"
#include "gguf.h"
#include "ggml.h"

#include "dsv4-ik-reference.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

static int g_fail = 0;
#define CHECK(cond, ...) do { if (!(cond)) { g_fail++; std::fprintf(stderr, "FAIL %s:%d  ", __FILE__, __LINE__); std::fprintf(stderr, __VA_ARGS__); std::fprintf(stderr, "\n"); } } while (0)

struct mmproj_info {
    std::string proj_type;
    int patch = 0, merge = 0, n_out = 0, min_pixels = 0, max_n_token = -1;
    std::vector<float> newline, start, end, pad;
};

static int kv_int(const gguf_context * g, const char * key, int def) {
    const int64_t k = gguf_find_key(g, key);
    if (k < 0) return def;
    return (int) gguf_get_val_u32(g, k);
}

static bool read_tensor_f32(const gguf_context * g, ggml_context * data, const char * name, std::vector<float> & out) {
    if (gguf_find_tensor(g, name) < 0) return false;
    ggml_tensor * t = ggml_get_tensor(data, name);
    if (!t || t->type != GGML_TYPE_F32) return false;
    out.resize(ggml_nelements(t));
    std::memcpy(out.data(), t->data, ggml_nbytes(t));
    return true;
}

static bool load_info(const char * path, mmproj_info & mi) {
    ggml_context * data = nullptr;
    gguf_init_params p = { /*no_alloc*/ false, &data };
    gguf_context * g = gguf_init_from_file(path, p);
    if (!g) return false;
    const int64_t kp = gguf_find_key(g, "clip.projector_type");
    mi.proj_type   = kp >= 0 ? gguf_get_val_str(g, kp) : "";
    mi.patch       = kv_int(g, "clip.vision.patch_size", 0);
    mi.merge       = kv_int(g, "clip.vision.projector.scale_factor", 0);
    mi.n_out       = kv_int(g, "clip.vision.projection_dim", 0);
    mi.min_pixels  = kv_int(g, "clip.vision.image_min_pixels", 0);
    mi.max_n_token = kv_int(g, "clip.vision.dsv4_max_n_token", -1);
    bool ok = read_tensor_f32(g, data, "v.image_newline", mi.newline)
           && read_tensor_f32(g, data, "v.token_embd.img_start", mi.start)
           && read_tensor_f32(g, data, "v.token_embd.img_end", mi.end);
    read_tensor_f32(g, data, "v.token_embd.img_pad", mi.pad); // optional for deepseek41
    gguf_free(g);
    ggml_free(data);
    return ok;
}

static std::vector<unsigned char> make_image(int w, int h) {
    std::vector<unsigned char> rgb((size_t) w * h * 3);
    for (int y = 0; y < h; y++) {
        for (int x = 0; x < w; x++) {
            unsigned char * px = &rgb[((size_t) y * w + x) * 3];
            px[0] = (unsigned char) ((x * 255) / std::max(1, w - 1));
            px[1] = (unsigned char) ((y * 255) / std::max(1, h - 1));
            px[2] = (unsigned char) (((x ^ y) * 7) & 255);
        }
    }
    return rgb;
}

static bool row_eq(const float * a, const std::vector<float> & b) {
    return std::memcmp(a, b.data(), b.size() * sizeof(float)) == 0;
}

struct run_out {
    bool ok = false;
    size_t n_tokens = 0;
    size_t n_before = 0;   // tokens of all chunks before the image chunk
    std::vector<float> embd;
};

// tokenize `prompt` (one media marker) with an RGB w x h image, encode the image chunk
static run_out run_case(mtmd_context * ctx, int n_embd, const std::string & prompt, int w, int h, bool add_special) {
    run_out r;
    std::vector<unsigned char> rgb = make_image(w, h);
    mtmd_bitmap * bmp = mtmd_bitmap_init(w, h, rgb.data());
    mtmd_input_chunks * chunks = mtmd_input_chunks_init();
    mtmd_input_text txt = { prompt.c_str(), prompt.size(), add_special, true };
    const mtmd_bitmap * bmps[1] = { bmp };
    const int32_t rc = mtmd_tokenize(ctx, chunks, &txt, bmps, 1);
    if (rc != 0) {
        std::fprintf(stderr, "mtmd_tokenize rc=%d\n", rc);
        mtmd_input_chunks_free(chunks);
        mtmd_bitmap_free(bmp);
        return r;
    }
    const mtmd_input_chunk * img = nullptr;
    for (size_t i = 0; i < mtmd_input_chunks_size(chunks); i++) {
        const mtmd_input_chunk * c = mtmd_input_chunks_get(chunks, i);
        if (mtmd_input_chunk_get_type(c) == MTMD_INPUT_CHUNK_TYPE_IMAGE) {
            img = c;
            break;
        }
        r.n_before += mtmd_input_chunk_get_n_tokens(c);
    }
    if (!img) {
        mtmd_input_chunks_free(chunks);
        mtmd_bitmap_free(bmp);
        return r;
    }
    r.n_tokens = mtmd_input_chunk_get_n_tokens(img);
    if (mtmd_encode_chunk(ctx, img) == 0) {
        const float * e = mtmd_get_output_embd(ctx);
        r.embd.assign(e, e + r.n_tokens * (size_t) n_embd);
        r.ok = true;
    }
    mtmd_input_chunks_free(chunks);
    mtmd_bitmap_free(bmp);
    return r;
}

int main(int argc, char ** argv) {
    std::string mmproj, text, peer;
    for (int i = 1; i < argc; i++) {
        if (!std::strcmp(argv[i], "--mmproj") && i + 1 < argc) mmproj = argv[++i];
        else if (!std::strcmp(argv[i], "--text") && i + 1 < argc) text = argv[++i];
        else if (!std::strcmp(argv[i], "--mmproj-peer") && i + 1 < argc) peer = argv[++i];
    }
    if (mmproj.empty()) {
        std::fprintf(stderr, "usage: %s --mmproj F [--text GGUF] [--mmproj-peer F]\n", argv[0]);
        return 2;
    }

    mmproj_info mi;
    if (!load_info(mmproj.c_str(), mi)) {
        std::fprintf(stderr, "cannot read %s\n", mmproj.c_str());
        return 2;
    }
    // "deepseek41v" is the canonical name (upstream); "deepseek41" is the fork's legacy spelling, still loadable via
    // PROJECTOR_TYPE_LEGACY_ALIASES -- accept both so files written before and after the rename are both checked.
    const bool is41 = mi.proj_type == "deepseek41v" || mi.proj_type == "deepseek41";
    const bool is4v = mi.proj_type == "deepseek4v";
    if (!is41 && !is4v) {
        std::fprintf(stderr, "projector_type is '%s', expected deepseek41v (or legacy deepseek41) or deepseek4v\n", mi.proj_type.c_str());
        return 2;
    }
    std::printf("projector=%s patch=%d merge=%d n_out=%d min_pixels=%d max_n_token(gguf)=%d\n",
                mi.proj_type.c_str(), mi.patch, mi.merge, mi.n_out, mi.min_pixels, mi.max_n_token);

    llama_model * model = nullptr;
    if (!text.empty()) {
        llama_backend_init();
        llama_model_params mp = llama_model_default_params();
        mp.n_gpu_layers = 0;
        model = llama_model_load_from_file(text.c_str(), mp);
        if (!model) {
            std::fprintf(stderr, "cannot load text model %s\n", text.c_str());
            return 2;
        }
    }

    mtmd_context_params cp = mtmd_context_params_default();
    cp.use_gpu = false;
    cp.n_threads = 4;
    cp.print_timings = false;
    cp.warmup = true;
    mtmd_context * ctx = mtmd_init_from_file(mmproj.c_str(), model, cp);
    CHECK(ctx != nullptr, "(a) mtmd_init_from_file failed for %s", mmproj.c_str());
    if (!ctx) {
        std::printf("RESULT: FAIL (projector did not load)\n");
        return 1;
    }
    std::printf("(a) projector loaded\n");

    // the solver parameters ik would use for this file
    const int ik_max_n_token  = is41 ? (mi.max_n_token > 0 ? mi.max_n_token : 1024) : 384;
    const int ik_max_wh_ratio = is41 ? 0 : 8;
    const int patch = mi.patch, merge = mi.merge, cell = patch * merge;

    // image sizes: aligned, unaligned, tall, wide, tiny, extreme aspect (the 4v stretch branch), large (solver kicks in)
    const int sizes[][2] = {
        { 64, 64 }, { 48, 80 }, { 80, 48 }, { 37, 53 }, { 100, 20 }, { 20, 100 }, { 8, 8 }, { 5, 5 },
        { 256, 32 }, { 33, 330 }, { 160, 160 }, { 200, 120 },
    };
    const std::string marker = mtmd_default_marker();
    const char * prefixes[] = { "", "a", "ab", "abc", "abcd", "hello world " };

    int n_cases = 0;
    for (const auto & sz : sizes) {
        for (const char * pre : prefixes) {
          // add_special = true makes mtmd prepend BOS (the vocabulary has add_bos); that shifts every position by one, which
          // the deepseek4v lead pad has to account for
          for (int sp = 0; sp < 2; sp++) {
            const bool add_special = sp == 1;
            // without a text vocabulary the bare marker is the only prompt that can be tokenized
            if (!model && (pre[0] != '\0' || add_special)) continue;
            if (!is4v && (add_special || (pre[0] != '\0' && std::strlen(pre) > 1))) continue; // the prefix / BOS only matter for deepseek4v (lead pad)
            const int w = sz[0], h = sz[1];
            run_out r = run_case(ctx, mi.n_out, std::string(pre) + marker, w, h, add_special);
            CHECK(r.ok, "(b) tokenize/encode failed for %dx%d prefix '%s'", w, h, pre);
            if (!r.ok) continue;
            n_cases++;

            const ik_ref_result ref = is41
                ? ik_ref_deepseek41(w, h, patch, merge, ik_max_n_token, ik_max_wh_ratio, mi.min_pixels)
                : ik_ref_deepseek4v(w, h, patch, merge, ik_max_n_token, ik_max_wh_ratio, mi.min_pixels);
            CHECK(!ref.assert_failed, "ik's own solver assertion would have fired for %dx%d", w, h);

            // 4v: the lead pad that puts START on position 3 (mod 4), from the tokens that really precede the image (BOS included)
            const int lead_pad = is4v ? (4 - 1 - (int) (r.n_before % 4)) : 0;
            const int ref_tokens = is41
                ? ik_ref_n_tokens_deepseek41(ref.best_width, ref.best_height, patch, merge)
                : ik_ref_n_tokens_deepseek4v(ref.best_width, ref.best_height, patch, merge, lead_pad);
            CHECK((int) r.n_tokens == ref_tokens, "(c) %dx%d prefix '%s': n_tokens %zu != ik reference %d (ik resized to %dx%d)",
                  w, h, pre, r.n_tokens, ref_tokens, ref.best_width, ref.best_height);

            // finite
            bool finite = true;
            for (float v : r.embd) { if (!std::isfinite(v)) { finite = false; break; } }
            CHECK(finite, "(b) non-finite embedding for %dx%d", w, h);

            // layout
            const int n_llm_w = CLIP_ALIGN(ref.best_width,  cell) / cell;
            const int n_llm_h = CLIP_ALIGN(ref.best_height, cell) / cell;
            const size_t n = r.n_tokens;
            const float * E = r.embd.data();
            auto row = [&](size_t i) { return E + i * (size_t) mi.n_out; };
            if (is41) {
                CHECK(row_eq(row(0), mi.start), "(d) %dx%d: row 0 is not START", w, h);
                CHECK(row_eq(row(n - 1), mi.end), "(d) %dx%d: last row is not END", w, h);
                for (int y = 0; y < n_llm_h; y++) {
                    const size_t i = 1 + (size_t) y * (n_llm_w + 1) + n_llm_w;
                    CHECK(row_eq(row(i), mi.newline), "(d) %dx%d: row %zu is not NEWLINE (grid row %d)", w, h, i, y);
                }
                CHECK(n == (size_t) n_llm_h * (n_llm_w + 1) + 2, "(d) %dx%d: n_tokens %zu != h*(w+1)+2 = %d", w, h, n, n_llm_h * (n_llm_w + 1) + 2);
            } else {
                // deepseek4v: [PAD]*lead [START] <interleaved rows> [PAD]*tail [END]
                CHECK(!mi.pad.empty(), "(d) deepseek4v mmproj has no img_pad tensor");
                bool lead_ok = true;
                for (int i = 0; i < lead_pad; i++) lead_ok = lead_ok && row_eq(row(i), mi.pad);
                CHECK(lead_ok, "(d) %dx%d prefix '%s': first %d rows are not PAD", w, h, pre, lead_pad);
                CHECK(row_eq(row(lead_pad), mi.start), "(d) %dx%d prefix '%s': row %d is not START", w, h, pre, lead_pad);
                CHECK(row_eq(row(n - 1), mi.end), "(d) %dx%d: last row is not END", w, h);
                // START must land on position == 3 (mod 4): the compressor window boundary
                CHECK(((r.n_before + lead_pad) % 4) == 3, "(d) %dx%d prefix '%s': START at position %zu is not = 3 mod 4", w, h, pre, r.n_before + lead_pad);
                int n_nl = 0;
                for (size_t i = (size_t) lead_pad + 1; i + 1 < n; i++) if (row_eq(row(i), mi.newline)) n_nl++;
                CHECK(n_nl == n_llm_h, "(d) %dx%d: %d NEWLINE rows, expected %d", w, h, n_nl, n_llm_h);
            }
            std::printf("  %4dx%-4d prefix '%-12s'%s -> resized %3dx%-3d grid %2dx%-2d tokens %4zu (ik ref %4d)%s\n",
                        w, h, pre, add_special ? " +BOS" : "     ", ref.best_width, ref.best_height, n_llm_w, n_llm_h, r.n_tokens, ref_tokens, ref.stretch ? " [stretch]" : "");
          }
        }
    }
    std::printf("(b)(c)(d) %d cases run\n", n_cases);

    // deepseek4v vs deepseek41 built from the same seed: the grid rows of the 4v block, read back in the reference
    // "two rows interleaved column-wise" order, must equal the 41 grid rows (same ViT + aligner weights, same input).
    if (!peer.empty() && is4v) {
        mmproj_info pi;
        if (!load_info(peer.c_str(), pi) || (pi.proj_type != "deepseek41v" && pi.proj_type != "deepseek41")) {
            CHECK(false, "peer mmproj must be a deepseek41 file");
        } else {
            mtmd_context * ctx41 = mtmd_init_from_file(peer.c_str(), model, cp);
            CHECK(ctx41 != nullptr, "peer deepseek41 mmproj did not load");
            if (ctx41) {
                const int wh[][2] = { { 64, 64 }, { 48, 80 }, { 80, 48 }, { 40, 24 } }; // multiples of the cell: both variants keep the size
                for (const auto & s : wh) {
                    const int w = s[0], h = s[1];
                    for (const char * pre : prefixes) {
                        if (!model && pre[0] != '\0') continue;
                        run_out a = run_case(ctx,   mi.n_out, std::string(pre) + marker, w, h, false);
                        run_out b = run_case(ctx41, mi.n_out, marker, w, h, false);
                        CHECK(a.ok && b.ok, "peer cross-check encode failed %dx%d", w, h);
                        if (!(a.ok && b.ok)) continue;
                        const int n_llm_w = w / cell, n_llm_h = h / cell;
                        const int lead = 4 - 1 - (int) (a.n_before % 4);
                        const int rows = n_llm_h + (n_llm_h % 2);
                        // independent statement of the layout: pairs of rows, column by column, newline as the last column
                        std::vector<int> expect; // index into the 41 grid, -1 = newline, -2 = pad
                        for (int g = 0; g < rows / 2; g++) {
                            for (int c = 0; c <= n_llm_w; c++) {
                                for (int k = 0; k < 2; k++) {
                                    const int rr = 2 * g + k;
                                    if (rr >= n_llm_h)      expect.push_back(-2);
                                    else if (c == n_llm_w)  expect.push_back(-1);
                                    else                    expect.push_back(rr * n_llm_w + c);
                                }
                            }
                        }
                        const int tail = (rows / 2 * (n_llm_w + 1)) % 2 * 2;
                        const size_t expect_n = (size_t) lead + 1 + expect.size() + tail + 1;
                        CHECK(a.n_tokens == expect_n, "peer: 4v n_tokens %zu != %zu", a.n_tokens, expect_n);
                        if (a.n_tokens != expect_n) continue;
                        const float * A = a.embd.data();
                        bool grid_ok = true;
                        for (size_t j = 0; j < expect.size() && grid_ok; j++) {
                            const float * ra = A + ((size_t) lead + 1 + j) * mi.n_out;
                            if (expect[j] >= 0) {
                                // 41 block: grid cell (y, x) is at row 1 + y*(w+1) + x
                                const int y = expect[j] / n_llm_w, x = expect[j] % n_llm_w;
                                const float * rb = b.embd.data() + (1 + (size_t) y * (n_llm_w + 1) + x) * mi.n_out;
                                grid_ok = std::memcmp(ra, rb, mi.n_out * sizeof(float)) == 0;
                            } else if (expect[j] == -1) {
                                grid_ok = row_eq(ra, mi.newline);
                            } else {
                                grid_ok = row_eq(ra, mi.pad);
                            }
                        }
                        CHECK(grid_ok, "peer: 4v grid rows differ from the 41 grid rows for %dx%d prefix '%s'", w, h, pre);
                    }
                }
                std::printf("(d) deepseek4v block == deepseek41 grid reordered per the reference layout: checked\n");
                mtmd_free(ctx41);
            }
        }
    }

    mtmd_free(ctx);
    if (model) llama_model_free(model);
    std::printf("RESULT: %s (%d failed checks)\n", g_fail ? "FAIL" : "PASS", g_fail);
    return g_fail ? 1 : 0;
}

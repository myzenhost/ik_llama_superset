// mtmd_tokenize with more media markers in the text than bitmaps: the extra markers must stay literal text (with a warning)
// instead of failing the whole prompt; fewer markers than bitmaps must still fail (a bitmap would be dropped).
// Port of ik_llama.cpp ed86eedaf (#2639). Fail-before: case 1 returned an error (the constructor threw "exceeds").
//
// usage: llama-test-mtmd-stray-marker <text-model.gguf> <mmproj.gguf>
// The text model is loaded vocab-only; mtmd_tokenize runs image preprocessing but never the vision encoder.
// GATE-CLAIM-OK: unit test, no resource claims.
#include "llama.h"
#include "mtmd.h"

#include <cstdio>
#include <string>
#include <vector>

static int fails = 0;
#define CHECK(cond, ...) do { if (!(cond)) { fprintf(stderr, "FAIL line %d: ", __LINE__); fprintf(stderr, __VA_ARGS__); fprintf(stderr, "\n"); fails++; } else { printf("ok   " #cond "\n"); } } while (0)

struct tok_result {
    int32_t rc = -1;
    std::vector<mtmd_input_chunk_type> types;
    std::vector<size_t> n_tokens;
};

static tok_result tokenize(mtmd_context * ctx, const std::string & text, std::vector<const mtmd_bitmap *> bms) {
    tok_result r;
    mtmd_input_text in;
    in.text          = text.c_str();
    in.text_len      = text.size();
    in.add_special   = false;
    in.parse_special = true;
    mtmd_input_chunks * chunks = mtmd_input_chunks_init();
    r.rc = mtmd_tokenize(ctx, chunks, &in, bms.data(), bms.size());
    if (r.rc == 0) {
        for (size_t i = 0; i < mtmd_input_chunks_size(chunks); i++) {
            const mtmd_input_chunk * c = mtmd_input_chunks_get(chunks, i);
            r.types.push_back(mtmd_input_chunk_get_type(c));
            r.n_tokens.push_back(mtmd_input_chunk_get_n_tokens(c));
        }
    }
    mtmd_input_chunks_free(chunks);
    return r;
}

static size_t count_type(const tok_result & r, mtmd_input_chunk_type t) {
    size_t n = 0;
    for (auto x : r.types) n += x == t;
    return n;
}

int main(int argc, char ** argv) {
    if (argc < 3) {
        fprintf(stderr, "usage: %s <text-model.gguf> <mmproj.gguf>\n", argv[0]);
        return 2;
    }
    llama_backend_init();
    llama_model_params mp = llama_model_default_params();
    mp.vocab_only   = true;
    mp.n_gpu_layers = 0;
    llama_model * model = llama_model_load_from_file(argv[1], mp);
    if (!model) {
        fprintf(stderr, "could not load text model %s\n", argv[1]);
        return 2;
    }
    mtmd_context_params cp = mtmd_context_params_default();
    cp.use_gpu       = false;
    cp.warmup        = false;
    cp.print_timings = false;
    cp.n_threads     = 4;
    mtmd_context * ctx = mtmd_init_from_file(argv[2], model, cp);
    if (!ctx) {
        fprintf(stderr, "could not load mmproj %s\n", argv[2]);
        llama_model_free(model);
        return 2;
    }
    const std::string marker = mtmd_get_marker(ctx);

    // a flat grey 64x64 RGB image: enough for the preprocessor, the encoder never runs
    std::vector<unsigned char> px(64 * 64 * 3, 128);
    mtmd_bitmap * bm1 = mtmd_bitmap_init(64, 64, px.data());
    mtmd_bitmap * bm2 = mtmd_bitmap_init(64, 64, px.data());

    // control: one marker, one bitmap -> text, image, text
    {
        tok_result r = tokenize(ctx, "describe " + marker + " please", { bm1 });
        CHECK(r.rc == 0, "1 marker / 1 bitmap: rc %d", r.rc);
        CHECK(count_type(r, MTMD_INPUT_CHUNK_TYPE_IMAGE) == 1, "1 marker / 1 bitmap: want 1 image chunk");
    }
    // the case: two markers, one bitmap -> succeeds, one image chunk, the second marker stays text
    size_t n_text_tokens_stray = 0;
    {
        tok_result r = tokenize(ctx, "a " + marker + " b " + marker + " c", { bm1 });
        CHECK(r.rc == 0, "2 markers / 1 bitmap must tokenize (was an error before the port), rc %d", r.rc);
        CHECK(count_type(r, MTMD_INPUT_CHUNK_TYPE_IMAGE) == 1, "2 markers / 1 bitmap: want exactly 1 image chunk, got %zu", count_type(r, MTMD_INPUT_CHUNK_TYPE_IMAGE));
        for (size_t i = 0; i < r.types.size(); i++) {
            if (r.types[i] == MTMD_INPUT_CHUNK_TYPE_TEXT) n_text_tokens_stray += r.n_tokens[i];
        }
    }
    // the stray marker must be present as text: more text tokens than the same prompt without it
    {
        tok_result r = tokenize(ctx, "a " + marker + " b  c", { bm1 });
        size_t n_text = 0;
        for (size_t i = 0; i < r.types.size(); i++) {
            if (r.types[i] == MTMD_INPUT_CHUNK_TYPE_TEXT) n_text += r.n_tokens[i];
        }
        CHECK(r.rc == 0 && n_text_tokens_stray > n_text, "the unmatched marker must survive as literal text: %zu text tokens with it vs %zu without", n_text_tokens_stray, n_text);
    }
    // a marker and no bitmap at all -> plain text prompt
    {
        tok_result r = tokenize(ctx, "just " + marker + " text", {});
        CHECK(r.rc == 0 && count_type(r, MTMD_INPUT_CHUNK_TYPE_IMAGE) == 0, "1 marker / 0 bitmaps: rc %d, want text only", r.rc);
    }
    // opposite: fewer markers than bitmaps must still fail (never drop an image silently)
    {
        tok_result r = tokenize(ctx, "one " + marker + " only", { bm1, bm2 });
        CHECK(r.rc != 0, "1 marker / 2 bitmaps must still fail, rc %d", r.rc);
    }

    mtmd_bitmap_free(bm1);
    mtmd_bitmap_free(bm2);
    mtmd_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    if (fails) {
        fprintf(stderr, "%d check(s) FAILED\n", fails);
        return 1;
    }
    printf("all stray-marker checks passed\n");
    return 0;
}

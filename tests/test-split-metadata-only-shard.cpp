// A split GGUF whose FIRST shard is metadata only (split.tensors.count = N, shard 00001 declares 0 tensors, as
// in the UD uploads and in convert_hf_to_gguf.py --no-tensor-first-split) must load, with and without
// mmap, and decode to exactly the logits of the same model as one unsplit file. A pair whose second shard is
// missing must be refused with a message that names the missing file.
// SOURCE-READ: tmp/research/findings/S69_UNSLOTH_MERGE_AND_MODEL_DATA_2026-09-01.md (raw 24-byte
// headers of the real shards: shard 00001 = 0 tensors / 67 KV, shard 00002 = 402 tensors / 3 KV)
//
// usage: test-split-metadata-only-shard <dir from tests/make_tiny_split_gguf.py>
// Row: S69-GGUF-SHARD1-IS-METADATA-ONLY-NTENSORS-ZERO-01.
#include "llama.h"

#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

static std::string g_log;
static void log_cb(ggml_log_level, const char * text, void *) { g_log += text; }

static int g_fail = 0;
#define CHECK(cond, ...) do { if (!(cond)) { g_fail++; printf("FAIL  %s:%d  %s  --  ", __FILE__, __LINE__, #cond); printf(__VA_ARGS__); printf("\n"); } else { printf("ok    %s\n", #cond); } } while (0)

static std::vector<float> decode(llama_model * model) {
    llama_context_params cp = llama_context_default_params();
    cp.n_ctx = 64; cp.n_batch = 16; cp.n_ubatch = 16; cp.n_threads = 2; cp.n_threads_batch = 2;
    llama_context * ctx = llama_init_from_model(model, cp);
    std::vector<float> res;
    if (!ctx) { return res; }
    const int T = 6;
    const llama_token toks[T] = { 3, 5, 7, 9, 11, 13 };
    llama_batch b = llama_batch_init(T, 0, 1);
    b.n_tokens = T;
    for (int i = 0; i < T; i++) {
        b.token[i] = toks[i]; b.pos[i] = i; b.n_seq_id[i] = 1; b.seq_id[i][0] = 0; b.logits[i] = 1;
    }
    if (llama_decode(ctx, b) == 0) {
        const int nv = llama_vocab_n_tokens(llama_model_get_vocab(model));
        const float * lg = llama_get_logits(ctx);
        if (lg) { res.assign(lg, lg + (size_t) T * nv); }
    }
    llama_batch_free(b);
    llama_free(ctx);
    return res;
}

static llama_model * load(const std::string & path, bool mmap) {
    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = 0;
    mp.use_mmap     = mmap;
    return llama_model_load_from_file(path.c_str(), mp);
}

int main(int argc, char ** argv) {
    if (argc < 2) { fprintf(stderr, "usage: %s <dir>\n", argv[0]); return 64; }
    const std::string dir = std::string(argv[1]) + "/";
    llama_backend_init();
    llama_log_set(log_cb, nullptr);

    llama_model * whole = load(dir + "whole.gguf", true);
    CHECK(whole != nullptr, "unsplit reference must load");
    if (!whole) { return 2; }
    const std::vector<float> ref = decode(whole);
    CHECK(!ref.empty(), "unsplit reference must decode");
    llama_model_free(whole);

    struct lay { const char * label; const char * first; };
    const lay layouts[] = {
        { "metadata-only shard 1, everything in shard 2", "meta1-00001-of-00002.gguf" },
        { "metadata-only shard 1, tensors over shards 2 and 3", "meta3-00001-of-00003.gguf" },
        { "control: tensors spread over both shards", "bal-00001-of-00002.gguf" },
    };
    for (const lay & l : layouts) {
        for (bool mmap : { true, false }) {
            g_log.clear();
            llama_model * m = load(dir + l.first, mmap);
            CHECK(m != nullptr, "%s (mmap=%d) must load", l.label, (int) mmap);
            if (m) {
                const std::vector<float> lg = decode(m);
                CHECK(!lg.empty() && lg.size() == ref.size() && memcmp(lg.data(), ref.data(), ref.size() * sizeof(float)) == 0,
                      "%s (mmap=%d) must decode to the unsplit logits", l.label, (int) mmap);
                llama_model_free(m);
            } else {
                printf("      log tail: %s\n", g_log.size() > 300 ? g_log.substr(g_log.size() - 300).c_str() : g_log.c_str());
            }
        }
    }

    // a metadata-only first shard whose partner is missing: refused, and the message names the file
    {
        g_log.clear();
        llama_model * m = load(dir + "lone-00001-of-00002.gguf", true);
        CHECK(m == nullptr, "a split pair with the tensor shard missing must be refused");
        CHECK(g_log.find("lone-00002-of-00002.gguf") != std::string::npos, "the refusal must name the missing shard file");
        if (m) { llama_model_free(m); }
    }

    llama_backend_free();
    printf(g_fail ? "RESULT: FAIL (%d)\n" : "RESULT: PASS\n", g_fail);
    return g_fail ? 1 : 0;
}

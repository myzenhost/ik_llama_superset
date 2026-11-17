// Ollama-published Gemma 4 blobs declare tokenizer.ggml.model = "llama" while carrying BPE merges.
// The vocab loader must promote that to the gemma4 BPE tokenizer (src/llama-vocab.cpp, impl::load).
// Fixture: models/ggml-vocab-gemma-4.gguf (vocab only); the "ollama blob" is the same header with the
// model key rewritten to "llama". Usage: test-ollama-gemma4-tokenizer <ggml-vocab-gemma-4.gguf> <tmp-dir>
#include "llama.h"
#include "gguf.h"

#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

static int g_fail = 0;
#define CHECK(c, ...) do { if (!(c)) { ++g_fail; std::printf("FAIL: "); std::printf(__VA_ARGS__); std::printf("\n"); } } while (0)

static std::vector<llama_token> tok(const llama_vocab * v, const char * s) {
    std::vector<llama_token> out(64);
    int n = llama_tokenize(v, s, (int) std::strlen(s), out.data(), (int) out.size(), false, true);
    out.resize(n < 0 ? 0 : n);
    return out;
}

int main(int argc, char ** argv) {
    if (argc < 3) { std::printf("SKIP: usage %s <gemma4-vocab.gguf> <tmpdir>\n", argv[0]); return 0; }
    const std::string src = argv[1], dst = std::string(argv[2]) + "/ollama-gemma4-llama-model.gguf";

    gguf_init_params ip = { /*no_alloc*/ true, /*ctx*/ nullptr };
    gguf_context * g = gguf_init_from_file(src.c_str(), ip);
    if (!g) { std::printf("FAIL: cannot read %s\n", src.c_str()); return 2; }
    int64_t k = gguf_find_key(g, "tokenizer.ggml.model");
    CHECK(k >= 0 && std::strcmp(gguf_get_val_str(g, k), "gemma4") == 0, "fixture tokenizer.ggml.model is gemma4");
    CHECK(gguf_find_key(g, "tokenizer.ggml.merges") >= 0, "fixture has merges");
    gguf_set_val_str(g, "tokenizer.ggml.model", "llama");
    CHECK(gguf_write_to_file(g, dst.c_str(), /*only_meta*/ false), "write rewritten header");
    gguf_free(g);

    llama_backend_init();
    llama_model_params mp = llama_model_default_params();
    mp.vocab_only = true;

    llama_model * ref = llama_model_load_from_file(src.c_str(), mp);
    llama_model * oll = llama_model_load_from_file(dst.c_str(), mp);
    CHECK(ref != nullptr, "reference loads");
    CHECK(oll != nullptr, "ollama-style (model=llama) blob loads");
    if (ref && oll) {
        const llama_vocab * vr = llama_model_get_vocab(ref);
        const llama_vocab * vo = llama_model_get_vocab(oll);
        CHECK(llama_vocab_type(vr) == LLAMA_VOCAB_TYPE_BPE, "reference vocab type BPE");
        CHECK(llama_vocab_type(vo) == LLAMA_VOCAB_TYPE_BPE, "ollama-style vocab type promoted to BPE (got %d)", (int) llama_vocab_type(vo));
        const char * probes[] = { "<|turn>user\nhello world<turn|>\n<|turn>model\n", "The quick brown fox", "<|channel>thought" };
        for (const char * p : probes) {
            auto a = tok(vr, p), b = tok(vo, p);
            CHECK(!a.empty() && a == b, "tokenization identical for '%s' (%zu vs %zu tokens)", p, a.size(), b.size());
        }
    }
    if (ref) llama_model_free(ref);
    if (oll) llama_model_free(oll);
    std::remove(dst.c_str());
    std::printf(g_fail ? "FAILED (%d)\n" : "PASS\n", g_fail);
    return g_fail ? 1 : 0;
}

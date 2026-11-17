// A GGUF of an architecture with no In-Place TTT graph route must not load silently with its "<arch>.ttt.*" keys ignored.
// Row S69-FORK-INPLACE-TTT-OTHER-ARCHS-2026-10-26-01. Writes tiny metadata-only GGUFs (no tensors); the load itself always fails
// (no tensors), so the check is on WHICH error the loader logs: the TTT guard message must appear for llama + ttt.chunk and
// must not appear for llama without the key or for qwen3 (the one routed arch) with it.
#include "llama.h"
#include "gguf.h"

#include <cstdio>
#include <string>

static std::string g_log;

static void capture(enum ggml_log_level, const char * text, void *) {
    g_log += text;
}

static int load_with(const char * arch, bool with_ttt) {
    const std::string path = std::string("test-ttt-arch-guard-") + arch + (with_ttt ? "-ttt" : "-plain") + ".gguf";
    gguf_context * g = gguf_init_empty();
    const std::string a = arch;
    gguf_set_val_str(g, "general.architecture", arch);
    gguf_set_val_u32(g, (a + ".block_count").c_str(), 2);
    gguf_set_val_u32(g, (a + ".context_length").c_str(), 64);
    gguf_set_val_u32(g, (a + ".embedding_length").c_str(), 32);
    gguf_set_val_u32(g, (a + ".attention.head_count").c_str(), 2);
    gguf_set_val_u32(g, (a + ".feed_forward_length").c_str(), 64);
    if (with_ttt) {
        gguf_set_val_u32(g, (a + ".ttt.chunk").c_str(), 8);
    }
    const char * toks[] = { "a", "b" };
    gguf_set_arr_str(g, "tokenizer.ggml.tokens", toks, 2);
    const bool ok = gguf_write_to_file(g, path.c_str(), false);
    gguf_free(g);
    if (!ok) {
        fprintf(stderr, "cannot write %s\n", path.c_str());
        return -1;
    }
    g_log.clear();
    llama_model_params mp = llama_model_default_params();
    llama_model * m = llama_model_load_from_file(path.c_str(), mp);
    if (m) {
        llama_model_free(m);
    }
    remove(path.c_str());
    return g_log.find("ttt: arch") != std::string::npos ? 1 : 0;
}

int main() {
    llama_log_set(capture, nullptr);
    llama_backend_init();
    int failed = 0;
    const int a = load_with("llama", true);
    const int b = load_with("llama", false);
    const int c = load_with("qwen3", true);
    if (a < 0 || b < 0 || c < 0) {
        return 2;
    }
    fprintf(stderr, "llama+ttt guard=%d (want 1), llama plain guard=%d (want 0), qwen3+ttt guard=%d (want 0)\n", a, b, c);
    if (a != 1) { fprintf(stderr, "FAIL: llama with ttt.chunk was not refused with the TTT message\n"); failed++; }
    if (b != 0) { fprintf(stderr, "FAIL: plain llama triggered the TTT guard\n"); failed++; }
    if (c != 0) { fprintf(stderr, "FAIL: qwen3 (routed) triggered the TTT guard\n"); failed++; }
    return failed ? 1 : 0;
}

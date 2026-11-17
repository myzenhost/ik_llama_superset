// common_vocab_token_table (common/common.cpp): the per-token byte table that common/llguidance.cpp hands to llguidance.
// Row S69-NO-STATIC-LIMITS-CPP-SERVER-SWEEP-2026-11-07-01 (D2). The table used to be built in a buffer of vocab*16 + 1 MiB bytes with a
// 1024 byte slot per token and GGML_ABORT when a token piece did not fit: a vocab with one long piece (a run of 3000 spaces, a long
// code-indent token) killed the process. The size a vocab needs is known from the vocab itself.
//
// Needs models/ggml-vocab-qwen2.gguf (run from the source root: WORKING_DIRECTORY is set by CMake). argv[1] = A | B | C runs one case.
#include "common.h"
#include "nolim-synth-vocab.h"
#include "llama.h"

#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

static int g_fails = 0;
#define CHECK(cond, ...) do { if (!(cond)) { ++g_fails; std::printf("FAIL: " __VA_ARGS__); std::printf("\n"); } } while (0)

static const char * BASE = "models/ggml-vocab-qwen2.gguf";

static llama_model * nolim_load_vocab_model(const char * path) {
    llama_backend_init();
    auto mparams = llama_model_default_params();
    mparams.vocab_only = true;
    llama_model * m = llama_model_load_from_file(path, mparams);
    if (m == nullptr) {
        throw std::runtime_error(std::string("cannot load ") + path);
    }
    return m;
}

static void check_table(const llama_vocab * vocab, const std::vector<uint32_t> & lens, const std::vector<uint8_t> & bytes) {
    const size_t n = (size_t) llama_vocab_n_tokens(vocab);
    CHECK(lens.size() == n, "lens has %zu entries for %zu tokens", lens.size(), n);
    size_t sum = 0;
    for (const uint32_t l : lens) { sum += l; }
    CHECK(sum == bytes.size(), "sum(lens)=%zu != bytes=%zu", sum, bytes.size());
    // every non-special token's slot equals its piece rendered mid-sequence (the table is the same bytes common_token_to_piece gives)
    size_t off = 0;
    size_t bad = 0;
    for (size_t i = 0; i < n && i < lens.size(); ++i) {
        if (lens[i] > 0 && bytes[off] != 0xff) {
            const std::string piece = common_token_to_piece(vocab, (llama_token) i, false);
            if (piece.size() != lens[i] || std::memcmp(piece.data(), bytes.data() + off, lens[i]) != 0) { ++bad; }
        }
        off += lens[i];
    }
    CHECK(bad == 0, "%zu token slot(s) differ from common_token_to_piece", bad);
}

// A: the real vocab, unchanged. Control: must pass before and after.
static void case_a() {
    llama_model * m = nolim_load_vocab_model(BASE);
    const llama_vocab * vocab = llama_model_get_vocab(m);
    std::vector<uint32_t> lens;
    std::vector<uint8_t>  bytes;
    common_vocab_token_table(vocab, lens, bytes);
    check_table(vocab, lens, bytes);
    std::printf("A: %zu tokens, %zu bytes\n", lens.size(), bytes.size());
    llama_model_free(m);
}

// B: one 3000-byte piece
static void case_b() {
    const std::string out = "nolim2_vocab_b.gguf";
    const std::string longp(3000, 'a');
    nolim_write_vocab_gguf(BASE, out.c_str(), { longp }, 1);
    llama_model * m = nolim_load_vocab_model(out.c_str());
    const llama_vocab * vocab = llama_model_get_vocab(m);
    const size_t idx = (size_t) llama_vocab_n_tokens(vocab) - 1;
    std::vector<uint32_t> lens;
    std::vector<uint8_t>  bytes;
    common_vocab_token_table(vocab, lens, bytes);
    CHECK(lens.size() == idx + 1, "lens size %zu", lens.size());
    if (lens.size() == idx + 1) {
        CHECK(lens[idx] == 3000, "long piece slot is %u bytes, want 3000", lens[idx]);
        size_t off = 0;
        for (size_t i = 0; i < idx; ++i) { off += lens[i]; }
        bool all_a = bytes.size() >= off + 3000;
        for (size_t k = 0; all_a && k < 3000; ++k) { all_a = bytes[off + k] == 'a'; }
        CHECK(all_a, "long piece bytes are not 3000 x 'a'");
    }
    check_table(vocab, lens, bytes);
    std::printf("B: long piece slot %u\n", lens.empty() ? 0u : lens.back());
    llama_model_free(m);
    std::remove(out.c_str());
}

// C: 3000 pieces of 900 bytes: 2.7 MB of text against the old buffer of 3000*16 + 1 MiB (every piece fits its slot, the SUM does not)
static void case_c() {
    const std::string out = "nolim2_vocab_c.gguf";
    std::vector<std::string> extra;
    for (int i = 0; i < 3000; ++i) {
        std::string t = "b" + std::to_string(i) + "_";
        t.resize(900, 'x');
        extra.push_back(t);
    }
    nolim_write_vocab_gguf(BASE, out.c_str(), extra, 1);
    llama_model * m = nolim_load_vocab_model(out.c_str());
    const llama_vocab * vocab = llama_model_get_vocab(m);
    std::vector<uint32_t> lens;
    std::vector<uint8_t>  bytes;
    common_vocab_token_table(vocab, lens, bytes);
    const size_t n = (size_t) llama_vocab_n_tokens(vocab);
    CHECK(lens.size() == n, "lens size %zu != %zu", lens.size(), n);
    size_t n900 = 0;
    for (const uint32_t l : lens) { n900 += l == 900 ? 1 : 0; }
    CHECK(n900 == 3000, "%zu slots of 900 bytes, want 3000", n900);
    check_table(vocab, lens, bytes);
    std::printf("C: %zu bytes for %zu tokens\n", bytes.size(), n);
    llama_model_free(m);
    std::remove(out.c_str());
}

int main(int argc, char ** argv) {
    const std::string only = argc > 1 ? argv[1] : "";
    try {
        if (only.empty() || only == "A") { case_a(); }
        if (only.empty() || only == "B") { case_b(); }
        if (only.empty() || only == "C") { case_c(); }
    } catch (const std::exception & e) {
        std::printf("EXCEPTION: %s\n", e.what());
        return 2;
    }
    if (g_fails) {
        std::printf("test-vocab-token-table: %d FAILED\n", g_fails);
        return 1;
    }
    std::printf("test-vocab-token-table: ok\n");
    return 0;
}

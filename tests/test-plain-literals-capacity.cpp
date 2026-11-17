// server_collect_plain_literals / server_plain_literal_base (tools/server/server-common.cpp): how many control-token literals the
// server can neutralise inside message text.
// Row S69-NO-STATIC-LIMITS-CPP-SERVER-SWEEP-2026-11-07-01 (D7). The collector kept the first 4000 literals (MAX_LITERALS = 4000,
// lits.resize) and dropped the rest WITHOUT a word: a control token past the 4000th was not neutralised, so request text could still
// inject it as a real token. The real bound is the private-use block each literal gets one code point of; past that the collector
// must fail loudly, never truncate.
//
// Needs models/ggml-vocab-qwen2.gguf (WORKING_DIRECTORY = source root, set by CMake). argv[1] = A | B | C | D runs one case.
#include "server-common.h"
#include "nolim-synth-vocab.h"
#include "llama.h"

#include <cstdio>
#include <set>
#include <stdexcept>
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

static std::vector<std::string> make_ctrl(size_t n) {
    std::vector<std::string> v;
    v.reserve(n);
    for (size_t i = 0; i < n; ++i) { v.push_back("<|nolim_" + std::to_string(i) + "|>"); }
    return v;
}

// A: the real vocab. Control: unchanged before and after.
static void case_a() {
    llama_model * m = nolim_load_vocab_model(BASE);
    const auto lits = server_collect_plain_literals(llama_model_get_vocab(m));
    bool end = false;
    for (const auto & l : lits) { end = end || l == "<|im_end|>"; }
    CHECK(end && lits.size() < 1000, "real vocab: %zu literals, <|im_end|> %s", lits.size(), end ? "present" : "MISSING");
    std::printf("A: %zu literals\n", lits.size());
    llama_model_free(m);
}

// B: 5000 extra control tokens (more than the old 4000, less than one private-use block): every one must be kept
static void case_b() {
    const std::string out = "nolim2_plain_b.gguf";
    const auto extra = make_ctrl(5000);
    nolim_write_vocab_gguf(BASE, out.c_str(), extra, 3);
    llama_model * m = nolim_load_vocab_model(out.c_str());
    std::vector<std::string> lits;
    try {
        lits = server_collect_plain_literals(llama_model_get_vocab(m));
    } catch (const std::exception & e) {
        std::printf("B: threw %s\n", e.what());
        CHECK(false, "5000 literals fit the private-use block and must not be refused");
    }
    const std::set<std::string> have(lits.begin(), lits.end());
    size_t missing = 0;
    for (const auto & t : extra) { missing += have.count(t) ? 0 : 1; }
    CHECK(missing == 0, "%zu of 5000 control-token literals dropped silently (got %zu literals)", missing, lits.size());
    std::printf("B: %zu literals, %zu of 5000 missing\n", lits.size(), missing);
    llama_model_free(m);
    std::remove(out.c_str());
}

// C: more control tokens than any block holds: refuse, do not truncate
static void case_c() {
    const std::string out = "nolim2_plain_c.gguf";
    nolim_write_vocab_gguf(BASE, out.c_str(), make_ctrl(70000), 3);
    llama_model * m = nolim_load_vocab_model(out.c_str());
    bool threw = false;
    size_t got = 0;
    try {
        got = server_collect_plain_literals(llama_model_get_vocab(m)).size();
    } catch (const std::exception & e) {
        threw = true;
        std::printf("C: threw: %s\n", e.what());
    }
    CHECK(threw, "70000 control tokens do not fit any block: must throw, but %zu literals came back silently", got);
    llama_model_free(m);
    std::remove(out.c_str());
}

// D: which placeholder run a request gets
static void case_d() {
    const json clean = json::parse(R"({"messages":[{"role":"user","content":"hello"}]})");
    auto base = [&](const json & b, size_t n) -> int64_t {
        try { return (int64_t) server_plain_literal_base(b, n); } catch (const std::exception &) { return -1; }
    };
    CHECK(base(clean, 1)     == 0xE000,  "1 literal -> BMP block");
    CHECK(base(clean, 4000)  == 0xE000,  "4000 literals -> BMP block");
    CHECK(base(clean, 6400)  == 0xE000,  "6400 literals fill the BMP block exactly (0xE000..0xF8FF)");
    // 6401 would run into 0xF900, ordinary CJK compatibility characters: it needs the plane-15 block
    CHECK(base(clean, 6401)  == 0xF0000, "6401 literals must move to plane 15, got 0x%llx", (unsigned long long) base(clean, 6401));
    CHECK(base(clean, 65534) == 0xF0000, "65534 literals fill plane 15 (0xF0000..0xFFFFD)");
    CHECK(base(clean, 65535) == -1,      "65535 literals fit no block: must throw");

    // a request that already uses a code point of the first run is moved to the other block, one that uses both is refused
    const json uses_bmp = json::parse("{\"messages\":[{\"role\":\"user\",\"content\":\"a\\ue005b\"}]}");
    CHECK(base(uses_bmp, 10) == 0xF0000, "request using U+E005 -> plane 15");
    const json uses_both = json::parse("{\"messages\":[{\"role\":\"user\",\"content\":\"a\\ue005b\\udb80\\udc05\"}]}");
    CHECK(base(uses_both, 10) == -1, "request using both blocks must be refused");
    std::printf("D: done\n");
}

int main(int argc, char ** argv) {
    const std::string only = argc > 1 ? argv[1] : "";
    try {
        if (only.empty() || only == "A") { case_a(); }
        if (only.empty() || only == "B") { case_b(); }
        if (only.empty() || only == "C") { case_c(); }
        if (only.empty() || only == "D") { case_d(); }
    } catch (const std::exception & e) {
        std::printf("EXCEPTION: %s\n", e.what());
        return 2;
    }
    if (g_fails) {
        std::printf("test-plain-literals-capacity: %d FAILED\n", g_fails);
        return 1;
    }
    std::printf("test-plain-literals-capacity: ok\n");
    return 0;
}

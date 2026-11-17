// common_gguf_scan_file_type / common_ftype_tag / common_gguf_probe_tag (common/gguf-header-probe.h).
// The GGUF heads here are written by the real gguf writer (gguf_write_to_file, only_meta), not hand-built,
// so the scan is checked against the format this tree actually emits.
// Fail-before: without common/gguf-header-probe.cpp this test does not link; the negative controls below
// (truncated head, non-GGUF, missing key, unknown ftype, depth bomb) each must NOT report found.

#include "gguf-header-probe.h"

#include "gguf.h"
#include "llama.h"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>

static int g_fail = 0;
#define CHECK(c)                                                         \
    do {                                                                 \
        if (!(c)) {                                                      \
            std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #c); \
            ++g_fail;                                                    \
        }                                                                \
    } while (0)

// Write a metadata-only GGUF: a token array of n_tok strings first (as real models have), then file_type.
static std::vector<uint8_t> make_head(int n_tok, bool with_file_type, uint32_t ftype) {
    gguf_context * ctx = gguf_init_empty();
    gguf_set_val_str(ctx, "general.architecture", "llama");
    std::vector<std::string>  toks;
    std::vector<const char *> ptrs;
    toks.reserve(n_tok);
    for (int i = 0; i < n_tok; ++i) {
        toks.push_back("tok" + std::to_string(i));
    }
    for (auto & t : toks) {
        ptrs.push_back(t.c_str());
    }
    gguf_set_arr_str(ctx, "tokenizer.ggml.tokens", ptrs.data(), ptrs.size());
    if (with_file_type) {
        gguf_set_val_u32(ctx, "general.file_type", ftype);
    }
    const char * path = "test-gguf-header-probe.tmp.gguf";
    const bool   ok   = gguf_write_to_file(ctx, path, /*only_meta =*/true);
    gguf_free(ctx);
    CHECK(ok);
    std::ifstream        f(path, std::ios::binary);
    std::vector<uint8_t> buf((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    f.close();
    std::remove(path);
    return buf;
}

int main() {
    // 1. full head -> found, fork-specific ftype (145 = IQ4_KS here; mainline has no 145)
    auto head = make_head(5000, true, LLAMA_FTYPE_MOSTLY_IQ4_KS);
    CHECK(head.size() > 40000);  // the key sits behind the token array, as in real files
    auto r = common_gguf_scan_file_type(head.data(), head.size());
    CHECK(r.status == common_gguf_scan_status::found);
    CHECK(r.file_type == (uint32_t) LLAMA_FTYPE_MOSTLY_IQ4_KS);
    CHECK(common_ftype_tag(r.file_type) == "IQ4_KS");

    // 2. every strict prefix that ends before the value is need_more, never found/absent
    //    (the u32 value ends somewhere before head.size(); find it once, then sweep prefixes below it)
    size_t first_found = 0;
    for (size_t n = 0; n <= head.size(); ++n) {
        if (common_gguf_scan_file_type(head.data(), n).status == common_gguf_scan_status::found) {
            first_found = n;
            break;
        }
    }
    CHECK(first_found > 0);
    int bad_prefix = 0;
    for (size_t n = 0; n < first_found; n += (n < 64 ? 1 : 997)) {
        if (common_gguf_scan_file_type(head.data(), n).status != common_gguf_scan_status::need_more) {
            ++bad_prefix;
        }
    }
    CHECK(bad_prefix == 0);

    // 3. tags: mainline-shared ids, ik ids, the guessed bit, unknown
    CHECK(common_ftype_tag(LLAMA_FTYPE_MOSTLY_Q4_K_M) == "Q4_K_M");
    CHECK(common_ftype_tag(LLAMA_FTYPE_ALL_F32) == "F32");
    CHECK(common_ftype_tag(LLAMA_FTYPE_MOSTLY_IQ2_KT) == "IQ2_KT");
    CHECK(common_ftype_tag(LLAMA_FTYPE_MOSTLY_Q8_0 | LLAMA_FTYPE_GUESSED) == "Q8_0");
    CHECK(common_ftype_tag(9999).empty());

    // 4. negative controls: each is final and NOT found
    const char not_gguf[] = "PK\x03\x04 not a model at all";
    CHECK(common_gguf_scan_file_type((const uint8_t *) not_gguf, sizeof(not_gguf)).status == common_gguf_scan_status::absent);
    auto no_key = make_head(10, false, 0);
    CHECK(common_gguf_scan_file_type(no_key.data(), no_key.size()).status == common_gguf_scan_status::absent);
    std::vector<uint8_t> be = {'G', 'G', 'U', 'F', 0, 0, 0, 3};  // big-endian version 3
    CHECK(common_gguf_scan_file_type(be.data(), be.size()).status == common_gguf_scan_status::absent);
    // absurd string length: must not read out of bounds, must ask for more (it can never be satisfied under the cap)
    std::vector<uint8_t> absurd = {'G', 'G', 'U', 'F', 3, 0, 0, 0};
    for (int i = 0; i < 8; ++i) absurd.push_back(0);               // n_tensors
    absurd.push_back(1); for (int i = 0; i < 7; ++i) absurd.push_back(0);  // n_kv = 1
    for (int i = 0; i < 8; ++i) absurd.push_back(0xff);            // key length = u64 max
    CHECK(common_gguf_scan_file_type(absurd.data(), absurd.size()).status == common_gguf_scan_status::need_more);

    // 5. ranged probe: chunked fetch appends, stops at found, reports bytes read; the cap stops a hopeless file
    int calls = 0;
    auto fetch = [&](uint64_t off, uint64_t len, std::vector<uint8_t> & out) {
        ++calls;
        if (off >= head.size()) return false;
        const uint64_t end = std::min<uint64_t>(head.size(), off + len);
        out.insert(out.end(), head.begin() + (ptrdiff_t) off, head.begin() + (ptrdiff_t) end);
        return true;
    };
    uint64_t got = 0;
    CHECK(common_gguf_probe_tag(fetch, head.size(), &got, 4096, COMMON_GGUF_PROBE_MAX) == "IQ4_KS");
    CHECK(got >= first_found && got < first_found + 4096);
    CHECK(calls == (int) ((got + 4095) / 4096));
    calls = 0;
    CHECK(common_gguf_probe_tag(fetch, head.size(), &got, 4096, 8192).empty());  // cap below the key
    CHECK(got == 8192 && calls == 2);
    auto failing = [](uint64_t, uint64_t, std::vector<uint8_t> &) { return false; };
    CHECK(common_gguf_probe_tag(failing, 0, &got).empty() && got == 0);

    std::printf("test-gguf-header-probe: %s (%d failures; key found at byte %zu of a %zu-byte head)\n",
                g_fail ? "FAIL" : "PASS", g_fail, first_found, head.size());
    return g_fail ? 1 : 0;
}

// GGUF v1 read: v1 (Aug-Oct 2023) is v2/v3 with 32-bit tensor/kv counts, string lengths, array lengths and tensor dims
// (tensor data offsets were already 64-bit). Upstream refuses v1; this fork reads it again with every v2/v3 check kept, and
// writes a context read from a v1 file back as the current version. Files are built byte by byte: the writer cannot
// produce v1. Row S69-ENGINE-REVIEW-1016-GGUF-V1-READER-01 (SPEC read: koboldcpp ggml/src/gguf.cpp, AGPL; no code copied).
// GATE-CLAIM-OK: unit test, no resource claims.
#include "ggml.h"
#include "gguf.h"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

static int fails = 0;
#define CHECK(cond, ...) do { if (!(cond)) { fprintf(stderr, "FAIL line %d: ", __LINE__); fprintf(stderr, __VA_ARGS__); fprintf(stderr, "\n"); fails++; } } while (0)

static void put32(std::vector<uint8_t> & b, uint32_t v) { for (int i = 0; i < 4; i++) b.push_back((v >> (8*i)) & 0xFF); }
static void put64(std::vector<uint8_t> & b, uint64_t v) { for (int i = 0; i < 8; i++) b.push_back((v >> (8*i)) & 0xFF); }
static void putf(std::vector<uint8_t> & b, float f) { uint32_t u; memcpy(&u, &f, 4); put32(b, u); }

// a GGUF string: length in 32 bits for v1, 64 bits otherwise
static void puts_(std::vector<uint8_t> & b, const std::string & s, bool v1) {
    if (v1) put32(b, (uint32_t) s.size()); else put64(b, s.size());
    b.insert(b.end(), s.begin(), s.end());
}
static void putn(std::vector<uint8_t> & b, uint64_t n, bool v1) { if (v1) put32(b, (uint32_t) n); else put64(b, n); }

enum class defect { none, huge_string, truncated, huge_array };

// 4 kv (string, u32, string array, f32 array) + 2 F32 tensors t0 [4 x 3], t1 [8]
static std::vector<uint8_t> make_file(uint32_t version, defect d = defect::none) {
    const bool v1 = version == 1;
    std::vector<uint8_t> f = {'G', 'G', 'U', 'F'};
    put32(f, version);
    putn(f, 2, v1);
    putn(f, 4, v1);

    puts_(f, "general.architecture", v1); put32(f, GGUF_TYPE_STRING);
    if (d == defect::huge_string) {
        putn(f, 0xFFFFFFF0u, v1); // a length that runs past the end of the file
        return f;
    }
    puts_(f, "llama", v1);
    puts_(f, "test.u32", v1);     put32(f, GGUF_TYPE_UINT32); put32(f, 7);
    puts_(f, "test.arr_str", v1); put32(f, GGUF_TYPE_ARRAY); put32(f, GGUF_TYPE_STRING);
    putn(f, d == defect::huge_array ? 0x7FFFFFFFu : 2, v1);
    puts_(f, "ab", v1); puts_(f, "cde", v1);
    puts_(f, "test.arr_f32", v1); put32(f, GGUF_TYPE_ARRAY); put32(f, GGUF_TYPE_FLOAT32); putn(f, 3, v1);
    putf(f, 1.5f); putf(f, -2.0f); putf(f, 3.25f);

    puts_(f, "t0", v1); put32(f, 2); putn(f, 4, v1); putn(f, 3, v1); put32(f, GGML_TYPE_F32); put64(f, 0);
    puts_(f, "t1", v1); put32(f, 1); putn(f, 8, v1);                  put32(f, GGML_TYPE_F32); put64(f, 64); // 48 B padded to 64

    while (f.size() % 32) f.push_back(0);
    for (int i = 0; i < 12; i++) putf(f, 0.5f*i);
    for (int i = 0; i < 4; i++) put32(f, 0);
    for (int i = 0; i < 8; i++) putf(f, -1.0f*i);
    if (d == defect::truncated) f.resize(f.size() - 7);
    return f;
}

static void check_content(gguf_context * g, ggml_context * ctx, uint32_t want_version, const char * what) {
    CHECK(gguf_get_version(g) == want_version, "%s: version %u, want %u", what, gguf_get_version(g), want_version);
    CHECK(gguf_get_n_kv(g) == 4, "%s: n_kv %lld", what, (long long) gguf_get_n_kv(g));
    CHECK(gguf_get_n_tensors(g) == 2, "%s: n_tensors %lld", what, (long long) gguf_get_n_tensors(g));
    int64_t k = gguf_find_key(g, "general.architecture");
    CHECK(k >= 0 && std::string(gguf_get_val_str(g, k)) == "llama", "%s: general.architecture", what);
    k = gguf_find_key(g, "test.u32");
    CHECK(k >= 0 && gguf_get_val_u32(g, k) == 7, "%s: test.u32", what);
    k = gguf_find_key(g, "test.arr_str");
    CHECK(k >= 0 && gguf_get_arr_n(g, k) == 2 && std::string(gguf_get_arr_str(g, k, 0)) == "ab" &&
          std::string(gguf_get_arr_str(g, k, 1)) == "cde", "%s: test.arr_str", what);
    k = gguf_find_key(g, "test.arr_f32");
    if (k >= 0 && gguf_get_arr_n(g, k) == 3) {
        const float * a = (const float *) gguf_get_arr_data(g, k);
        CHECK(a[0] == 1.5f && a[1] == -2.0f && a[2] == 3.25f, "%s: test.arr_f32 values", what);
    } else {
        CHECK(false, "%s: test.arr_f32 missing or wrong length", what);
    }
    CHECK(gguf_get_tensor_offset(g, gguf_find_tensor(g, "t1")) == 64, "%s: t1 offset", what);
    if (ctx == nullptr) {
        return; // meta only
    }
    ggml_tensor * t0 = ggml_get_tensor(ctx, "t0");
    ggml_tensor * t1 = ggml_get_tensor(ctx, "t1");
    CHECK(t0 && t0->ne[0] == 4 && t0->ne[1] == 3 && t0->ne[2] == 1, "%s: t0 shape", what);
    CHECK(t1 && t1->ne[0] == 8 && t1->ne[1] == 1, "%s: t1 shape", what);
    if (t0 && t1) {
        bool ok = true;
        for (int i = 0; i < 12; i++) ok = ok && ((const float *) t0->data)[i] == 0.5f*i;
        for (int i = 0; i < 8; i++)  ok = ok && ((const float *) t1->data)[i] == -1.0f*i;
        CHECK(ok, "%s: tensor data", what);
    }
}

static gguf_context * load(const std::vector<uint8_t> & f, ggml_context ** ctx) {
    gguf_init_params ip = { /*.no_alloc =*/ false, /*.ctx =*/ ctx };
    return gguf_init_from_buffer(f.data(), f.size(), ip);
}

int main() {
    // 1. v1 file loads, every value right; gguf_get_version reports 1
    const std::vector<uint8_t> f1 = make_file(1);
    ggml_context * ctx1 = nullptr;
    gguf_context * g1 = load(f1, &ctx1);
    CHECK(g1 != nullptr, "v1 file did not load");
    if (g1) {
        check_content(g1, ctx1, 1, "v1");

        // 2. written back (meta: a read context holds no tensor data to write): the current version (64-bit layout), never a
        //    v1 header over v3 fields; re-reads identically
        const char * tmp = "test-gguf-v1-read.tmp.gguf";
        CHECK(gguf_write_to_file(g1, tmp, true), "write v1-read context");
        FILE * fp = fopen(tmp, "rb");
        std::vector<uint8_t> w;
        if (fp) { int c; while ((c = fgetc(fp)) != EOF) w.push_back((uint8_t) c); fclose(fp); }
        remove(tmp);
        uint32_t wver = 0;
        if (w.size() >= 8) memcpy(&wver, w.data() + 4, 4);
        CHECK(wver == GGUF_VERSION, "written header version %u, want %d", wver, GGUF_VERSION);
        gguf_init_params mp = { /*.no_alloc =*/ true, /*.ctx =*/ nullptr };
        gguf_context * g3 = gguf_init_from_buffer(w.data(), w.size(), mp);
        CHECK(g3 != nullptr, "re-read of the written meta failed");
        if (g3) { check_content(g3, nullptr, GGUF_VERSION, "v1 written back"); gguf_free(g3); }
        gguf_free(g1); ggml_free(ctx1);
    }

    // 3. control: the same content as v3 loads too (the test file builder is right)
    {
        ggml_context * ctx = nullptr;
        gguf_context * g = load(make_file(3), &ctx);
        CHECK(g != nullptr, "v3 control did not load");
        if (g) { check_content(g, ctx, 3, "v3"); gguf_free(g); ggml_free(ctx); }
    }

    // 4. v1 validation is kept: oversized string, oversized array and truncated data are refused; v4 still refused
    {
        ggml_context * ctx = nullptr;
        gguf_context * g = load(make_file(1, defect::huge_string), &ctx);
        CHECK(g == nullptr, "v1 string longer than the file accepted");
        if (g) { gguf_free(g); ggml_free(ctx); }
        ctx = nullptr;
        g = load(make_file(1, defect::huge_array), &ctx);
        CHECK(g == nullptr, "v1 array longer than the file accepted");
        if (g) { gguf_free(g); ggml_free(ctx); }
        ctx = nullptr;
        g = load(make_file(1, defect::truncated), &ctx);
        CHECK(g == nullptr, "truncated v1 file accepted");
        if (g) { gguf_free(g); ggml_free(ctx); }
        ctx = nullptr;
        g = load(make_file(GGUF_VERSION + 1), &ctx);
        CHECK(g == nullptr, "future version accepted");
        if (g) { gguf_free(g); ggml_free(ctx); }
    }

    // 5. a v1 file and a v3 file read at the same time do not share the width setting
    {
        const std::vector<uint8_t> a = make_file(1), b = make_file(3);
        ggml_context * ca = nullptr; ggml_context * cb = nullptr;
        gguf_context * ga = load(a, &ca);
        gguf_context * gb = load(b, &cb);
        CHECK(ga && gb, "v1 then v3 in one process: %p %p", (void *) ga, (void *) gb);
        if (ga) { gguf_free(ga); ggml_free(ca); }
        if (gb) { gguf_free(gb); ggml_free(cb); }
    }

    printf("%s: %d failure(s)\n", fails ? "FAIL" : "OK", fails);
    return fails ? 1 : 0;
}

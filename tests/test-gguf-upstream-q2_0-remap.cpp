// A GGUF tensor with id 42 is OUR Q2_0_G128 (128 elem / 34 B) in every file we wrote, but UPSTREAM's Q2_0 (64 elem / 18 B: Strata, GSQ-RCO,
// mainline) also uses 42. The reader must tell them apart by block geometry (byte span between tensor offsets), retype upstream files to
// GGML_TYPE_Q2_0 (401), leave our files alone, and REFUSE mixed or unexplained evidence. Files are built byte by byte here because the
// writer would size an id-42 tensor with 128-element blocks. Spec read: Strata serve gguf_reader.hpp (MIT); re-implemented, no code copied.
// GATE-CLAIM-OK: unit test, no resource claims.
#include "ggml.h"
#include "gguf.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

static int fails = 0;
#define CHECK(cond, ...) do { if (!(cond)) { fprintf(stderr, "FAIL line %d: ", __LINE__); fprintf(stderr, __VA_ARGS__); fprintf(stderr, "\n"); fails++; } } while (0)

struct tspec { std::string name; std::vector<int64_t> ne; uint32_t type; uint32_t block_elems; uint32_t block_bytes; };

static void put32(std::vector<uint8_t> & b, uint32_t v) { for (int i = 0; i < 4; i++) b.push_back((v >> (8*i)) & 0xFF); }
static void put64(std::vector<uint8_t> & b, uint64_t v) { for (int i = 0; i < 8; i++) b.push_back((v >> (8*i)) & 0xFF); }

static uint64_t pad32(uint64_t x) { return (x + 31) & ~uint64_t(31); }

// block of `elems` 2-bit codes (block_bytes = 2 + elems/4): d = 0.5 (fp16 0x3800), codes from a position-dependent pattern
static std::vector<uint8_t> make_block(uint32_t elems, uint32_t seed) {
    std::vector<uint8_t> blk(2 + elems/4);
    blk[0] = 0x00; blk[1] = 0x38;
    for (uint32_t i = 0; i < elems/4; i++) blk[2 + i] = (uint8_t) ((i*37 + seed*11 + 5) & 0xFF);
    return blk;
}

static std::vector<uint8_t> make_file(const std::vector<tspec> & ts, std::vector<uint64_t> & offsets) {
    std::vector<uint8_t> f;
    const char magic[4] = {'G', 'G', 'U', 'F'};
    f.insert(f.end(), magic, magic + 4);
    put32(f, 3);
    put64(f, ts.size());
    put64(f, 0); // no kv: default alignment 32
    std::vector<uint64_t> spans;
    uint64_t off = 0;
    for (const tspec & t : ts) {
        uint64_t n = 1; for (int64_t e : t.ne) n *= e;
        const uint64_t nbytes = n / t.block_elems * t.block_bytes;
        offsets.push_back(off);
        put64(f, t.name.size()); f.insert(f.end(), t.name.begin(), t.name.end());
        put32(f, (uint32_t) t.ne.size()); for (int64_t e : t.ne) put64(f, e);
        put32(f, t.type);
        put64(f, off);
        spans.push_back(pad32(nbytes));
        off += pad32(nbytes);
    }
    while (f.size() % 32) f.push_back(0);
    const size_t data0 = f.size();
    f.resize(data0 + off, 0);
    for (size_t i = 0; i < ts.size(); i++) {
        uint64_t n = 1; for (int64_t e : ts[i].ne) n *= e;
        uint8_t * p = f.data() + data0 + offsets[i];
        for (uint64_t b = 0; b < n / ts[i].block_elems; b++) {
            const std::vector<uint8_t> blk = make_block(ts[i].block_elems, (uint32_t) (i*131 + b));
            memcpy(p + b*ts[i].block_bytes, blk.data(), blk.size());
        }
    }
    return f;
}

static gguf_context * load(const std::vector<uint8_t> & f) {
    gguf_init_params ip = { /*.no_alloc =*/ true, /*.ctx =*/ nullptr };
    return gguf_init_from_buffer(f.data(), f.size(), ip);
}

int main() {
    // ggml_init fills the fp16->fp32 table the dequantisers read; without it every scale decodes as 0
    ggml_free(ggml_init({ /*.mem_size =*/ 1024, /*.mem_buffer =*/ nullptr, /*.no_alloc =*/ true }));
    const uint32_t T42 = (uint32_t) GGML_TYPE_Q2_0_G128;

    // 1. upstream-written file: 64-element rows, plus a wide one that only upstream's geometry explains
    {
        std::vector<tspec> ts = {
            { "up.a", {64, 3},   T42, 64, 18 },
            { "up.b", {256, 40}, T42, 64, 18 },
            { "up.c", {128, 2},  T42, 64, 18 },
        };
        std::vector<uint64_t> offs;
        std::vector<uint8_t> f = make_file(ts, offs);
        gguf_context * g = load(f);
        CHECK(g != nullptr, "upstream-geometry file must load (before the remap: 'not a multiple of block size' abort)");
        if (g) {
            CHECK(gguf_get_n_tensors(g) == 3, "n_tensors");
            const ggml_type_traits * tr = ggml_get_type_traits(GGML_TYPE_Q2_0);
            for (int i = 0; i < 3 && i < gguf_get_n_tensors(g); i++) {
                CHECK(gguf_get_tensor_type(g, i) == GGML_TYPE_Q2_0, "tensor %d type %d want %d", i, (int) gguf_get_tensor_type(g, i), (int) GGML_TYPE_Q2_0);
                CHECK(gguf_get_tensor_offset(g, i) == offs[i], "tensor %d offset", i);
                // values: (code-1)*d with d = 0.5, from the bytes we wrote
                const size_t data0 = gguf_get_data_offset(g);
                uint64_t n = 1; for (int64_t e : ts[i].ne) n *= e;
                std::vector<float> y(64);
                const uint8_t * blk = f.data() + data0 + gguf_get_tensor_offset(g, i);
                tr->to_float(blk, y.data(), 64);
                double worst = 0;
                for (int j = 0; j < 64; j++) {
                    const int q = (blk[2 + j/4] >> ((j % 4)*2)) & 3;
                    worst = std::fmax(worst, std::fabs(y[j] - (q - 1)*0.5));
                }
                CHECK(worst == 0.0, "tensor %d first block dequant differs from (code-1)*d by %g (n=%llu)", i, worst, (unsigned long long) n);
                if (worst != 0.0) {
                    for (int j = 0; j < 8; j++) fprintf(stderr, "   j=%d y=%g want=%g qs=%02x d=%02x%02x\n", j, y[j], (((blk[2 + j/4] >> ((j % 4)*2)) & 3) - 1)*0.5, blk[2 + j/4], blk[1], blk[0]);
                }
            }
            gguf_free(g);
        }
    }

    // 2. our own G128 file stays id 42 (control: the remap must not touch what we wrote)
    {
        std::vector<tspec> ts = {
            { "ours.a", {256, 40}, T42, 128, 34 },
            { "ours.b", {128, 5},  T42, 128, 34 },
        };
        std::vector<uint64_t> offs;
        std::vector<uint8_t> f = make_file(ts, offs);
        gguf_context * g = load(f);
        CHECK(g != nullptr, "our G128 file must load");
        if (g) {
            const ggml_type_traits * tr = ggml_get_type_traits(GGML_TYPE_Q2_0_G128);
            for (int i = 0; i < 2; i++) {
                CHECK(gguf_get_tensor_type(g, i) == GGML_TYPE_Q2_0_G128, "tensor %d must stay Q2_0_G128, got %d", i, (int) gguf_get_tensor_type(g, i));
                CHECK(gguf_get_tensor_offset(g, i) == offs[i], "tensor %d offset", i);
                // values through OUR decoder: a 128-element block, (code-1)*d with d = 0.5 (ik merge-owed item 5: both
                // kinds of id-42 file are checked for type AND values)
                const size_t data0 = gguf_get_data_offset(g);
                std::vector<float> y(128);
                const uint8_t * blk = f.data() + data0 + gguf_get_tensor_offset(g, i);
                tr->to_float(blk, y.data(), 128);
                double worst = 0;
                for (int j = 0; j < 128; j++) {
                    const int q = (blk[2 + j/4] >> ((j % 4)*2)) & 3;
                    worst = std::fmax(worst, std::fabs(y[j] - (q - 1)*0.5));
                }
                CHECK(worst == 0.0, "G128 tensor %d first block dequant differs from (code-1)*d by %g", i, worst);
            }
            gguf_free(g);
        }
    }

    // 3. mixed evidence: one tensor fits ours only, one fits upstream only -> refuse, never guess
    {
        std::vector<tspec> ts = {
            { "mix.a", {256, 40}, T42, 128, 34 },
            { "mix.b", {256, 40}, T42, 64, 18 },
        };
        std::vector<uint64_t> offs;
        std::vector<uint8_t> f = make_file(ts, offs);
        gguf_context * g = load(f);
        CHECK(g == nullptr, "mixed geometry must be refused");
        if (g) gguf_free(g);
    }

    // 4. a 64-wide row whose span fits neither geometry -> refuse (was: refused for the wrong reason, still refused)
    {
        std::vector<tspec> ts = {
            { "bad.a", {64, 40}, T42, 64, 20 }, // 20-byte blocks: neither 18 nor 34
        };
        std::vector<uint64_t> offs;
        std::vector<uint8_t> f = make_file(ts, offs);
        gguf_context * g = load(f);
        CHECK(g == nullptr, "unexplained geometry must be refused");
        if (g) gguf_free(g);
    }

    // 5. the renumbered type itself (401) written as such loads unchanged
    {
        std::vector<tspec> ts = { { "direct.a", {64, 3}, (uint32_t) GGML_TYPE_Q2_0, 64, 18 } };
        std::vector<uint64_t> offs;
        std::vector<uint8_t> f = make_file(ts, offs);
        gguf_context * g = load(f);
        CHECK(g != nullptr && gguf_get_tensor_type(g, 0) == GGML_TYPE_Q2_0, "type 401 loads as 401");
        if (g) gguf_free(g);
    }

    // 6. optional real-file control: GGUF_Q2_0_REAL_FILE=<a file we wrote with Q2_0_G128 tensors, e.g. Ternary-Bonsai-27B-Q2_0.gguf>
    //    must keep every id-42 tensor as OUR type (the remap must never touch what we hold on disk)
    if (const char * real = getenv("GGUF_Q2_0_REAL_FILE")) {
        gguf_init_params ip = { /*.no_alloc =*/ true, /*.ctx =*/ nullptr };
        gguf_context * g = gguf_init_from_file(real, ip);
        CHECK(g != nullptr, "real file '%s' must load", real);
        if (g) {
            int64_t n_ours = 0, n_up = 0;
            for (int64_t i = 0; i < gguf_get_n_tensors(g); i++) {
                n_ours += gguf_get_tensor_type(g, i) == GGML_TYPE_Q2_0_G128;
                n_up   += gguf_get_tensor_type(g, i) == GGML_TYPE_Q2_0;
            }
            printf("real file: %lld tensors, %lld Q2_0_G128, %lld upstream Q2_0\n", (long long) gguf_get_n_tensors(g), (long long) n_ours, (long long) n_up);
            CHECK(n_ours > 0 && n_up == 0, "real file: want Q2_0_G128 > 0 and upstream Q2_0 == 0, got %lld / %lld", (long long) n_ours, (long long) n_up);
            gguf_free(g);
        }
    }

    if (fails) {
        fprintf(stderr, "%d check(s) FAILED\n", fails);
        return 1;
    }
    printf("all upstream Q2_0 remap checks passed\n");
    return 0;
}

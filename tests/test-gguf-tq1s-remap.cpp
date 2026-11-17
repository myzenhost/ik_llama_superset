// A GGUF tensor with id 45 / 46 is OUR GGML_TYPE_TURBO4_0 / TURBO2_0 (128 elem / 66 B, 128 elem / 34 B: KV-cache types,
// but llama-quantize CAN write them as weights through --custom-q / --tensor-type, so they are not runtime-only), while
// TheTom/llama-cpp-turboquant (ca28b4bc0, ggml/include/ggml.h:435-436) writes its TQ3_1S weights as 45 (32 elem / 16 B)
// and TQ4_1S as 46 (32 elem / 20 B). The reader must tell them apart by block geometry (byte span between tensor
// offsets), exactly like the upstream Q2_0 id-42 remap (test-gguf-upstream-q2_0-remap.cpp), retype their files to this
// fork's tq3_1s / tq4_1s (ids 74 / 75), leave files holding our turbo types alone, and REFUSE mixed or unexplained
// evidence. Files are built byte by byte because our writer would never emit 45 with a 16-byte block.
// Type lookup is BY NAME at run time, so this compiles on a tree without the TQ types and FAILS there.
// GATE-CLAIM-OK: unit test, no resource claims.
#include "ggml.h"
#include "gguf.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

static int fails = 0;
#define CHECK(cond, ...) do { if (!(cond)) { fprintf(stderr, "FAIL line %d: ", __LINE__); fprintf(stderr, __VA_ARGS__); fprintf(stderr, "\n"); fails++; } } while (0)

struct tspec { std::string name; std::vector<int64_t> ne; uint32_t type; uint32_t block_elems; uint32_t block_bytes; };

static void put32(std::vector<uint8_t> & b, uint32_t v) { for (int i = 0; i < 4; i++) b.push_back((v >> (8*i)) & 0xFF); }
static void put64(std::vector<uint8_t> & b, uint64_t v) { for (int i = 0; i < 8; i++) b.push_back((v >> (8*i)) & 0xFF); }
static uint64_t pad32(uint64_t x) { return (x + 31) & ~uint64_t(31); }

// Real TQ blocks (their reference quantiser's output for test-tq1s-quant's 8-block vector, first 4 blocks each) so the
// remapped tensors can be checked by dequantising them; other geometries get a byte pattern.
static const uint8_t k_tq3_blocks[4][16] = {
    {0xc5,0x24,0x3a,0x23,0x74,0xa3,0x05,0x6b,0x37,0x88,0x14,0xc3,0x85,0x9c,0x04,0xea},
    {0x43,0x36,0x73,0x38,0x23,0xd9,0xb0,0xf3,0x58,0x23,0xab,0xda,0xfa,0x1b,0x25,0x4e},
    {0xbf,0x3b,0x98,0x39,0x69,0xb9,0x35,0x60,0xd3,0x18,0x94,0x69,0xd6,0x1f,0x88,0x79},
    {0x4d,0x42,0xeb,0x42,0x1a,0xe6,0x34,0xa3,0xb8,0x91,0xdb,0x67,0x53,0x25,0xb9,0x7a},
};
static const uint8_t k_tq4_blocks[4][20] = {
    {0x47,0x24,0xd7,0x21,0xc9,0x3a,0x64,0x03,0xb6,0x7a,0x17,0x95,0x58,0x29,0x68,0x93,0x68,0x55,0x90,0xf5},
    {0xa8,0x35,0xab,0x36,0x87,0x99,0x2a,0xb9,0xc6,0x97,0xdb,0x31,0xb5,0xc5,0xba,0xfd,0x76,0x49,0x84,0x36},
    {0xeb,0x39,0x15,0x38,0xb3,0x8b,0x55,0x3c,0x90,0x1c,0x2b,0x0c,0x49,0x9d,0xae,0xdc,0x6f,0x90,0x70,0x7d},
    {0xc4,0x40,0x5f,0x41,0x73,0x50,0x2d,0x2b,0x85,0x84,0x67,0x89,0x56,0x6f,0xdd,0x49,0x8b,0x89,0xb7,0x5c},
};

static std::vector<uint8_t> make_file(const std::vector<tspec> & ts, std::vector<uint64_t> & offsets) {
    std::vector<uint8_t> f;
    const char magic[4] = {'G', 'G', 'U', 'F'};
    f.insert(f.end(), magic, magic + 4);
    put32(f, 3);
    put64(f, ts.size());
    put64(f, 0); // no kv: default alignment 32
    uint64_t off = 0;
    for (const tspec & t : ts) {
        uint64_t n = 1; for (int64_t e : t.ne) n *= e;
        const uint64_t nbytes = n / t.block_elems * t.block_bytes;
        offsets.push_back(off);
        put64(f, t.name.size()); f.insert(f.end(), t.name.begin(), t.name.end());
        put32(f, (uint32_t) t.ne.size()); for (int64_t e : t.ne) put64(f, e);
        put32(f, t.type);
        put64(f, off);
        off += pad32(nbytes);
    }
    while (f.size() % 32) f.push_back(0);
    const size_t data0 = f.size();
    f.resize(data0 + off, 0);
    for (size_t i = 0; i < ts.size(); i++) {
        uint64_t n = 1; for (int64_t e : ts[i].ne) n *= e;
        uint8_t * p = f.data() + data0 + offsets[i];
        for (uint64_t b = 0; b < n / ts[i].block_elems; b++) {
            uint8_t * blk = p + b*ts[i].block_bytes;
            if (ts[i].block_elems == 32 && ts[i].block_bytes == 16) memcpy(blk, k_tq3_blocks[b % 4], 16);
            else if (ts[i].block_elems == 32 && ts[i].block_bytes == 20) memcpy(blk, k_tq4_blocks[b % 4], 20);
            else for (uint32_t k = 0; k < ts[i].block_bytes; k++) blk[k] = (uint8_t) ((k*37 + b*11 + i*5 + 3) & 0xFF);
        }
    }
    return f;
}

static gguf_context * load(const std::vector<uint8_t> & f) {
    gguf_init_params ip = { /*.no_alloc =*/ true, /*.ctx =*/ nullptr };
    return gguf_init_from_buffer(f.data(), f.size(), ip);
}

static int type_by_name(const char * name) {
    for (int t = 0; t < GGML_TYPE_COUNT; ++t) {
        const char * s = ggml_type_name((ggml_type) t);
        if (s && strcmp(s, name) == 0) return t;
    }
    return -1;
}

int main() {
    ggml_free(ggml_init({ /*.mem_size =*/ 1024, /*.mem_buffer =*/ nullptr, /*.no_alloc =*/ true }));
    const uint32_t T45 = 45, T46 = 46;
    // the on-disk ids are OUR turbo KV types: the precondition of this whole test
    CHECK((int) GGML_TYPE_TURBO4_0 == 45 && (int) GGML_TYPE_TURBO2_0 == 46, "turbo ids moved: TURBO4_0=%d TURBO2_0=%d", (int) GGML_TYPE_TURBO4_0, (int) GGML_TYPE_TURBO2_0);
    CHECK(ggml_blck_size(GGML_TYPE_TURBO4_0) == 128 && ggml_type_size(GGML_TYPE_TURBO4_0) == 66, "turbo4 geometry %lld/%zu", (long long) ggml_blck_size(GGML_TYPE_TURBO4_0), ggml_type_size(GGML_TYPE_TURBO4_0));
    CHECK(ggml_blck_size(GGML_TYPE_TURBO2_0) == 128 && ggml_type_size(GGML_TYPE_TURBO2_0) == 34, "turbo2 geometry %lld/%zu", (long long) ggml_blck_size(GGML_TYPE_TURBO2_0), ggml_type_size(GGML_TYPE_TURBO2_0));
    const int TQ3 = type_by_name("tq3_1s"), TQ4 = type_by_name("tq4_1s");
    CHECK(TQ3 == 74 && TQ4 == 75, "tq3_1s / tq4_1s must exist at 74 / 75, got %d / %d", TQ3, TQ4);

    // 1. a file from their quantiser: 45 = TQ3_1S, 46 = TQ4_1S; a 96-wide row (not a multiple of 128) needs the deferral;
    //    an f32 neighbour in between must be untouched
    {
        std::vector<tspec> ts = {
            { "tq.a", {256, 4}, T45, 32, 16 },
            { "f.c",  {8},      0,   1,  4  },
            { "tq.b", {96, 3},  T46, 32, 20 },
            { "tq.d", {32, 1},  T45, 32, 16 },
            { "tq.e", {512, 2}, T46, 32, 20 },
        };
        std::vector<uint64_t> offs;
        std::vector<uint8_t> f = make_file(ts, offs);
        gguf_context * g = load(f);
        CHECK(g != nullptr, "their-geometry file must load (before the remap: 96-wide row is 'not a multiple of block size 128')");
        if (g) {
            const int want[5] = { TQ3, 0, TQ4, TQ3, TQ4 };
            for (int i = 0; i < 5 && i < gguf_get_n_tensors(g); i++) {
                CHECK((int) gguf_get_tensor_type(g, i) == want[i], "tensor %s type %d (%s) want %d", ts[i].name.c_str(), (int) gguf_get_tensor_type(g, i), ggml_type_name(gguf_get_tensor_type(g, i)), want[i]);
                CHECK(gguf_get_tensor_offset(g, i) == offs[i], "tensor %d offset", i);
            }
            if (TQ3 > 0 && TQ4 > 0 && (int) gguf_get_tensor_type(g, 0) == TQ3 && (int) gguf_get_tensor_type(g, 2) == TQ4) {
                // dequantise the remapped bytes with the remapped type: equals dequantising the same blocks directly
                const size_t data0 = gguf_get_data_offset(g);
                for (int i : { 0, 2 }) {
                    const ggml_type t = gguf_get_tensor_type(g, i);
                    std::vector<float> y(128), z(128);
                    ggml_get_type_traits(t)->to_float(f.data() + data0 + gguf_get_tensor_offset(g, i), y.data(), 96);
                    std::vector<uint8_t> raw;
                    for (int b = 0; b < 3; ++b) {
                        if (i == 0) raw.insert(raw.end(), k_tq3_blocks[b], k_tq3_blocks[b] + 16);
                        else        raw.insert(raw.end(), k_tq4_blocks[b], k_tq4_blocks[b] + 20);
                    }
                    ggml_get_type_traits(t)->to_float(raw.data(), z.data(), 96);
                    double mag = 0; bool same = true;
                    for (int j = 0; j < 96; ++j) { same = same && y[j] == z[j]; mag += std::fabs(y[j]); }
                    CHECK(same && mag > 0.1, "tensor %d dequant of the remapped data differs or is empty (sum|y| %g)", i, mag);
                }
            }
            gguf_free(g);
        }
    }

    // 2. a file holding OUR turbo blocks stays 45 / 46. llama-quantize cannot produce one (measured 2026-10-17: it stops on
    //    "ggml_validate_row_data: invalid type 45"), but the gguf C API can, and the geometry decision must not retype it
    {
        std::vector<tspec> ts = {
            { "ours.a", {256, 4}, T45, 128, 66 },
            { "ours.b", {128, 3}, T46, 128, 34 },
        };
        std::vector<uint64_t> offs;
        std::vector<uint8_t> f = make_file(ts, offs);
        gguf_context * g = load(f);
        CHECK(g != nullptr, "our turbo file must load");
        if (g) {
            CHECK(gguf_get_tensor_type(g, 0) == GGML_TYPE_TURBO4_0, "ours.a must stay TURBO4_0, got %d", (int) gguf_get_tensor_type(g, 0));
            CHECK(gguf_get_tensor_type(g, 1) == GGML_TYPE_TURBO2_0, "ours.b must stay TURBO2_0, got %d", (int) gguf_get_tensor_type(g, 1));
            gguf_free(g);
        }
    }

    // 3. mixed evidence for one id: refuse, never guess
    {
        std::vector<tspec> ts = {
            { "mix.a", {256, 4}, T45, 128, 66 },
            { "mix.b", {256, 4}, T45, 32,  16 },
        };
        std::vector<uint64_t> offs;
        std::vector<uint8_t> f = make_file(ts, offs);
        gguf_context * g = load(f);
        CHECK(g == nullptr, "mixed id-45 geometry must be refused");
        if (g) gguf_free(g);
    }

    // 4. the two ids are decided independently: their TQ3_1S (45) next to our TURBO2_0 (46) is a legal file
    {
        std::vector<tspec> ts = {
            { "ind.a", {256, 4}, T45, 32,  16 },
            { "ind.b", {256, 4}, T46, 128, 34 },
        };
        std::vector<uint64_t> offs;
        std::vector<uint8_t> f = make_file(ts, offs);
        gguf_context * g = load(f);
        CHECK(g != nullptr, "independent decision file must load");
        if (g) {
            CHECK((int) gguf_get_tensor_type(g, 0) == TQ3, "ind.a want tq3_1s got %d", (int) gguf_get_tensor_type(g, 0));
            CHECK(gguf_get_tensor_type(g, 1) == GGML_TYPE_TURBO2_0, "ind.b want TURBO2_0 got %d", (int) gguf_get_tensor_type(g, 1));
            gguf_free(g);
        }
    }

    // 5. a 96-wide id-45 row whose span fits neither geometry -> refuse
    {
        std::vector<tspec> ts = { { "bad.a", {96, 4}, T45, 32, 18 } };
        std::vector<uint64_t> offs;
        std::vector<uint8_t> f = make_file(ts, offs);
        gguf_context * g = load(f);
        CHECK(g == nullptr, "unexplained id-45 geometry must be refused");
        if (g) gguf_free(g);
    }

    // 6. the fork's own ids (74 / 75) written as such load unchanged, and the Q2_0 id-42 remap still works beside them
    if (TQ3 > 0 && TQ4 > 0) {
        std::vector<tspec> ts = {
            { "direct.a", {64, 3}, (uint32_t) TQ3, 32, 16 },
            { "direct.b", {64, 3}, (uint32_t) TQ4, 32, 20 },
            { "up.q2",    {64, 3}, 42,             64, 18 },
        };
        std::vector<uint64_t> offs;
        std::vector<uint8_t> f = make_file(ts, offs);
        gguf_context * g = load(f);
        CHECK(g != nullptr, "direct-id file must load");
        if (g) {
            CHECK((int) gguf_get_tensor_type(g, 0) == TQ3 && (int) gguf_get_tensor_type(g, 1) == TQ4, "74/75 load as 74/75");
            CHECK(gguf_get_tensor_type(g, 2) == GGML_TYPE_Q2_0, "id 42 upstream geometry still remaps to Q2_0, got %d", (int) gguf_get_tensor_type(g, 2));
            gguf_free(g);
        }
    }

    if (fails) {
        fprintf(stderr, "%d check(s) FAILED\n", fails);
        return 1;
    }
    printf("all TQ3_1S / TQ4_1S id 45/46 remap checks passed\n");
    return 0;
}

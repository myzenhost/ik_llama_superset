// Standalone test for server-embd-quant.h. Build: g++ -std=c++17 -I.. test_server_embd_quant.cpp -o t && ./t
#include "server-embd-quant.h"

#include <cstdio>

static int fails = 0;
#define CHECK(c) do { if (!(c)) { std::printf("FAIL line %d: %s\n", __LINE__, #c); ++fails; } } while (0)

int main() {
    // ---- name mapping -------------------------------------------------------------------------------------------------
    CHECK(server_embd_quant_from_name("binary")  == SERVER_EMBD_QUANT_BINARY);
    CHECK(server_embd_quant_from_name("ubinary") == SERVER_EMBD_QUANT_UBINARY);
    CHECK(server_embd_quant_from_name("int8")    == SERVER_EMBD_QUANT_INT8);
    CHECK(server_embd_quant_from_name("float")   == SERVER_EMBD_QUANT_NONE);   // controls: the existing formats are not quantised
    CHECK(server_embd_quant_from_name("base64")  == SERVER_EMBD_QUANT_NONE);
    CHECK(server_embd_quant_from_name("")        == SERVER_EMBD_QUANT_NONE);
    CHECK(server_embd_quant_from_name("Binary")  == SERVER_EMBD_QUANT_NONE);   // names are exact, like "base64"

    // ---- dimension rule: binary needs a positive multiple of 8, int8 only positive -----------------------------------
    CHECK( server_embd_quant_dim_ok(SERVER_EMBD_QUANT_BINARY, 384));
    CHECK( server_embd_quant_dim_ok(SERVER_EMBD_QUANT_UBINARY, 8));
    CHECK(!server_embd_quant_dim_ok(SERVER_EMBD_QUANT_BINARY, 12));
    CHECK(!server_embd_quant_dim_ok(SERVER_EMBD_QUANT_UBINARY, 9));
    CHECK(!server_embd_quant_dim_ok(SERVER_EMBD_QUANT_BINARY, 0));
    CHECK(!server_embd_quant_dim_ok(SERVER_EMBD_QUANT_BINARY, -8));
    CHECK( server_embd_quant_dim_ok(SERVER_EMBD_QUANT_INT8, 12));              // int8 has no packing constraint
    CHECK(!server_embd_quant_dim_ok(SERVER_EMBD_QUANT_INT8, 0));

    // ---- pack bits: first value -> highest bit, strictly positive only ------------------------------------------------
    {
        // 1 0 1 1 0 0 0 1 = 0xB1 ; the second byte 0 0 0 0 0 0 0 1 = 0x01
        const std::vector<float> v = { 0.5f, -0.5f, 0.1f, 2.0f, -3.0f, -0.001f, -9.0f, 7.0f,
                                       -1.f, -1.f, -1.f, -1.f, -1.f, -1.f, -1.f, 0.25f };
        const std::vector<uint8_t> p = server_embd_pack_bits(v, 16);
        CHECK(p.size() == 2);
        CHECK(p.size() == 2 && p[0] == 0xB1 && p[1] == 0x01);
        // Matryoshka head: only the first 8 values are packed
        const std::vector<uint8_t> h = server_embd_pack_bits(v, 8);
        CHECK(h.size() == 1 && h[0] == 0xB1);
    }
    {
        // zero is NOT positive (Vespa packInto: value <= 0 leaves the bit clear); so are -0.0 and NaN
        const std::vector<float> v = { 0.0f, -0.0f, std::nanf(""), 1e-30f, 0.0f, 0.0f, 0.0f, 0.0f };
        const std::vector<uint8_t> p = server_embd_pack_bits(v, 8);
        CHECK(p.size() == 1 && p[0] == 0x10);   // only index 3 (1e-30) is set: bit 7-3 = 4 -> 0x10
    }
    {
        // all positive -> 0xFF, all negative -> 0x00
        CHECK(server_embd_pack_bits(std::vector<float>(8, 1.0f), 8)[0] == 0xFF);
        CHECK(server_embd_pack_bits(std::vector<float>(8, -1.0f), 8)[0] == 0x00);
    }
    // illegal sizes give an empty vector, never an out-of-range read
    CHECK(server_embd_pack_bits(std::vector<float>(16, 1.0f), 12).empty());    // not a multiple of 8
    CHECK(server_embd_pack_bits(std::vector<float>(8, 1.0f), 16).empty());     // longer than the vector
    CHECK(server_embd_pack_bits(std::vector<float>(8, 1.0f), 0).empty());

    // ---- signed view of a packed byte ---------------------------------------------------------------------------------
    CHECK(server_embd_as_int8(0x00) == 0);
    CHECK(server_embd_as_int8(0x7F) == 127);
    CHECK(server_embd_as_int8(0x80) == -128);
    CHECK(server_embd_as_int8(0xB1) == -79);
    CHECK(server_embd_as_int8(0xFF) == -1);

    // ---- quantize(): binary signed, ubinary unsigned, same bytes -------------------------------------------------------
    {
        const std::vector<float> v = { 0.5f, -0.5f, 0.1f, 2.0f, -3.0f, -0.001f, -9.0f, 7.0f };
        const std::vector<int> b = server_embd_quantize(v, 8, SERVER_EMBD_QUANT_BINARY);
        const std::vector<int> u = server_embd_quantize(v, 8, SERVER_EMBD_QUANT_UBINARY);
        CHECK(b.size() == 1 && b[0] == -79);
        CHECK(u.size() == 1 && u[0] == 177);
        CHECK(b.size() == 1 && u.size() == 1 && ((b[0] & 0xFF) == u[0]));
        // sign(float) bits equal the unpacked binary, bit by bit
        for (size_t i = 0; i < 8; i++) {
            const int bit = (u[0] >> (7 - i)) & 1;
            CHECK(bit == (v[i] > 0.0f ? 1 : 0));
        }
        CHECK(server_embd_quantize(v, 8, SERVER_EMBD_QUANT_NONE).empty());     // NONE never quantises
        CHECK(server_embd_quantize(v, 12, SERVER_EMBD_QUANT_BINARY).empty());  // illegal dim -> empty
    }

    // ---- int8: round(v*127), symmetric clamp, NaN -> 0 ------------------------------------------------------------------
    {
        const std::vector<float> v = { 1.0f, -1.0f, 0.0f, 0.5f, -0.5f, 0.004f, 5.0f, -5.0f, std::nanf("") };
        const std::vector<int> q = server_embd_to_int8(v, v.size());
        CHECK(q.size() == 9);
        CHECK(q.size() == 9 && q[0] == 127);      // unit component -> full scale
        CHECK(q.size() == 9 && q[1] == -127);     // symmetric: -128 is never produced
        CHECK(q.size() == 9 && q[2] == 0);
        CHECK(q.size() == 9 && q[3] == 64);       // 63.5 rounds away from zero
        CHECK(q.size() == 9 && q[4] == -64);
        CHECK(q.size() == 9 && q[5] == 1);        // 0.508 -> 1
        CHECK(q.size() == 9 && q[6] == 127);      // saturates, does not wrap
        CHECK(q.size() == 9 && q[7] == -127);
        CHECK(q.size() == 9 && q[8] == 0);
        // head only
        CHECK(server_embd_to_int8(v, 3).size() == 3);
        CHECK(server_embd_to_int8(v, 99).empty());   // longer than the vector
        CHECK(server_embd_to_int8(v, 0).empty());
        CHECK(server_embd_quantize(v, v.size(), SERVER_EMBD_QUANT_INT8) == q);
    }

    // ---- per-token rows (pooling 'none', ColBERT-style multi-vector) ----------------------------------------------------
    // SOURCE-READ: upstream/vespa/model-integration/src/main/java/ai/vespa/embedding/ColBertEmbedder.java
    // (toBitTensor: every token row packed on its own from its first 8*dim floats, bit = value > 0, first value -> bit 7;
    // embedDocument packs, embedQuery refuses int8). vespa-engine/vespa@f4223f13, Apache-2.0.
    // SOURCE-NUANCE: int8 there is ONLY the bit-packed form; a token row is never scaled to bytes, because token vectors are
    // not normalised (send_embedding normalises pooled vectors only), so a x127 scale would saturate. int8 stays refused per token.
    {
        CHECK( server_embd_quant_rows_ok(SERVER_EMBD_QUANT_BINARY));
        CHECK( server_embd_quant_rows_ok(SERVER_EMBD_QUANT_UBINARY));
        CHECK(!server_embd_quant_rows_ok(SERVER_EMBD_QUANT_INT8));    // opposite-answer control
        CHECK(!server_embd_quant_rows_ok(SERVER_EMBD_QUANT_NONE));

        const std::vector<std::vector<float>> rows = {
            { 1.f, -1.f,  1.f, -1.f,  1.f, -1.f,  1.f, -1.f,   0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.5f },
            {-1.f, -1.f, -1.f, -1.f, -1.f, -1.f, -1.f,  2.f,   3.f, 3.f, 3.f, 3.f, 3.f, 3.f, 3.f, 3.f  },
            { 0.f,  0.f,  0.f,  0.f,  0.f,  0.f,  0.f,  0.f,   9.f, 9.f, 9.f, 9.f, 9.f, 9.f, 9.f, 9.f  },
        };
        const auto q = server_embd_quantize_rows(rows, 16, SERVER_EMBD_QUANT_UBINARY);
        CHECK(q.size() == 3);
        CHECK(q.size() == 3 && q[0] == std::vector<int>({0xAA, 0x01}));   // 10101010, then only the last value positive
        CHECK(q.size() == 3 && q[1] == std::vector<int>({0x01, 0xFF}));
        CHECK(q.size() == 3 && q[2] == std::vector<int>({0x00, 0xFF}));   // zero is not positive
        // each row equals the single-vector path: rows are independent
        for (size_t r = 0; r < rows.size() && q.size() == 3; r++) {
            CHECK(q[r] == server_embd_quantize(rows[r], 16, SERVER_EMBD_QUANT_UBINARY));
        }
        const auto qs = server_embd_quantize_rows(rows, 16, SERVER_EMBD_QUANT_BINARY);
        CHECK(qs.size() == 3 && qs[0][0] == 0xAA - 256);                     // the signed view
        // Matryoshka head per token: 8 of 16 dims -> one byte per token, the prefix of the full packing
        const auto h = server_embd_quantize_rows(rows, 8, SERVER_EMBD_QUANT_UBINARY);
        CHECK(h.size() == 3 && h[0] == std::vector<int>({0xAA}) && h[1] == std::vector<int>({0x01}));
        // refusals: int8 per token, illegal dim, a row shorter than the head, no rows
        CHECK(server_embd_quantize_rows(rows, 16, SERVER_EMBD_QUANT_INT8).empty());
        CHECK(server_embd_quantize_rows(rows, 12, SERVER_EMBD_QUANT_BINARY).empty());
        std::vector<std::vector<float>> ragged = rows;
        ragged[1].resize(8);
        CHECK(server_embd_quantize_rows(ragged, 16, SERVER_EMBD_QUANT_BINARY).empty());   // one bad row fails the whole result
        CHECK(server_embd_quantize_rows({}, 16, SERVER_EMBD_QUANT_BINARY).empty());
    }

    std::printf(fails ? "FAILED: %d\n" : "OK\n", fails);
    return fails ? 1 : 0;
}

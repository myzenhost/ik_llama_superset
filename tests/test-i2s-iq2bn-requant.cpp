// I2_S -> IQ2_BN requantisation on a synthetic ternary tensor (row S69-ENGREV-BITNET-NORMIN-2026-10-27-01).
//
// llama-quantize (src/llama-quantize.cpp, llama_tensor_dequantize_impl) dequantizes a whole I2_S tensor to f32 and
// the generic path then quantises that f32 data to the target type. I2_S carries one tensor-wide scale after the packed
// codes; IQ2_BN carries a per-row scale equal to the row's max |x|. For an EXACT ternary tensor d * {-1, 0, +1} that
// chain must be lossless as long as each row holds at least one non-zero value. This test builds such a tensor by hand
// in the I2_S layout (read from dequantize_row_ms_i2s), runs the two steps through the public ggml API and checks:
//   (1) the I2_S dequantisation returns d * t exactly,
//   (2) the IQ2_BN requantisation + per-row dequantisation returns d * t exactly,
//   (3) a negative control: one flipped IQ2_BN code is detected, so the comparison is not vacuous.
// It does NOT load a vendor BitNet-embedding GGUF or measure cos-sim / nDCG (needs the model files).

#include "ggml.h"
#include "ggml-cpu.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

static int g_fail = 0;
#define CHECK(cond, ...) do { if (!(cond)) { std::printf("FAIL %s:%d: ", __FILE__, __LINE__); std::printf(__VA_ARGS__); std::printf("\n"); ++g_fail; } } while (0)

int main() {
    const int64_t nrows = 4, n_per_row = 256;      // 256 = 2 I2_S blocks of 128 and 4 IQ2_BN blocks of 64
    const int64_t n = nrows * n_per_row;
    const float d = 0.375f;                        // exactly representable, so every comparison is exact

    // ternary pattern in {-1,0,+1}; row 3 is all zero (IQ2_BN scale 0 -> must still dequantise to 0)
    std::vector<int> t((size_t) n);
    uint32_t s = 12345u;
    for (int64_t r = 0; r < nrows; ++r) {
        for (int64_t i = 0; i < n_per_row; ++i) {
            s = s * 1664525u + 1013904223u;
            t[(size_t) (r * n_per_row + i)] = (r == 3) ? 0 : (int) ((s >> 16) % 3) - 1;
        }
    }
    t[0] = 1;                                      // rows 0..2 each hold a non-zero value for certain
    t[(size_t) n_per_row] = -1;
    t[(size_t) (2 * n_per_row)] = 1;

    // I2_S buffer: n/4 bytes of 2-bit codes (code = t + 1), then one f32 scale. Within a 128-element block byte j holds
    // elements j, 32+j, 64+j, 96+j at shifts 6, 4, 2, 0 (dequantize_row_ms_i2s, iqk_quantize_base.cpp).
    std::vector<uint8_t> i2s((size_t) (n / 4) + sizeof(float), 0);
    for (int64_t e = 0; e < n; ++e) {
        const int64_t ib = e / 128, rem = e % 128, ig = rem / 32, j = rem % 32;
        i2s[(size_t) (ib * 32 + j)] |= (uint8_t) ((t[(size_t) e] + 1) << (6 - 2 * ig));
    }
    std::memcpy(i2s.data() + n / 4, &d, sizeof(float));

    ggml_quantize_init(GGML_TYPE_IQ2_BN);

    // (1) whole-tensor I2_S dequantisation, as llama_tensor_dequantize_impl does
    std::vector<float> f32((size_t) n, -999.0f);
    const ggml_type_traits_t i2s_tt = ggml_internal_get_type_traits(GGML_TYPE_I2_S);
    CHECK(i2s_tt.to_float != nullptr, "I2_S has no to_float");
    if (!i2s_tt.to_float) { return 1; }
    i2s_tt.to_float(i2s.data(), f32.data(), n);
    int bad1 = 0;
    for (int64_t e = 0; e < n; ++e) { if (f32[(size_t) e] != d * (float) t[(size_t) e]) ++bad1; }
    CHECK(bad1 == 0, "I2_S dequantise: %d of %lld values differ from d*t", bad1, (long long) n);

    // (2) requantise to IQ2_BN, then dequantise row by row (row meta is per row)
    const size_t row_bytes = ggml_row_size(GGML_TYPE_IQ2_BN, n_per_row);
    std::vector<uint8_t> bn((size_t) nrows * row_bytes, 0);
    const size_t wrote = ggml_quantize_chunk(GGML_TYPE_IQ2_BN, f32.data(), bn.data(), 0, nrows, n_per_row, nullptr);
    CHECK(wrote == bn.size(), "IQ2_BN wrote %zu bytes, expected %zu", wrote, bn.size());
    const ggml_type_traits_t bn_tt = ggml_internal_get_type_traits(GGML_TYPE_IQ2_BN);
    CHECK(bn_tt.to_float != nullptr, "IQ2_BN has no to_float");
    if (!bn_tt.to_float) { return 1; }

    auto count_bad = [&](const std::vector<uint8_t> & q) {
        int bad = 0;
        std::vector<float> back((size_t) n_per_row);
        for (int64_t r = 0; r < nrows; ++r) {
            bn_tt.to_float(q.data() + (size_t) r * row_bytes, back.data(), n_per_row);
            for (int64_t i = 0; i < n_per_row; ++i) {
                if (back[(size_t) i] != d * (float) t[(size_t) (r * n_per_row + i)]) ++bad;
            }
        }
        return bad;
    };
    const int bad2 = count_bad(bn);
    CHECK(bad2 == 0, "I2_S->IQ2_BN round trip: %d of %lld values differ from d*t", bad2, (long long) n);

    // (3) negative control: flip the first code of row 0 (element 0 is +1, code 2; xor 3 gives code 1 = 0)
    std::vector<uint8_t> broken = bn;
    broken[sizeof(float)] ^= 0x03;
    const int bad3 = count_bad(broken);
    CHECK(bad3 > 0, "negative control: a corrupted IQ2_BN code was NOT detected");

    if (g_fail) { std::printf("test-i2s-iq2bn-requant: %d FAILED\n", g_fail); return 1; }
    std::printf("test-i2s-iq2bn-requant: OK (I2_S dequant exact, IQ2_BN requant exact over %lld values, control detects %d)\n",
                (long long) n, bad3);
    return 0;
}

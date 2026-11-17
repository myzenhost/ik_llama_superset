// ggml_validate_row_data must ACCEPT every row its own quantizer produces, for every type llama-quantize can emit.
//
// WHY: llama-quantize validates every quantized row unconditionally (src/llama-quantize.cpp:978, 1009, 1095) and the
// model loader does so under --check-tensors. A type that is registered, has a quantizer and is offered as an ftype
// can still produce NO FILE if validate_row_data has no case for it: the switch falls to `default`, prints
// "invalid type N" and returns false. That happened to TQ1_0/TQ2_0 (see the comment above their cases in
// ggml-quants.c) and, found 2026-09-27, to NVFP4.
//
// Checks, both directions:
//   * each listed type: quantize a random row with ggml_quantize_chunk, then validate it -> must be true
//   * POSITIVE CONTROL Q4_0 (long-standing case) -> true
//   * NEGATIVE CONTROL an F16 row holding a NaN -> false (proves the function can say no)
//   * NVFP4 decode: every one of the 256 UE4M3 scale bytes decodes to a finite float, which is the fact that makes
//     "nothing to validate" correct for NVFP4 (a NaN/inf-producing byte would need a real check instead)
#include "ggml.h"
#include "ggml-cpu.h"

#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

// The UE4M3 decoder lives in internal ggml-impl.h, so it is exercised through the PUBLIC path instead: dequantize
// (type_traits.to_float) a row whose scale bytes cover all 256 values.

static int n_fail = 0;

static void check(bool ok, const char * what) {
    printf("  %-4s %s\n", ok ? "OK" : "FAIL", what);
    if (!ok) { n_fail++; }
}

static bool quantize_and_validate(ggml_type t, int64_t n_per_row) {
    std::mt19937 rng(1234);
    std::uniform_real_distribution<float> u(-1.0f, 1.0f);
    std::vector<float> src(n_per_row);
    for (auto & v : src) { v = u(rng); }
    const size_t row = ggml_row_size(t, n_per_row);
    std::vector<uint8_t> dst(row + 64);
    const size_t wrote = ggml_quantize_chunk(t, src.data(), dst.data(), 0, 1, n_per_row, nullptr, nullptr);
    if (wrote != row) {
        printf("       %s: quantize wrote %zu bytes, row size %zu\n", ggml_type_name(t), wrote, row);
        return false;
    }
    return ggml_validate_row_data(t, dst.data(), row);
}

int main() {
    ggml_cpu_init();  // CPU traits (from_float etc.) used by some quantizers
    const int64_t n = 256;  // a multiple of every block size used here (32, 64, 256)

    printf("validate(quantize(row)) must be true:\n");
    const ggml_type types[] = { GGML_TYPE_NVFP4, GGML_TYPE_Q1_0, GGML_TYPE_Q2_0, GGML_TYPE_TQ1_0, GGML_TYPE_TQ2_0,
                                GGML_TYPE_MXFP4 };
    for (ggml_type t : types) {
        char msg[128];
        snprintf(msg, sizeof(msg), "%s", ggml_type_name(t));
        check(quantize_and_validate(t, n), msg);
    }

    printf("controls:\n");
    check(quantize_and_validate(GGML_TYPE_Q4_0, n), "POSITIVE: q4_0 accepted");
    {
        std::vector<ggml_fp16_t> f(n, ggml_fp32_to_fp16(1.0f));
        f[7] = ggml_fp32_to_fp16(NAN);
        check(!ggml_validate_row_data(GGML_TYPE_F16, f.data(), f.size() * sizeof(ggml_fp16_t)),
              "NEGATIVE: f16 row with a NaN rejected");
    }

    printf("nvfp4: all 256 scale bytes decode finite (why 'nothing to validate' is correct):\n");
    {
        // one block = 4 scale bytes + 32 bytes of E2M1 codes; build 64 blocks so the 256 scale bytes are 0..255,
        // and set every code to 0x77 (value +6.0, the largest E2M1 magnitude) so a scale's full range is exercised.
        const int64_t k = 64 * 64;
        const size_t row = ggml_row_size(GGML_TYPE_NVFP4, k);
        std::vector<uint8_t> blk(row);
        const size_t bsz = row / 64;
        for (int b = 0; b < 64; ++b) {
            uint8_t * p = blk.data() + b * bsz;
            for (int s = 0; s < 4; ++s) { p[s] = (uint8_t)(b * 4 + s); }
            for (size_t j = 4; j < bsz; ++j) { p[j] = 0x77; }
        }
        std::vector<float> out(k);
        const auto * tt = ggml_get_type_traits(GGML_TYPE_NVFP4);
        bool have = tt && tt->to_float;
        check(have, "nvfp4 has a to_float");
        if (have) {
            tt->to_float(blk.data(), out.data(), k);
            int bad = 0;
            for (float v : out) { bad += !std::isfinite(v); }
            char msg[96];
            snprintf(msg, sizeof(msg), "0 non-finite values over all 256 scale bytes (got %d)", bad);
            check(bad == 0, msg);
        }
    }

    printf("%d failed\n", n_fail);
    return n_fail ? 1 : 0;
}

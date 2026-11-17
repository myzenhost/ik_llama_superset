// Unit tests for quantization specific functions - quantize, dequantize and dot product

#include "ggml.h"
#include "ggml-cpu.h"

#undef NDEBUG
#include <assert.h>
#include <algorithm>
#include <cmath>
#include <limits>
#include <math.h>
#include <stdio.h>
#include <string.h>
#include <algorithm>
#include <string>
#include <vector>
#include <random>

#if defined(_MSC_VER)
#pragma warning(disable: 4244 4267) // possible loss of data
#endif

constexpr float MAX_QUANTIZATION_REFERENCE_ERROR = 0.0001f;
constexpr float MAX_QUANTIZATION_TOTAL_ERROR = 0.002f;
constexpr float MAX_QUANTIZATION_TOTAL_ERROR_BINARY = 0.025f;
constexpr float MAX_QUANTIZATION_TOTAL_ERROR_TERNARY = 0.01f;
constexpr float MAX_QUANTIZATION_TOTAL_ERROR_2BITS = 0.0075f;
constexpr float MAX_QUANTIZATION_TOTAL_ERROR_3BITS = 0.0040f;
constexpr float MAX_QUANTIZATION_TOTAL_ERROR_3BITS_XXS = 0.0050f;
constexpr float MAX_QUANTIZATION_TOTAL_ERROR_FP4 = 0.0030f;
// TRELLIS (KT) TYPES ON THIS TEST'S INPUT. generate_data() is a cosine -- a structured, non-weight-like signal. KT
// codebooks are FITTED to a weight-like distribution, so on this input they trail their bpw-matched round siblings,
// and the trail is a property of the INPUT, not a defect. Measured 2026-09-26, same run, same input:
//     iq2_kt 0.007600 / iq2_ks 0.004803 = 1.58x      iq3_kt 0.004605 / iq3_ks 0.002487 = 1.85x
//     iq4_kt 0.003478 / iq4_kss 0.001563 = 2.23x
// and tests/test-quant-roundtrip-distribution.cpp shows the SAME types trailing on uniform input (1.39/1.55/1.73x) but
// BEATING their siblings on Laplace, the shape of real weights (iq2_kt 0.90x, iq4_kt 0.83x), while test-mulmat-gemv
// shows their kernels agree with a double reference to ~0.004. So each KT type gets its bpw class bound times this
// factor: the largest KT/sibling ratio measured on THIS input, rounded up. It is a calibration with both populations
// stated, not a widening to clear a red line -- a broken KT quantiser (output uncorrelated with the input) lands orders
// of magnitude above any of these bounds, and the distribution test fails if KT stops beating its siblings on
// weight-like data.
constexpr float KT_ON_COSINE_INPUT_FACTOR = 2.25f;
constexpr float MAX_DOT_PRODUCT_ERROR = 0.02f;
constexpr float MAX_DOT_PRODUCT_ERROR_LOWBIT = 0.04f;
constexpr float MAX_DOT_PRODUCT_ERROR_FP4 = 0.03f;
constexpr float MAX_DOT_PRODUCT_ERROR_BINARY = 0.40f;
constexpr float MAX_DOT_PRODUCT_ERROR_TERNARY = 0.15f;

// The turbo KV types store each 128-group in the WHT-ROTATED basis (quantise rotates, to_float does not rotate back, the K dot
// product expects an already rotated Q: src/llama-turbo-graph.h rotates it in the graph). The generic checks below therefore
// compare in that basis: dequantised row vs the rotated input, vec_dot vs the exact dot of the unrotated vectors (rotation is
// orthonormal), see turbo_rotate_groups(). Before this the rows were compared in the wrong basis and reported error ~0.03 / ~1.2.
extern "C" void turbo_cpu_fwht_forward(float * x, int group_size);
static bool is_turbo_type(ggml_type t) { return t == GGML_TYPE_TURBO2_0 || t == GGML_TYPE_TURBO3_0 || t == GGML_TYPE_TURBO4_0; }
static std::vector<float> turbo_rotate_groups(const float * x, size_t n) {
    std::vector<float> r(x, x + n);
    for (size_t i = 0; i + 128 <= n; i += 128) {
        turbo_cpu_fwht_forward(r.data() + i, 128);
    }
    return r;
}

static const char* RESULT_STR[] = {"ok", "FAILED"};


// Generate synthetic data
static void generate_data(float offset, size_t n, float * dst, float amplitude = 2.0f) {
    for (size_t i = 0; i < n; i++) {
        dst[i] = 0.1 + amplitude*cosf(i + offset);
    }
}
static void generate_bitnet_data(size_t n, float * dst) {
    std::mt19937 rndm(1234);
    for (size_t i = 0; i < n; i++) {
        auto r = rndm();
        dst[i] = r > std::mt19937::max()/2 ? 0.f : r < std::mt19937::max()/4 ? -1.f : 1.f;
    }
}


// Calculate RMSE between two float arrays
static float array_rmse(const float * a1, const float * a2, size_t n) {
    double sum = 0;
    for (size_t i = 0; i < n; i++) {
        double diff = a1[i] - a2[i];
        sum += diff * diff;
    }
    return sqrtf(sum) / n;
}

// Total quantization error on test data
// ref (optional): what the dequantised row is compared with when it is not the input itself (turbo types: the rotated input).
static float total_quantization_error(const ggml_type_traits * qfns, const ggml_type_traits_cpu * qfns_cpu, size_t test_size, const float * test_data,
                                      const float * ref = nullptr) {
    std::vector<uint8_t> tmp_q(2*test_size);
    std::vector<float> tmp_out(test_size);

    qfns_cpu->from_float(test_data, tmp_q.data(), test_size);
    qfns->to_float(tmp_q.data(), tmp_out.data(), test_size);
    return array_rmse(ref ? ref : test_data, tmp_out.data(), test_size);
}

// Total quantization error on test data
static float reference_quantization_error(const ggml_type_traits * qfns, const ggml_type_traits_cpu * qfns_cpu, size_t test_size, const float * test_data) {
    std::vector<uint8_t> tmp_q(2*test_size);
    std::vector<float> tmp_out(test_size);
    std::vector<float> tmp_out_ref(test_size);

    // FIXME: why is done twice?
    qfns_cpu->from_float(test_data, tmp_q.data(), test_size);
    qfns->to_float(tmp_q.data(), tmp_out.data(), test_size);

    qfns->from_float_ref(test_data, tmp_q.data(), test_size);
    qfns->to_float(tmp_q.data(), tmp_out_ref.data(), test_size);

    return array_rmse(tmp_out.data(), tmp_out_ref.data(), test_size);
}

static float dot_product(const float * a1, const float * a2, size_t test_size) {
    double sum = 0;
    for (size_t i = 0; i < test_size; i++) {
        sum += a1[i] * a2[i];
    }
    return sum;
}

// Total dot product error
// ref1/ref2 (optional): the vectors the exact dot product is taken over when the quantised pair works in another basis (turbo types:
// test_data1 is quantised in its original basis, test_data2 is the ROTATED vector, the reference is dot(original1, original2)).
static float dot_product_error(const ggml_type_traits_cpu * qfns_cpu, ggml_type src0_type, size_t test_size,
                               const float * test_data1, const float * test_data2,
                               const float * test_data3, const float * test_data4,
                               const int nrc, const float * ref1 = nullptr, const float * ref2 = nullptr) {
    const auto * vdot = ggml_get_type_traits_cpu(qfns_cpu->vec_dot_type);
    const size_t pad  = 64;
    const size_t bx   = ggml_row_size(src0_type, test_size) + pad;
    const size_t by   = ggml_row_size(qfns_cpu->vec_dot_type, test_size) + pad;

    std::vector<uint8_t> tmp_q1(bx * nrc);
    std::vector<uint8_t> tmp_q2(by * nrc);

    qfns_cpu->from_float(test_data1, tmp_q1.data(), test_size);
    vdot->from_float(test_data2, tmp_q2.data(), test_size);

    if (nrc == 1) {
        float result = INFINITY;
        qfns_cpu->vec_dot(test_size, &result, 0, tmp_q1.data(), 0, tmp_q2.data(), 0, 1);

        const float dot_ref = dot_product(ref1 ? ref1 : test_data1, ref2 ? ref2 : test_data2, test_size);
        return fabsf(result - dot_ref) / test_size;
    }

    // nrc == 2: kernel computes a 2x2 dot product matrix
    // Output layout: s[0]=dot(vx0,vy0), s[1]=dot(vx1,vy0), s[bs]=dot(vx0,vy1), s[bs+1]=dot(vx1,vy1)
    // row and output strides are padded, same as in the mul_mat path
    qfns_cpu->from_float(test_data3, tmp_q1.data() + bx, test_size);
    vdot->from_float(test_data4, tmp_q2.data() + by, test_size);

    const size_t bs = 16;
    std::vector<float> result(bs + 2, INFINITY);
    qfns_cpu->vec_dot(test_size, result.data(), bs, tmp_q1.data(), bx, tmp_q2.data(), by, 2);

    const float ref00 = dot_product(test_data1, test_data2, test_size);
    const float ref10 = dot_product(test_data3, test_data2, test_size);
    const float ref01 = dot_product(test_data1, test_data4, test_size);
    const float ref11 = dot_product(test_data3, test_data4, test_size);

    const auto err = [test_size](float val, float ref) {
        const float e = fabsf(val - ref) / test_size;
        return std::isfinite(e) ? e : INFINITY;
    };

    return std::max({err(result[0], ref00), err(result[1], ref10), err(result[bs], ref01), err(result[bs + 1], ref11)});
}

static int test_vec_dot_f32(bool verbose) {
    const auto * f32 = ggml_get_type_traits_cpu(GGML_TYPE_F32);
    int num_failed = 0;
    for (int n : {1, 2, 3, 5, 7, 8, 15, 16, 17, 31, 33, 63, 67, 127, 129, 193, 255, 1023}) {
        std::vector<float> a(n);
        std::vector<float> b(n);
        generate_data(0.0, n, a.data());
        generate_data(1.0, n, b.data());

        float result = 0.0f;
        f32->vec_dot(n, &result, 0, a.data(), 0, b.data(), 0, 1);
        const float ref = dot_product(a.data(), b.data(), n);
        const float error = fabsf(result - ref) / n;

        const bool failed = !(error < MAX_QUANTIZATION_REFERENCE_ERROR);
        num_failed += failed;
        if (failed || verbose) {
            printf(" f32 vec_dot n=%4d:                 %s (ref=%f got=%f err=%f)\n",
                   n, RESULT_STR[failed], ref, result, error);
        }
    }
    return num_failed;
}

// Positive finite E4M3 values in encoding order, from OCP FP8 rev. 1.1, section 5.1.
static const float F8_E4M3_VALUES[] = {
    0.0f, 0.001953125f, 0.00390625f, 0.005859375f, 0.0078125f, 0.009765625f, 0.01171875f, 0.013671875f,
    0.015625f, 0.017578125f, 0.01953125f, 0.021484375f, 0.0234375f, 0.025390625f, 0.02734375f, 0.029296875f,
    0.03125f, 0.03515625f, 0.0390625f, 0.04296875f, 0.046875f, 0.05078125f, 0.0546875f, 0.05859375f,
    0.0625f, 0.0703125f, 0.078125f, 0.0859375f, 0.09375f, 0.1015625f, 0.109375f, 0.1171875f,
    0.125f, 0.140625f, 0.15625f, 0.171875f, 0.1875f, 0.203125f, 0.21875f, 0.234375f,
    0.25f, 0.28125f, 0.3125f, 0.34375f, 0.375f, 0.40625f, 0.4375f, 0.46875f,
    0.5f, 0.5625f, 0.625f, 0.6875f, 0.75f, 0.8125f, 0.875f, 0.9375f,
    1.0f, 1.125f, 1.25f, 1.375f, 1.5f, 1.625f, 1.75f, 1.875f,
    2.0f, 2.25f, 2.5f, 2.75f, 3.0f, 3.25f, 3.5f, 3.75f,
    4.0f, 4.5f, 5.0f, 5.5f, 6.0f, 6.5f, 7.0f, 7.5f,
    8.0f, 9.0f, 10.0f, 11.0f, 12.0f, 13.0f, 14.0f, 15.0f,
    16.0f, 18.0f, 20.0f, 22.0f, 24.0f, 26.0f, 28.0f, 30.0f,
    32.0f, 36.0f, 40.0f, 44.0f, 48.0f, 52.0f, 56.0f, 60.0f,
    64.0f, 72.0f, 80.0f, 88.0f, 96.0f, 104.0f, 112.0f, 120.0f,
    128.0f, 144.0f, 160.0f, 176.0f, 192.0f, 208.0f, 224.0f, 240.0f,
    256.0f, 288.0f, 320.0f, 352.0f, 384.0f, 416.0f, 448.0f,
};

static uint8_t fp32_to_f8_e4m3_ref(float value) {
    if (isnan(value)) {
        return 0x7F;
    }
    const uint8_t sign = signbit(value) ? 0x80 : 0;
    const double magnitude = fabs((double) value);
    if (magnitude >= F8_E4M3_VALUES[126]) {
        return sign | 126;
    }
    int nearest = 0;
    double distance = magnitude;
    for (int i = 1; i < 127; ++i) {
        const double candidate = fabs(magnitude - F8_E4M3_VALUES[i]);
        if (candidate < distance || (candidate == distance && (i & 1) == 0)) {
            nearest = i;
            distance = candidate;
        }
    }
    return sign | (uint8_t) nearest;
}

static int test_f8_e4m3(bool verbose) {
    static_assert(sizeof(ggml_fp8_e4m3_t) == 1);

    const auto * traits = ggml_get_type_traits(GGML_TYPE_F8_E4M3);
    const auto * traits_cpu = ggml_get_type_traits_cpu(GGML_TYPE_F8_E4M3);
    std::vector<uint8_t> encoded(256);
    std::vector<uint8_t> roundtrip(256);
    std::vector<float> decoded(256);
    int num_failed = 0;

    const bool traits_failed = traits->blck_size != 1 || traits->type_size != sizeof(ggml_fp8_e4m3_t) || traits->is_quantized;
    num_failed += traits_failed;
    if (verbose || traits_failed) {
        printf("f8_e4m3 scalar type traits:          %s\n", RESULT_STR[traits_failed]);
    }

    for (int i = 0; i < 256; ++i) {
        encoded[i] = (uint8_t) i;
    }
    traits->to_float(encoded.data(), decoded.data(), decoded.size());
    traits_cpu->from_float(decoded.data(), roundtrip.data(), decoded.size());

    for (int i = 0; i < 256; ++i) {
        const bool is_nan = (i & 0x7F) == 0x7F;
        const float expected = is_nan ? NAN : copysignf(F8_E4M3_VALUES[i & 0x7F], i & 0x80 ? -1.0f : 1.0f);
        const bool decode_failed = is_nan ? !isnan(decoded[i])
            : decoded[i] != expected || (expected == 0.0f && signbit(decoded[i]) != signbit(expected));
        const bool encode_failed = !is_nan && roundtrip[i] != encoded[i];
        if (decode_failed || encode_failed) {
            num_failed++;
            if (verbose) {
                printf("f8_e4m3 code 0x%02x failed: decoded=%f expected=%f roundtrip=0x%02x\n",
                        i, decoded[i], expected, roundtrip[i]);
            }
        }
    }

    std::vector<float> inputs;
    for (float value : F8_E4M3_VALUES) {
        inputs.push_back(value);
        inputs.push_back(-value);
    }
    for (int i = 0; i < 126; ++i) {
        const float midpoint = (F8_E4M3_VALUES[i] + F8_E4M3_VALUES[i + 1]) / 2.0f;
        for (float value : {nextafterf(midpoint, -INFINITY), midpoint, nextafterf(midpoint, INFINITY)}) {
            inputs.push_back(value);
            inputs.push_back(-value);
        }
    }
    for (float value : {std::numeric_limits<float>::denorm_min(), std::numeric_limits<float>::max(),
                       nextafterf(448.0f, INFINITY), nextafterf(464.0f, -INFINITY), 464.0f,
                       nextafterf(464.0f, INFINITY), INFINITY, NAN}) {
        inputs.push_back(value);
        inputs.push_back(-value);
    }

    std::vector<uint8_t> quantized(inputs.size());
    std::vector<uint8_t> quantized_ref(inputs.size());
    traits_cpu->from_float(inputs.data(), quantized.data(), inputs.size());
    traits->from_float_ref(inputs.data(), quantized_ref.data(), inputs.size());
    for (size_t i = 0; i < inputs.size(); ++i) {
        const uint8_t expected = fp32_to_f8_e4m3_ref(inputs[i]);
        if (quantized[i] != expected || quantized_ref[i] != expected) {
            num_failed++;
            if (verbose) {
                printf("f8_e4m3 input %a failed: encoded=0x%02x reference=0x%02x expected=0x%02x\n",
                       inputs[i], quantized[i], quantized_ref[i], expected);
            }
        }
    }

    for (int n : {0, 1, 7, 8, 15, 16, 31, 32, 63, 64, 65, 129, 257}) {
        std::vector<uint8_t> x(n + 1);
        std::vector<float> y(n + 1);
        double expected = 0.0;
        for (int i = 0; i < n; ++i) {
            const uint8_t code = (uint8_t) (i % 254 + (i % 254 >= 127));
            x[i + 1] = code;
            y[i + 1] = (i % 7 - 3) * 0.125f;
            expected += copysignf(F8_E4M3_VALUES[code & 0x7F], code & 0x80 ? -1.0f : 1.0f) * y[i + 1];
        }
        float actual;
        traits_cpu->vec_dot(n, &actual, 0, x.data() + 1, 0, y.data() + 1, 0, 1);
        const bool failed = fabs(actual - expected) > 1e-5 * std::max(1.0, fabs(expected));
        num_failed += failed;
        if (verbose || failed) {
            printf("f8_e4m3 dot n=%d:                  %s (ref=%f got=%f)\n", n, RESULT_STR[failed], expected, actual);
        }
    }

    for (uint8_t code : {0x7F, 0xFF}) {
        std::vector<uint8_t> x(65, 0);
        std::vector<float> y(65, 1.0f);
        x[0] = x.back() = code;
        float actual;
        traits_cpu->vec_dot(x.size(), &actual, 0, x.data(), 0, y.data(), 0, 1);
        num_failed += !isnan(actual);
    }

    if (verbose || num_failed) {
        printf("f8_e4m3 exhaustive conversion:      %s (%d failures)\n", RESULT_STR[num_failed != 0], num_failed);
    }
    return num_failed;
}

static int test_vec_dot_q(bool verbose) {
    int num_failed = 0;

    const size_t test_size = 32 * 128;

    std::vector<float> test_data(test_size);
    std::vector<float> test_data2(test_size);
    // Fork-only. IQ1_BN / IQ2_BN cannot represent the 0.1 + 2*cos(i) data the other types use, so
    // this fork generates ternary data for them rather than skipping the two types, which is what
    // it did before. Upstream has neither the types nor this helper; the call sites moved here
    // when upstream restructured main() into test_vec_dot_q().
    std::vector<float> test_data_bitnet(test_size);

    std::vector<float> test_data3(test_size);
    std::vector<float> test_data4(test_size);


    generate_data(0.0, test_data.size(), test_data.data());
    generate_data(1.0, test_data2.size(), test_data2.data());
    generate_data(3.0, test_data3.size(), test_data3.data(), 1.0f);
    generate_data(4.0, test_data4.size(), test_data4.data(), 1.5f);
    generate_bitnet_data(test_data_bitnet.size(), test_data_bitnet.data());

    for (int i = 0; i < GGML_TYPE_COUNT; i++) {
        ggml_type type = (ggml_type) i;
        const auto * qfns = ggml_get_type_traits(type);
        const auto * qfns_cpu = ggml_get_type_traits_cpu(type);

        // deprecated - skip
        if (qfns->blck_size == 0) {
            continue;
        }

        const ggml_type ei = (ggml_type)i;

        // see test_data_bitnet above: these two types need representable input
        const float * test_data_quantize = test_data.data();
        const float * test_data_vecdot   = test_data2.data();
        if (ei == GGML_TYPE_IQ1_BN || ei == GGML_TYPE_IQ2_BN) {
            test_data_quantize = test_data_bitnet.data();
            test_data_vecdot   = test_data_bitnet.data();
        }

        // Fork-only. Row-interleaved repacks (q4_0_r8, iq4_xs_r8, bf16_r16, ...: name ends _r<N>)
        // store N rows per block. This test treats its buffer as ONE row, so for them the
        // quantiser writes past it (iq4_xs_r8: heap corruption, 0xC0000374) and the one-row
        // vec_dot aborts ("iqk_mul_mat declined this pair") -- either of which killed the loop
        // before it reached any later type (upstream Q1_0/Q2_0 are 400/401). They need a
        // multi-row harness (test-backend-ops); here they are skipped VISIBLY, never silently.
        {
            const char * nm = ggml_type_name(type);
            const char * r  = strrchr(nm, '_');
            if (r && r[1] == 'r' && r[2] >= '0' && r[2] <= '9') {
                printf("Testing %s: SKIPPED (row-interleaved type; needs a multi-row harness)\n", nm);
                continue;
            }
        }
        // Trellis (KT) quantization aborts BY DESIGN on a CPU build without AVX2/AVX512 (iqk_quantize.cpp
        // iqk_kt_can_quantize: "use --cuda-quantize"), which took the whole run down at iq2_kt on aarch64
        // (qemu-user lane, engq-c-1019). Skipped VISIBLY there; every later type still runs.
        if ((ei == GGML_TYPE_IQ1_KT || ei == GGML_TYPE_IQ2_KT || ei == GGML_TYPE_IQ3_KT || ei == GGML_TYPE_IQ4_KT) &&
                !ggml_cpu_has_avx2() && !ggml_cpu_has_avx512()) {
            printf("Testing %s: SKIPPED (trellis quantization needs AVX2/AVX512 on the CPU; this build has neither)\n", ggml_type_name(type));
            continue;
        }
        printf("Testing %s\n", ggml_type_name((ggml_type) i));
        ggml_quantize_init(ei);

        if (qfns_cpu->from_float && qfns->to_float) {
            const bool turbo = is_turbo_type(type);
            const std::vector<float> rot_q = turbo ? turbo_rotate_groups(test_data_quantize, test_size) : std::vector<float>();
            const std::vector<float> rot_v = turbo ? turbo_rotate_groups(test_data_vecdot,   test_size) : std::vector<float>();
            const float total_error = total_quantization_error(qfns, qfns_cpu, test_size, test_data_quantize, turbo ? rot_q.data() : nullptr);
            const float max_quantization_error =
                type == GGML_TYPE_Q1_0    ? MAX_QUANTIZATION_TOTAL_ERROR_BINARY :
                type == GGML_TYPE_TQ1_0   ? MAX_QUANTIZATION_TOTAL_ERROR_TERNARY :
                type == GGML_TYPE_TQ2_0   ? MAX_QUANTIZATION_TOTAL_ERROR_TERNARY :
                type == GGML_TYPE_Q2_0    ? MAX_QUANTIZATION_TOTAL_ERROR_TERNARY :
                type == GGML_TYPE_Q2_K    ? MAX_QUANTIZATION_TOTAL_ERROR_2BITS :
                type == GGML_TYPE_IQ2_S   ? MAX_QUANTIZATION_TOTAL_ERROR_2BITS :
                type == GGML_TYPE_Q3_K    ? MAX_QUANTIZATION_TOTAL_ERROR_3BITS :
                type == GGML_TYPE_IQ3_S   ? MAX_QUANTIZATION_TOTAL_ERROR_3BITS :
                type == GGML_TYPE_IQ3_XXS ? MAX_QUANTIZATION_TOTAL_ERROR_3BITS_XXS :
                type == GGML_TYPE_NVFP4 || type == GGML_TYPE_NVFP4_KS ? MAX_QUANTIZATION_TOTAL_ERROR_FP4 :   // NVFP4_KS [0.002456]
                // TurboQuant KV (rotated basis, see turbo_rotate_groups): the bound of the bit-class (turbo4 4 bit -> the default 0.002)
                // Measured 2026-10-11 (rotated basis, this cosine input): turbo4 0.002067, turbo3 passes the 3-bit class bound, turbo2 0.007635.
                // turbo4 and turbo2 sit just above their class bounds (the Lloyd-Max centroids are fitted to N(0,1/128) coordinates, a
                // cosine is not Gaussian), so those two get the class bound x 1.25; a broken rotation or table lands ~15x higher (0.031 before the fix).
                type == GGML_TYPE_TURBO4_0 ? MAX_QUANTIZATION_TOTAL_ERROR * 1.25f :
                type == GGML_TYPE_TURBO3_0 ? MAX_QUANTIZATION_TOTAL_ERROR_3BITS :
                type == GGML_TYPE_TURBO2_0 ? MAX_QUANTIZATION_TOTAL_ERROR_2BITS * 1.25f :
                // Fork-only types, each mapped to the EXISTING upstream bound of its bits-per-weight class
                // (bpw from ggml_type_size/blck_size, measured 2026-09-26). No bound was loosened or added.
                // Measured value on this data in brackets, so a regression is visible against the margin.
                type == GGML_TYPE_Q1_0_G128 ? MAX_QUANTIZATION_TOTAL_ERROR_BINARY :     // 1.125 bpw [0.009696]
                type == GGML_TYPE_Q2_0_G128 ? MAX_QUANTIZATION_TOTAL_ERROR_TERNARY :    // 2.125 bpw [0.006238]
                // Prism ternary (ik_llama #2550), mapped to the EXISTING ternary bound. These rows only run since the merge gained
                // their CPU traits entries (before that the loop skipped them silently: no from_float on the CPU side).
                type == GGML_TYPE_PQ2_0     ? MAX_QUANTIZATION_TOTAL_ERROR_TERNARY :    // 2.125 bpw, 3 levels in 2 bits [0.008678]
                type == GGML_TYPE_PTQ1_0    ? MAX_QUANTIZATION_TOTAL_ERROR_TERNARY :    // 1.75 bpw, base-3 [0.008678, same trits as PQ2_0]
                type == GGML_TYPE_IQ1_KT    ? MAX_QUANTIZATION_TOTAL_ERROR_TERNARY :    // 1.75  bpw [0.008847]
                type == GGML_TYPE_IQ2_K     ? MAX_QUANTIZATION_TOTAL_ERROR_2BITS :      // 2.375 bpw [0.004809]
                type == GGML_TYPE_IQ2_KS    ? MAX_QUANTIZATION_TOTAL_ERROR_2BITS :      // 2.1875 bpw [0.004803]
                // KT types: class bound x KT_ON_COSINE_INPUT_FACTOR -- see the constant for the measurements.
                // (iq2_kt was 0.007367 before to_float gained its x1.05 run-time scale; 0.007600 with it.)
                type == GGML_TYPE_IQ2_KT    ? MAX_QUANTIZATION_TOTAL_ERROR_2BITS * KT_ON_COSINE_INPUT_FACTOR :  // 2.125 bpw [0.007600]
                type == GGML_TYPE_IQ2_KL    ? MAX_QUANTIZATION_TOTAL_ERROR_2BITS :      // 2.6875 bpw [0.004990]
                type == GGML_TYPE_IQ3_K     ? MAX_QUANTIZATION_TOTAL_ERROR_3BITS :      // 3.4375 bpw [0.002438]
                type == GGML_TYPE_IQ3_KS    ? MAX_QUANTIZATION_TOTAL_ERROR_3BITS :      // 3.1875 bpw [0.002487]
                type == GGML_TYPE_IQ3_KT    ? MAX_QUANTIZATION_TOTAL_ERROR_3BITS * KT_ON_COSINE_INPUT_FACTOR :  // 3.125 bpw [0.004605 with x1.01]
                type == GGML_TYPE_IQ4_KT    ? MAX_QUANTIZATION_TOTAL_ERROR * KT_ON_COSINE_INPUT_FACTOR :        // 4.0 bpw [0.003478]
                // TQ3_1S (TurboQuant WHT-rotated, 74): 4.0 bpw by size but 3-bit codes -- 1 of its 4 bpw is two fp16 scales per 32.
                // It FAILS its bpw class (0.002) on this input [0.003371, measured 2026-10-17], so it gets the 3-bit CODE class, the
                // same bound IQ3_S / TURBO3_0 use. Stated, not hidden: at equal bpw it is a 3-bit quantiser. TQ4_1S passes the default.
                type == GGML_TYPE_TQ3_1S    ? MAX_QUANTIZATION_TOTAL_ERROR_3BITS :
                // NOT mapped, deliberately: iq1_bn / iq2_bn [~0.0156] on their own ternary data.
                MAX_QUANTIZATION_TOTAL_ERROR;
            bool failed = !(total_error < max_quantization_error);
            num_failed += failed;
            if (failed || verbose) {
                printf("%5s absolute quantization error:    %s (%f)\n", ggml_type_name(type), RESULT_STR[failed], total_error);
            }

            const float reference_error = reference_quantization_error(qfns, qfns_cpu, test_size, test_data_quantize);
            failed = !(reference_error < MAX_QUANTIZATION_REFERENCE_ERROR);
            // Fork-only. Some fast quantisers are DELIBERATELY not byte-equal to the _ref one: q4_0's
            // from_float is iqk_quantize_q4_0, which refits the scale by weighted least squares instead
            // of round-to-nearest. "Fast == ref" is the wrong oracle for those; "fast is no WORSE than
            // ref" is the right one. So a fast/ref mismatch passes only when the fast path's own error
            // against the input is <= the reference path's -- a fast path that drifts AND loses
            // accuracy still fails, which is the defect this check exists to catch.
            float ref_path_error = -1.0f;
            if (failed && qfns->from_float_ref) {
                std::vector<uint8_t> tq(2*test_size);
                std::vector<float>   tout(test_size);
                qfns->from_float_ref(test_data_quantize, tq.data(), test_size);
                qfns->to_float(tq.data(), tout.data(), test_size);
                ref_path_error = array_rmse(test_data_quantize, tout.data(), test_size);
                if (total_error <= ref_path_error) {
                    failed = false;
                }
            }
            num_failed += failed;
            if (failed || verbose) {
                if (ref_path_error >= 0.0f) {
                    printf("%5s reference implementation error: %s (%f; differs from ref by design? fast err %f vs ref err %f)\n",
                           ggml_type_name(type), RESULT_STR[failed], reference_error, total_error, ref_path_error);
                } else {
                    printf("%5s reference implementation error: %s (%f)\n", ggml_type_name(type), RESULT_STR[failed], reference_error);
                }
            }

            if (type == GGML_TYPE_NVFP4_KS) {
                // vec_dot_type is the row-GROUPED Q8_2_X4 activation (ggml_cpu_act_type_is_grouped): this harness quantises ONE row
                // and hands it to vec_dot, which is not that layout (measured: nan, same class as tests/test-bonsai-scalar-vecdot.cpp).
                // The dot product is covered through ggml_mul_mat by tests/test-nvfp4-ks.cpp (A4, n=1..64) and test-mulmat-gemv (nvfp4_ks).
                printf("%5s dot product error:              SKIPPED (grouped Q8_2_X4 activations; see test-nvfp4-ks A4)\n", ggml_type_name(type));
                continue;
            }
            const float vec_dot_error = turbo
                ? dot_product_error(qfns_cpu, type, test_size, test_data.data(), rot_v.data(), nullptr, nullptr, 1, test_data.data(), test_data_vecdot)
                : dot_product_error(qfns_cpu, type, test_size, test_data.data(), test_data_vecdot, nullptr, nullptr, 1);
            // turbo2 dot product measured 0.066167 (rotated basis), over the 0.04 low-bit class; bound 0.08 (x2 of the class, 1.2x the measurement)
            const float max_allowed_error = type == GGML_TYPE_TURBO2_0 ? 0.08f : type == GGML_TYPE_Q2_K || type == GGML_TYPE_IQ2_XS || type == GGML_TYPE_IQ2_XXS ||
                type == GGML_TYPE_IQ3_XXS || type == GGML_TYPE_IQ3_S || type == GGML_TYPE_IQ2_S ||
                type == GGML_TYPE_IQ2_KL || type == GGML_TYPE_TURBO3_0  // TurboQuant 3 bit (turbo4 keeps the default, turbo2 has its own bound above)
                || type == GGML_TYPE_TQ3_1S  // 3-bit codes at 4.0 bpw [0.034097, measured 2026-10-17]; TQ4_1S keeps the default
                    // IQ2_KL: fork, 2.6875 bpw, 2-bit class [0.027234]
                ? MAX_DOT_PRODUCT_ERROR_LOWBIT
                : type == GGML_TYPE_Q1_0 || type == GGML_TYPE_Q1_0_G128   // fork twin, same layout [0.211374]
                ? MAX_DOT_PRODUCT_ERROR_BINARY
                : type == GGML_TYPE_TQ1_0 || type == GGML_TYPE_TQ2_0 || type == GGML_TYPE_Q2_0 ||
                  type == GGML_TYPE_Q2_0_G128 ||  // fork, 2.125 bpw ternary-class [0.089892]
                  type == GGML_TYPE_PQ2_0 || type == GGML_TYPE_PTQ1_0   // Prism ternary [0.141130 / 0.141111], under the 0.15 bound
                ? MAX_DOT_PRODUCT_ERROR_TERNARY
                : type == GGML_TYPE_NVFP4 || type == GGML_TYPE_NVFP4_KS
                ? MAX_DOT_PRODUCT_ERROR_FP4
                : MAX_DOT_PRODUCT_ERROR;
            failed = !(vec_dot_error < max_allowed_error);
            num_failed += failed;
            if (failed || verbose) {
                printf("%5s dot product error:              %s (%f)\n", ggml_type_name(type), RESULT_STR[failed], vec_dot_error);
            }

            // Test nrc=2 path for types that support it
            if (qfns_cpu->nrows == 2) {
                const float vec_dot_error_nrc2 = dot_product_error(qfns_cpu, type, test_size, test_data.data(), test_data2.data(), test_data3.data(), test_data4.data(), 2);
                failed = !(vec_dot_error_nrc2 < max_allowed_error);
                num_failed += failed;
                if (failed || verbose) {
                    printf("%5s dot product error (nrc=2):    %s (%f)\n", ggml_type_name(type), RESULT_STR[failed], vec_dot_error_nrc2);
                }
            }
        }
    }

    return num_failed;
}

// In every group, all values with importance are equal, and the max (0) has none.
// The scale search in make_qkx3_quants then fits min == max and passes inf/nan to nearest_int (#29804).
static int test_quantize_imatrix_degenerate(bool verbose) {
    const int64_t n = 256;
    std::vector<float> x(n);
    std::vector<float> imatrix(n);
    std::vector<float> out(n);
    int num_failed = 0;

    printf("Testing degenerate imatrix:\n");
    for (ggml_type type : {GGML_TYPE_Q2_K, GGML_TYPE_Q4_K, GGML_TYPE_Q5_K, GGML_TYPE_Q4_1, GGML_TYPE_Q5_1}) {
        const int64_t group = type == GGML_TYPE_Q2_K ? 16 : 32;
        for (int64_t i = 0; i < n; ++i) {
            const int64_t g = i / group;
            const int64_t p = i % group;
            const bool important = p % 3 == 1;
            x[i] = important ? -0.02f*(g + 1) : (p % 2 ? -1.0f : 0.0f);
            imatrix[i] = important ? 1.0f : 0.0f;
        }
        printf("  - %s\n", ggml_type_name(type));

        std::vector<uint8_t> q(ggml_row_size(type, n));
        ggml_quantize_init(type);
        ggml_quantize_chunk(type, x.data(), q.data(), 0, 1, n, imatrix.data());
        ggml_get_type_traits(type)->to_float(q.data(), out.data(), n);

        const bool failed = !std::all_of(out.begin(), out.end(), [](float v) { return std::isfinite(v); });
        num_failed += failed;
        if (failed || verbose) {
            printf("%5s imatrix degenerate groups:      %s\n", ggml_type_name(type), RESULT_STR[failed]);
        }
    }

    return num_failed;
}

int main(int argc, char * argv[]) {
    bool verbose = false;

    std::string arg;
    for (int i = 1; i < argc; i++) {
        arg = argv[i];

        if (arg == "-v") {
            verbose = true;
        } else {
            fprintf(stderr, "error: unknown argument: %s\n", arg.c_str());
            return 1;
        }
    }

    ggml_cpu_init();

    int num_failed = 0;

    num_failed += test_vec_dot_f32(verbose);
    num_failed += test_f8_e4m3(verbose);
    num_failed += test_vec_dot_q(verbose);
    num_failed += test_quantize_imatrix_degenerate(verbose);

    if (num_failed || verbose) {
        printf("%d tests failed\n", num_failed);
    }

    return num_failed > 0;
}

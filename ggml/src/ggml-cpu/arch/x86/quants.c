#define GGML_COMMON_IMPL_C
#include "ggml-common.h"
#include "ggml-quants.h"
#include "ggml-impl.h"
#include "ggml-cpu.h"
#include "simd-mappings.h"

#include "../../quants.h"
#include "../../ggml-cpu-impl.h"

#include <math.h>
#include <string.h>
#include <assert.h>
#include <stdlib.h> // for qsort
#include <stdio.h>  // for GGML_ASSERT

#define GROUP_MAX_EPS 1e-15f
#define GROUP_MAX_EPS_IQ3_XXS 1e-8f
#define GROUP_MAX_EPS_IQ2_S 1e-8f
#define GROUP_MAX_EPS_IQ1_M 1e-7f
#define GROUP_MAX_EPS_IQ1_S 1e-12f

#define UNUSED GGML_UNUSED

// some compilers don't provide _mm256_set_m128i, e.g. gcc 7
#define MM256_SET_M128I(a, b) _mm256_insertf128_si256(_mm256_castsi128_si256(b), (a), 1)

#if defined(__AVX__) || defined(__AVX2__) || defined(__AVX512F__) || defined(__SSSE3__)
// multiply int8_t, add results pairwise twice
static inline __m128i mul_sum_i8_pairs(const __m128i x, const __m128i y) {
    // Get absolute values of x vectors
    const __m128i ax = _mm_sign_epi8(x, x);
    // Sign the values of the y vectors
    const __m128i sy = _mm_sign_epi8(y, x);
    // Perform multiplication and create 16-bit values
    const __m128i dot = _mm_maddubs_epi16(ax, sy);
    const __m128i ones = _mm_set1_epi16(1);
    return _mm_madd_epi16(ones, dot);
}

#if __AVX__ || __AVX2__ || __AVX512F__
// horizontally add 8 floats
static inline float hsum_float_8(const __m256 x) {
    __m128 res = _mm256_extractf128_ps(x, 1);
    res = _mm_add_ps(res, _mm256_castps256_ps128(x));
    res = _mm_add_ps(res, _mm_movehl_ps(res, res));
    res = _mm_add_ss(res, _mm_movehdup_ps(res));
    return _mm_cvtss_f32(res);
}

// horizontally add 8 int32_t
static inline int hsum_i32_8(const __m256i a) {
    const __m128i sum128 = _mm_add_epi32(_mm256_castsi256_si128(a), _mm256_extractf128_si256(a, 1));
    const __m128i hi64 = _mm_unpackhi_epi64(sum128, sum128);
    const __m128i sum64 = _mm_add_epi32(hi64, sum128);
    const __m128i hi32  = _mm_shuffle_epi32(sum64, _MM_SHUFFLE(2, 3, 0, 1));
    return _mm_cvtsi128_si32(_mm_add_epi32(sum64, hi32));
}

// horizontally add 4 int32_t
static inline int hsum_i32_4(const __m128i a) {
    const __m128i hi64 = _mm_unpackhi_epi64(a, a);
    const __m128i sum64 = _mm_add_epi32(hi64, a);
    const __m128i hi32  = _mm_shuffle_epi32(sum64, _MM_SHUFFLE(2, 3, 0, 1));
    return _mm_cvtsi128_si32(_mm_add_epi32(sum64, hi32));
}

#if defined(__AVX2__) || defined(__AVX512F__)
static inline __m256i mul_add_epi8(const __m256i x, const __m256i y) {
    const __m256i ax = _mm256_sign_epi8(x, x);
    const __m256i sy = _mm256_sign_epi8(y, x);
    return _mm256_maddubs_epi16(ax, sy);
}

// spread 32 bits to 32 bytes { 0x00, 0xFF }
static inline __m256i bytes_from_bits_32(const uint8_t * x) {
    uint32_t x32;
    memcpy(&x32, x, sizeof(uint32_t));
    const __m256i shuf_mask = _mm256_set_epi64x(
            0x0303030303030303, 0x0202020202020202,
            0x0101010101010101, 0x0000000000000000);
    __m256i bytes = _mm256_shuffle_epi8(_mm256_set1_epi32(x32), shuf_mask);
    const __m256i bit_mask = _mm256_set1_epi64x(0x7fbfdfeff7fbfdfe);
    bytes = _mm256_or_si256(bytes, bit_mask);
    return _mm256_cmpeq_epi8(bytes, _mm256_set1_epi64x(-1));
}

// Unpack 32 4-bit fields into 32 bytes
// The output vector contains 32 bytes, each one in [ 0 .. 15 ] interval
static inline __m256i bytes_from_nibbles_32(const uint8_t * rsi)
{
    const __m128i tmp = _mm_loadu_si128((const __m128i *)rsi);
    const __m256i bytes = MM256_SET_M128I(_mm_srli_epi16(tmp, 4), tmp);
    const __m256i lowMask = _mm256_set1_epi8( 0xF );
    return _mm256_and_si256(lowMask, bytes);
}

// add int16_t pairwise and return as float vector
static inline __m256 sum_i16_pairs_float(const __m256i x) {
    const __m256i ones = _mm256_set1_epi16(1);
    const __m256i summed_pairs = _mm256_madd_epi16(ones, x);
    return _mm256_cvtepi32_ps(summed_pairs);
}

static inline __m256 mul_sum_us8_pairs_float(const __m256i ax, const __m256i sy) {
#if defined(__AVX512VNNI__) && defined(__AVX512VL__)
    const __m256i zero = _mm256_setzero_si256();
    const __m256i summed_pairs = _mm256_dpbusd_epi32(zero, ax, sy);
    return _mm256_cvtepi32_ps(summed_pairs);
#elif defined(__AVXVNNI__)
    const __m256i zero = _mm256_setzero_si256();
    const __m256i summed_pairs = _mm256_dpbusd_avx_epi32(zero, ax, sy);
    return _mm256_cvtepi32_ps(summed_pairs);
#else
    // Perform multiplication and create 16-bit values
    const __m256i dot = _mm256_maddubs_epi16(ax, sy);
    return sum_i16_pairs_float(dot);
#endif
}

// multiply int8_t, add results pairwise twice and return as float vector
static inline __m256 mul_sum_i8_pairs_float(const __m256i x, const __m256i y) {
#if __AVXVNNIINT8__
    const __m256i zero = _mm256_setzero_si256();
    const __m256i summed_pairs = _mm256_dpbssd_epi32(zero, x, y);
    return _mm256_cvtepi32_ps(summed_pairs);
#else
    // Get absolute values of x vectors
    const __m256i ax = _mm256_sign_epi8(x, x);
    // Sign the values of the y vectors
    const __m256i sy = _mm256_sign_epi8(y, x);
    return mul_sum_us8_pairs_float(ax, sy);
#endif
}

static inline __m128i packNibbles( __m256i bytes )
{
    // Move bits within 16-bit lanes from 0000_abcd_0000_efgh into 0000_0000_abcd_efgh
#if __AVX512F__
    const __m256i bytes_srli_4 = _mm256_srli_epi16(bytes, 4);   // 0000_0000_abcd_0000
    bytes = _mm256_or_si256(bytes, bytes_srli_4);               // 0000_abcd_abcd_efgh
    return _mm256_cvtepi16_epi8(bytes);                         // abcd_efgh
#else
    const __m256i lowByte = _mm256_set1_epi16( 0xFF );
    __m256i high = _mm256_andnot_si256( lowByte, bytes );
    __m256i low = _mm256_and_si256( lowByte, bytes );
    high = _mm256_srli_epi16( high, 4 );
    bytes = _mm256_or_si256( low, high );

    // Compress uint16_t lanes into bytes
    __m128i r0 = _mm256_castsi256_si128( bytes );
    __m128i r1 = _mm256_extracti128_si256( bytes, 1 );
    return _mm_packus_epi16( r0, r1 );
#endif
}
#elif defined(__AVX__)
static inline __m128i packNibbles( __m128i bytes1, __m128i bytes2 )
{
    // Move bits within 16-bit lanes from 0000_abcd_0000_efgh into 0000_0000_abcd_efgh
    const __m128i lowByte = _mm_set1_epi16( 0xFF );
    __m128i high = _mm_andnot_si128( lowByte, bytes1 );
    __m128i low = _mm_and_si128( lowByte, bytes1 );
    high = _mm_srli_epi16( high, 4 );
    bytes1 = _mm_or_si128( low, high );
    high = _mm_andnot_si128( lowByte, bytes2 );
    low = _mm_and_si128( lowByte, bytes2 );
    high = _mm_srli_epi16( high, 4 );
    bytes2 = _mm_or_si128( low, high );

    return _mm_packus_epi16( bytes1, bytes2);
}

static inline __m128i mul_add_epi8_sse(const __m128i x, const __m128i y) {
    const __m128i ax = _mm_sign_epi8(x, x);
    const __m128i sy = _mm_sign_epi8(y, x);
    return _mm_maddubs_epi16(ax, sy);
}

// spread 32 bits to 32 bytes { 0x00, 0xFF }
static inline __m256i bytes_from_bits_32(const uint8_t * x) {
    uint32_t x32;
    memcpy(&x32, x, sizeof(uint32_t));
    const __m128i shuf_maskl = _mm_set_epi64x(0x0101010101010101, 0x0000000000000000);
    const __m128i shuf_maskh = _mm_set_epi64x(0x0303030303030303, 0x0202020202020202);
    __m128i bytesl = _mm_shuffle_epi8(_mm_set1_epi32(x32), shuf_maskl);
    __m128i bytesh = _mm_shuffle_epi8(_mm_set1_epi32(x32), shuf_maskh);
    const __m128i bit_mask = _mm_set1_epi64x(0x7fbfdfeff7fbfdfe);
    bytesl = _mm_or_si128(bytesl, bit_mask);
    bytesh = _mm_or_si128(bytesh, bit_mask);
    bytesl = _mm_cmpeq_epi8(bytesl, _mm_set1_epi64x(-1));
    bytesh = _mm_cmpeq_epi8(bytesh, _mm_set1_epi64x(-1));
    return MM256_SET_M128I(bytesh, bytesl);
}

// Unpack 32 4-bit fields into 32 bytes
// The output vector contains 32 bytes, each one in [ 0 .. 15 ] interval
static inline __m256i bytes_from_nibbles_32(const uint8_t * rsi)
{
    // Load 16 bytes from memory
    __m128i tmpl = _mm_loadu_si128((const __m128i *)rsi);
    __m128i tmph = _mm_srli_epi16(tmpl, 4);
    const __m128i lowMask = _mm_set1_epi8(0xF);
    tmpl = _mm_and_si128(lowMask, tmpl);
    tmph = _mm_and_si128(lowMask, tmph);
    return MM256_SET_M128I(tmph, tmpl);
}

// add int16_t pairwise and return as float vector
static inline __m256 sum_i16_pairs_float(const __m128i xh, const __m128i xl) {
    const __m128i ones = _mm_set1_epi16(1);
    const __m128i summed_pairsl = _mm_madd_epi16(ones, xl);
    const __m128i summed_pairsh = _mm_madd_epi16(ones, xh);
    const __m256i summed_pairs = MM256_SET_M128I(summed_pairsh, summed_pairsl);
    return _mm256_cvtepi32_ps(summed_pairs);
}

static inline __m256 mul_sum_us8_pairs_float(const __m256i ax, const __m256i sy) {
    const __m128i axl = _mm256_castsi256_si128(ax);
    const __m128i axh = _mm256_extractf128_si256(ax, 1);
    const __m128i syl = _mm256_castsi256_si128(sy);
    const __m128i syh = _mm256_extractf128_si256(sy, 1);
    // Perform multiplication and create 16-bit values
    const __m128i dotl = _mm_maddubs_epi16(axl, syl);
    const __m128i doth = _mm_maddubs_epi16(axh, syh);
    return sum_i16_pairs_float(doth, dotl);
}

// multiply int8_t, add results pairwise twice and return as float vector
static inline __m256 mul_sum_i8_pairs_float(const __m256i x, const __m256i y) {
    const __m128i xl = _mm256_castsi256_si128(x);
    const __m128i xh = _mm256_extractf128_si256(x, 1);
    const __m128i yl = _mm256_castsi256_si128(y);
    const __m128i yh = _mm256_extractf128_si256(y, 1);
    // Get absolute values of x vectors
    const __m128i axl = _mm_sign_epi8(xl, xl);
    const __m128i axh = _mm_sign_epi8(xh, xh);
    // Sign the values of the y vectors
    const __m128i syl = _mm_sign_epi8(yl, xl);
    const __m128i syh = _mm_sign_epi8(yh, xh);
    // Perform multiplication and create 16-bit values
    const __m128i dotl = _mm_maddubs_epi16(axl, syl);
    const __m128i doth = _mm_maddubs_epi16(axh, syh);
    return sum_i16_pairs_float(doth, dotl);
}

// larger version of mul_sum_i8_pairs_float where x and y are each represented by four 128-bit vectors
static inline __m256 mul_sum_i8_quad_float(const __m128i x_1_0, const __m128i x_1_1, const __m128i x_2_0, const __m128i x_2_1,
                                           const __m128i y_1_0, const __m128i y_1_1, const __m128i y_2_0, const __m128i y_2_1) {
    const __m128i mone = _mm_set1_epi16(1);

    const __m128i p16_1_0 = mul_add_epi8_sse(x_1_0, y_1_0);
    const __m128i p16_1_1 = mul_add_epi8_sse(x_1_1, y_1_1);
    const __m128i p16_2_0 = mul_add_epi8_sse(x_2_0, y_2_0);
    const __m128i p16_2_1 = mul_add_epi8_sse(x_2_1, y_2_1);
    const __m128i p_1_0 = _mm_madd_epi16(p16_1_0, mone);
    const __m128i p_1_1 = _mm_madd_epi16(p16_1_1, mone);
    const __m128i p_2_0 = _mm_madd_epi16(p16_2_0, mone);
    const __m128i p_2_1 = _mm_madd_epi16(p16_2_1, mone);
    const __m128i p_1 = _mm_add_epi32(p_1_0, p_1_1);
    const __m128i p_2 = _mm_add_epi32(p_2_0, p_2_1);
    return _mm256_cvtepi32_ps(MM256_SET_M128I(p_2, p_1));
}

// quad fp16 delta calculation
static inline __m256 quad_fp16_delta_float(const float x0, const float y0, const float x1, const float y1) {
    // GGML_CPU_FP16_TO_FP32 is faster than Intel F16C
    return _mm256_set_m128(_mm_set1_ps(GGML_CPU_FP16_TO_FP32(x1) * GGML_CPU_FP16_TO_FP32(y1)),
                           _mm_set1_ps(GGML_CPU_FP16_TO_FP32(x0) * GGML_CPU_FP16_TO_FP32(y0)));
}

static inline __m256 quad_mx_delta_float(const uint8_t x0, const float y0, const uint8_t x1, const float y1) {
    return _mm256_set_m128(_mm_set1_ps(GGML_CPU_E8M0_TO_FP32_HALF(x1) * GGML_CPU_FP16_TO_FP32(y1)),
                           _mm_set1_ps(GGML_CPU_E8M0_TO_FP32_HALF(x0) * GGML_CPU_FP16_TO_FP32(y0)));
}
#endif
#elif defined(__SSSE3__)
static inline __m128i bytes_from_bits_16(const uint8_t * x) {
    uint16_t x16;
    memcpy(&x16, x, sizeof(uint16_t));

    const __m128i shuf_mask = _mm_set_epi64x(0x0101010101010101, 0x0000000000000000);
    __m128i bytes = _mm_shuffle_epi8(_mm_set1_epi16((short) x16), shuf_mask);
    const __m128i bit_mask = _mm_set_epi64x(0x7fbfdfeff7fbfdfe, 0x7fbfdfeff7fbfdfe);
    bytes = _mm_or_si128(bytes, bit_mask);

    return _mm_cmpeq_epi8(bytes, _mm_set1_epi64x(-1));
}

// horizontally add 4x4 floats
static inline float hsum_float_4x4(const __m128 a, const __m128 b, const __m128 c, const __m128 d) {
    __m128 res_0 =_mm_hadd_ps(a, b);
    __m128 res_1 =_mm_hadd_ps(c, d);
    __m128 res =_mm_hadd_ps(res_0, res_1);
    res =_mm_hadd_ps(res, res);
    res =_mm_hadd_ps(res, res);

    return _mm_cvtss_f32(res);
}
#endif // __AVX__ || __AVX2__ || __AVX512F__
#endif // defined(__AVX__) || defined(__AVX2__) || defined(__AVX512F__) || defined(__SSSE3__)





// placeholder implementation for Apple targets


//===================================== Dot products =================================

//
// Helper functions
//

#if __AVX__ || __AVX2__ || __AVX512F__

// shuffles to pick the required scales in dot products
static inline __m256i get_scale_shuffle_q3k(int i) {
    static const uint8_t k_shuffle[128] = {
         0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1,     2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3,
         4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5,     6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7,
         8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9,    10,11,10,11,10,11,10,11,10,11,10,11,10,11,10,11,
        12,13,12,13,12,13,12,13,12,13,12,13,12,13,12,13,    14,15,14,15,14,15,14,15,14,15,14,15,14,15,14,15,
    };
    return _mm256_loadu_si256((const __m256i*)k_shuffle + i);
}
static inline __m256i get_scale_shuffle_k4(int i) {
    static const uint8_t k_shuffle[256] = {
         0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1,
         2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3,
         4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5,
         6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7,
         8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9,
        10,11,10,11,10,11,10,11,10,11,10,11,10,11,10,11,10,11,10,11,10,11,10,11,10,11,10,11,10,11,10,11,
        12,13,12,13,12,13,12,13,12,13,12,13,12,13,12,13,12,13,12,13,12,13,12,13,12,13,12,13,12,13,12,13,
        14,15,14,15,14,15,14,15,14,15,14,15,14,15,14,15,14,15,14,15,14,15,14,15,14,15,14,15,14,15,14,15
    };
    return _mm256_loadu_si256((const __m256i*)k_shuffle + i);
}
static inline __m128i get_scale_shuffle(int i) {
    static const uint8_t k_shuffle[128] = {
         0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1,
         2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3,
         4, 4, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 5, 5,
         6, 6, 6, 6, 6, 6, 6, 6, 7, 7, 7, 7, 7, 7, 7, 7,
         8, 8, 8, 8, 8, 8, 8, 8, 9, 9, 9, 9, 9, 9, 9, 9,
        10,10,10,10,10,10,10,10, 11,11,11,11,11,11,11,11,
        12,12,12,12,12,12,12,12, 13,13,13,13,13,13,13,13,
        14,14,14,14,14,14,14,14, 15,15,15,15,15,15,15,15
    };
    return _mm_loadu_si128((const __m128i*)k_shuffle + i);
}
#endif

void ggml_vec_dot_q1_0_q8_0(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc) {
    const int qk = QK1_0;
    const int nb = n / qk;

    assert(n % qk == 0);
    assert(nrc == 1);
    UNUSED(nrc);
    UNUSED(bx);
    UNUSED(by);
    UNUSED(bs);

    const block_q1_0 * GGML_RESTRICT x = vx;
    const block_q8_0 * GGML_RESTRICT y = vy;

#if defined(__AVX2__)
    const __m256i ones_8 = _mm256_set1_epi8(1);
    const __m256i ones_16 = _mm256_set1_epi16(1);
    const __m256i byte_shuf = _mm256_setr_epi8(
            0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1,
            2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3);
    const __m256i bit_masks = _mm256_setr_epi8(
            1, 2, 4, 8, 16, 32, 64, (char) -128, 1, 2, 4, 8, 16, 32, 64, (char) -128,
            1, 2, 4, 8, 16, 32, 64, (char) -128, 1, 2, 4, 8, 16, 32, 64, (char) -128);
    const __m256i zero = _mm256_setzero_si256();
    __m256 acc = _mm256_setzero_ps();

    for (int ib = 0; ib < nb; ++ib) {
        const float d0 = GGML_CPU_FP16_TO_FP32(x[ib].d);
        // qs sits at byte offset 2 of the block: read the 32-bit words with memcpy (a uint32_t* load is misaligned, UB; UBSan flagged it)
        const uint8_t * GGML_RESTRICT qs8 = x[ib].qs;
        const block_q8_0 * GGML_RESTRICT y_ptr = &y[ib * 4];
        uint32_t qw;

        __m256 acc_block;
        {
            const __m256i qy = _mm256_loadu_si256((const __m256i *) y_ptr[0].qs);
            const __m256i sm = _mm256_cmpeq_epi8(
                    _mm256_and_si256(_mm256_shuffle_epi8(_mm256_set1_epi32((memcpy(&qw, qs8 + 0, 4), (int) qw)), byte_shuf), bit_masks), zero);
            const __m256i sy = _mm256_sub_epi8(_mm256_xor_si256(qy, sm), sm);
            const __m256i s32 = _mm256_madd_epi16(_mm256_maddubs_epi16(ones_8, sy), ones_16);
            acc_block = _mm256_mul_ps(_mm256_set1_ps(GGML_CPU_FP16_TO_FP32(y_ptr[0].d)), _mm256_cvtepi32_ps(s32));
        }
        for (int K = 1; K < 4; ++K) {
            const __m256i qy = _mm256_loadu_si256((const __m256i *) y_ptr[K].qs);
            const __m256i sm = _mm256_cmpeq_epi8(
                    _mm256_and_si256(_mm256_shuffle_epi8(_mm256_set1_epi32((memcpy(&qw, qs8 + 4 * K, 4), (int) qw)), byte_shuf), bit_masks), zero);
            const __m256i sy = _mm256_sub_epi8(_mm256_xor_si256(qy, sm), sm);
            const __m256i s32 = _mm256_madd_epi16(_mm256_maddubs_epi16(ones_8, sy), ones_16);
            acc_block = _mm256_fmadd_ps(_mm256_set1_ps(GGML_CPU_FP16_TO_FP32(y_ptr[K].d)), _mm256_cvtepi32_ps(s32), acc_block);
        }
        acc = _mm256_fmadd_ps(_mm256_set1_ps(d0), acc_block, acc);
    }

    *s = hsum_float_8(acc);
#elif defined(__AVX__)
    const __m128i ones_8 = _mm_set1_epi8(1);
    const __m128i ones_16 = _mm_set1_epi16(1);
    const __m128i zero = _mm_setzero_si128();
    __m256 acc = _mm256_setzero_ps();

    for (int ib = 0; ib < nb; ++ib) {
        const float d0 = GGML_CPU_FP16_TO_FP32(x[ib].d);
        const block_q8_0 * GGML_RESTRICT y_ptr = &y[ib * 4];
        __m256 acc_block;
        {
            const __m256i bit_mask = bytes_from_bits_32(&x[ib].qs[0]);
            const __m128i bit_mask_0 = _mm256_castsi256_si128(bit_mask);
            const __m128i bit_mask_1 = _mm256_extractf128_si256(bit_mask, 1);
            const __m128i qy_0 = _mm_loadu_si128((const __m128i *) &y_ptr[0].qs[0]);
            const __m128i qy_1 = _mm_loadu_si128((const __m128i *) &y_ptr[0].qs[16]);
            const __m128i sign_mask_0 = _mm_cmpeq_epi8(bit_mask_0, zero);
            const __m128i sign_mask_1 = _mm_cmpeq_epi8(bit_mask_1, zero);
            const __m128i sy_0 = _mm_sub_epi8(_mm_xor_si128(qy_0, sign_mask_0), sign_mask_0);
            const __m128i sy_1 = _mm_sub_epi8(_mm_xor_si128(qy_1, sign_mask_1), sign_mask_1);
            const __m128i sum16_0 = _mm_maddubs_epi16(ones_8, sy_0);
            const __m128i sum16_1 = _mm_maddubs_epi16(ones_8, sy_1);
            const __m128i sum32_0 = _mm_madd_epi16(sum16_0, ones_16);
            const __m128i sum32_1 = _mm_madd_epi16(sum16_1, ones_16);
            const __m256 q = _mm256_cvtepi32_ps(MM256_SET_M128I(sum32_1, sum32_0));
            acc_block = _mm256_mul_ps(_mm256_set1_ps(GGML_CPU_FP16_TO_FP32(y_ptr[0].d)), q);
        }
        for(int K = 1; K < 4; ++K) {
            const __m256i bit_mask = bytes_from_bits_32(&x[ib].qs[(K) * 4]);
            const __m128i bit_mask_0 = _mm256_castsi256_si128(bit_mask);
            const __m128i bit_mask_1 = _mm256_extractf128_si256(bit_mask, 1);
            const __m128i qy_0 = _mm_loadu_si128((const __m128i *) &y_ptr[(K)].qs[0]);
            const __m128i qy_1 = _mm_loadu_si128((const __m128i *) &y_ptr[(K)].qs[16]);
            const __m128i sign_mask_0 = _mm_cmpeq_epi8(bit_mask_0, zero);
            const __m128i sign_mask_1 = _mm_cmpeq_epi8(bit_mask_1, zero);
            const __m128i sy_0 = _mm_sub_epi8(_mm_xor_si128(qy_0, sign_mask_0), sign_mask_0);
            const __m128i sy_1 = _mm_sub_epi8(_mm_xor_si128(qy_1, sign_mask_1), sign_mask_1);
            const __m128i sum16_0 = _mm_maddubs_epi16(ones_8, sy_0);
            const __m128i sum16_1 = _mm_maddubs_epi16(ones_8, sy_1);
            const __m128i sum32_0 = _mm_madd_epi16(sum16_0, ones_16);
            const __m128i sum32_1 = _mm_madd_epi16(sum16_1, ones_16);
            const __m256 q = _mm256_cvtepi32_ps(MM256_SET_M128I(sum32_1, sum32_0));
            acc_block = _mm256_add_ps(acc_block, _mm256_mul_ps(_mm256_set1_ps(GGML_CPU_FP16_TO_FP32(y_ptr[(K)].d)), q));
        }
#undef Q1_AVX_BLOCK

        acc = _mm256_add_ps(acc, _mm256_mul_ps(_mm256_set1_ps(d0), acc_block));
    }

    *s = hsum_float_8(acc);
#elif defined(__SSSE3__)
    const __m128i ones_8 = _mm_set1_epi8(1);
    const __m128i ones_16 = _mm_set1_epi16(1);
    const __m128i zero = _mm_setzero_si128();
    __m128 acc_0 = _mm_setzero_ps();
    __m128 acc_1 = _mm_setzero_ps();
    __m128 acc_2 = _mm_setzero_ps();
    __m128 acc_3 = _mm_setzero_ps();

    for (int ib = 0; ib < nb; ++ib) {
        const __m128 d0 = _mm_set1_ps(GGML_CPU_FP16_TO_FP32(x[ib].d));
        const block_q8_0 * GGML_RESTRICT y_ptr = &y[ib * 4];

#define Q1_SSSE3_BLOCK(QS_OFF, Y_IDX, ACC) \
        { \
            const __m128i bit_mask_0 = bytes_from_bits_16(&x[ib].qs[(QS_OFF) + 0]); \
            const __m128i bit_mask_1 = bytes_from_bits_16(&x[ib].qs[(QS_OFF) + 2]); \
            const __m128i qy_0 = _mm_loadu_si128((const __m128i *) &y_ptr[(Y_IDX)].qs[0]); \
            const __m128i qy_1 = _mm_loadu_si128((const __m128i *) &y_ptr[(Y_IDX)].qs[16]); \
            const __m128i sign_mask_0 = _mm_cmpeq_epi8(bit_mask_0, zero); \
            const __m128i sign_mask_1 = _mm_cmpeq_epi8(bit_mask_1, zero); \
            const __m128i sy_0 = _mm_sub_epi8(_mm_xor_si128(qy_0, sign_mask_0), sign_mask_0); \
            const __m128i sy_1 = _mm_sub_epi8(_mm_xor_si128(qy_1, sign_mask_1), sign_mask_1); \
            const __m128i sum_0 = _mm_madd_epi16(_mm_maddubs_epi16(ones_8, sy_0), ones_16); \
            const __m128i sum_1 = _mm_madd_epi16(_mm_maddubs_epi16(ones_8, sy_1), ones_16); \
            const __m128 q = _mm_cvtepi32_ps(_mm_add_epi32(sum_0, sum_1)); \
            (ACC) = _mm_add_ps((ACC), _mm_mul_ps(_mm_mul_ps(d0, _mm_set1_ps(GGML_CPU_FP16_TO_FP32(y_ptr[(Y_IDX)].d))), q)); \
        }
        Q1_SSSE3_BLOCK(0,  0, acc_0)
        Q1_SSSE3_BLOCK(4,  1, acc_1)
        Q1_SSSE3_BLOCK(8,  2, acc_2)
        Q1_SSSE3_BLOCK(12, 3, acc_3)
#undef Q1_SSSE3_BLOCK
    }

    *s = hsum_float_4x4(acc_0, acc_1, acc_2, acc_3);
#else
    UNUSED(nb);
    UNUSED(x);
    UNUSED(y);
    ggml_vec_dot_q1_0_q8_0_generic(n, s, bs, vx, bx, vy, by, nrc);
#endif
}

#if defined(__AVX2__)
// four consecutive Q2_0 weights per byte: byte b becomes b & 3, (b >> 2) & 3, (b >> 4) & 3, (b >> 6) & 3 (OR, never add: the overlapping copies only touch masked-off bits)
static inline __m256i q2_0_unpack_32_avx2(const uint8_t * qs) {
    __m256i v = _mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i *) qs));
    v = _mm256_or_si256(v, _mm256_slli_epi32(v, 12));
    v = _mm256_or_si256(v, _mm256_slli_epi32(v,  6));
    return _mm256_and_si256(v, _mm256_set1_epi32(0x03030303));
}

// 32 weights against one Q8_0 block: sum((v - 1) * y) = sum(v * y) - sum(y), exact for every int8 y
static inline __m256 q2_0_dot_32_avx2(const uint8_t * qs, const int8_t * qy) {
    const __m256i v = q2_0_unpack_32_avx2(qs);
    const __m256i y = _mm256_loadu_si256((const __m256i *) qy);
    const __m256i p = _mm256_sub_epi16(_mm256_maddubs_epi16(v, y), _mm256_maddubs_epi16(_mm256_set1_epi8(1), y));
    return _mm256_cvtepi32_ps(_mm256_madd_epi16(p, _mm256_set1_epi16(1)));
}

static inline __m256 q2_0_acc_block_avx2(const block_q2_0 * x, const block_q8_0 * y, __m256 acc) {
    const float  dx = GGML_CPU_FP16_TO_FP32(x->d);
    const __m256 d0 = _mm256_set1_ps(dx * GGML_CPU_FP16_TO_FP32(y[0].d));
    const __m256 d1 = _mm256_set1_ps(dx * GGML_CPU_FP16_TO_FP32(y[1].d));
    const __m256 p0 = q2_0_dot_32_avx2(x->qs + 0, y[0].qs);
    const __m256 p1 = q2_0_dot_32_avx2(x->qs + 8, y[1].qs);
#if defined(__FMA__)
    acc = _mm256_fmadd_ps(d0, p0, acc);
    return _mm256_fmadd_ps(d1, p1, acc);
#else
    acc = _mm256_add_ps(acc, _mm256_mul_ps(d0, p0));
    return _mm256_add_ps(acc, _mm256_mul_ps(d1, p1));
#endif
}
#endif

void ggml_vec_dot_q2_0_q8_0(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc) {
#if defined(__AVX2__)
    assert(n % QK2_0 == 0);
    assert(nrc == 1);
    UNUSED(nrc);
    UNUSED(bx);
    UNUSED(by);
    UNUSED(bs);

    const block_q2_0 * GGML_RESTRICT x = vx;
    const block_q8_0 * GGML_RESTRICT y = vy;
    const int nb = n / QK2_0;

    // two Q2_0 blocks per step, each into its own accumulator (two FMA chains)
    __m256 acc0 = _mm256_setzero_ps();
    __m256 acc1 = _mm256_setzero_ps();
    int i = 0;
    for (; i + 1 < nb; i += 2) {
        acc0 = q2_0_acc_block_avx2(x + i + 0, y + 2*i + 0, acc0);
        acc1 = q2_0_acc_block_avx2(x + i + 1, y + 2*i + 2, acc1);
    }
    if (i < nb) {
        acc0 = q2_0_acc_block_avx2(x + i, y + 2*i, acc0);
    }

    *s = hsum_float_8(_mm256_add_ps(acc0, acc1));
#else
    ggml_vec_dot_q2_0_q8_0_generic(n, s, bs, vx, bx, vy, by, nrc);
#endif
}




void ggml_vec_dot_mxfp4_q8_0(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc) {
    assert(nrc == 1);
    UNUSED(nrc);
    UNUSED(bx);
    UNUSED(by);
    UNUSED(bs);
    assert(n % QK_MXFP4 == 0);
    static_assert(QK_MXFP4 == QK8_0, "QK_MXFP4 and QK8_0 must be the same");

    const block_mxfp4 * GGML_RESTRICT x = vx;
    const block_q8_0 * GGML_RESTRICT y = vy;

    const int nb = n / QK_MXFP4;

    int ib = 0;
    float sumf = 0;

#if defined __AVX2__

    const __m128i values128 = _mm_loadu_si128((const __m128i*)kvalues_fp4);
    const __m128i m4b  = _mm_set1_epi8(0x0f);
    const __m256i mone = _mm256_set1_epi16(1);

    __m256 accum1 = _mm256_setzero_ps();
    __m256 accum2 = _mm256_setzero_ps();

    for (; ib + 1 < nb; ib += 2) {
        const __m128i q4bits_1 = _mm_loadu_si128((const __m128i*)x[ib + 0].qs);
        const __m128i q4bits_2 = _mm_loadu_si128((const __m128i*)x[ib + 1].qs);
        const __m256i q8b_1 = _mm256_loadu_si256((const __m256i *)y[ib + 0].qs);
        const __m256i q8b_2 = _mm256_loadu_si256((const __m256i *)y[ib + 1].qs);
        const __m256i q4b_1 = MM256_SET_M128I(_mm_shuffle_epi8(values128, _mm_and_si128(_mm_srli_epi16(q4bits_1, 4), m4b)),
                                              _mm_shuffle_epi8(values128, _mm_and_si128(q4bits_1, m4b)));
        const __m256i q4b_2 = MM256_SET_M128I(_mm_shuffle_epi8(values128, _mm_and_si128(_mm_srli_epi16(q4bits_2, 4), m4b)),
                                              _mm_shuffle_epi8(values128, _mm_and_si128(q4bits_2, m4b)));
        const __m256i p16_1 = mul_add_epi8(q4b_1, q8b_1);
        const __m256i p16_2 = mul_add_epi8(q4b_2, q8b_2);
        const __m256i p_1 = _mm256_madd_epi16(p16_1, mone);
        const __m256i p_2 = _mm256_madd_epi16(p16_2, mone);
        const __m256 scale0 = _mm256_set1_ps(GGML_CPU_FP16_TO_FP32(y[ib + 0].d)*GGML_CPU_E8M0_TO_FP32_HALF(x[ib + 0].e));
        const __m256 scale1 = _mm256_set1_ps(GGML_CPU_FP16_TO_FP32(y[ib + 1].d)*GGML_CPU_E8M0_TO_FP32_HALF(x[ib + 1].e));
        accum1 = _mm256_fmadd_ps(scale0, _mm256_cvtepi32_ps(p_1), accum1);
        accum2 = _mm256_fmadd_ps(scale1, _mm256_cvtepi32_ps(p_2), accum2);
    }

    sumf = hsum_float_8(_mm256_add_ps(accum1, accum2));

#elif defined __AVX__
    const __m128i values128 = _mm_loadu_si128((const __m128i*)kvalues_fp4);
    const __m128i m4b  = _mm_set1_epi8(0x0f);

    __m256 accum = _mm256_setzero_ps();
    for (; ib + 1 < nb; ib += 2) {
        const __m128i q4bits_1 = _mm_loadu_si128((const __m128i *)x[ib + 0].qs);
        const __m128i q4bits_2 = _mm_loadu_si128((const __m128i *)x[ib + 1].qs);
        const __m128i q8b_1_0 = _mm_loadu_si128((const __m128i *)y[ib + 0].qs);
        const __m128i q8b_1_1 = _mm_loadu_si128((const __m128i *)y[ib + 0].qs + 1);
        const __m128i q8b_2_0 = _mm_loadu_si128((const __m128i *)y[ib + 1].qs);
        const __m128i q8b_2_1 = _mm_loadu_si128((const __m128i *)y[ib + 1].qs + 1);

        const __m128i q4b_1_0 = _mm_shuffle_epi8(values128, _mm_and_si128(q4bits_1, m4b));
        const __m128i q4b_1_1 = _mm_shuffle_epi8(values128, _mm_and_si128(_mm_srli_epi16(q4bits_1, 4), m4b));
        const __m128i q4b_2_0 = _mm_shuffle_epi8(values128, _mm_and_si128(q4bits_2, m4b));
        const __m128i q4b_2_1 = _mm_shuffle_epi8(values128, _mm_and_si128(_mm_srli_epi16(q4bits_2, 4), m4b));

        const __m256 p = mul_sum_i8_quad_float(q4b_1_0, q4b_1_1, q4b_2_0, q4b_2_1, q8b_1_0, q8b_1_1, q8b_2_0, q8b_2_1);
        const __m256 deltas = quad_mx_delta_float(x[ib].e, y[ib].d, x[ib + 1].e, y[ib + 1].d);
        accum = _mm256_add_ps(_mm256_mul_ps(deltas, p), accum);
    }

    sumf = hsum_float_8(accum);

#endif
    for (; ib < nb; ++ib) {
        const float d = GGML_CPU_FP16_TO_FP32(y[ib].d)*GGML_CPU_E8M0_TO_FP32_HALF(x[ib].e);
        int sumi1 = 0;
        int sumi2 = 0;
        for (int j = 0; j < QK_MXFP4/2; ++j) {
            sumi1 += y[ib].qs[j +          0] * kvalues_fp4[x[ib].qs[j] & 0xf];
            sumi2 += y[ib].qs[j + QK_MXFP4/2] * kvalues_fp4[x[ib].qs[j] >>  4];
        }
        sumf += d * (sumi1 + sumi2);
    }
    *s = sumf;
}

void ggml_vec_dot_nvfp4_q8_0(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc) {
    assert(nrc == 1);
    UNUSED(nrc);
    UNUSED(bx);
    UNUSED(by);
    UNUSED(bs);
    assert(n % QK_NVFP4 == 0);

    const block_nvfp4 * GGML_RESTRICT x = vx;
    const block_q8_0 * GGML_RESTRICT y = vy;

    const int nb = n / QK_NVFP4;
    int ib = 0;
    float sumf = 0;

#if defined(__AVX2__)

    const __m128i values128 = _mm_loadu_si128((const __m128i*)kvalues_fp4);
    const __m128i m4b  = _mm_set1_epi8(0x0f);
    const __m256i mone = _mm256_set1_epi16(1);

    __m256 accum = _mm256_setzero_ps();
    for(; ib < nb; ib++){

        const __m128i q4bits_01 = _mm_loadu_si128((const __m128i *)(x[ib].qs + 0));
        const __m128i q4bits_23 = _mm_loadu_si128((const __m128i *)(x[ib].qs + 16));

        const __m256i q8_01 = _mm256_loadu_si256((const __m256i *)y[2*ib + 0].qs);
        const __m256i q8_23 = _mm256_loadu_si256((const __m256i *)y[2*ib + 1].qs);

        const __m128i q4_01_lo = _mm_shuffle_epi8(values128, _mm_and_si128(q4bits_01, m4b));
        const __m128i q4_01_hi = _mm_shuffle_epi8(values128, _mm_and_si128(_mm_srli_epi16(q4bits_01, 4), m4b));
        const __m128i q4_23_lo = _mm_shuffle_epi8(values128, _mm_and_si128(q4bits_23, m4b));
        const __m128i q4_23_hi = _mm_shuffle_epi8(values128, _mm_and_si128(_mm_srli_epi16(q4bits_23, 4), m4b));

        //reordering
        const __m256i q4_01 = MM256_SET_M128I(_mm_unpackhi_epi64(q4_01_lo,q4_01_hi), _mm_unpacklo_epi64(q4_01_lo,q4_01_hi));
        const __m256i q4_23 = MM256_SET_M128I(_mm_unpackhi_epi64(q4_23_lo,q4_23_hi),_mm_unpacklo_epi64(q4_23_lo,q4_23_hi));

        const __m256i p01 = mul_add_epi8(q4_01,q8_01);
        const __m256i p_1 = _mm256_madd_epi16(p01, mone);

        const __m256i p23 = mul_add_epi8(q4_23,q8_23);
        const __m256i p_2 = _mm256_madd_epi16(p23, mone);

        const float dy0 = GGML_CPU_FP16_TO_FP32(y[2*ib].d);
        const float dy1 = GGML_CPU_FP16_TO_FP32(y[2*ib+1].d);

        const float s0 = GGML_CPU_UE4M3_TO_FP32(x[ib].d[0]) * dy0;
        const float s1 = GGML_CPU_UE4M3_TO_FP32(x[ib].d[1]) * dy0;
        const float s2 = GGML_CPU_UE4M3_TO_FP32(x[ib].d[2]) * dy1;
        const float s3 = GGML_CPU_UE4M3_TO_FP32(x[ib].d[3]) * dy1;

        const __m256 scales01 = _mm256_set_m128(_mm_set1_ps(s1), _mm_set1_ps(s0));
        const __m256 scales23 = _mm256_set_m128(_mm_set1_ps(s3), _mm_set1_ps(s2));

        accum = _mm256_fmadd_ps(scales01, _mm256_cvtepi32_ps(p_1), accum);
        accum = _mm256_fmadd_ps(scales23, _mm256_cvtepi32_ps(p_2), accum);
    }
    sumf = hsum_float_8(accum);

#elif defined(__AVX__)

    const __m128i values128 = _mm_loadu_si128((const __m128i*)kvalues_fp4);
    const __m128i m4b  = _mm_set1_epi8(0x0f);

    __m256 accum = _mm256_setzero_ps();
    for(; ib < nb; ib++){

        const __m128i q4bits_01 = _mm_loadu_si128((const __m128i *)(x[ib].qs + 0));
        const __m128i q4bits_23 = _mm_loadu_si128((const __m128i *)(x[ib].qs + 16));

        const __m128i q8_0 = _mm_loadu_si128((const __m128i *)(y[2*ib + 0].qs + 0));
        const __m128i q8_1 = _mm_loadu_si128((const __m128i *)(y[2*ib + 0].qs + 16));
        const __m128i q8_2 = _mm_loadu_si128((const __m128i *)(y[2*ib + 1].qs + 0));
        const __m128i q8_3 = _mm_loadu_si128((const __m128i *)(y[2*ib + 1].qs + 16));

        const __m128i q4_01_lo = _mm_shuffle_epi8(values128, _mm_and_si128(q4bits_01, m4b));
        const __m128i q4_01_hi = _mm_shuffle_epi8(values128, _mm_and_si128(_mm_srli_epi16(q4bits_01, 4), m4b));
        const __m128i q4_23_lo = _mm_shuffle_epi8(values128, _mm_and_si128(q4bits_23, m4b));
        const __m128i q4_23_hi = _mm_shuffle_epi8(values128, _mm_and_si128(_mm_srli_epi16(q4bits_23, 4), m4b));

        const __m128i q4_0 = _mm_unpacklo_epi64(q4_01_lo, q4_01_hi);
        const __m128i q4_1 = _mm_unpackhi_epi64(q4_01_lo, q4_01_hi);
        const __m128i q4_2 = _mm_unpacklo_epi64(q4_23_lo, q4_23_hi);
        const __m128i q4_3 = _mm_unpackhi_epi64(q4_23_lo, q4_23_hi);

        const __m128i p0_i32 = mul_sum_i8_pairs(q4_0, q8_0);
        const __m128i p1_i32 = mul_sum_i8_pairs(q4_1, q8_1);
        const __m128i p2_i32 = mul_sum_i8_pairs(q4_2, q8_2);
        const __m128i p3_i32 = mul_sum_i8_pairs(q4_3, q8_3);

        const __m128 p0 = _mm_cvtepi32_ps(p0_i32);
        const __m128 p1 = _mm_cvtepi32_ps(p1_i32);
        const __m128 p2 = _mm_cvtepi32_ps(p2_i32);
        const __m128 p3 = _mm_cvtepi32_ps(p3_i32);

        const __m256 p01 = _mm256_set_m128(p1, p0);
        const __m256 p23 = _mm256_set_m128(p3, p2);

        const float dy0 = GGML_CPU_FP16_TO_FP32(y[2*ib].d);
        const float dy1 = GGML_CPU_FP16_TO_FP32(y[2*ib+1].d);

        const float s0 = GGML_CPU_UE4M3_TO_FP32(x[ib].d[0]) * dy0;
        const float s1 = GGML_CPU_UE4M3_TO_FP32(x[ib].d[1]) * dy0;
        const float s2 = GGML_CPU_UE4M3_TO_FP32(x[ib].d[2]) * dy1;
        const float s3 = GGML_CPU_UE4M3_TO_FP32(x[ib].d[3]) * dy1;

        const __m256 scales01 = _mm256_set_m128(_mm_set1_ps(s1), _mm_set1_ps(s0));
        const __m256 scales23 = _mm256_set_m128(_mm_set1_ps(s3), _mm_set1_ps(s2));

        accum = _mm256_add_ps(accum, _mm256_mul_ps(p01, scales01));
        accum = _mm256_add_ps(accum, _mm256_mul_ps(p23, scales23));
    }
    sumf = hsum_float_8(accum);

#endif

    for (;ib < nb; ++ib) {
        for (int s_idx = 0; s_idx < 4; ++s_idx) {
            const float d = GGML_CPU_UE4M3_TO_FP32(x[ib].d[s_idx]);
            const int q8_block = s_idx / 2;
            const int q8_off   = (s_idx % 2) * QK_NVFP4_SUB;
            const float dy = GGML_CPU_FP16_TO_FP32(y[2*ib + q8_block].d);

            int sumi_lo = 0, sumi_hi = 0;
            for (int j = 0; j < QK_NVFP4_SUB/2; ++j) {
                const uint8_t qv = x[ib].qs[s_idx*(QK_NVFP4_SUB/2) + j];
                sumi_lo += y[2*ib + q8_block].qs[q8_off + j +               0] * kvalues_fp4[qv & 0xf];
                sumi_hi += y[2*ib + q8_block].qs[q8_off + j + QK_NVFP4_SUB/2] * kvalues_fp4[qv >>  4];
            }

            sumf += dy * d * (sumi_lo + sumi_hi);
        }
    }
    *s = sumf;
}





















#if defined (__AVX__) || defined (__AVX2__)
static const int8_t keven_signs_q2xs[1024] = {
     1,  1,  1,  1,  1,  1,  1,  1, -1,  1,  1,  1,  1,  1,  1, -1,  1, -1,  1,  1,  1,  1,  1, -1, -1, -1,  1,  1,  1,  1,  1,  1,
     1,  1, -1,  1,  1,  1,  1, -1, -1,  1, -1,  1,  1,  1,  1,  1,  1, -1, -1,  1,  1,  1,  1,  1, -1, -1, -1,  1,  1,  1,  1, -1,
     1,  1,  1, -1,  1,  1,  1, -1, -1,  1,  1, -1,  1,  1,  1,  1,  1, -1,  1, -1,  1,  1,  1,  1, -1, -1,  1, -1,  1,  1,  1, -1,
     1,  1, -1, -1,  1,  1,  1,  1, -1,  1, -1, -1,  1,  1,  1, -1,  1, -1, -1, -1,  1,  1,  1, -1, -1, -1, -1, -1,  1,  1,  1,  1,
     1,  1,  1,  1, -1,  1,  1, -1, -1,  1,  1,  1, -1,  1,  1,  1,  1, -1,  1,  1, -1,  1,  1,  1, -1, -1,  1,  1, -1,  1,  1, -1,
     1,  1, -1,  1, -1,  1,  1,  1, -1,  1, -1,  1, -1,  1,  1, -1,  1, -1, -1,  1, -1,  1,  1, -1, -1, -1, -1,  1, -1,  1,  1,  1,
     1,  1,  1, -1, -1,  1,  1,  1, -1,  1,  1, -1, -1,  1,  1, -1,  1, -1,  1, -1, -1,  1,  1, -1, -1, -1,  1, -1, -1,  1,  1,  1,
     1,  1, -1, -1, -1,  1,  1, -1, -1,  1, -1, -1, -1,  1,  1,  1,  1, -1, -1, -1, -1,  1,  1,  1, -1, -1, -1, -1, -1,  1,  1, -1,
     1,  1,  1,  1,  1, -1,  1, -1, -1,  1,  1,  1,  1, -1,  1,  1,  1, -1,  1,  1,  1, -1,  1,  1, -1, -1,  1,  1,  1, -1,  1, -1,
     1,  1, -1,  1,  1, -1,  1,  1, -1,  1, -1,  1,  1, -1,  1, -1,  1, -1, -1,  1,  1, -1,  1, -1, -1, -1, -1,  1,  1, -1,  1,  1,
     1,  1,  1, -1,  1, -1,  1,  1, -1,  1,  1, -1,  1, -1,  1, -1,  1, -1,  1, -1,  1, -1,  1, -1, -1, -1,  1, -1,  1, -1,  1,  1,
     1,  1, -1, -1,  1, -1,  1, -1, -1,  1, -1, -1,  1, -1,  1,  1,  1, -1, -1, -1,  1, -1,  1,  1, -1, -1, -1, -1,  1, -1,  1, -1,
     1,  1,  1,  1, -1, -1,  1,  1, -1,  1,  1,  1, -1, -1,  1, -1,  1, -1,  1,  1, -1, -1,  1, -1, -1, -1,  1,  1, -1, -1,  1,  1,
     1,  1, -1,  1, -1, -1,  1, -1, -1,  1, -1,  1, -1, -1,  1,  1,  1, -1, -1,  1, -1, -1,  1,  1, -1, -1, -1,  1, -1, -1,  1, -1,
     1,  1,  1, -1, -1, -1,  1, -1, -1,  1,  1, -1, -1, -1,  1,  1,  1, -1,  1, -1, -1, -1,  1,  1, -1, -1,  1, -1, -1, -1,  1, -1,
     1,  1, -1, -1, -1, -1,  1,  1, -1,  1, -1, -1, -1, -1,  1, -1,  1, -1, -1, -1, -1, -1,  1, -1, -1, -1, -1, -1, -1, -1,  1,  1,
     1,  1,  1,  1,  1,  1, -1, -1, -1,  1,  1,  1,  1,  1, -1,  1,  1, -1,  1,  1,  1,  1, -1,  1, -1, -1,  1,  1,  1,  1, -1, -1,
     1,  1, -1,  1,  1,  1, -1,  1, -1,  1, -1,  1,  1,  1, -1, -1,  1, -1, -1,  1,  1,  1, -1, -1, -1, -1, -1,  1,  1,  1, -1,  1,
     1,  1,  1, -1,  1,  1, -1,  1, -1,  1,  1, -1,  1,  1, -1, -1,  1, -1,  1, -1,  1,  1, -1, -1, -1, -1,  1, -1,  1,  1, -1,  1,
     1,  1, -1, -1,  1,  1, -1, -1, -1,  1, -1, -1,  1,  1, -1,  1,  1, -1, -1, -1,  1,  1, -1,  1, -1, -1, -1, -1,  1,  1, -1, -1,
     1,  1,  1,  1, -1,  1, -1,  1, -1,  1,  1,  1, -1,  1, -1, -1,  1, -1,  1,  1, -1,  1, -1, -1, -1, -1,  1,  1, -1,  1, -1,  1,
     1,  1, -1,  1, -1,  1, -1, -1, -1,  1, -1,  1, -1,  1, -1,  1,  1, -1, -1,  1, -1,  1, -1,  1, -1, -1, -1,  1, -1,  1, -1, -1,
     1,  1,  1, -1, -1,  1, -1, -1, -1,  1,  1, -1, -1,  1, -1,  1,  1, -1,  1, -1, -1,  1, -1,  1, -1, -1,  1, -1, -1,  1, -1, -1,
     1,  1, -1, -1, -1,  1, -1,  1, -1,  1, -1, -1, -1,  1, -1, -1,  1, -1, -1, -1, -1,  1, -1, -1, -1, -1, -1, -1, -1,  1, -1,  1,
     1,  1,  1,  1,  1, -1, -1,  1, -1,  1,  1,  1,  1, -1, -1, -1,  1, -1,  1,  1,  1, -1, -1, -1, -1, -1,  1,  1,  1, -1, -1,  1,
     1,  1, -1,  1,  1, -1, -1, -1, -1,  1, -1,  1,  1, -1, -1,  1,  1, -1, -1,  1,  1, -1, -1,  1, -1, -1, -1,  1,  1, -1, -1, -1,
     1,  1,  1, -1,  1, -1, -1, -1, -1,  1,  1, -1,  1, -1, -1,  1,  1, -1,  1, -1,  1, -1, -1,  1, -1, -1,  1, -1,  1, -1, -1, -1,
     1,  1, -1, -1,  1, -1, -1,  1, -1,  1, -1, -1,  1, -1, -1, -1,  1, -1, -1, -1,  1, -1, -1, -1, -1, -1, -1, -1,  1, -1, -1,  1,
     1,  1,  1,  1, -1, -1, -1, -1, -1,  1,  1,  1, -1, -1, -1,  1,  1, -1,  1,  1, -1, -1, -1,  1, -1, -1,  1,  1, -1, -1, -1, -1,
     1,  1, -1,  1, -1, -1, -1,  1, -1,  1, -1,  1, -1, -1, -1, -1,  1, -1, -1,  1, -1, -1, -1, -1, -1, -1, -1,  1, -1, -1, -1,  1,
     1,  1,  1, -1, -1, -1, -1,  1, -1,  1,  1, -1, -1, -1, -1, -1,  1, -1,  1, -1, -1, -1, -1, -1, -1, -1,  1, -1, -1, -1, -1,  1,
     1,  1, -1, -1, -1, -1, -1, -1, -1,  1, -1, -1, -1, -1, -1,  1,  1, -1, -1, -1, -1, -1, -1,  1, -1, -1, -1, -1, -1, -1, -1, -1,
};
#endif

// ---------------------------------------------------------------------------------------------
// DUPLICATE DEFINITIONS REMOVED 2026-09-13. The definitions that stood here duplicated ggml-quants.c,
// which is compiled into ggml-base, so BOTH archives defined them and every executable in the tree
// failed to link:
//
//     ld: ggml-base.a(ggml-quants.c.obj): multiple definition of `ggml_vec_dot_q4_0_q8_0';
//         ggml-cpu.a(quants.c.obj): first defined here                      ... 37 in total
//
// THE FORK'S COPIES ARE THE ONES THAT SURVIVE, and that direction is measured, not assumed. Ten of
// the thirty-seven call iqk_mul_mat / iqk_quantize_q4_0 / iqk_quantize_row_q8_K, or carry a
// commented-out upstream call beside the replacement -- ggml-quants.c:709 is
// `quantize_row_q4_0(...) { iqk_quantize_q4_0(x, y, k); //quantize_row_q4_0_ref(x, y, k); }`.
// Twenty-two more differ in body size by 2-4x. Removing THOSE instead would have swapped this
// fork's IQK kernels for upstream's generic ones with nothing in the build saying so.
//
// Before this change both definitions existed and LINK ORDER decided which one ran. This does not
// introduce a choice; it removes an accident.
//
// KEPT HERE: the _generic fallbacks, and the eight symbols this file uniquely provides --
// ggml_vec_dot_{mxfp4,nvfp4,q1_0,q2_0}_q8_0 and quantize_row_{mxfp4,nvfp4,q1_0,q2_0} -- which are
// this fork's own quant types and are referenced by ggml-cpu.c and ggml.c. Those references are why
// this file could not simply be dropped from the target; measured with nm -u, see
// temp/_cpuside_out.txt.
// ---------------------------------------------------------------------------------------------

// ---------------------------------------------------------------------------------------------
// quantize_row_q8_0_cpu -- the AVX/AVX2 activation quantiser for Q8_0, restored 2026-09-26.
//
// Upstream's x86 quantize_row_q8_0 (this file, verbatim arms below; ggml-org/llama.cpp, MIT) was
// removed by the de-duplication pass that left quants.c.dup37.bak beside this file, because
// ggml-base's ggml-quants.c defines the same symbol. ggml-base is compiled WITHOUT -mavx2
// (build.ninja: ggml-quants.c.obj has no -m flags), so the survivor runs its scalar arm,
// quantize_row_q8_0_ref: id = 1/d with roundf (half away from zero). Upstream's CPU backend
// multiplies by 127/amax and rounds half-to-even. Every Q8_0 activation the generic mul_mat path
// quantises on x86 therefore differed from upstream's, and ran scalar.
// Named _cpu so it cannot collide with ggml-base's symbol again; type_traits_cpu[Q8_0].from_float
// points here on AVX/AVX2 builds (ggml-cpu.c). Row
// S69-FORK-GENERIC-Q8-0-PATH-DEVIATES-0-47PCT-FROM-EXACT-2026-09-26-01.
// ---------------------------------------------------------------------------------------------
void quantize_row_q8_0_cpu(const float * GGML_RESTRICT x, void * GGML_RESTRICT vy, int64_t k) {
    assert(QK8_0 == 32);
    assert(k % QK8_0 == 0);
    const int nb = k / QK8_0;

    block_q8_0 * GGML_RESTRICT y = vy;

#if defined(__AVX2__) || defined(__AVX__)
    for (int i = 0; i < nb; i++) {
        // Load elements into 4 AVX vectors
        __m256 v0 = _mm256_loadu_ps( x );
        __m256 v1 = _mm256_loadu_ps( x + 8 );
        __m256 v2 = _mm256_loadu_ps( x + 16 );
        __m256 v3 = _mm256_loadu_ps( x + 24 );
        x += 32;

        // Compute max(abs(e)) for the block
        const __m256 signBit = _mm256_set1_ps( -0.0f );
        __m256 maxAbs = _mm256_andnot_ps( signBit, v0 );
        maxAbs = _mm256_max_ps( maxAbs, _mm256_andnot_ps( signBit, v1 ) );
        maxAbs = _mm256_max_ps( maxAbs, _mm256_andnot_ps( signBit, v2 ) );
        maxAbs = _mm256_max_ps( maxAbs, _mm256_andnot_ps( signBit, v3 ) );

        __m128 max4 = _mm_max_ps( _mm256_extractf128_ps( maxAbs, 1 ), _mm256_castps256_ps128( maxAbs ) );
        max4 = _mm_max_ps( max4, _mm_movehl_ps( max4, max4 ) );
        max4 = _mm_max_ss( max4, _mm_movehdup_ps( max4 ) );
        const float maxScalar = _mm_cvtss_f32( max4 );

        // Quantize these floats
        const float d = maxScalar / 127.f;
        y[i].d = GGML_CPU_FP32_TO_FP16(d);
        const float id = ( maxScalar != 0.0f ) ? 127.f / maxScalar : 0.0f;
        const __m256 mul = _mm256_set1_ps( id );

        // Apply the multiplier
        v0 = _mm256_mul_ps( v0, mul );
        v1 = _mm256_mul_ps( v1, mul );
        v2 = _mm256_mul_ps( v2, mul );
        v3 = _mm256_mul_ps( v3, mul );

        // Round to nearest integer
        v0 = _mm256_round_ps( v0, _MM_ROUND_NEAREST );
        v1 = _mm256_round_ps( v1, _MM_ROUND_NEAREST );
        v2 = _mm256_round_ps( v2, _MM_ROUND_NEAREST );
        v3 = _mm256_round_ps( v3, _MM_ROUND_NEAREST );

        // Convert floats to integers
        __m256i i0 = _mm256_cvtps_epi32( v0 );
        __m256i i1 = _mm256_cvtps_epi32( v1 );
        __m256i i2 = _mm256_cvtps_epi32( v2 );
        __m256i i3 = _mm256_cvtps_epi32( v3 );

#if defined(__AVX2__)
        // Convert int32 to int16, then int16 to int8
        i0 = _mm256_packs_epi32( i0, i1 );
        i2 = _mm256_packs_epi32( i2, i3 );
        i0 = _mm256_packs_epi16( i0, i2 );

        // The AVX2 pack instructions work on 16-byte halves independently; restore the order
        const __m256i perm = _mm256_setr_epi32( 0, 4, 1, 5, 2, 6, 3, 7 );
        i0 = _mm256_permutevar8x32_epi32( i0, perm );

        _mm256_storeu_si256((__m256i *)y[i].qs, i0);
#else
        // AVX without AVX2: split the registers and use the SSE packs
        __m128i ni0 = _mm256_castsi256_si128( i0 );
        __m128i ni1 = _mm256_extractf128_si256( i0, 1);
        __m128i ni2 = _mm256_castsi256_si128( i1 );
        __m128i ni3 = _mm256_extractf128_si256( i1, 1);
        __m128i ni4 = _mm256_castsi256_si128( i2 );
        __m128i ni5 = _mm256_extractf128_si256( i2, 1);
        __m128i ni6 = _mm256_castsi256_si128( i3 );
        __m128i ni7 = _mm256_extractf128_si256( i3, 1);

        ni0 = _mm_packs_epi32( ni0, ni1 );
        ni2 = _mm_packs_epi32( ni2, ni3 );
        ni4 = _mm_packs_epi32( ni4, ni5 );
        ni6 = _mm_packs_epi32( ni6, ni7 );
        ni0 = _mm_packs_epi16( ni0, ni2 );
        ni4 = _mm_packs_epi16( ni4, ni6 );

        _mm_storeu_si128((__m128i *)(y[i].qs +  0), ni0);
        _mm_storeu_si128((__m128i *)(y[i].qs + 16), ni4);
#endif
    }
#else
    GGML_UNUSED(nb);
    GGML_UNUSED(y);
    quantize_row_q8_0(x, vy, k);
#endif
}

// AVX2 dot product for TQ2_0 x q8_K (row S69-TQ2-0-CPU-PATH-5X-SLOWER-THAN-IQ2-BN-2026-10-16-01).
// WHY IT LIVES HERE. type_traits_cpu[TQ2_0].vec_dot named ggml_vec_dot_tq2_0_q8_K, and on x86 that
// name is NOT remapped by arch-fallback.h, so it linked to the ggml-base copy in ggml-quants.c --
// and ggml-base is compiled without any -m ISA flag, so that copy's #if defined(__AVX2__) branch
// (ggml-quants.c:1370) was dead code and the SCALAR loop ran. MEASURED: objdump of
// ggml-base.dir/ggml-quants.c.obj, ggml_vec_dot_tq2_0_q8_K = 433 lines, 0 ymm, 0 pmaddubsw.
// This file is compiled with the CPU backend's ISA flags, like the IQ1_XS kernel below.
// SOURCE-READ ggml-quants.c:1362-1416 (the mainline llama.cpp b9590 x86 kernel, MIT, already in
// this tree): same body. Identity used: codes are {0,1,2} for {-1,0,+1}, so
// sum((q-1)*y) == sum(q*y) - sum(y), and sum(y) per 16 is what q8_K caches in bsums -- which is
// why maddubs (unsigned left operand) applies. The float accumulation is NOT the mainline kernel's
// (see the note in the loop): it reproduces the scalar reference bit for bit. iqk's
// mul_mat_tq2_0_q8_K (iqk_gemm_1bit.cpp) does the same; tests/test-tq2_0-mulmat.cpp holds the GEMM,
// this function and the ggml-base scalar loop to the same bits.
void ggml_vec_dot_tq2_0_q8_K_avx2(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc) {
#if defined(__AVX2__)
    assert(n % QK_K == 0);
    assert(nrc == 1);
    UNUSED(nrc);
    UNUSED(bx);
    UNUSED(by);
    UNUSED(bs);

    const block_tq2_0 * GGML_RESTRICT x = vx;
    const block_q8_K  * GGML_RESTRICT y = vy;

    const int nb = n / QK_K;

    float sumf = 0.0f;
    for (int i = 0; i < nb; ++i) {
        // 16-bit accumulators are safe: a lane sums 8 maddubs results of at most 2*2*128.
        __m256i sumi0 = _mm256_setzero_si256();
        __m256i sumi1 = _mm256_setzero_si256();

        for (size_t j = 0; j < sizeof(x->qs); j += 32) {
            __m256i qx0 = _mm256_loadu_si256((const __m256i *) (x[i].qs + j));
            __m256i qx1 = _mm256_srli_epi16(qx0, 2);
            __m256i qx2 = _mm256_srli_epi16(qx0, 4);
            __m256i qx3 = _mm256_srli_epi16(qx0, 6);

            // 0, 1, 2 (3 is not a valid code)
            qx0 = _mm256_and_si256(qx0, _mm256_set1_epi8(3));
            qx1 = _mm256_and_si256(qx1, _mm256_set1_epi8(3));
            qx2 = _mm256_and_si256(qx2, _mm256_set1_epi8(3));
            qx3 = _mm256_and_si256(qx3, _mm256_set1_epi8(3));

            const __m256i qy0 = _mm256_loadu_si256((const __m256i *) (y[i].qs + j*4 +  0));
            const __m256i qy1 = _mm256_loadu_si256((const __m256i *) (y[i].qs + j*4 + 32));
            const __m256i qy2 = _mm256_loadu_si256((const __m256i *) (y[i].qs + j*4 + 64));
            const __m256i qy3 = _mm256_loadu_si256((const __m256i *) (y[i].qs + j*4 + 96));

            qx0 = _mm256_maddubs_epi16(qx0, qy0);
            qx1 = _mm256_maddubs_epi16(qx1, qy1);
            qx2 = _mm256_maddubs_epi16(qx2, qy2);
            qx3 = _mm256_maddubs_epi16(qx3, qy3);

            sumi0 = _mm256_add_epi16(sumi0, _mm256_add_epi16(qx0, qx1));
            sumi1 = _mm256_add_epi16(sumi1, _mm256_add_epi16(qx2, qx3));
        }

        const __m256i ysum = _mm256_loadu_si256((const __m256i *) y[i].bsums);
        const float   d    = y[i].d * GGML_CPU_FP16_TO_FP32(x[i].d);

        sumi0 = _mm256_add_epi16(sumi0, sumi1);
        sumi0 = _mm256_sub_epi16(sumi0, ysum);            // apply the deferred -1
        sumi0 = _mm256_madd_epi16(sumi0, _mm256_set1_epi16(1));

        // The float part is the SCALAR reference's, step for step: one exact int32 total per
        // block, then sumf += (float) sumi * d with the product rounded on its own. The integer
        // sums are exact whatever their order, so this returns the bit pattern the ggml-base
        // scalar loop returned. The barrier stops -mfma contracting the multiply-add into one
        // rounding, which the scalar build (no -m flags) cannot do. MEASURED 2026-10-16: the
        // per-lane float accumulation the mainline kernel uses moved BitNet-2B TQ2_0
        // wikitext-2 PPL (--chunks 10) from 20.1866 to 20.2233 -- not a defect, but not the
        // same answer either, and the row asked for the same answer.
        float prod = (float) hsum_i32_8(sumi0) * d;
#if defined(__GNUC__) || defined(__clang__)
        __asm__("" : "+x"(prod));
#endif
        sumf += prod;
    }

    *s = sumf;
#else
    ggml_vec_dot_tq2_0_q8_K(n, s, bs, vx, bx, vy, by, nrc);
#endif
}

// AVX2 dot product for TQ1_0 x q8_K (row S69-TQ1-0-CPU-SCALAR-VEC-DOT-AND-NO-IQK-GEMM-2026-10-17-01).
// WHY IT LIVES HERE: the same defect TQ2_0 had (see ggml_vec_dot_tq2_0_q8_K_avx2 above).
// type_traits_cpu[TQ1_0].vec_dot named ggml_vec_dot_tq1_0_q8_K; arch-fallback.h has no x86 remap
// for it, so it linked to the ggml-base copy in ggml-quants.c, and ggml-base is compiled without
// any -m ISA flag, so that copy's #if defined(__AVX2__) branch (ggml-quants.c:1063) is dead and
// its scalar loop (ggml-quants.c:1177-1209) runs. This file has the CPU backend's ISA flags.
//
// SOURCE-READ ggml-quants.c:1063-1176 (mainline llama.cpp b9590 x86 kernel, MIT, already in this
// tree) for the trit decode; the float accumulation is NOT that kernel's (see the loop note).
// Layout (block_tq1_0, ggml-common.h:347): qs[48] then qh[4] then d. qs[0..32) holds 5 trits per
// byte for elements 32*l + m (l = 0..4, m = 0..31); qs[32..48) 5 trits per byte for elements
// 160 + 16*l + m (m = 0..15); qh[0..4) 4 trits per byte for elements 240 + 4*l + j. Trit l of a
// byte q is ((uint8_t)(q * 3^l) * 3) >> 8. AVX2 has no 8-bit multiply, so q * 3^l (mod 256) is
// built from shift-mask-add chains (x*9 = ((x << 3) & 0xF8) + x per byte), and the final
// (q * 3) >> 8 is avg_epu8(q', avg_epu8(q', 0)) >> 6 with q' = subs_epu8(q, 1). Both identities
// were checked exhaustively over the 256 byte values (temp/w12-1017/trit_check.py, 0 mismatches).
// The 8 decoded 32-byte vectors line up with q8_K qs[32*k .. 32*k + 31], k = 0..7.
#if defined(__AVX2__)
static inline __m256i tq1_0_trits_avx2(__m256i q) {
    q = _mm256_subs_epu8(q, _mm256_set1_epi8(1));
    q = _mm256_avg_epu8(q, _mm256_avg_epu8(q, _mm256_setzero_si256()));
    return _mm256_and_si256(_mm256_srli_epi16(q, 6), _mm256_set1_epi8(3));
}

static inline void tq1_0_decode_avx2(const block_tq1_0 * GGML_RESTRICT x, __m256i * GGML_RESTRICT v) {
    const __m256i m8 = _mm256_set1_epi8(-8);
    const __m256i b0 = _mm256_loadu_si256((const __m256i *) x->qs);
    const __m256i b1 = _mm256_add_epi8(b0, _mm256_add_epi8(b0, b0));                          // * 3
    const __m256i b2 = _mm256_add_epi8(_mm256_and_si256(_mm256_slli_epi16(b0, 3), m8), b0);   // * 9
    const __m256i b3 = _mm256_add_epi8(_mm256_and_si256(_mm256_slli_epi16(b1, 3), m8), b1);   // * 27
    const __m256i b4 = _mm256_add_epi8(_mm256_and_si256(_mm256_slli_epi16(b2, 3), m8), b2);   // * 81
    v[0] = tq1_0_trits_avx2(b0);
    v[1] = tq1_0_trits_avx2(b1);
    v[2] = tq1_0_trits_avx2(b2);
    v[3] = tq1_0_trits_avx2(b3);
    v[4] = tq1_0_trits_avx2(b4);

    const __m128i m8h = _mm_set1_epi8(-8);
    const __m128i t0 = _mm_loadu_si128((const __m128i *) (x->qs + 32));
    const __m128i t1 = _mm_add_epi8(t0, _mm_add_epi8(t0, t0));
    const __m128i t2 = _mm_add_epi8(_mm_and_si128(_mm_slli_epi16(t0, 3), m8h), t0);
    const __m128i t3 = _mm_add_epi8(_mm_and_si128(_mm_slli_epi16(t1, 3), m8h), t1);
    const __m128i t4 = _mm_add_epi8(_mm_and_si128(_mm_slli_epi16(t2, 3), m8h), t2);
    uint32_t qh;
    memcpy(&qh, x->qh, sizeof(qh));   // potentially unaligned
    // lane 4*l + j = qh[j] * 3^l (mod 256): 16-bit multiply, masked and packed back to bytes
    __m256i h16 = _mm256_cvtepu8_epi16(_mm_set1_epi32((int) qh));
    h16 = _mm256_mullo_epi16(h16, _mm256_set_epi16(27, 27, 27, 27, 9, 9, 9, 9, 3, 3, 3, 3, 1, 1, 1, 1));
    h16 = _mm256_and_si256(h16, _mm256_set1_epi16(0xFF));
    const __m128i t5 = _mm_packus_epi16(_mm256_castsi256_si128(h16), _mm256_extracti128_si256(h16, 1));
    v[5] = tq1_0_trits_avx2(MM256_SET_M128I(t1, t0));   // elements 160..191
    v[6] = tq1_0_trits_avx2(MM256_SET_M128I(t3, t2));   // elements 192..223
    v[7] = tq1_0_trits_avx2(MM256_SET_M128I(t5, t4));   // elements 224..255
}
#endif // __AVX2__

void ggml_vec_dot_tq1_0_q8_K_avx2(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc) {
#if defined(__AVX2__)
    assert(n % QK_K == 0);
    assert(nrc == 1);
    UNUSED(nrc);
    UNUSED(bx);
    UNUSED(by);
    UNUSED(bs);

    const block_tq1_0 * GGML_RESTRICT x = vx;
    const block_q8_K  * GGML_RESTRICT y = vy;

    const int nb = n / QK_K;

    float sumf = 0.0f;
    for (int i = 0; i < nb; ++i) {
        __m256i v[8];
        tq1_0_decode_avx2(x + i, v);
        // int16 lanes are exact: a lane sums 8 maddubs results of |2*128 + 2*128| <= 512 and one
        // bsum (|16*128|), at most 6144 in magnitude.
        __m256i sumi = _mm256_setzero_si256();
        for (int k = 0; k < 8; ++k) {
            sumi = _mm256_add_epi16(sumi, _mm256_maddubs_epi16(v[k], _mm256_loadu_si256((const __m256i *) (y[i].qs + 32*k))));
        }
        sumi = _mm256_sub_epi16(sumi, _mm256_loadu_si256((const __m256i *) y[i].bsums));   // the deferred -1
        sumi = _mm256_madd_epi16(sumi, _mm256_set1_epi16(1));

        // The float part is the SCALAR reference's (ggml-quants.c:1207), step for step: one exact
        // int32 total per block, then sumf += (float) sum * (d_x * d_y) with the product rounded on
        // its own. The barrier stops -mfma contracting the multiply-add into one rounding, which the
        // scalar build (ggml-base, no -m flags) cannot do. So this returns the scalar's bit pattern
        // and a model's perplexity cannot move (TQ2_0 MEASURED 2026-10-16: the mainline per-lane
        // float accumulation moved BitNet-2B PPL 20.1866 -> 20.2233).
        const float d = GGML_CPU_FP16_TO_FP32(x[i].d) * y[i].d;
        float prod = (float) hsum_i32_8(sumi) * d;
#if defined(__GNUC__) || defined(__clang__)
        __asm__("" : "+x"(prod));
#endif
        sumf += prod;
    }

    *s = sumf;
#else
    ggml_vec_dot_tq1_0_q8_K(n, s, bs, vx, bx, vy, by, nrc);
#endif
}

// AVX2 dot product for IQ1_XS x q8_K. Same lattice code as IQ1_S's AVX2 branch (gather of 8-weight grid rows + maddubs): the narrow
// grids are subsets of iq1s_grid stored the same way, so only the index / scale / sign decode differs. The scalar reference is
// ggml_vec_dot_iq1_xs_q8_K in ggml-quants.c; tests/test-iq1-narrow-quants.cpp holds the two to the same answer.
void ggml_vec_dot_iq1_xs_q8_K_avx2(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc) {
#if defined(__AVX2__)
    assert(n % QK_K == 0);
    assert(nrc == 1);
    UNUSED(nrc);
    UNUSED(bx);
    UNUSED(by);
    UNUSED(bs);

    const block_iq1_xs * GGML_RESTRICT x = vx;
    const block_q8_K  * GGML_RESTRICT y = vy;

    const int nb = n / QK_K;

    __m256 accum = _mm256_setzero_ps();
    float accum1 = 0;
    for (int i = 0; i < nb; ++i) {

        const int8_t  * q8 = y[i].qs;
        const uint8_t * qs = x[i].qs;
        const uint8_t * qh = x[i].qh;
        const uint8_t * sc = x[i].sc;

        __m256i sumi = _mm256_setzero_si256();
        int sumi1 = 0;
        for (int ib = 0; ib < QK_K/32; ib += 2) {
            const uint8_t h0 = qh[ib+0], h1 = qh[ib+1];
            const int nib1 = sc[ib/2] & 0xf, nib2 = sc[ib/2] >> 4;
            const int ls1 = 2*(nib1 & 7) + 1, ls2 = 2*(nib2 & 7) + 1;
            const int neg1 = nib1 & 8, neg2 = nib2 & 8;
            const __m256i q1b_1 = _mm256_set_epi64x(iq1_xs_grid[qs[3] | (((h0 >> 6) & 3) << 8)], iq1_xs_grid[qs[2] | (((h0 >> 4) & 3) << 8)],
                                                    iq1_xs_grid[qs[1] | (((h0 >> 2) & 3) << 8)], iq1_xs_grid[qs[0] | ((h0 & 3) << 8)]);
            const __m256i q1b_2 = _mm256_set_epi64x(iq1_xs_grid[qs[7] | (((h1 >> 6) & 3) << 8)], iq1_xs_grid[qs[6] | (((h1 >> 4) & 3) << 8)],
                                                    iq1_xs_grid[qs[5] | (((h1 >> 2) & 3) << 8)], iq1_xs_grid[qs[4] | ((h1 & 3) << 8)]);
            qs += 8;
            const __m256i q8b_1 = _mm256_loadu_si256((const __m256i*)q8); q8 += 32;
            const __m256i q8b_2 = _mm256_loadu_si256((const __m256i*)q8); q8 += 32;

            const __m256i dot1 = mul_add_epi8(q1b_1, q8b_1);
            const __m256i dot2 = mul_add_epi8(q1b_2, q8b_2);
            const __m256i p1 = _mm256_madd_epi16(dot1, _mm256_set1_epi16((int16_t) ls1));
            const __m256i p2 = _mm256_madd_epi16(dot2, _mm256_set1_epi16((int16_t) ls2));

            sumi = _mm256_add_epi32(sumi, _mm256_add_epi32(p1, p2));
            sumi1 += (y[i].bsums[2*ib+0] + y[i].bsums[2*ib+1]) * (neg1 ? -1 : 1) * ls1
                   + (y[i].bsums[2*ib+2] + y[i].bsums[2*ib+3]) * (neg2 ? -1 : 1) * ls2;
        }

        const float d = y[i].d * GGML_CPU_FP16_TO_FP32(x[i].d);
        accum = _mm256_fmadd_ps(_mm256_set1_ps(d), _mm256_cvtepi32_ps(sumi), accum);
        accum1 += d * sumi1;
    }

    *s = hsum_float_8(accum) + IQ1S_DELTA * accum1;
#else
    ggml_vec_dot_iq1_xs_q8_K(n, s, bs, vx, bx, vy, by, nrc);
#endif
}

// AVX2 dot product for IQ1_XXS x q8_K. Same lattice code as IQ1_S's AVX2 branch (gather of 8-weight grid rows + maddubs): the narrow
// grids are subsets of iq1s_grid stored the same way, so only the index / scale / sign decode differs. The scalar reference is
// ggml_vec_dot_iq1_xxs_q8_K in ggml-quants.c; tests/test-iq1-narrow-quants.cpp holds the two to the same answer.
void ggml_vec_dot_iq1_xxs_q8_K_avx2(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc) {
#if defined(__AVX2__)
    assert(n % QK_K == 0);
    assert(nrc == 1);
    UNUSED(nrc);
    UNUSED(bx);
    UNUSED(by);
    UNUSED(bs);

    const block_iq1_xxs * GGML_RESTRICT x = vx;
    const block_q8_K  * GGML_RESTRICT y = vy;

    const int nb = n / QK_K;

    __m256 accum = _mm256_setzero_ps();
    float accum1 = 0;
    for (int i = 0; i < nb; ++i) {

        const int8_t  * q8 = y[i].qs;
        const uint8_t * qs = x[i].qs;
        const uint8_t * qh = x[i].qh;

        __m256i sumi = _mm256_setzero_si256();
        int sumi1 = 0;
        for (int ib = 0; ib < QK_K/32; ib += 2) {
            const uint8_t h0 = qh[ib+0], h1 = qh[ib+1];
            const int ls1 = 2*((h0 >> 4) & 7) + 1, ls2 = 2*((h1 >> 4) & 7) + 1;
            const int neg1 = h0 & 0x80, neg2 = h1 & 0x80;
            const __m256i q1b_1 = _mm256_set_epi64x(iq1_xxs_grid[qs[3] | (((h0 >> 3) & 1) << 8)], iq1_xxs_grid[qs[2] | (((h0 >> 2) & 1) << 8)],
                                                    iq1_xxs_grid[qs[1] | (((h0 >> 1) & 1) << 8)], iq1_xxs_grid[qs[0] | ((h0 & 1) << 8)]);
            const __m256i q1b_2 = _mm256_set_epi64x(iq1_xxs_grid[qs[7] | (((h1 >> 3) & 1) << 8)], iq1_xxs_grid[qs[6] | (((h1 >> 2) & 1) << 8)],
                                                    iq1_xxs_grid[qs[5] | (((h1 >> 1) & 1) << 8)], iq1_xxs_grid[qs[4] | ((h1 & 1) << 8)]);
            qs += 8;
            const __m256i q8b_1 = _mm256_loadu_si256((const __m256i*)q8); q8 += 32;
            const __m256i q8b_2 = _mm256_loadu_si256((const __m256i*)q8); q8 += 32;

            const __m256i dot1 = mul_add_epi8(q1b_1, q8b_1);
            const __m256i dot2 = mul_add_epi8(q1b_2, q8b_2);
            const __m256i p1 = _mm256_madd_epi16(dot1, _mm256_set1_epi16((int16_t) ls1));
            const __m256i p2 = _mm256_madd_epi16(dot2, _mm256_set1_epi16((int16_t) ls2));

            sumi = _mm256_add_epi32(sumi, _mm256_add_epi32(p1, p2));
            sumi1 += (y[i].bsums[2*ib+0] + y[i].bsums[2*ib+1]) * (neg1 ? -1 : 1) * ls1
                   + (y[i].bsums[2*ib+2] + y[i].bsums[2*ib+3]) * (neg2 ? -1 : 1) * ls2;
        }

        const float d = y[i].d * GGML_CPU_FP16_TO_FP32(x[i].d);
        accum = _mm256_fmadd_ps(_mm256_set1_ps(d), _mm256_cvtepi32_ps(sumi), accum);
        accum1 += d * sumi1;
    }

    *s = hsum_float_8(accum) + IQ1S_DELTA * accum1;
#else
    ggml_vec_dot_iq1_xxs_q8_K(n, s, bs, vx, bx, vy, by, nrc);
#endif
}

// AVX2 dot product for IQ1_XXXS x q8_K. Same lattice code as IQ1_S's AVX2 branch (gather of 8-weight grid rows + maddubs): the narrow
// grids are subsets of iq1s_grid stored the same way, so only the index / scale / sign decode differs. The scalar reference is
// ggml_vec_dot_iq1_xxxs_q8_K in ggml-quants.c; tests/test-iq1-narrow-quants.cpp holds the two to the same answer.
void ggml_vec_dot_iq1_xxxs_q8_K_avx2(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc) {
#if defined(__AVX2__)
    assert(n % QK_K == 0);
    assert(nrc == 1);
    UNUSED(nrc);
    UNUSED(bx);
    UNUSED(by);
    UNUSED(bs);

    const block_iq1_xxxs * GGML_RESTRICT x = vx;
    const block_q8_K  * GGML_RESTRICT y = vy;

    const int nb = n / QK_K;

    __m256 accum = _mm256_setzero_ps();
    float accum1 = 0;
    for (int i = 0; i < nb; ++i) {

        const int8_t  * q8 = y[i].qs;
        const uint8_t * qs = x[i].qs;
        const uint8_t * sc = x[i].sc;

        __m256i sumi = _mm256_setzero_si256();
        int sumi1 = 0;
        for (int ib = 0; ib < QK_K/32; ib += 2) {
            const int nib1 = sc[ib/2] & 0xf, nib2 = sc[ib/2] >> 4;
            const int ls1 = 2*(nib1 & 7) + 1, ls2 = 2*(nib2 & 7) + 1;
            const int neg1 = nib1 & 8, neg2 = nib2 & 8;
            const __m256i q1b_1 = _mm256_set_epi64x(iq1_xxxs_grid[qs[3]], iq1_xxxs_grid[qs[2]], iq1_xxxs_grid[qs[1]], iq1_xxxs_grid[qs[0]]);
            const __m256i q1b_2 = _mm256_set_epi64x(iq1_xxxs_grid[qs[7]], iq1_xxxs_grid[qs[6]], iq1_xxxs_grid[qs[5]], iq1_xxxs_grid[qs[4]]);
            qs += 8;
            const __m256i q8b_1 = _mm256_loadu_si256((const __m256i*)q8); q8 += 32;
            const __m256i q8b_2 = _mm256_loadu_si256((const __m256i*)q8); q8 += 32;

            const __m256i dot1 = mul_add_epi8(q1b_1, q8b_1);
            const __m256i dot2 = mul_add_epi8(q1b_2, q8b_2);
            const __m256i p1 = _mm256_madd_epi16(dot1, _mm256_set1_epi16((int16_t) ls1));
            const __m256i p2 = _mm256_madd_epi16(dot2, _mm256_set1_epi16((int16_t) ls2));

            sumi = _mm256_add_epi32(sumi, _mm256_add_epi32(p1, p2));
            sumi1 += (y[i].bsums[2*ib+0] + y[i].bsums[2*ib+1]) * (neg1 ? -1 : 1) * ls1
                   + (y[i].bsums[2*ib+2] + y[i].bsums[2*ib+3]) * (neg2 ? -1 : 1) * ls2;
        }

        const float d = y[i].d * GGML_CPU_FP16_TO_FP32(x[i].d);
        accum = _mm256_fmadd_ps(_mm256_set1_ps(d), _mm256_cvtepi32_ps(sumi), accum);
        accum1 += d * sumi1;
    }

    *s = hsum_float_8(accum) + IQ1S_DELTA * accum1;
#else
    ggml_vec_dot_iq1_xxxs_q8_K(n, s, bs, vx, bx, vy, by, nrc);
#endif
}

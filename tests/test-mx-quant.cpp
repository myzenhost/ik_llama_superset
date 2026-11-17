// OCP Microscaling MXFP6 (E2M3, E3M2) and MXFP8 (E4M3, E5M2): block of 32 elements + one E8M0 scale.
// Enum ids 70-73. Model-free; everything here is synthetic.
//
// WHAT THIS PROVES
//   1. LAYOUT  ids, names, block size (32), type size (25 B FP6 / 33 B FP8), ggml_row_size, is_quantized, CPU traits row.
//   2. DECODE  every code (64 per FP6 format, 256 per FP8 format) at every one of the 32 positions of a block, against a
//              decode table written HERE from OCP MX v1.0 (sign/exponent/mantissa formula, subnormals, zero, saturation
//              value, NaN/Inf per format), bit-exact including -0, +-Inf and NaN-ness; landmark values hard-coded.
//   3. ENCODE  every finite code survives encode(decode(c)); every adjacent midpoint rounds to-nearest-even, the values just
//              either side of it round to the near neighbour; overflow saturates; NaN/Inf per format.
//   4. SCALE   E8M0 rule recomputed independently (double math) for gaussian blocks, extremes e=0 / 254 / 255, all-zero and
//              denormal blocks.
//   5. ROUND TRIP on gaussian data: per-element error <= half a top-binade step, rel-RMS bounds, and agreement with an
//              INDEPENDENT brute-force nearest-value quantizer built from the decode table.
//   6. VEC_DOT against Q8_0 equals dequantise-then-dot (double) within fp32 summation tolerance, for several row lengths.
//   7. BLOCKS  row lengths that are multiples of 32 (1..64 blocks): row quantiser == per-block quantiser, a huge block does not
//              disturb its neighbours, no write past the row, ggml_quantize_chunk agrees.
//   8. GRAPH   the CPU backend claims and computes MUL_MAT on these types and agrees with the dequantised weights.
// It says nothing about GPU backends (none exist for these types) or model quality.

#define GGML_COMMON_DECL_C
#include "ggml-common.h"

#include "ggml.h"
#include "ggml-cpu.h"
#include "ggml-backend.h"
#include "ggml-quants.h"

#include <cfloat>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

static int g_fail = 0;
#define CHECK(cond, ...) do { if (!(cond)) { ++g_fail; if (g_fail < 40) { printf("FAIL %s:%d  ", __FILE__, __LINE__); printf(__VA_ARGS__); printf("\n"); } } } while (0)

struct Fmt {
    const char * name;
    ggml_type    type;
    int          id;
    int          ebits, mbits, bias, emax;
    int          kind;       // 0 none, 1 E4M3 (NaN only), 2 E5M2 (Inf+NaN)
    bool         fp6;
    size_t       block_bytes;
    double       max_val;
    int          max_code;
    int          min_sub_code;       // code 1
    double       min_sub;            // smallest positive subnormal
    double       min_normal;
};

static const Fmt k_fmts[] = {
    { "mxfp6_e2m3", GGML_TYPE_MXFP6_E2M3, 70, 2, 3,  1,  2, 0, true,  25,   7.5,  0x1F, 1, 0.125,        1.0 },
    { "mxfp6_e3m2", GGML_TYPE_MXFP6_E3M2, 71, 3, 2,  3,  4, 0, true,  25,  28.0,  0x1F, 1, 0.0625,       0.25 },
    { "mxfp8_e4m3", GGML_TYPE_MXFP8_E4M3, 72, 4, 3,  7,  8, 1, false, 33, 448.0,  0x7E, 1, 1.0/512,      1.0/64 },
    { "mxfp8_e5m2", GGML_TYPE_MXFP8_E5M2, 73, 5, 2, 15, 15, 2, false, 33, 57344., 0x7B, 1, 1.0/65536,    1.0/16384 },
};

// ---- the spec decode table, written independently of the library ----
struct Dec { double v; bool nan; bool neg; };
static Dec spec_decode(const Fmt & f, int code) {
    const int total = 1 + f.ebits + f.mbits;
    const int s = (code >> (total - 1)) & 1;
    const int E = (code >> f.mbits) & ((1 << f.ebits) - 1);
    const int m = code & ((1 << f.mbits) - 1);
    Dec d{0, false, s == 1};
    if (f.kind == 1 && E == 15 && m == 7) { d.nan = true; return d; }
    if (f.kind == 2 && E == 31) { if (m) d.nan = true; else d.v = s ? -INFINITY : INFINITY; return d; }
    double mag = E == 0 ? std::ldexp((double) m, 1 - f.bias - f.mbits)
                        : std::ldexp(1.0 + (double) m / (1 << f.mbits), E - f.bias);
    d.v = s ? -mag : mag;
    return d;
}

static float e8m0(int e) { return e == 255 ? NAN : (float) std::ldexp(1.0, e - 127); }

typedef void (*q_fn)(const float *, void *, int64_t);
typedef void (*dq_fn)(const void *, float *, int64_t);
static q_fn q_of(const Fmt & f) {
    switch (f.type) {
        case GGML_TYPE_MXFP6_E2M3: return quantize_row_mxfp6_e2m3;
        case GGML_TYPE_MXFP6_E3M2: return quantize_row_mxfp6_e3m2;
        case GGML_TYPE_MXFP8_E4M3: return quantize_row_mxfp8_e4m3;
        default:                   return quantize_row_mxfp8_e5m2;
    }
}
static dq_fn dq_of(const Fmt & f) {
    switch (f.type) {
        case GGML_TYPE_MXFP6_E2M3: return (dq_fn) dequantize_row_mxfp6_e2m3;
        case GGML_TYPE_MXFP6_E3M2: return (dq_fn) dequantize_row_mxfp6_e3m2;
        case GGML_TYPE_MXFP8_E4M3: return (dq_fn) dequantize_row_mxfp8_e4m3;
        default:                   return (dq_fn) dequantize_row_mxfp8_e5m2;
    }
}

// raw block helpers (independent of the library's packing code): FP6 is an LSB-first 6-bit stream
static void raw_set(const Fmt & f, uint8_t * blk, int j, int code) {
    uint8_t * qs = blk + 1;
    if (!f.fp6) { qs[j] = (uint8_t) code; return; }
    for (int b = 0; b < 6; ++b) {
        const int bit = 6*j + b;
        if ((code >> b) & 1) qs[bit / 8] |= (uint8_t)(1u << (bit % 8));
    }
}

static bool same(const Dec & d, float y) {
    if (d.nan) return std::isnan(y);
    if (std::isinf(d.v)) return std::isinf(y) && (y < 0) == (d.v < 0);
    return !std::isnan(y) && !std::isinf(y) && y == (float) d.v && std::signbit(y) == d.neg;
}

static void test_layout() {
    printf("[layout]\n");
    for (const auto & f : k_fmts) {
        CHECK((int) f.type == f.id, "%s enum id %d != %d", f.name, (int) f.type, f.id);
        const auto * tt = ggml_get_type_traits(f.type);
        CHECK(tt && tt->is_quantized && tt->blck_size == 32 && tt->type_size == f.block_bytes, "%s traits", f.name);
        CHECK(!strcmp(ggml_type_name(f.type), f.name), "%s name '%s'", f.name, ggml_type_name(f.type));
        CHECK(ggml_row_size(f.type, 96) == 3*f.block_bytes, "%s row_size", f.name);
        CHECK(ggml_is_quantized(f.type), "%s ggml_is_quantized", f.name);
        const auto * tc = ggml_get_type_traits_cpu(f.type);
        CHECK(tc && tc->from_float && tc->vec_dot && tc->vec_dot_type == GGML_TYPE_Q8_0, "%s cpu traits row", f.name);
        printf("  %-11s id=%d block=%zu B/32 elem (%.2f bpw)\n", f.name, f.id, f.block_bytes, 8.0*f.block_bytes/32);
    }
    // ids must be pairwise distinct from every other member checked via the type-name table: no other type has these names
    for (int t = 0; t < GGML_TYPE_COUNT; ++t) {
        for (const auto & f : k_fmts) {
            if (t == f.id) continue;
            const char * n = ggml_type_name((ggml_type) t);
            CHECK(!n || strcmp(n, f.name), "type %d also named %s", t, f.name);
        }
    }
}

static void test_decode_table() {
    printf("[decode: every code at every position]\n");
    for (const auto & f : k_fmts) {
        const int ncodes = f.fp6 ? 64 : 256;
        int bad = 0;
        for (int code = 0; code < ncodes; ++code) {
            const Dec d = spec_decode(f, code);
            for (int j = 0; j < 32; ++j) {
                uint8_t blk[33] = {0};
                blk[0] = 127;
                raw_set(f, blk, j, code);
                float y[32];
                dq_of(f)(blk, y, 32);
                for (int i = 0; i < 32; ++i) {
                    const bool ok = i == j ? same(d, y[i]) : (y[i] == 0.0f && !std::signbit(y[i]));
                    if (!ok) { if (bad++ < 5) printf("  mismatch %s code=0x%02x pos=%d lane=%d got %g want %g\n", f.name, code, j, i, y[i], d.v); }
                }
            }
        }
        CHECK(bad == 0, "%s decode mismatches %d", f.name, bad);
        // landmarks, hard-coded from the spec
        CHECK(spec_decode(f, f.max_code).v == f.max_val, "%s table max", f.name);
        CHECK(spec_decode(f, 1).v == f.min_sub, "%s table min subnormal", f.name);
        CHECK(spec_decode(f, 1 << f.mbits).v == f.min_normal, "%s table min normal", f.name);
        CHECK(spec_decode(f, 0).v == 0 && !spec_decode(f, 0).neg, "%s +0", f.name);
        const int sbit = f.fp6 ? 0x20 : 0x80;
        CHECK(spec_decode(f, sbit).v == 0 && spec_decode(f, sbit).neg, "%s -0", f.name);
        int nnan = 0, ninf = 0;
        for (int c = 0; c < ncodes; ++c) { const Dec d = spec_decode(f, c); nnan += d.nan; ninf += std::isinf(d.v); }
        CHECK(nnan == (f.kind == 0 ? 0 : f.kind == 1 ? 2 : 6), "%s NaN code count %d", f.name, nnan);
        CHECK(ninf == (f.kind == 2 ? 2 : 0), "%s Inf code count %d", f.name, ninf);
        printf("  %-11s %3d codes x 32 positions: %d mismatches; %d NaN codes, %d Inf codes\n", f.name, ncodes, bad, nnan, ninf);
    }
}

// quantise one block whose scale is forced to 2^0 by an anchor element holding the format max at lane 31
static void quant_scale1(const Fmt & f, const float * x31, uint8_t * blk) {
    float x[32];
    for (int j = 0; j < 31; ++j) x[j] = x31[j];
    x[31] = (float) f.max_val;
    memset(blk, 0, 33);
    q_of(f)(x, blk, 32);
}
static int raw_get(const Fmt & f, const uint8_t * blk, int j) {
    if (!f.fp6) return blk[1 + j];
    int c = 0;
    for (int b = 0; b < 6; ++b) { const int bit = 6*j + b; c |= ((blk[1 + bit/8] >> (bit % 8)) & 1) << b; }
    return c;
}

static void test_encode() {
    printf("[encode: exhaustive round trip, ties-to-even, saturation, specials]\n");
    for (const auto & f : k_fmts) {
        const int ncodes = f.fp6 ? 64 : 256;
        int bad = 0, n = 0;
        for (int code = 0; code < ncodes; ++code) {
            const Dec d = spec_decode(f, code);
            if (d.nan || std::isinf(d.v)) continue;
            float x[31] = {0};
            const int pos = code % 31;
            x[pos] = (float) d.v;
            uint8_t blk[33];
            quant_scale1(f, x, blk);
            ++n;
            if (blk[0] != 127 || !same(spec_decode(f, raw_get(f, blk, pos)), (float) d.v) ) {
                if (bad++ < 5) printf("  %s code 0x%02x: e=%d got code 0x%02x\n", f.name, code, blk[0], raw_get(f, blk, pos));
            }
        }
        CHECK(bad == 0, "%s encode(decode(c)) failures %d", f.name, bad);
        printf("  %-11s %3d finite codes re-encode: %d failures\n", f.name, n, bad);

        // positive finite values sorted by code == sorted by magnitude
        std::vector<int> pos_codes;
        for (int c = 0; c <= f.max_code; ++c) pos_codes.push_back(c);
        int tie_bad = 0, nties = 0;
        for (size_t i = 0; i + 1 < pos_codes.size(); ++i) {
            const double lo = spec_decode(f, pos_codes[i]).v, hi = spec_decode(f, pos_codes[i + 1]).v;
            const double mid = 0.5 * (lo + hi);
            const double step = hi - lo;
            const int even = (pos_codes[i] & 1) == 0 ? pos_codes[i] : pos_codes[i + 1];
            for (int sgn = 0; sgn < 2; ++sgn) {
                const double s = sgn ? -1.0 : 1.0;
                float x[31] = {0};
                x[0] = (float)(s * mid); x[1] = (float)(s * (mid - step * 0.25)); x[2] = (float)(s * (mid + step * 0.25));
                uint8_t blk[33];
                quant_scale1(f, x, blk);
                const double g0 = spec_decode(f, raw_get(f, blk, 0)).v;
                const double g1 = spec_decode(f, raw_get(f, blk, 1)).v;
                const double g2 = spec_decode(f, raw_get(f, blk, 2)).v;
                ++nties;
                if (g0 != s * spec_decode(f, even).v || g1 != s * lo || g2 != s * hi) {
                    if (tie_bad++ < 5) printf("  %s tie between 0x%02x/0x%02x sign %g: mid->%g (want %g) below->%g (want %g) above->%g (want %g)\n",
                        f.name, pos_codes[i], pos_codes[i+1], s, g0, s * spec_decode(f, even).v, g1, s * lo, g2, s * hi);
                }
            }
        }
        CHECK(tie_bad == 0, "%s rounding failures %d", f.name, tie_bad);
        printf("  %-11s %d midpoint/neighbour groups: %d failures\n", f.name, nties, tie_bad);

        // saturation (never Inf) and specials
        float x[31] = {0};
        x[0] = (float)(f.max_val * 1.01); x[1] = (float)(-f.max_val * 1.01);   // inside the window where the scale does not step up, so the clamp is what saturates
        uint8_t blk[33];
        quant_scale1(f, x, blk);
        CHECK(spec_decode(f, raw_get(f, blk, 0)).v == f.max_val && spec_decode(f, raw_get(f, blk, 1)).v == -f.max_val, "%s saturation", f.name);
        float xs[32] = {0};
        for (int j = 0; j < 32; ++j) xs[j] = 1.0f;
        xs[3] = NAN; xs[4] = INFINITY; xs[5] = -INFINITY;
        uint8_t sb[33] = {0};
        q_of(f)(xs, sb, 32);
        if (f.fp6) {
            CHECK(sb[0] == 255, "%s non-finite block must get NaN scale, got e=%d", f.name, sb[0]);
            float y[32]; dq_of(f)(sb, y, 32);
            bool all_nan = true; for (int j = 0; j < 32; ++j) all_nan &= std::isnan(y[j]);
            CHECK(all_nan, "%s NaN-scale block must dequantise to all NaN", f.name);
        } else {
            CHECK(sb[0] != 255, "%s fp8 non-finite must not poison the scale", f.name);
            float y[32]; dq_of(f)(sb, y, 32);
            CHECK(std::isnan(y[3]), "%s NaN lane", f.name);
            if (f.kind == 2) CHECK(std::isinf(y[4]) && y[4] > 0 && std::isinf(y[5]) && y[5] < 0, "%s Inf lanes", f.name);
            else             CHECK(std::isnan(y[4]) && std::isnan(y[5]), "%s Inf->NaN lanes (E4M3 has no Inf)", f.name);
            CHECK(y[0] == 1.0f && y[31] == 1.0f, "%s finite lanes beside specials: %g %g", f.name, y[0], y[31]);
        }
    }
}

static double expected_scale_exp(const Fmt & f, double amax) {
    // independent of the library: "even mode" = round amax at the element mantissa boundary, take its exponent, minus emax
    if (amax == 0) return 0;
    int e; std::frexp(amax, &e); e -= 1;                         // amax in [2^e, 2^(e+1))
    const double rounded = amax + std::ldexp(1.0, e - f.mbits - 1);
    int er; std::frexp(rounded, &er); er -= 1;
    int s = er + 127 - f.emax;                                    // biased exponent of rounded max, minus emax
    if (s < 0) s = 0;
    if (s > 254) s = 254;
    return s;
}

static void test_scale_extremes() {
    printf("[E8M0 scale]\n");
    for (const auto & f : k_fmts) {
        const int one = f.bias << f.mbits;   // code of +1.0 (E == bias, m == 0)
        CHECK(spec_decode(f, one).v == 1.0, "%s test code for 1.0", f.name);
        for (int e : {0, 1, 2, 126, 127, 128, 253, 254, 255}) {
            uint8_t blk[33] = {0};
            blk[0] = (uint8_t) e;
            raw_set(f, blk, 7, one);
            float y[32]; dq_of(f)(blk, y, 32);
            if (e == 255) {
                bool all = true; for (int j = 0; j < 32; ++j) all &= std::isnan(y[j]);
                CHECK(all, "%s e=255 not all NaN", f.name);
            } else {
                CHECK(y[7] == e8m0(e) && y[7] > 0 && y[6] == 0.0f, "%s e=%d: dequant %g want %g", f.name, e, y[7], e8m0(e));
            }
        }
        // 2^-127 is a fp32 denormal and must survive; 2^127 is the top finite scale
        CHECK(e8m0(0) == std::ldexp(1.0f, -127) && e8m0(254) == std::ldexp(1.0f, 127), "scale constants");

        // quantiser picks the independently computed exponent over many magnitudes
        std::mt19937_64 rng(5);
        std::normal_distribution<double> N(0, 1);
        int bad = 0, n = 0;
        for (int it = 0; it < 4000; ++it) {
            const double mag = std::ldexp(1.0, (int)(rng() % 250) - 125) * (0.5 + (rng() % 1000) / 1000.0);
            float x[32]; double amax = 0;
            for (int j = 0; j < 32; ++j) { x[j] = (float)(N(rng) * mag); amax = std::max(amax, (double) std::fabs(x[j])); }
            uint8_t blk[33]; memset(blk, 0, 33);
            q_of(f)(x, blk, 32);
            ++n; if (blk[0] != (int) expected_scale_exp(f, amax)) { if (bad++ < 3) printf("  %s amax=%g e=%d want %d\n", f.name, amax, blk[0], (int) expected_scale_exp(f, amax)); }
        }
        CHECK(bad == 0, "%s scale exponent mismatches %d/%d", f.name, bad, n);
        printf("  %-11s scale rule vs independent calc over %d blocks: %d mismatches\n", f.name, n, bad);

        // all zeros, denormals, FLT_MAX
        float z[32] = {0}; uint8_t blk[33]; memset(blk, 0xAA, 33);
        q_of(f)(z, blk, 32);
        float y[32]; dq_of(f)(blk, y, 32);
        bool allz = true; for (float v : y) allz &= (v == 0.0f);
        CHECK(blk[0] == 0 && allz, "%s all-zero block e=%d", f.name, blk[0]);
        float dn[32]; for (int j = 0; j < 32; ++j) dn[j] = (j & 1 ? -1.0f : 1.0f) * 1e-45f;
        q_of(f)(dn, blk, 32); dq_of(f)(blk, y, 32);
        bool fin = true; for (float v : y) fin &= std::isfinite(v);
        CHECK(blk[0] == 0 && fin, "%s denormal-only block: e=%d finite=%d", f.name, blk[0], (int) fin);
        // FLT_MAX/4: finite round trip. FLT_MAX itself: the scaled max rounds to 4.0 * 2^126 == 2^128 which fp32 cannot hold,
        // so +-Inf comes back from dequantise (documented edge); the scale must still not be the NaN scale and no lane is NaN.
        float big[32]; for (int j = 0; j < 32; ++j) big[j] = (j & 1 ? -1.0f : 1.0f) * (FLT_MAX / 4) * (0.5f + j / 64.0f);
        q_of(f)(big, blk, 32); dq_of(f)(blk, y, 32);
        fin = true; for (float v : y) fin &= std::isfinite(v);
        CHECK(blk[0] != 255 && fin && std::fabs(y[31]) > 1e37f, "%s FLT_MAX/4 block: e=%d finite=%d y31=%g", f.name, blk[0], (int) fin, y[31]);
        for (int j = 0; j < 32; ++j) big[j] = (j & 1 ? -1.0f : 1.0f) * FLT_MAX * (0.5f + j / 64.0f);
        q_of(f)(big, blk, 32); dq_of(f)(blk, y, 32);
        bool nonan = true; for (float v : y) nonan &= !std::isnan(v);
        CHECK(blk[0] != 255 && nonan, "%s FLT_MAX block: e=%d", f.name, blk[0]);
        printf("  %-11s zero/denormal/FLT_MAX blocks ok=%d\n", f.name, (int)(allz && fin));
    }
}

static void test_roundtrip() {
    printf("[round trip on gaussian data, vs independent nearest-value quantiser]\n");
    for (const auto & f : k_fmts) {
        const int ncodes = f.fp6 ? 64 : 256;
        std::vector<int> finite;
        for (int c = 0; c < ncodes; ++c) { const Dec d = spec_decode(f, c); if (!d.nan && !std::isinf(d.v)) finite.push_back(c); }
        const int nb = 2048;
        std::mt19937_64 rng(11);
        std::normal_distribution<double> N(0, 1);
        std::vector<float> x(32 * nb), y(32 * nb);
        for (int b = 0; b < nb; ++b) { const double sc = std::ldexp(1.0, (int)(rng() % 21) - 10); for (int j = 0; j < 32; ++j) x[32*b + j] = (float)(N(rng) * sc); }
        std::vector<uint8_t> q(f.block_bytes * nb);
        q_of(f)(x.data(), q.data(), 32 * nb);
        dq_of(f)(q.data(), y.data(), 32 * nb);
        double se = 0, sx = 0, worst = 0; long nearest_bad = 0;
        for (int b = 0; b < nb; ++b) {
            const uint8_t * blk = q.data() + f.block_bytes * b;
            const double sc = e8m0(blk[0]);
            const double step_top = sc * std::ldexp(1.0, f.emax - f.mbits);
            for (int j = 0; j < 32; ++j) {
                const double err = std::fabs((double) y[32*b + j] - x[32*b + j]);
                se += err * err; sx += (double) x[32*b + j] * x[32*b + j];
                worst = std::max(worst, err / step_top);
                // brute-force nearest finite value (ties -> even code), saturating by construction
                const double v = x[32*b + j] / sc;
                double best = 1e300; int bc = 0;
                for (int c : finite) {
                    const double dd = std::fabs(v - spec_decode(f, c).v);
                    if (dd < best || (dd == best && (c & 1) == 0 && (bc & 1) == 1)) { best = dd; bc = c; }
                }
                if (spec_decode(f, bc).v != spec_decode(f, raw_get(f, blk, j)).v) ++nearest_bad;
            }
        }
        const double rel = std::sqrt(se / sx);
        // in-range elements err <= 0.5 step; an element clipped at the max code errs by at most (T - max) where T is the amax the
        // even-mode scale still accepts: T = 2^(emax+1) - 2^(emax-mbits-1)  (E4M3: 496 vs max 448 = 1.5 steps; others 0.5 step)
        const double T = std::ldexp(1.0, f.emax + 1) - std::ldexp(1.0, f.emax - f.mbits - 1);
        const double err_bound = std::max(0.5, (T - f.max_val) / std::ldexp(1.0, f.emax - f.mbits));
        const double bound = f.mbits == 3 ? 0.045 : 0.085;
        printf("  %-11s rel-RMS %.5f (bound %.3f)  max err / top-binade step %.4f (bound %.2f)  brute-force disagreements %ld/%d\n",
            f.name, rel, bound, worst, err_bound, nearest_bad, 32 * nb);
        CHECK(rel < bound, "%s rel-RMS %.5f", f.name, rel);
        CHECK(worst <= err_bound + 1e-6, "%s max abs error %.4f of a top-binade step", f.name, worst);
        CHECK(nearest_bad == 0, "%s disagrees with nearest-value quantiser on %ld elements", f.name, nearest_bad);
    }
}

static void test_vec_dot() {
    printf("[vec_dot vs dequantise-then-dot]\n");
    typedef void (*vd_fn)(int, float *, size_t, const void *, size_t, const void *, size_t, int);
    vd_fn vds[] = { ggml_vec_dot_mxfp6_e2m3_q8_0, ggml_vec_dot_mxfp6_e3m2_q8_0, ggml_vec_dot_mxfp8_e4m3_q8_0, ggml_vec_dot_mxfp8_e5m2_q8_0 };
    int fi = 0;
    for (const auto & f : k_fmts) {
        double worst = 0;
        for (int nblk : {1, 2, 3, 7, 16, 64}) {
            const int n = 32 * nblk;
            std::mt19937_64 rng(100 + nblk);
            std::normal_distribution<double> N(0, 1);
            std::vector<float> a(n), b(n);
            for (int i = 0; i < n; ++i) { a[i] = (float)(N(rng) * 3.0); b[i] = (float) N(rng); }
            std::vector<uint8_t> qa(f.block_bytes * nblk);
            std::vector<block_q8_0> qb(nblk);
            q_of(f)(a.data(), qa.data(), n);
            quantize_row_q8_0_ref(b.data(), qb.data(), n);
            std::vector<float> da(n), db(n);
            dq_of(f)(qa.data(), da.data(), n);
            dequantize_row_q8_0(qb.data(), db.data(), n);
            double ref = 0, mag = 0;
            for (int i = 0; i < n; ++i) { ref += (double) da[i] * db[i]; mag += std::fabs((double) da[i] * db[i]); }
            float got = 0;
            vds[fi](n, &got, 0, qa.data(), 0, qb.data(), 0, 1);
            CHECK(mag > 1.0 && std::fabs(ref) > 1e-3, "%s n=%d vacuous dot: sum|terms|=%g ref=%g", f.name, n, mag, ref);
            const double err = std::fabs(got - ref) / (mag + 1e-30);
            worst = std::max(worst, err);
            CHECK(err < 1e-5, "%s n=%d vec_dot %.7g vs %.7g (err/sum|.| %.2e)", f.name, n, got, ref, err);
        }
        // NaN scale poisons the dot
        uint8_t blk[33] = {0}; blk[0] = 255;
        block_q8_0 qy; memset(&qy, 0, sizeof qy); qy.d = ggml_fp32_to_fp16(1.0f); for (int j = 0; j < 32; ++j) qy.qs[j] = 1;
        float got = 0; vds[fi](32, &got, 0, blk, 0, &qy, 0, 1);
        CHECK(std::isnan(got), "%s NaN scale must give NaN dot, got %g", f.name, got);
        printf("  %-11s block counts 1,2,3,7,16,64: worst |vec_dot - ref| / sum|terms| = %.2e (tol 1e-5)\n", f.name, worst);
        ++fi;
    }
}


// ---- 6b. the CPU-backend dot (AVX2+FMA when the build has it, else a forward to the scalar reference) vs the scalar reference
extern "C" {
void ggml_vec_dot_mxfp6_e2m3_q8_0_cpu(int, float *, size_t, const void *, size_t, const void *, size_t, int);
void ggml_vec_dot_mxfp6_e3m2_q8_0_cpu(int, float *, size_t, const void *, size_t, const void *, size_t, int);
void ggml_vec_dot_mxfp8_e4m3_q8_0_cpu(int, float *, size_t, const void *, size_t, const void *, size_t, int);
void ggml_vec_dot_mxfp8_e5m2_q8_0_cpu(int, float *, size_t, const void *, size_t, const void *, size_t, int);
}

static void test_vec_dot_cpu() {
    printf("[CPU vec_dot (SIMD where built) vs scalar reference]\n");
    typedef void (*vd_fn)(int, float *, size_t, const void *, size_t, const void *, size_t, int);
    vd_fn ref_fn[] = { ggml_vec_dot_mxfp6_e2m3_q8_0, ggml_vec_dot_mxfp6_e3m2_q8_0, ggml_vec_dot_mxfp8_e4m3_q8_0, ggml_vec_dot_mxfp8_e5m2_q8_0 };
    vd_fn cpu_fn[] = { ggml_vec_dot_mxfp6_e2m3_q8_0_cpu, ggml_vec_dot_mxfp6_e3m2_q8_0_cpu, ggml_vec_dot_mxfp8_e4m3_q8_0_cpu, ggml_vec_dot_mxfp8_e5m2_q8_0_cpu };
    int fi = 0;
    for (const auto & f : k_fmts) {
        std::mt19937_64 rng(777 + fi);
        const int ncodes = f.fp6 ? 64 : 256;
        double worst = 0; int n_special = 0, n_cmp = 0;
        // (a) every code at every one of the 32 positions, scale 2^(e-127) for several e (incl. 0 and 254), random Q8_0 activations
        for (int code = 0; code < ncodes; ++code) {
            for (int e : {0, 1, 100, 127, 130, 254}) {
                for (int pos = 0; pos < 32; ++pos) {
                    std::vector<uint8_t> blk(f.block_bytes, 0);
                    blk[0] = (uint8_t) e;
                    for (int j = 0; j < 32; ++j) {
                        const int c = (j == pos) ? code : (int)(rng() % ncodes);
                        if (f.fp6) { const int bit = 6*j, b = 1 + (bit >> 3), sh = bit & 7; blk[b] |= (uint8_t)(c << sh); if (sh > 2) blk[b + 1] |= (uint8_t)(c >> (8 - sh)); }
                        else blk[1 + j] = (uint8_t) c;
                    }
                    block_q8_0 qy; qy.d = ggml_fp32_to_fp16(0.05f);
                    for (int j = 0; j < 32; ++j) qy.qs[j] = (int8_t)((int)(rng() % 255) - 127);
                    float a = 0, b = 0;
                    ref_fn[fi](32, &a, 0, blk.data(), 0, &qy, 0, 1);
                    cpu_fn[fi](32, &b, 0, blk.data(), 0, &qy, 0, 1);
                    ++n_cmp;
                    if (std::isnan(a) || std::isinf(a) || std::isnan(b) || std::isinf(b)) {
                        ++n_special;
                        CHECK(std::isnan(a) == std::isnan(b) && std::isinf(a) == std::isinf(b) && (std::isnan(a) || a == b), "%s code %d pos %d e %d special: ref %g cpu %g", f.name, code, pos, e, a, b);
                    } else {
                        // fp32 sums of 32 terms in two different orders differ by at most ~32 * 2^-24 of sum|terms| (+ the FMA's single rounding)
                        std::vector<float> dv(32);
                        std::vector<uint8_t> blk127 = blk; blk127[0] = 127;   // dequantise at scale 1 (no float overflow at e=254), rescale in double
                        dq_of(f)(blk127.data(), dv.data(), 32);
                        double mag = 0;
                        for (int j = 0; j < 32; ++j) mag += std::fabs((double) dv[j] * qy.qs[j] * 0.05);
                        mag = std::ldexp(mag, e - 127);
                        const double err = std::fabs((double) a - b);
                        worst = std::max(worst, err / (mag + 1e-30));
                        CHECK(err <= 2e-6 * mag + 1e-30, "%s code %d pos %d e %d: ref %.9g cpu %.9g (err %.3g, sum|terms| %.3g)", f.name, code, pos, e, a, b, err, mag);
                    }
                }
            }
        }
        // (b) random rows of 1..64 blocks, relative to sum|terms| via dequantise
        double worst_rel = 0;
        for (int nblk : {1, 2, 3, 5, 16, 64}) {
            const int n = 32 * nblk;
            std::normal_distribution<double> N(0, 1);
            std::vector<float> a(n), b(n);
            for (int i = 0; i < n; ++i) { a[i] = (float)(N(rng) * 3.0); b[i] = (float) N(rng); }
            std::vector<uint8_t> qa(f.block_bytes * nblk);
            std::vector<block_q8_0> qb(nblk);
            q_of(f)(a.data(), qa.data(), n);
            quantize_row_q8_0_ref(b.data(), qb.data(), n);
            std::vector<float> da(n), db(n);
            dq_of(f)(qa.data(), da.data(), n);
            dequantize_row_q8_0(qb.data(), db.data(), n);
            double mag = 0;
            for (int i = 0; i < n; ++i) mag += std::fabs((double) da[i] * db[i]);
            float r = 0, c = 0;
            ref_fn[fi](n, &r, 0, qa.data(), 0, qb.data(), 0, 1);
            cpu_fn[fi](n, &c, 0, qa.data(), 0, qb.data(), 0, 1);
            const double rel = std::fabs((double) r - c) / (mag + 1e-30);
            worst_rel = std::max(worst_rel, rel);
            CHECK(rel < 2e-6, "%s n=%d cpu %.8g vs scalar %.8g (rel to sum|terms| %.2e)", f.name, n, c, r, rel);
        }
        printf("  %-11s %d (code,pos,scale) blocks incl. %d Inf/NaN results: classification identical, finite within fp32 summation-order bound; random rows worst rel %.2e, worst per-block rel %.2e (tol 2e-6)\n",
               f.name, n_cmp, n_special, worst_rel, worst);
        ++fi;
    }
}

static void test_blocks() {
    printf("[block boundaries / row lengths]\n");
    for (const auto & f : k_fmts) {
        for (int nblk : {1, 2, 3, 33, 64}) {
            const int n = 32 * nblk;
            std::mt19937_64 rng(77 + nblk);
            std::normal_distribution<double> N(0, 1);
            std::vector<float> x(n);
            for (int i = 0; i < n; ++i) x[i] = (float)(N(rng) * std::ldexp(1.0, (int)(i / 32) % 9 - 4));
            const size_t row = (size_t) nblk * f.block_bytes;
            std::vector<uint8_t> whole(row + 16, 0xCD), perblk(row, 0);
            q_of(f)(x.data(), whole.data(), n);
            for (int b = 0; b < nblk; ++b) q_of(f)(x.data() + 32*b, perblk.data() + b * f.block_bytes, 32);
            CHECK(!memcmp(whole.data(), perblk.data(), row), "%s nblk=%d row != per-block", f.name, nblk);
            bool guard = true; for (int i = 0; i < 16; ++i) guard &= whole[row + i] == 0xCD;
            CHECK(guard, "%s nblk=%d wrote past the row", f.name, nblk);
            // one huge block must leave the neighbours' bytes unchanged
            if (nblk >= 3) {
                std::vector<float> x2 = x; for (int j = 0; j < 32; ++j) x2[32 + j] *= 1e6f;
                std::vector<uint8_t> w2(row);
                q_of(f)(x2.data(), w2.data(), n);
                CHECK(!memcmp(w2.data(), whole.data(), f.block_bytes) &&
                      !memcmp(w2.data() + 2*f.block_bytes, whole.data() + 2*f.block_bytes, row - 2*f.block_bytes),
                      "%s nblk=%d neighbour blocks disturbed", f.name, nblk);
                CHECK(memcmp(w2.data() + f.block_bytes, whole.data() + f.block_bytes, f.block_bytes) != 0, "%s mutated block unchanged?", f.name);
            }
            // dequantise whole row == per block
            std::vector<float> yw(n), yb(n);
            dq_of(f)(whole.data(), yw.data(), n);
            for (int b = 0; b < nblk; ++b) dq_of(f)(whole.data() + b * f.block_bytes, yb.data() + 32*b, 32);
            CHECK(!memcmp(yw.data(), yb.data(), n * sizeof(float)), "%s nblk=%d dequant row != per-block", f.name, nblk);
        }
        // ggml_quantize_chunk: 3 rows of 96, then compare against the row quantiser
        const int nr = 3, nper = 96;
        std::vector<float> x(nr * nper); std::mt19937 rng(3); std::normal_distribution<float> N(0, 1);
        for (auto & v : x) v = N(rng);
        std::vector<uint8_t> a(nr * ggml_row_size(f.type, nper)), b(a.size());
        const size_t got = ggml_quantize_chunk(f.type, x.data(), a.data(), 0, nr, nper, nullptr, nullptr);
        for (int r = 0; r < nr; ++r) q_of(f)(x.data() + r * nper, b.data() + r * ggml_row_size(f.type, nper), nper);
        CHECK(got == a.size() && a == b, "%s ggml_quantize_chunk size %zu want %zu / bytes equal %d", f.name, got, a.size(), (int)(a == b));
        CHECK(ggml_validate_row_data(f.type, a.data(), a.size()), "%s ggml_validate_row_data", f.name);
        printf("  %-11s rows of 1,2,3,33,64 blocks and ggml_quantize_chunk: ok\n", f.name);
    }
}

static void test_graph() {
    printf("[CPU backend MUL_MAT]\n");
    ggml_backend_t be = ggml_backend_cpu_init();
    CHECK(be != nullptr, "no cpu backend");
    if (!be) return;
    for (const auto & f : k_fmts) {
        for (int64_t N : {1, 7}) {
            const int64_t K = 160, M = 24; // K = 5 blocks
            ggml_init_params ip = { 16u*1024*1024, nullptr, false };
            ggml_context * ctx = ggml_init(ip);
            std::mt19937 rng(9); std::normal_distribution<float> G(0, 1);
            std::vector<float> w(K * M); for (auto & v : w) v = G(rng);
            ggml_tensor * W = ggml_new_tensor_2d(ctx, f.type, K, M);
            for (int64_t r = 0; r < M; ++r) q_of(f)(w.data() + r*K, (uint8_t *) W->data + r * W->nb[1], K);
            std::vector<float> wd(K * M);
            for (int64_t r = 0; r < M; ++r) dq_of(f)((uint8_t *) W->data + r * W->nb[1], wd.data() + r*K, K);
            ggml_tensor * X = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, K, N);
            for (int64_t i = 0; i < K*N; ++i) ((float *) X->data)[i] = G(rng);
            ggml_tensor * Y = ggml_mul_mat(ctx, W, X);
            CHECK(ggml_backend_supports_op(be, Y), "%s MUL_MAT not claimed", f.name);
            ggml_cgraph * gf = ggml_new_graph(ctx);
            ggml_build_forward_expand(gf, Y);
            CHECK(ggml_backend_graph_compute(be, gf) == GGML_STATUS_SUCCESS, "%s MUL_MAT compute", f.name);
            double se = 0, sr = 0;
            for (int64_t j = 0; j < N; ++j) for (int64_t i = 0; i < M; ++i) {
                double s = 0; for (int64_t k = 0; k < K; ++k) s += (double) wd[i*K + k] * ((const float *) X->data)[j*K + k];
                const double d = ((const float *) Y->data)[j*M + i] - s; se += d*d; sr += s*s;
            }
            const double rel = std::sqrt(se / sr);
            printf("  %-11s MUL_MAT N=%lld rel-rms vs double(dequant weights) = %.5f (bound 0.01: Q8_0 activation rounding)\n", f.name, (long long) N, rel);
            CHECK(std::isfinite(rel) && rel < 0.01, "%s MUL_MAT N=%lld rel-rms %.5f", f.name, (long long) N, rel);
            ggml_free(ctx);
        }
    }
    ggml_backend_free(be);
}

int main() {
    // GGML_FP16_TO_FP32 reads a table that ggml_init() fills: without this the Q8_0 scales below decode to 0 and every dot is trivially 0
    { ggml_init_params ip = { 1024, nullptr, true }; ggml_free(ggml_init(ip)); }
    ggml_cpu_init(); // fills the CPU backend's own fp16 table that GGML_CPU_FP16_TO_FP32 reads (the *_cpu dots below use it)
    test_layout();
    test_decode_table();
    test_encode();
    test_scale_extremes();
    test_roundtrip();
    test_vec_dot();
    test_vec_dot_cpu();
    test_blocks();
    test_graph();
    if (g_fail) { printf("test-mx-quant: %d FAILED checks\n", g_fail); return 1; }
    printf("test-mx-quant: all checks passed\n");
    return 0;
}

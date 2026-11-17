// Row-interleaved quant types (_r4 / _r8 / _r16): quantize one interleaved group of rows with ggml_quantize_chunk,
// dequantize it with the type's to_float, and check that (a) neither step writes past its buffer (guard canaries
// after both the quantized and the float buffer) and (b) the values round-trip.
//
// Why: dequantize_row_mxfp4_r8 computed nblock = k/QK_MXFP4 where one block_mxfp4_r8 holds 8 rows x 32 values,
// so it ran 8x too many blocks and wrote up to ~7*n_per_row floats past its output (iqk_quantize.cpp; the same line
// is in upstream ik_llama origin/main cdf232cc1 at iqk_quantize.cpp:4353). test-quantize-perf only ever showed the
// consequence -- a heap-corruption exit (0xc0000374) several types LATER -- which a bisection over its type order
// traced to mxfp4_r8 (temp/_qperf_0928_bisect.txt). This test names the faulting type directly, for every
// row-interleaved type, so a sibling with the same slip cannot hide behind a delayed heap check.
//
// Its first run found a second, unrelated fault through the round-trip check: q8_k_r16 decoded to rel RMS 1.89 on
// AVX2, because its repack flipped the quants to the declared unsigned format only under HAVE_FANCY_SIMD while its
// dequantizer always assumes unsigned.

#include "ggml.h"
#include "ggml-cpu.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

static int interleave_of(const std::string & name) {
    if (name.size() > 4 && name.compare(name.size() - 4, 4, "_r16") == 0) return 16;
    if (name.size() > 3 && name.compare(name.size() - 3, 3, "_r8") == 0)  return 8;
    if (name.size() > 3 && name.compare(name.size() - 3, 3, "_r4") == 0)  return 4;
    return 0;
}

static std::string lower(std::string s) {
    for (auto & c : s) c = (char) std::tolower((unsigned char) c);
    return s;
}

static double rel_rms_of(const std::vector<float> & ref, const float * got, int64_t n) {
    double se = 0, ss = 0;
    for (int64_t j = 0; j < n; ++j) {
        const double d = (double) got[j] - ref[j];
        se += d * d;
        ss += (double) ref[j] * ref[j];
    }
    return ss > 0 ? std::sqrt(se / ss) : 0.0;
}

// The non-interleaved type the _rN type repacks ("iq2_bn_r4" -> "iq2_bn", "q8_k_r8" -> "q8_K"), or GGML_TYPE_COUNT.
static ggml_type base_of(const std::string & name, int R) {
    const std::string want = lower(name.substr(0, name.size() - (R == 16 ? 4 : 3)));
    for (int i = 0; i < GGML_TYPE_COUNT; ++i) {
        const char * c = ggml_type_name((ggml_type) i);
        if (c && lower(c) == want) return (ggml_type) i;
    }
    return GGML_TYPE_COUNT;
}

// Round-trip error of the BASE type on the same data, or -1 when the base cannot be driven here. ggml_quantize_chunk's
// default arm is assert(false) with result 0 -- a no-op in Release -- so an unsupported base is detected by its size.
static double base_rel_rms(ggml_type base, const std::vector<float> & src, int R, int64_t n_per_row,
                           const std::vector<float> & imatrix) {
    if (base == GGML_TYPE_COUNT) return -1;
    const auto * tb = ggml_get_type_traits(base);
    if (!tb || !tb->to_float || n_per_row % ggml_blck_size(base) != 0) return -1;
    const size_t rs = ggml_row_size(base, n_per_row);
    std::vector<uint8_t> qb(rs * R);
    ggml_quantize_init(base);
    if (ggml_quantize_chunk(base, src.data(), qb.data(), 0, R, n_per_row, imatrix.data(), nullptr) != rs * R) return -1;
    std::vector<float> ob(src.size());
    for (int r = 0; r < R; ++r) tb->to_float(qb.data() + r * rs, ob.data() + r * n_per_row, n_per_row);
    return rel_rms_of(src, ob.data(), (int64_t) src.size());
}

int main() {
    ggml_cpu_init();

    const int64_t n_per_row = 1024;                    // a multiple of every block size used by these types (32, 256)
    // Guards are sized to the whole class of the bug, not to one instance: an R-row group whose dequant loops over
    // k/QK blocks instead of (k/R)/QK runs R x too far, writing up to (R-1)*n_per_row floats past the output and
    // reading (R-1) x the quantized size past the input. 8x the buffer covers every R <= 9 with no write outside the
    // canaries, and R=16 still lands its first bad writes in them -- so an overrun is REPORTED here rather than
    // corrupting the heap and surfacing as an unrelated crash later, which is how test-quantize-perf saw it.
    const uint32_t kCanary  = 0xDEADBEEFu;

    int n_checked = 0, n_skipped = 0, n_failed = 0;

    for (int i = 0; i < GGML_TYPE_COUNT; ++i) {
        const ggml_type type = (ggml_type) i;
        const char * cname = ggml_type_name(type);
        if (!cname) continue;
        const std::string name = cname;
        const int R = interleave_of(name);
        if (R == 0) continue;

        const auto * tr = ggml_get_type_traits(type);
        if (!tr || !tr->to_float) {
            printf("  skip %-14s (no to_float)\n", cname);
            ++n_skipped;
            continue;
        }
        if (n_per_row % ggml_blck_size(type) != 0) {
            printf("  skip %-14s (block size %lld does not divide %lld)\n", cname, (long long) ggml_blck_size(type),
                   (long long) n_per_row);
            ++n_skipped;
            continue;
        }

        const int64_t n = R * n_per_row;
        std::vector<float> src(n);
        for (int64_t j = 0; j < n; ++j) {
            src[j] = 0.1f * std::sin(0.37f * (float) j) + 0.05f * std::cos(1.3f * (float) (j % 97));
        }

        const size_t qbytes = ggml_row_size(type, n_per_row) * R;
        const size_t kGuard = 8 * qbytes / sizeof(uint32_t);                // uint32 canaries after the quants
        std::vector<uint8_t> q(qbytes + kGuard * sizeof(uint32_t));
        for (size_t g = 0; g < kGuard; ++g) memcpy(q.data() + qbytes + g * sizeof(uint32_t), &kCanary, sizeof(kCanary));

        // A uniform importance matrix: every type accepts one, and the imatrix-requiring types are then checked
        // instead of skipped. The weights only steer the quantizer's search; bounds are what this test is about.
        const std::vector<float> imatrix(n_per_row, 1.0f);
        ggml_quantize_init(type);
        ggml_quantize_chunk(type, src.data(), q.data(), 0, R, n_per_row, imatrix.data(), nullptr);

        bool q_guard_ok = true;
        for (size_t g = 0; g < kGuard; ++g) {
            uint32_t v;
            memcpy(&v, q.data() + qbytes + g * sizeof(uint32_t), sizeof(v));
            q_guard_ok &= v == kCanary;
        }

        const size_t kFGuard = 8 * (size_t) n;                                 // float-sized canaries after the output
        std::vector<float> out(n + kFGuard);
        for (size_t g = 0; g < kFGuard; ++g) memcpy(&out[n + g], &kCanary, sizeof(kCanary));
        tr->to_float(q.data(), out.data(), n);

        size_t first_bad = kFGuard;
        for (size_t g = 0; g < kFGuard; ++g) {
            uint32_t v;
            memcpy(&v, &out[n + g], sizeof(v));
            if (v != kCanary) { first_bad = g; break; }
        }
        const bool f_guard_ok = first_bad == kFGuard;

        const double rel_rms = rel_rms_of(src, out.data(), n);
        // Round trip, judged against the BASE type on the same data where the base can be driven: interleaving changes
        // the layout, not the quantization, so a layout/decode slip shows as the _rN type doing much worse than its
        // base. A coarse type (iq2_bn is ternary; this data is not) is then not failed for being coarse. Where no base
        // can be driven, a loose absolute bound still catches garbage and offset reads.
        const double base_rel = base_rel_rms(base_of(name, R), src, R, n_per_row, imatrix);
        const double limit = base_rel >= 0 ? std::max(1.25 * base_rel, base_rel + 0.02) : 0.5;
        const bool rt_ok = std::isfinite(rel_rms) && rel_rms <= limit;

        const bool ok = q_guard_ok && f_guard_ok && rt_ok;
        char base_txt[64];
        if (base_rel >= 0) snprintf(base_txt, sizeof(base_txt), "base %.4f", base_rel);
        else snprintf(base_txt, sizeof(base_txt), "no base");
        printf("  %-4s %-14s R=%-2d quant-guard %s  float-guard %s%s  rel_rms %.4f (%s, limit %.4f)\n",
               ok ? "ok" : "FAIL", cname, R, q_guard_ok ? "intact" : "OVERWRITTEN",
               f_guard_ok ? "intact" : "OVERWRITTEN",
               f_guard_ok ? "" : (" (first bad canary at float +" + std::to_string(first_bad) + ")").c_str(), rel_rms,
               base_txt, limit);
        ++n_checked;
        n_failed += !ok;
    }

    printf("%d row-interleaved types checked, %d skipped, %d failed\n", n_checked, n_skipped, n_failed);
    if (n_checked == 0) {
        printf("FAIL: no row-interleaved type was checked -- the type scan found nothing, so this run proves nothing\n");
        return 1;
    }
    return n_failed == 0 ? 0 : 1;
}

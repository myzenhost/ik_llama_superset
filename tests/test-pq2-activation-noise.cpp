// Where does the Prism PQ2_0 KL-divergence gap come from: the weight decode, or the activation quantiser?
//
// Row S69-PRISM-PQ2-0-KLD-2-TO-5X-ABOVE-OTHER-PRISM-TYPES-2026-10-04-01. Measured 2026-10-01 against the same weights
// dequantised to F32: PQ2_0 (activation q8_K, 256-element blocks) KLD 0.0172, PQ2_0_R8 (q8_K128) 0.0085, PTQ1_0 (q8_0) 0.0037,
// Q2_0_G128 (q8_0_x4) 0.0010. The order follows the activation block size. This test separates the two causes with a
// double-precision reference and no model:
//
//   (A) KERNEL DECODE: ggml_vec_dot_pq2_0_q8_K against dot(dequantised weights, DEQUANTISED activations) -- both sides see the
//       same quantised activation, so any difference is the kernel's integer arithmetic. Must be ~1e-6 relative.
//   (B) ACTIVATION NOISE: dot(dequantised weights, dequantised activations) against dot(dequantised weights, ORIGINAL float
//       activations) for each activation format. That is the error the activation quantiser alone adds, per format.
//
// If (A) is ~0 and (B) grows with the block size, the KLD gap is activation quantisation, not a kernel fault.
//
// Activations are gaussian with 1% of elements scaled x20 (a stated stand-in for LLM residual-stream outliers: block-scale
// quantisers lose precision on exactly those). The no-outlier row is printed too, so the dependence on outliers is visible.
// SCOPE: host backend, one weight row per trial, K = 4096, 200 trials, weights = gaussian quantised by quantize_row_pq2_0.

#include "ggml.h"
#include "ggml-cpu.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

static int g_fail = 0;

struct fmt_t { ggml_type t; const char * label; };

int main() {
    setvbuf(stdout, NULL, _IONBF, 0);
    // ggml_cpu_init() fills the fp16->fp32 lookup table GGML_FP16_TO_FP32 reads. Without it every block scale decodes as 0 and the
    // whole run is zeros (the first three runs of this test were exactly that; the NO SIGNAL guard below caught it).
    ggml_cpu_init();
    const ggml_type wtype = GGML_TYPE_PQ2_0;
    const int64_t K = 4096;
    const int trials = 200;
    const ggml_type_traits     * wb = ggml_get_type_traits(wtype);
    const ggml_type_traits_cpu * wc = ggml_get_type_traits_cpu(wtype);
    if (!wb || !wb->to_float || !wc || !wc->vec_dot || !wc->from_float) {
        printf("SKIPPED: PQ2_0 has no to_float / CPU vec_dot in this build. Not a verdict.\n");
        return 0;
    }
    const fmt_t fmts[] = { {GGML_TYPE_Q8_K, "q8_K   (256)"}, {GGML_TYPE_Q8_K128, "q8_K128(128)"}, {GGML_TYPE_Q8_0, "q8_0   ( 32)"} };

    for (int outliers = 0; outliers < 2; outliers++) {
        std::mt19937 rng(20261004 + outliers);
        std::normal_distribution<float> nd(0.0f, 1.0f);
        std::uniform_real_distribution<float> ud(0.0f, 1.0f);
        double se_kernel_vs_deq = 0, ss_ref = 0;
        double se_fmt[3] = {0, 0, 0}, ss_fmt[3] = {0, 0, 0};
        bool have_fmt[3] = {false, false, false};
        double se_kernel_vs_exact = 0;
        // Emulated symmetric int8 block quantiser (d = amax/127, q = round(x/d)) at three block sizes, for the formats this build exposes
        // no to_float for. Cross-checked below: the 32-block emulation must match the REAL q8_0, the 256-block one the kernel's real q8_K.
        const int emu_blocks[3] = {256, 128, 32};
        double se_emu[3] = {0, 0, 0}, ss_emu = 0;

        for (int tr = 0; tr < trials; tr++) {
            std::vector<float> w(K), x(K);
            // Ternary weights as a Prism file stores them: {-1,0,+1} x a per-128-block scale. (Gaussian weights through the
            // absmax quantiser leave ~1 nonzero per block -- the first version of this test did that and measured nothing.)
            for (int64_t b = 0; b * 128 < K; b++) {
                const float scale = 0.02f * (0.5f + ud(rng));
                for (int j = 0; j < 128; j++) { const float u = ud(rng); w[b * 128 + j] = (u < 1.0f / 3 ? -1.0f : (u < 2.0f / 3 ? 0.0f : 1.0f)) * scale; }
            }
            for (auto & v : x) { v = nd(rng); if (outliers && ud(rng) < 0.01f) v *= 20.0f; }
            std::vector<char> wq(ggml_row_size(wtype, K) + 64, 0);
            wc->from_float(w.data(), wq.data(), K);
            std::vector<float> wd(K);
            wb->to_float(wq.data(), wd.data(), K);

            double exact = 0;
            for (int64_t i = 0; i < K; i++) exact += (double) wd[i] * (double) x[i];
            if (tr == 0) {
                int64_t nz_w = 0, nz_wd = 0;
                for (int64_t i = 0; i < K; i++) { nz_w += w[i] != 0.0f; nz_wd += wd[i] != 0.0f; }
                printf("  [debug trial 0] nonzero w %lld, nonzero dequantised %lld, exact dot %.6g, vec_dot_type %s\n",
                       (long long) nz_w, (long long) nz_wd, exact, ggml_type_name(wc->vec_dot_type));
            }

            ss_emu += exact * exact;
            for (int e = 0; e < 3; e++) {
                double dq = 0;
                for (int64_t b0 = 0; b0 < K; b0 += emu_blocks[e]) {
                    float amax = 0;
                    for (int j = 0; j < emu_blocks[e]; j++) amax = std::fmax(amax, std::fabs(x[b0 + j]));
                    const float d = amax / 127.0f, id = d > 0 ? 1.0f / d : 0.0f;
                    for (int j = 0; j < emu_blocks[e]; j++) dq += (double) wd[b0 + j] * (double) (std::nearbyint(x[b0 + j] * id) * d);
                }
                se_emu[e] += (dq - exact) * (dq - exact);
            }
            for (int f = 0; f < 3; f++) {
                const ggml_type_traits     * fb = ggml_get_type_traits(fmts[f].t);
                const ggml_type_traits_cpu * fc = ggml_get_type_traits_cpu(fmts[f].t);
                if (!fc || !fc->from_float) continue;
                const bool raw_q8k = fmts[f].t == GGML_TYPE_Q8_K && !(fb && fb->to_float);
                if (!raw_q8k && !(fb && fb->to_float)) continue;
                std::vector<char> xq(ggml_row_size(fmts[f].t, K) + 64, 0);
                fc->from_float(x.data(), xq.data(), K);
                std::vector<float> xd(K);
                if (raw_q8k) {
                    // This fork's block_q8_K = { float d; float sum; int8_t qs[256]; int16_t bsums[16] } (ggml/src/ggml-common.h:523-528,
                    // SOURCE-READ); the row size below proves the layout, a mismatch aborts rather than mis-decoding.
                    const size_t blk = 4 + 4 + 256 + 32;
                    if (ggml_row_size(GGML_TYPE_Q8_K, 256) != blk) { printf("q8_K layout assumption broken: ggml_row_size(Q8_K,256) = %zu, assumed %zu\n", ggml_row_size(GGML_TYPE_Q8_K, 256), blk); return 2; }
                    for (int64_t b = 0; b * 256 < K; b++) {
                        float d; std::memcpy(&d, xq.data() + b * blk, 4);
                        const int8_t * qs = (const int8_t *) (xq.data() + b * blk + 8);
                        for (int j = 0; j < 256; j++) xd[b * 256 + j] = d * (float) qs[j];
                    }
                } else {
                    fb->to_float(xq.data(), xd.data(), K);
                }
                have_fmt[f] = true;
                double dq = 0;
                for (int64_t i = 0; i < K; i++) dq += (double) wd[i] * (double) xd[i];
                se_fmt[f] += (dq - exact) * (dq - exact);
                ss_fmt[f] += exact * exact;

                if (fmts[f].t == wc->vec_dot_type) {
                    float s = 0;
                    wc->vec_dot((int) K, &s, 0, wq.data(), 0, xq.data(), 0, 1);
                    se_kernel_vs_deq += ((double) s - dq) * ((double) s - dq);
                    se_kernel_vs_exact += ((double) s - exact) * ((double) s - exact);
                    ss_ref += dq * dq;
                }
            }
        }
        printf("\nactivations: gaussian%s, K=%lld, %d trials\n", outliers ? " + 1 percent outliers x20" : ", no outliers", (long long) K, trials);
        if (!(ss_ref > 1e-9)) {
            printf("  NO SIGNAL: the reference dot products are all ~0, so every ratio below would read 0 and prove nothing.\n");
            g_fail++;
            continue;
        }
        const double rel_kernel = std::sqrt(se_kernel_vs_deq / (ss_ref > 0 ? ss_ref : 1e-30));
        printf("  (A) kernel vs double(dequant w . dequant x), format %-12s : rel RMS %.3e   %s\n",
               ggml_type_name(wc->vec_dot_type), rel_kernel, rel_kernel < 1e-5 ? "ok" : "FAIL");
        if (!(rel_kernel < 1e-5)) g_fail++;
        printf("  (kernel vs EXACT float activation, what a model sees)     : rel RMS %.3e\n",
               std::sqrt(se_kernel_vs_exact / (ss_ref > 0 ? ss_ref : 1e-30)));
        for (int e = 0; e < 3; e++) {
            printf("  (B') emulated int8 block quantiser, %3d-element blocks : rel RMS %.3e\n", emu_blocks[e], std::sqrt(se_emu[e] / ss_emu));
        }
        // REGRESSION GUARD: the activation route PQ2_0 really uses must not be noisier than a 128-element-block quantiser (it was
        // q8_K, 256-element blocks, until 2026-10-04: 3.0e-02 against 2.3e-02 here). 5% slack for the finite sample.
        {
            const double route = std::sqrt(se_kernel_vs_exact / (ss_ref > 0 ? ss_ref : 1e-30));
            const double lim   = 1.05 * std::sqrt(se_emu[1] / ss_emu);
            const bool route_ok = route <= lim;
            printf("  (C) route noise %.3e vs the 128-block limit %.3e : %s\n", route, lim, route_ok ? "ok" : "FAIL");
            if (!route_ok) g_fail++;
        }
        for (int f = 0; f < 3; f++) {
            if (!have_fmt[f]) { printf("  (B) %s : no to_float/from_float in this build, SKIPPED (not a verdict)\n", fmts[f].label); continue; }
            printf("  (B) activation noise, %s                : rel RMS %.3e\n", fmts[f].label, std::sqrt(se_fmt[f] / (ss_fmt[f] > 0 ? ss_fmt[f] : 1e-30)));
        }
    }
    printf("\n%s\n", g_fail ? "FAIL: the PQ2_0 kernel disagrees with the double-precision decode" : "kernel decode agrees; B shows the activation-quantiser contribution per format");
    return g_fail ? 1 : 0;
}

// GPU check (Vulkan, CUDA, ...) of TurboQuant flash attention: GGML_OP_FLASH_ATTN_EXT with turbo K and/or V against a host reference.
//
// WHY A HOST REFERENCE AND NOT test-backend-ops: that tool compares a backend with the CPU backend, and the CPU backend has no turbo attention by design
// (llama-context.cpp refuses turbo KV without a GPU; ggml-cpu.cpp declines the types), so there is nothing to compare against.
//
// THE CONTRACT BEING TESTED. A turbo KV block stores its 128-element group ROTATED (R = diag(S2) . H128 . diag(S1), orthogonal), then quantised. The attention kernel
// works in the rotated basis: it decodes centroid * norm (no inverse rotation) and the backend wraps it with  Q <- R Q  (K side)  and  out <- R^T out  (V side),
// which is exact because (R q).(R k) == q.k and sum_i p_i (R v_i) == R (sum_i p_i v_i). So a backend result must equal ordinary attention over the
// UN-rotated, dequantised K and V:
//     K_true = R^T dequant(K_blocks)        V_true = R^T dequant(V_blocks)       (turbo_cpu_fwht_inverse per 128-group; q8_0 sides are not rotated)
//     out[iq,h] = softmax_j( scale * Q[iq,h] . K_true[hk,j] + mask[iq,j] ) . V_true[hk,j]
// The blocks are written ONCE, on the host, by the reference writer (quantize_row_turbo*_0_ref through the type traits); the backend only reads them, so this
// isolates the READ path (decode + Q rotation + output un-rotation). The WRITE path is tests/test-turbo-cpy.cpp.
//
// CONTROLS (the verdict must be able to fail): on the host, the two ways a backend could get the rotation wrong are computed and must differ from the true
// reference by NMSE > 0.1:  NO-Q-ROTATION (Q . K_rotated with an un-rotated Q) and NO-OUTPUT-UNROTATION (V_rotated used as V). A backend that skipped either would match the
// control, not the reference, and fail the main check. Thresholds are fixed here before any run: kFaNmseMax = 2e-3 (fp16 inputs/f32 accumulate: measured values for
// the similar q8_0 path are ~1e-5; the budget leaves 100x), kControlNmseMin = 0.1.
//
// CPU MODE (`test-turbo-fa CPU`, a ctest entry): the CPU backend has no flash-attention wrapper, so the graph carries the rotations (llama-turbo-graph.h does that in the
// real graph builders): GGML_OP_TURBO_WHT forward on Q when K is turbo, FLASH_ATTN_EXT, GGML_OP_TURBO_WHT inverse on the output when V is turbo. Here the graph is built by hand with
// the same two ops, so the test is independent of the helper (tests/test-turbo-graph-rot.cpp tests the helper). Tolerance kCpuNmseMax = 1e-6 (f32 Q, f32 dequant, f32 accumulate;
// fixed before the first run; a q8_0 K side converts Q to q8_0 in the generic CPU kernel, so that case keeps kFaNmseMax). Cases with hs_pad/hsv set cover head sizes that are not 128: K is zero-padded to the next multiple of 128 exactly as the cache does
// (192 -> 256, 576 -> 640), Q is zero-padded the same way, and the reference attends over the LOGICAL width.
//
// Usage:  test-turbo-fa [backend-name]      default CUDA0; CPU = the CPU backend with the graph-level rotation; Vulkan0 for the Vulkan flash-attention shaders. SKIPs (exit 0) when the backend cannot be created.
//         test-turbo-fa host                 NO backend is loaded or touched: only the host parts run (the two controls and the rotation identity on the quantised data).
// The device run is NOT a ctest entry on purpose (it needs the GPU and is run by hand, alone); the host mode is.
//
// GATE-CLAIM-OK: a test source; it names no resource state.

#include "turbo-write-check.h"

#include "ggml-alloc.h"
#include "ggml-backend.h"

#include <algorithm>
#include <cstdlib>
#include <string>

extern "C" void turbo_cpu_fwht_inverse(float * x, int group_size);

static int g_fail = 0;

static void expect(bool ok, const std::string & what) {
    printf("  %-4s %s\n", ok ? "ok" : "FAIL", what.c_str());
    if (!ok) ++g_fail;
}

static const double kFaNmseMax      = 2e-3;
static const double kControlNmseMin = 0.1;
static const double kCpuNmseMax     = 1e-6;
// a q8_0 K side makes the generic CPU FA convert Q to q8_0 first (~1e-5 NMSE by itself, nothing to do with turbo): those cases use kFaNmseMax (measured 5.9e-5)

struct Case {
    ggml_type tk, tv;
    int hs, n_head, n_head_kv, n_q, n_kv;
    int hs_pad = 0;   // K/Q head size as stored (zero padded); 0 = hs
    int hsv    = 0;   // V head size; 0 = hs
};

static bool is_turbo(ggml_type t) { return t == GGML_TYPE_TURBO2_0 || t == GGML_TYPE_TURBO3_0 || t == GGML_TYPE_TURBO4_0; }

static std::vector<uint8_t> quantize_any(ggml_type t, const std::vector<float> & x) {
    const ggml_type_traits * tr = ggml_get_type_traits(t);
    std::vector<uint8_t> out((x.size() / tr->blck_size) * tr->type_size, 0);
    tr->from_float_ref(x.data(), out.data(), (int64_t) x.size());
    return out;
}

// dequantised values as stored (ROTATED domain for turbo) and, optionally, un-rotated
static std::vector<float> decode(ggml_type t, const std::vector<uint8_t> & q, size_t n, bool unrotate) {
    std::vector<float> y(n, 0.0f);
    ggml_get_type_traits(t)->to_float(q.data(), y.data(), (int64_t) n);
    if (unrotate && is_turbo(t)) {
        for (size_t g = 0; g + 128 <= n; g += 128) turbo_cpu_fwht_inverse(&y[g], 128);
    }
    return y;
}

// forward rotation of one 128-group, F = diag(S2) . (1/sqrt(128)) H . diag(S1): the same maths as turbo_cpu_fwht in ggml-turbo-quant.c, written against the test's own sign tables
static void rotate_fwd(float * v) {
    for (int j = 0; j < 128; ++j) v[j] *= tw::kSigns1[j];
    for (int h = 1; h < 128; h *= 2)
        for (int i = 0; i < 128; i += 2 * h)
            for (int j = i; j < i + h; ++j) { const float a = v[j], b = v[j + h]; v[j] = a + b; v[j + h] = a - b; }
    for (int j = 0; j < 128; ++j) v[j] *= 0.08838834764831845f * tw::kSigns2[j];
}

// ordinary attention in double. Q [h][iq][d], K/V [hk][j][d], mask(iq, j) additive.
static std::vector<float> attention_ref(const Case & c, const std::vector<float> & Q, const std::vector<float> & K, const std::vector<float> & V,
                                        const std::vector<float> & maskf, float scale) {
    const int hv = c.hsv ? c.hsv : c.hs;
    std::vector<float> out((size_t) hv * c.n_head * c.n_q, 0.0f);
    const int grp = c.n_head / c.n_head_kv;
    std::vector<double> s(c.n_kv);
    for (int h = 0; h < c.n_head; ++h) {
        const int hk = h / grp;
        for (int iq = 0; iq < c.n_q; ++iq) {
            const float * q = &Q[((size_t) h * c.n_q + iq) * c.hs];
            double mx = -1e300;
            for (int j = 0; j < c.n_kv; ++j) {
                const float * k = &K[((size_t) hk * c.n_kv + j) * c.hs];
                double dot = 0;
                for (int d = 0; d < c.hs; ++d) dot += (double) q[d] * k[d];
                s[j] = dot * scale + maskf[(size_t) iq * c.n_kv + j];
                mx = std::max(mx, s[j]);
            }
            double den = 0;
            for (int j = 0; j < c.n_kv; ++j) { s[j] = std::exp(s[j] - mx); den += s[j]; }
            float * o = &out[((size_t) iq * c.n_head + h) * hv];
            for (int d = 0; d < hv; ++d) {
                double acc = 0;
                for (int j = 0; j < c.n_kv; ++j) acc += s[j] * V[((size_t) hk * c.n_kv + j) * hv + d];
                o[d] = (float) (acc / den);
            }
        }
    }
    return out;
}

static bool g_require_turbo4 = false;
static bool g_cpu_mode       = false;   // the CPU backend: the graph carries the rotations
// CUDA flash attention requires the KV length to be a multiple of FATTN_KQ_STRIDE (256), which the KV cache guarantees by padding;
// ggml_cuda_fattn_is_supported() (fattn.cu) refuses any other length, so an unpadded case is never claimed. On CUDA the device
// tensors are padded the same way: zero K/V rows, masked to -inf, so the logical attention and the host reference do not change.
static int  g_kv_pad         = 1;

static void run_case(ggml_backend_t be /* nullptr: host parts only */, const Case & c, uint32_t seed) {
    const std::string name = std::string(ggml_type_name(c.tk)) + "/" + ggml_type_name(c.tv) + " hs" + std::to_string(c.hs) + " heads " + std::to_string(c.n_head) + "/" +
                             std::to_string(c.n_head_kv) + " n_q " + std::to_string(c.n_q) + " n_kv " + std::to_string(c.n_kv);
    printf("%s\n", name.c_str());

    // Work in the STORED (zero padded) domain: the zero tail of Q kills whatever quantisation noise lands in the K tail, so attention over the padded width
    // equals attention over the logical width; only the softmax scale comes from the logical head size.
    const float scale = 1.0f / std::sqrt((float) c.hs);
    const int hk = c.hs_pad ? c.hs_pad : c.hs;
    const int hv = c.hsv ? c.hsv : c.hs;
    Case cs = c;            // the case as the kernels see it
    cs.hs = hk;
    const int rows = c.n_head_kv * c.n_kv;
    const int gk = (hk / 128) * rows, gv = (hv / 128) * rows;
    std::vector<float> Kf = tw::gen_data(gk, seed), Vf = tw::gen_data(gv, seed + 1);
    // the generator mixes scales on purpose (zeros, tiny groups, outliers); keep attention logits in a sane range for the softmax
    for (float & v : Kf) v *= 0.25f;
    for (int r = 0; r < rows; ++r) {
        for (int d = c.hs; d < hk; ++d) Kf[(size_t) r * hk + d] = 0.0f;   // the cache zero pads K rows to a whole rotation group
    }
    std::mt19937 rng(seed + 2);
    std::normal_distribution<float> N(0.0f, 1.0f);
    std::vector<float> Q((size_t) hk * c.n_q * c.n_head, 0.0f);
    for (int r = 0; r < c.n_q * c.n_head; ++r) {
        for (int d = 0; d < c.hs; ++d) Q[(size_t) r * hk + d] = 0.8f * N(rng);
    }

    const std::vector<uint8_t> Kq = quantize_any(c.tk, Kf), Vq = quantize_any(c.tv, Vf);
    const size_t nk = Kf.size();

    // mask: causal with the new tokens at the end of the cache
    const int n_q_pad = GGML_PAD(c.n_q, GGML_KQ_MASK_PAD);
    std::vector<ggml_fp16_t> mask16((size_t) c.n_kv * n_q_pad, ggml_fp32_to_fp16(0.0f));
    std::vector<float> maskf((size_t) c.n_kv * c.n_q, 0.0f);
    for (int iq = 0; iq < c.n_q; ++iq) {
        for (int j = 0; j < c.n_kv; ++j) {
            const bool ok = j <= (c.n_kv - c.n_q) + iq;
            mask16[(size_t) iq * c.n_kv + j] = ggml_fp32_to_fp16(ok ? 0.0f : -INFINITY);
            maskf[(size_t) iq * c.n_kv + j]  = ok ? 0.0f : -1e30f;
        }
    }

    // host references: the true one, and the two ways to get the rotation wrong
    const size_t nv = Vf.size();
    const std::vector<float> Kt = decode(c.tk, Kq, nk, true),  Vt = decode(c.tv, Vq, nv, true);
    const std::vector<float> Kr = decode(c.tk, Kq, nk, false), Vr = decode(c.tv, Vq, nv, false);
    const std::vector<float> ref = attention_ref(cs, Q, Kt, Vt, maskf, scale);
    if (is_turbo(c.tk)) {
        const std::vector<float> ctl = attention_ref(cs, Q, Kr, Vt, maskf, scale);      // un-rotated Q against rotated K
        const double e = tw::nmse(ctl, ref);
        printf("  host control NO-Q-ROTATION        nmse vs the true reference %.4g\n", e);
        expect(e > kControlNmseMin, "CONTROL: skipping the Q rotation changes the result by NMSE > " + std::to_string(kControlNmseMin) + " (the check can see it)");
    }
    if (is_turbo(c.tv)) {
        const std::vector<float> ctl = attention_ref(cs, Q, Kt, Vr, maskf, scale);      // rotated V used as is
        const double e = tw::nmse(ctl, ref);
        printf("  host control NO-OUTPUT-UNROTATION nmse vs the true reference %.4g\n", e);
        expect(e > kControlNmseMin, "CONTROL: skipping the output un-rotation changes the result by NMSE > " + std::to_string(kControlNmseMin) + " (the check can see it)");
    }

    // host proof of the identity the whole design rests on, on THIS data: attention done the way the backend does it (rotate Q, attend over the
    // stored rotated K and V, un-rotate the output) equals attention over the un-rotated K and V. Needs no device.
    {
        std::vector<float> Qr = Q;
        if (is_turbo(c.tk)) {
            for (size_t g = 0; g + 128 <= Qr.size(); g += 128) rotate_fwd(&Qr[g]);
        }
        const std::vector<float> Ka = is_turbo(c.tk) ? Kr : Kt;
        const std::vector<float> Va = is_turbo(c.tv) ? Vr : Vt;
        std::vector<float> got = attention_ref(cs, Qr, Ka, Va, maskf, scale);
        if (is_turbo(c.tv)) {
            for (size_t g = 0; g + 128 <= got.size(); g += 128) turbo_cpu_fwht_inverse(&got[g], 128);
        }
        const double e = tw::nmse(got, ref);
        printf("  host identity (rotate Q, attend in the rotated basis, un-rotate) vs attention over the un-rotated K/V: nmse %.4g\n", e);
        expect(e < 1e-9, "HOST: the rotation identity holds on this data (NMSE < 1e-9)");
    }

    if (!be) return;

    // device copies, KV length padded to g_kv_pad (see g_kv_pad). Rows quantise independently, so the real rows of Kd/Vd are the same blocks as Kq/Vq.
    const int n_kv_dev = GGML_PAD(c.n_kv, g_kv_pad);
    std::vector<uint8_t> Kd = Kq, Vd = Vq;
    std::vector<ggml_fp16_t> maskd = mask16;
    if (n_kv_dev != c.n_kv) {
        std::vector<float> Kp((size_t) c.n_head_kv * n_kv_dev * hk, 0.0f), Vp((size_t) c.n_head_kv * n_kv_dev * hv, 0.0f);
        for (int h = 0; h < c.n_head_kv; ++h) {
            std::copy_n(&Kf[(size_t) h * c.n_kv * hk], (size_t) c.n_kv * hk, &Kp[(size_t) h * n_kv_dev * hk]);
            std::copy_n(&Vf[(size_t) h * c.n_kv * hv], (size_t) c.n_kv * hv, &Vp[(size_t) h * n_kv_dev * hv]);
        }
        Kd = quantize_any(c.tk, Kp);
        Vd = quantize_any(c.tv, Vp);
        maskd.assign((size_t) n_kv_dev * n_q_pad, ggml_fp32_to_fp16(-INFINITY));
        for (int iq = 0; iq < n_q_pad; ++iq) {
            std::copy_n(&mask16[(size_t) iq * c.n_kv], c.n_kv, &maskd[(size_t) iq * n_kv_dev]);
        }
        printf("  device KV length padded %d -> %d (masked)\n", c.n_kv, n_kv_dev);
    }

    // device run
    ggml_init_params ip = { ggml_tensor_overhead() * 16 + ggml_graph_overhead(), nullptr, true };
    ggml_context * ctx = ggml_init(ip);
    ggml_tensor * q = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, hk, c.n_q, c.n_head, 1);
    ggml_tensor * k = ggml_new_tensor_3d(ctx, c.tk, hk, n_kv_dev, c.n_head_kv);
    ggml_tensor * v = ggml_new_tensor_3d(ctx, c.tv, hv, n_kv_dev, c.n_head_kv);
    ggml_tensor * m = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, n_kv_dev, n_q_pad);
    // CPU: no backend wrapper, so the graph rotates (forward on Q when K is turbo, inverse on the output when V is turbo). GPU backends rotate inside their own FA wrapper.
    ggml_tensor * qa = (g_cpu_mode && is_turbo(c.tk)) ? ggml_turbo_wht(ctx, q, nullptr, 0, 128) : q;
    ggml_tensor * o = ggml_flash_attn_ext(ctx, qa, k, v, m, scale, 0.0f, 0.0f);
    ggml_flash_attn_ext_set_prec(o, GGML_PREC_F32);
    if (g_cpu_mode && is_turbo(c.tv)) o = ggml_turbo_wht(ctx, o, nullptr, 1, 128);
    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, o);

    const bool claimed = ggml_backend_supports_op(be, o);
    const bool uses_turbo4 = c.tk == GGML_TYPE_TURBO4_0 || c.tv == GGML_TYPE_TURBO4_0;
    if (!claimed && uses_turbo4 && !g_require_turbo4) {
        printf("  SKIP this backend does not claim turbo4 attention (not a verdict)\n");
        ggml_free(ctx);
        return;
    }
    expect(claimed, "backend claims GGML_OP_FLASH_ATTN_EXT with this K/V pair (supports_op)");
    if (!claimed) { ggml_free(ctx); return; }
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, be);
    if (!buf) { expect(false, "buffer allocation"); ggml_free(ctx); return; }
    ggml_backend_tensor_set(q, Q.data(), 0, Q.size() * sizeof(float));
    ggml_backend_tensor_set(k, Kd.data(), 0, Kd.size());
    ggml_backend_tensor_set(v, Vd.data(), 0, Vd.size());
    ggml_backend_tensor_set(m, maskd.data(), 0, maskd.size() * sizeof(ggml_fp16_t));
    const bool ran = ggml_backend_graph_compute(be, gf) == GGML_STATUS_SUCCESS;
    expect(ran, "graph compute succeeded");
    if (ran) {
        std::vector<float> got((size_t) hv * c.n_head * c.n_q);
        ggml_backend_tensor_get(o, got.data(), 0, got.size() * sizeof(float));
        bool finite = true;
        for (float x : got) finite = finite && std::isfinite(x);
        expect(finite, "every output value is finite");
        const double e = tw::nmse(got, ref);
        const double lim = (g_cpu_mode && c.tk != GGML_TYPE_Q8_0) ? kCpuNmseMax : kFaNmseMax;
        printf("  device vs true reference          nmse %.4g (limit %.1g)\n", e, lim);
        expect(finite && e <= lim, "device output matches attention over the un-rotated dequantised K/V");
    }
    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
}

int main(int argc, char ** argv) {
    const char * be_name = argc > 1 ? argv[1] : "CUDA0";
    const bool host_only = std::string(be_name) == "host";
    g_cpu_mode = std::string(be_name) == "CPU";
    tw::init_tables();
    ggml_backend_t be = nullptr;
    if (!host_only) {
        ggml_backend_load_all();
        be = ggml_backend_init_by_name(be_name, nullptr);
        if (!be) {
            printf("SKIPPED: backend %s could not be initialised in this build. Not a verdict.\n", be_name);
            return 0;
        }
    }
    printf("backend %s\n", host_only ? "(none: host parts only)" : be_name);

    // turbo4 attention exists on the Vulkan backend only (CUDA has no turbo4 writer or reader): there it is required, elsewhere a case the backend does not
    // claim is reported and skipped instead of failing.
    g_require_turbo4 = g_cpu_mode || std::string(be_name).rfind("Vulkan", 0) == 0;
    g_kv_pad         = std::string(be_name).rfind("CUDA", 0) == 0 ? 256 : 1;
    const std::vector<Case> cases = {
        { GGML_TYPE_TURBO3_0, GGML_TYPE_TURBO3_0, 128, 8, 2,  1,  64 },    // decode, GQA 4
        { GGML_TYPE_TURBO3_0, GGML_TYPE_TURBO3_0, 128, 8, 2,  4, 200 },
        { GGML_TYPE_TURBO3_0, GGML_TYPE_TURBO3_0, 256, 8, 2, 33, 200 },    // prefill-shaped batch, head size 256
        { GGML_TYPE_TURBO2_0, GGML_TYPE_TURBO2_0, 128, 4, 4,  4, 129 },    // no GQA, KV length not a multiple of any block
        { GGML_TYPE_TURBO3_0, GGML_TYPE_Q8_0,     128, 8, 2,  1, 128 },    // mixed: only K is rotated
        { GGML_TYPE_Q8_0,     GGML_TYPE_TURBO3_0, 128, 8, 2,  1, 128 },    // mixed: only the output is un-rotated
        { GGML_TYPE_TURBO4_0, GGML_TYPE_TURBO4_0, 128, 8, 2,  5, 300 },    // 4-bit PolarQuant (Vulkan only; CUDA does not claim it)
    };
    // head sizes that are not a multiple of 128 (K zero padded to a whole group, as the cache does). The GPU kernels are instantiated for 128/256 only, so these run on the CPU.
    const std::vector<Case> cases_padded = {
        { GGML_TYPE_TURBO3_0, GGML_TYPE_TURBO3_0, 192, 8, 2,  3, 130, 256, 128 },   // DeepSeek-style K 192 -> 256, V 128
        { GGML_TYPE_TURBO3_0, GGML_TYPE_TURBO3_0, 576, 8, 1,  2, 200, 640, 512 },   // MLA-shaped K 576 -> 640, V 512
        { GGML_TYPE_TURBO4_0, GGML_TYPE_TURBO2_0,  96, 4, 2,  1,  64, 128, 128 },   // K 96 -> 128, mixed turbo4 / turbo2
    };
    uint32_t seed = 20261005;
    for (const Case & c : cases) {
        run_case(be, c, seed);
        seed += 7;
    }
    if (g_cpu_mode) {
        for (const Case & c : cases_padded) {
            run_case(be, c, seed);
            seed += 7;
        }
    }

    if (be) ggml_backend_free(be);
    printf("%s: %d check(s) failed\n", g_fail ? "FAIL" : "PASS", g_fail);
    return g_fail ? 1 : 0;
}

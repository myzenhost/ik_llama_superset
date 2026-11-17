// TurboQuant KV on the CPU backend: how many rotations does ONE attention call get, on each path?
//
// The GPU backends (CUDA, Vulkan, Metal, SYCL) rotate Q and the attention output inside their own flash-attention wrapper. The CPU backend has no wrapper, so the
// graph builders (llm_build_kqv, llm_graph_context::build_attn_mha) emit GGML_OP_TURBO_WHT through the helpers of src/llama-turbo-graph.h -- only when the
// cache is in host memory. A cache in a device buffer must get NO graph rotation, or it would be rotated twice (once here, once by the wrapper). This test
// counts the GGML_OP_TURBO_WHT nodes the helpers put in the graph and, for the host path, runs the graph on the CPU backend against an f32 reference.
//
//   path            cache buffer     K / V            expected WHT nodes in the graph (fwd on Q, inv on out)
//   host            host             turbo / turbo    2  (1 forward, 1 inverse)
//   host            host             turbo / q8_0     1  (forward only: only K is rotated)
//   host            host             q8_0 / turbo     1  (inverse only)
//   host            host             f16 / f16        0
//   device (fake)   NOT host         turbo / turbo    0  (the backend wrapper rotates; GPU numerics are UNMEASURED here, no GPU)
//
// Numerics (host path, CPU backend): out == softmax(scale Q K_true^T + mask) V_true with K_true = R^T dequant(K) (kCpuNmseMax = 1e-6, fixed before the first run).
// Controls, each must differ from the reference by NMSE > 0.1:  NO rotation (device policy on a host run) and DOUBLE rotation (the helpers applied twice) -- the two
// ways a conflict between graph and wrapper would show up.
//
// GATE-CLAIM-OK: a test source; it names no resource state.

#include "../src/llama-turbo-graph.h"
#include "../ggml/src/ggml-backend-impl.h"

#include "ggml-alloc.h"
#include "ggml-cpu.h"

#include <cmath>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <random>
#include <string>
#include <vector>

extern "C" void turbo_cpu_fwht_inverse(float * x, int group_size);

static int g_fail = 0;
static void expect(bool ok, const std::string & what) {
    printf("  %-4s %s\n", ok ? "ok" : "FAIL", what.c_str());
    if (!ok) ++g_fail;
}

static const double kCpuNmseMax     = 1e-6;
// A q8_0 K side is a CPU kernel property, not a turbo one: the generic FA converts Q to q8_0 (the vec_dot type of q8_0) before the dot, which costs ~1e-5 NMSE by
// itself. Those cases use the 2e-3 budget test-turbo-fa.cpp states for the fp16/q8 paths (measured on this CPU path: 5.9e-5).
static const double kCpuQ8NmseMax   = 2e-3;
static const double kControlNmseMin = 0.1;

static double nmse(const std::vector<float> & a, const std::vector<float> & b) {
    double num = 0, den = 0;
    for (size_t i = 0; i < a.size(); ++i) { const double d = (double) a[i] - b[i]; num += d*d; den += (double) b[i]*b[i]; }
    return den > 0 ? num/den : num;
}

// a buffer type that is not host memory (stands in for a device buffer; nothing is ever allocated from it)
static bool fake_is_host(ggml_backend_buffer_type_t) { return false; }
static const char * fake_name(ggml_backend_buffer_type_t) { return "FAKE_DEVICE"; }

static int count_wht(ggml_cgraph * gf) {
    int n = 0;
    for (int i = 0; i < ggml_graph_n_nodes(gf); ++i) n += ggml_graph_node(gf, i)->op == GGML_OP_TURBO_WHT;
    return n;
}

static std::vector<uint8_t> quantize(ggml_type t, const std::vector<float> & x) {
    const ggml_type_traits * tr = ggml_get_type_traits(t);
    std::vector<uint8_t> out((x.size() / tr->blck_size) * tr->type_size, 0);
    tr->from_float_ref(x.data(), out.data(), (int64_t) x.size());
    return out;
}

static std::vector<float> true_values(ggml_type t, const std::vector<uint8_t> & q, size_t n) {
    std::vector<float> y(n);
    ggml_get_type_traits(t)->to_float(q.data(), y.data(), (int64_t) n);
    if (llama_turbo_is_type(t)) {
        for (size_t g = 0; g + 128 <= n; g += 128) turbo_cpu_fwht_inverse(&y[g], 128);
    }
    return y;
}

struct Case { const char * name; ggml_type tk, tv; int hs, n_head, n_head_kv, n_q, n_kv; int want; };

// which path: 0 = host cache + helpers, 1 = fake device cache + helpers, 2 = host cache, no rotation (control), 3 = host cache, helpers twice (control)
static const char * kPathName[] = { "host", "device (fake buffer)", "CONTROL host, no rotation", "CONTROL host, rotated twice" };

static void run(ggml_backend_t cpu, const Case & c, int path, uint32_t seed) {
    printf("%s  [%s]\n", c.name, kPathName[path]);
    const float scale = 1.0f / std::sqrt((float) c.hs);
    std::mt19937 rng(seed);
    std::normal_distribution<float> N(0.0f, 1.0f);
    const size_t nrow = (size_t) c.n_head_kv * c.n_kv;
    std::vector<float> Kf(nrow * c.hs), Vf(nrow * c.hs), Q((size_t) c.hs * c.n_q * c.n_head);
    for (float & v : Kf) v = 0.35f * N(rng);
    for (float & v : Vf) v = N(rng);
    for (float & v : Q)  v = 0.8f * N(rng);
    const std::vector<uint8_t> Kq = quantize(c.tk, Kf), Vq = quantize(c.tv, Vf);
    const std::vector<float> Kt = true_values(c.tk, Kq, Kf.size()), Vt = true_values(c.tv, Vq, Vf.size());

    // reference in double
    std::vector<float> ref((size_t) c.hs * c.n_head * c.n_q);
    const int grp = c.n_head / c.n_head_kv;
    for (int h = 0; h < c.n_head; ++h) for (int iq = 0; iq < c.n_q; ++iq) {
        const float * q = &Q[((size_t) h * c.n_q + iq) * c.hs];
        std::vector<double> s(c.n_kv);
        double mx = -1e300;
        for (int j = 0; j < c.n_kv; ++j) {
            const float * k = &Kt[((size_t) (h / grp) * c.n_kv + j) * c.hs];
            double d = 0; for (int e = 0; e < c.hs; ++e) d += (double) q[e] * k[e];
            s[j] = d * scale; mx = std::max(mx, s[j]);
        }
        double den = 0; for (int j = 0; j < c.n_kv; ++j) { s[j] = std::exp(s[j] - mx); den += s[j]; }
        float * o = &ref[((size_t) iq * c.n_head + h) * c.hs];
        for (int e = 0; e < c.hs; ++e) {
            double a = 0; for (int j = 0; j < c.n_kv; ++j) a += s[j] * Vt[((size_t) (h / grp) * c.n_kv + j) * c.hs + e];
            o[e] = (float) (a / den);
        }
    }

    ggml_init_params ip = { ggml_tensor_overhead() * 24 + ggml_graph_overhead(), nullptr, true };
    ggml_context * ctx = ggml_init(ip);
    const int n_q_pad = GGML_PAD(c.n_q, GGML_KQ_MASK_PAD);
    ggml_tensor * q = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, c.hs, c.n_q, c.n_head, 1);
    ggml_tensor * k = ggml_new_tensor_3d(ctx, c.tk, c.hs, c.n_kv, c.n_head_kv);
    ggml_tensor * v = ggml_new_tensor_3d(ctx, c.tv, c.hs, c.n_kv, c.n_head_kv);
    ggml_tensor * m = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, c.n_kv, n_q_pad);

    // the cache tensors' buffers: the real one (host) allocates; the fake one only has to answer is_host
    ggml_backend_buffer_t buf = nullptr;
    ggml_backend_buffer_type_t fake_buft = nullptr;
    ggml_backend_buffer_t fake_buf = nullptr;
    ggml_backend_buffer_type_i fake_iface = {};
    fake_iface.get_name = fake_name; fake_iface.is_host = fake_is_host;
    ggml_backend_buffer_type fake_buft_s = { fake_iface, nullptr, nullptr };
    fake_buft = &fake_buft_s;
    buf = ggml_backend_alloc_ctx_tensors(ctx, cpu);
    if (!buf) { expect(false, "buffer allocation"); ggml_free(ctx); return; }
    ggml_backend_tensor_set(q, Q.data(), 0, Q.size() * sizeof(float));
    ggml_backend_tensor_set(k, Kq.data(), 0, Kq.size());
    ggml_backend_tensor_set(v, Vq.data(), 0, Vq.size());
    std::vector<ggml_fp16_t> mask((size_t) c.n_kv * n_q_pad, ggml_fp32_to_fp16(0.0f));
    ggml_backend_tensor_set(m, mask.data(), 0, mask.size() * sizeof(ggml_fp16_t));

    // path 1: the cache the builders see lives in a non-host buffer. Only the buffer pointer is swapped for the helpers' decision; data stays where it is.
    ggml_backend_buffer_t real_k = k->buffer, real_v = v->buffer;
    if (path == 1) {
        fake_buf = ggml_backend_buffer_init(fake_buft, {}, nullptr, 0);
        k->buffer = fake_buf; v->buffer = fake_buf;
    }
    const bool use_helpers = path != 2;
    ggml_tensor * qa = use_helpers ? llama_turbo_rotate_q(ctx, q, k) : q;
    if (path == 3) qa = llama_turbo_rotate_q(ctx, qa, k);
    ggml_tensor * o = ggml_flash_attn_ext(ctx, qa, k, v, m, scale, 0.0f, 0.0f);
    ggml_flash_attn_ext_set_prec(o, GGML_PREC_F32);
    if (use_helpers) o = llama_turbo_unrotate_out(ctx, o, v);
    if (path == 3) o = llama_turbo_unrotate_out(ctx, o, v);
    k->buffer = real_k; v->buffer = real_v;     // restore before compute / free
    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, o);
    // the inputs are allocated; the nodes created above are not
    ggml_gallocr_t galloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(cpu));
    if (path != 1 && !ggml_gallocr_alloc_graph(galloc, gf)) { expect(false, "graph allocation"); }

    const int n_wht = count_wht(gf);
    const int want = path == 0 ? c.want : path == 1 ? 0 : path == 2 ? 0 : 2 * c.want;
    printf("  GGML_OP_TURBO_WHT nodes in the graph: %d (expected %d)\n", n_wht, want);
    expect(n_wht == want, "rotation count per attention call is " + std::to_string(want));

    if (path != 1) {
        ggml_tensor * fa = nullptr;
        for (int i = 0; i < ggml_graph_n_nodes(gf); ++i) if (ggml_graph_node(gf, i)->op == GGML_OP_FLASH_ATTN_EXT) fa = ggml_graph_node(gf, i);
        expect(fa && ggml_backend_supports_op(cpu, fa), "CPU backend claims GGML_OP_FLASH_ATTN_EXT with this K/V pair");
        const bool ran = ggml_backend_graph_compute(cpu, gf) == GGML_STATUS_SUCCESS;
        expect(ran, "graph compute succeeded");
        if (ran) {
            std::vector<float> got(ref.size());
            ggml_backend_tensor_get(o, got.data(), 0, got.size() * sizeof(float));
            const double e = nmse(got, ref);
            if (path == 0) {
                const double lim = c.tk == GGML_TYPE_Q8_0 ? kCpuQ8NmseMax : kCpuNmseMax;
                printf("  CPU vs f32 reference           nmse %.4g (limit %.1g)\n", e, lim);
                expect(e <= lim, "CPU output matches attention over the un-rotated dequantised K/V");
            } else if (c.want > 0) {
                printf("  control vs f32 reference       nmse %.4g (must exceed %.1g)\n", e, kControlNmseMin);
                expect(e > kControlNmseMin, "CONTROL differs from the reference (the check can see a wrong rotation count)");
            }
        }
    }
    ggml_gallocr_free(galloc);
    if (fake_buf) { delete fake_buf; }
    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
}

// the KV WRITE on the CPU backend: GGML_OP_CPY F32 -> turbo (how llm_build_kv_store stores a K / V row of one head or of n_tokens heads) must produce exactly the blocks
// the reference writer makes row by row. Before the turbo traits rows existed this aborted ("not implemented") in ops.cpp.
static void check_write(ggml_backend_t cpu, ggml_type t, int hs, int rows) {
    printf("CPU CPY f32 -> %s, %d rows of %d\n", ggml_type_name(t), rows, hs);
    ggml_init_params ip = { ggml_tensor_overhead() * 8 + ggml_graph_overhead(), nullptr, true };
    ggml_context * ctx = ggml_init(ip);
    ggml_tensor * src = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, hs, rows);
    ggml_tensor * dst = ggml_new_tensor_2d(ctx, t, hs, rows);
    ggml_tensor * cp  = ggml_cpy(ctx, src, dst);
    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, cp);
    expect(ggml_backend_supports_op(cpu, cp), "CPU backend claims GGML_OP_CPY f32 -> turbo");
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, cpu);
    std::mt19937 rng(99);
    std::normal_distribution<float> N(0.0f, 1.0f);
    std::vector<float> x((size_t) hs * rows);
    for (float & v : x) v = N(rng);
    ggml_backend_tensor_set(src, x.data(), 0, x.size() * sizeof(float));
    const bool ran = ggml_backend_graph_compute(cpu, gf) == GGML_STATUS_SUCCESS;
    expect(ran, "graph compute succeeded");
    if (ran) {
        const size_t rs = ggml_row_size(t, hs);
        std::vector<uint8_t> got(rs * rows), want(rs * rows);
        ggml_backend_tensor_get(dst, got.data(), 0, got.size());
        for (int r = 0; r < rows; ++r) ggml_get_type_traits(t)->from_float_ref(&x[(size_t) r * hs], want.data() + r * rs, hs);
        expect(got == want, "stored blocks are byte-identical to the reference writer, row by row");
    }
    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
}

int main() {
    ggml_backend_t cpu = ggml_backend_cpu_init();
    if (!cpu) { printf("FAIL: no CPU backend\n"); return 1; }
    check_write(cpu, GGML_TYPE_TURBO3_0, 128, 6);
    check_write(cpu, GGML_TYPE_TURBO2_0, 256, 5);
    check_write(cpu, GGML_TYPE_TURBO4_0, 128, 3);
    const std::vector<Case> cases = {
        { "turbo3/turbo3 hs128", GGML_TYPE_TURBO3_0, GGML_TYPE_TURBO3_0, 128, 8, 2, 4, 200, 2 },
        { "turbo2/turbo4 hs256", GGML_TYPE_TURBO2_0, GGML_TYPE_TURBO4_0, 256, 4, 4, 3, 130, 2 },
        { "turbo3/q8_0   hs128", GGML_TYPE_TURBO3_0, GGML_TYPE_Q8_0,     128, 8, 2, 1, 128, 1 },
        { "q8_0/turbo3   hs128", GGML_TYPE_Q8_0,     GGML_TYPE_TURBO3_0, 128, 8, 2, 1, 128, 1 },
        { "f16/f16       hs128", GGML_TYPE_F16,      GGML_TYPE_F16,      128, 8, 2, 2, 64,  0 },
    };
    uint32_t seed = 7001;
    for (const Case & c : cases) {
        for (int path = 0; path < 4; ++path) {
            if (path == 1 && c.want == 0) continue;     // nothing to count on a device path without turbo
            run(cpu, c, path, seed);
        }
        seed += 11;
    }
    ggml_backend_free(cpu);
    printf("%s: %d check(s) failed\n", g_fail ? "FAIL" : "PASS", g_fail);
    return g_fail ? 1 : 0;
}

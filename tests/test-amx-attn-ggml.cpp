// test-amx-attn-ggml -- GGML_OP_FLASH_ATTN_EXT on the CPU backend with bf16 K/V, checked against a double reference.
// Row S69-OV-PICK-AMX-BF16-ATTENTION-DECODE-GEMV-AND-PREFILL-2026-10-06-01 (wiring half).
//
// The op must be right whichever CPU path runs it. In a build with GGML_AMX_TILE + GGML_AMX_BF16 and GGML_AMX_ATTN=1
// the AMX kernel (ggml/src/ggml-cpu/amx/attn-bf16.h, dispatched from ops.cpp ggml_compute_forward_flash_attn_ext)
// takes the supported cases and prints "AMX-BF16 flash attention FIRED" once on stderr; without the env (or in any
// other build) iqk or the generic kernel takes them. The runner checks the FIRED line, runs both settings under
// `sde -spr`, and the forced setting natively (a CPU without AMX must die with an illegal instruction there).
#include "ggml.h"
#include "ggml-cpu.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

static float bf16_to_f32(ggml_bf16_t b) {
    const uint32_t u = (uint32_t) b.bits << 16;
    float f;
    std::memcpy(&f, &u, 4);
    return f;
}

struct Case {
    int D, Dv, n_q, n_head, n_head_kv, n_kv;
    bool causal;
    float softcap;
    bool sinks;
    const char * name;
};

static bool run(const Case & cs, int n_threads) {
    ggml_init_params ip = { 256u * 1024 * 1024, nullptr, false };
    ggml_context * ctx = ggml_init(ip);
    const int n_q_pad = GGML_PAD(cs.n_q, GGML_KQ_MASK_PAD);
    ggml_tensor * q = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, cs.D, cs.n_q, cs.n_head);
    ggml_tensor * k = ggml_new_tensor_3d(ctx, GGML_TYPE_BF16, cs.D, cs.n_kv, cs.n_head_kv);
    ggml_tensor * v = ggml_new_tensor_3d(ctx, GGML_TYPE_BF16, cs.Dv, cs.n_kv, cs.n_head_kv);
    ggml_tensor * m = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, cs.n_kv, n_q_pad);
    ggml_tensor * s = cs.sinks ? ggml_new_tensor_1d(ctx, GGML_TYPE_F32, cs.n_head) : nullptr;

    std::mt19937 rng(42 + cs.D + cs.n_q);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    for (int64_t i = 0; i < ggml_nelements(q); ++i) ((float *) q->data)[i] = nd(rng);
    for (int64_t i = 0; i < ggml_nelements(k); ++i) ((ggml_bf16_t *) k->data)[i] = ggml_fp32_to_bf16(nd(rng));
    for (int64_t i = 0; i < ggml_nelements(v); ++i) ((ggml_bf16_t *) v->data)[i] = ggml_fp32_to_bf16(nd(rng));
    std::vector<float> maskf((size_t) n_q_pad * cs.n_kv, 0.0f);
    for (int iq = 0; iq < n_q_pad; ++iq) {
        for (int n = 0; n < cs.n_kv; ++n) {
            float x = (float) ((n * 5 + iq) % 4) * -0.5f;
            if (cs.causal && n > cs.n_kv - cs.n_q + iq) x = -INFINITY;
            if (!cs.causal && n >= cs.n_kv - 40) x = -INFINITY;   // decode: a padded tail of empty cells
            maskf[(size_t) iq * cs.n_kv + n] = x;
            ((ggml_fp16_t *) m->data)[(size_t) iq * cs.n_kv + n] = ggml_fp32_to_fp16(x);
        }
    }
    if (s) for (int h = 0; h < cs.n_head; ++h) ((float *) s->data)[h] = nd(rng);

    const float scale = 1.0f / std::sqrt((float) cs.D);
    ggml_tensor * out = ggml_flash_attn_ext(ctx, q, k, v, m, scale, 0.0f, cs.softcap);
    ggml_flash_attn_ext_set_prec(out, GGML_PREC_F32);
    if (s) ggml_flash_attn_ext_add_sinks(out, s);
    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, out);
    if (ggml_graph_compute_with_ctx(ctx, gf, n_threads) != GGML_STATUS_SUCCESS) {
        std::printf("FAIL %s: compute status\n", cs.name);
        ggml_free(ctx);
        return false;
    }

    const int gqa = cs.n_head / cs.n_head_kv;
    double err = 0.0, ref2 = 0.0;
    for (int h = 0; h < cs.n_head; ++h) {
        const int hk = h / gqa;
        for (int iq = 0; iq < cs.n_q; ++iq) {
            std::vector<double> sc(cs.n_kv);
            double mx = -INFINITY;
            for (int n = 0; n < cs.n_kv; ++n) {
                double a = 0.0;
                for (int d = 0; d < cs.D; ++d) {
                    const float qv = bf16_to_f32(ggml_fp32_to_bf16(((float *) q->data)[((size_t) h * cs.n_q + iq) * cs.D + d]));
                    a += (double) qv * bf16_to_f32(((ggml_bf16_t *) k->data)[((size_t) hk * cs.n_kv + n) * cs.D + d]);
                }
                double x = a * scale;
                if (cs.softcap > 0) x = cs.softcap * std::tanh(a * scale / cs.softcap);
                x += ggml_fp16_to_fp32(ggml_fp32_to_fp16(maskf[(size_t) iq * cs.n_kv + n]));
                sc[n] = x;
                mx = x > mx ? x : mx;
            }
            const double sk = s ? ((float *) s->data)[h] : -INFINITY;
            if (s && sk > mx) mx = sk;
            double l = s ? std::exp(sk - mx) : 0.0;
            std::vector<double> o(cs.Dv, 0.0);
            for (int n = 0; n < cs.n_kv; ++n) {
                const double e = std::isinf(sc[n]) ? 0.0 : std::exp(sc[n] - mx);
                l += e;
                for (int c = 0; c < cs.Dv; ++c) o[c] += e * bf16_to_f32(((ggml_bf16_t *) v->data)[((size_t) hk * cs.n_kv + n) * cs.Dv + c]);
            }
            const float * got = (const float *) ((const char *) out->data + h * out->nb[1] + iq * out->nb[2]);
            for (int c = 0; c < cs.Dv; ++c) {
                const double r = o[c] / l;
                err += (got[c] - r) * (got[c] - r);
                ref2 += r * r;
            }
        }
    }
    const double nmse = err / ref2;
    const bool ok = std::isfinite(nmse) && nmse < 1e-3;
    std::printf("%s %-44s NMSE %.3e\n", ok ? "ok  " : "FAIL", cs.name, nmse);
    ggml_free(ctx);
    return ok;
}

int main() {
    const Case cases[] = {
        { 128, 128, 1, 8, 2, 256, false, 0.0f, true,  "decode GQA 4 D=128 n_kv=256 sinks" },
        { 128, 128, 1, 4, 4, 128, false, 0.0f, false, "decode MHA (rows path, M=1)" },
        { 64,  64,  20, 4, 2, 64,  true,  30.0f, false, "prefill n_q=20 causal softcap" },
        { 128, 64,  17, 2, 1, 96,  true,  0.0f, true,  "prefill Dv<D n_q=17 sinks" },
        { 80,  80,  3, 2, 1, 64,  true,  0.0f, false, "D=80: AMX declines (can_run), other path" },
    };
    std::setvbuf(stdout, nullptr, _IONBF, 0);   // a crash must not swallow the cases that already ran
    int n_threads = 4;
    if (const char * e = std::getenv("GGML_AMX_TEST_THREADS")) {
        n_threads = std::max(1, std::atoi(e));
    }
    int fails = 0;
    for (const auto & cs : cases) {
        fails += run(cs, n_threads) ? 0 : 1;
    }
    std::printf("%s: %d failure(s)\n", fails ? "FAILED" : "PASSED", fails);
    return fails ? 1 : 0;
}

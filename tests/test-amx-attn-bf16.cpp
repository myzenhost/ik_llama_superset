// test-amx-attn-bf16 -- numerical test of ggml/src/ggml-cpu/amx/attn-bf16.h against a double-precision reference.
// Row S69-OV-PICK-AMX-BF16-ATTENTION-DECODE-GEMV-AND-PREFILL-2026-10-06-01.
//
// Needs AMX-TILE + AMX-BF16 (Sapphire Rapids or later) or Intel SDE: `sde -spr -- test-amx-attn-bf16`.
// Without them it exits 77 (ctest SKIP) after reading CPUID; GGML_AMX_TEST_FORCE=1 runs the kernel anyway, which on a
// CPU without AMX must die with an illegal instruction (the native negative control).
//
// The reference rounds Q, K and V to bf16 exactly as the kernel does and then computes in double; the only error the
// kernel may add is the bf16 rounding of the softmax weights P before the P.V product (relative 2^-9 per weight), so
// NMSE stays far under the 1e-3 the row asks for. A control case recomputes the reference WITHOUT the mask and must
// disagree, so a kernel that ignored the mask could not pass.
#if defined(AMX_ATTN_HEADER)
#include AMX_ATTN_HEADER   // mutation runs compile the test against a deliberately broken copy
#else
#include "../ggml/src/ggml-cpu/amx/attn-bf16.h"
#endif

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

#if defined(__x86_64__) || defined(_M_X64)
#include <cpuid.h>
#endif
#if defined(__linux__)
#include <sys/syscall.h>
#include <unistd.h>
#endif

static bool cpu_has_amx_bf16() {
#if defined(__x86_64__)
    unsigned a, b, c, d;
    if (!__get_cpuid_count(7, 0, &a, &b, &c, &d)) {
        return false;
    }
    const bool tile = (d >> 24) & 1u;
    const bool bf16 = (d >> 22) & 1u;
    return tile && bf16;
#else
    return false;
#endif
}

static bool os_allows_amx() {
#if defined(__linux__)
    // same request as ggml/src/ggml-cpu/amx/amx.cpp ggml_amx_init (ARCH_REQ_XCOMP_PERM, XFEATURE_XTILEDATA)
    return syscall(SYS_arch_prctl, 0x1023, 18) == 0;
#else
    return true;   // Windows enables the AMX tile state for every process (XFD)
#endif
}

static float bf16_round(float x) {
    const uint16_t h = ggml_amx_attn::f32_to_bf16_rne(x);
    const uint32_t u = (uint32_t) h << 16;
    float f;
    std::memcpy(&f, &u, 4);
    return f;
}

static uint16_t f32_to_fp16(float f) {   // test-side encoder for mask values: 0, -inf, small negatives
    if (std::isinf(f)) {
        return f < 0 ? 0xfc00 : 0x7c00;
    }
    if (f == 0.0f) {
        return 0;
    }
    uint32_t u;
    std::memcpy(&u, &f, 4);
    const uint16_t sign = (u >> 16) & 0x8000u;
    const int      e    = (int) ((u >> 23) & 0xff) - 127 + 15;
    const uint32_t man  = u & 0x7fffffu;
    if (e <= 0 || e >= 31) {
        return sign;   // the test only feeds normal-range values; anything else becomes signed zero
    }
    return (uint16_t) (sign | (e << 10) | (man >> 13));
}

struct Case {
    int M, D, Dv, n_kv;
    bool mask;
    float softcap;
    bool sinks;
    const char * name;
};

static double run_case(const Case & cs, std::mt19937 & rng, bool apply_mask_in_ref, double * nmse_out) {
    const int M = cs.M, D = cs.D, Dv = cs.Dv, n_kv = cs.n_kv;
    std::normal_distribution<float> nd(0.0f, 1.0f);
    // strides deliberately larger than the rows, to catch a kernel that assumed packed rows
    const size_t sq = (size_t) D + 3, sk = (size_t) D + 8, sv = (size_t) Dv + 16, so = (size_t) Dv + 5;
    std::vector<float> q((size_t) M * sq);
    std::vector<ggml_bf16_t> K((size_t) n_kv * sk), V((size_t) n_kv * sv);
    std::vector<ggml_fp16_t> mask((size_t) M * n_kv, 0);
    std::vector<float> sinks(M), out((size_t) M * so, NAN);
    for (auto & x : q) {
        x = nd(rng);
    }
    for (auto & x : K) {
        x.bits = ggml_amx_attn::f32_to_bf16_rne(nd(rng));
    }
    for (auto & x : V) {
        x.bits = ggml_amx_attn::f32_to_bf16_rne(nd(rng));
    }
    if (cs.mask) {
        // causal-style: query m sees keys < n_kv - M + 1 + m; plus one whole 32-key block masked out, plus soft biases
        for (int m = 0; m < M; ++m) {
            for (int n = 0; n < n_kv; ++n) {
                float v = (float) ((n * 7 + m * 3) % 5) * -0.25f;
                if (n >= n_kv - M + 1 + m) {
                    v = -INFINITY;
                }
                if (n_kv >= 64 && n >= 32 && n < 64) {
                    v = -INFINITY;
                }
                mask[(size_t) m * n_kv + n] = f32_to_fp16(v);
            }
        }
    }
    for (auto & s : sinks) {
        s = nd(rng) * 2.0f;
    }
    const float scale = 1.0f / std::sqrt((float) D);

    std::vector<char> scratch(ggml_amx_attn::scratch_size(D, Dv));
#if defined(GGML_AMX_ATTN_BF16)
    ggml_amx_attn::attn_rows(M, D, Dv, n_kv, q.data(), sq, K.data(), sk * 2, V.data(), sv * 2,
                             cs.mask ? mask.data() : nullptr, (size_t) n_kv * 2, scale, cs.softcap,
                             cs.sinks ? sinks.data() : nullptr, out.data(), so, scratch.data());
#endif

    double err = 0.0, ref2 = 0.0;
    for (int m = 0; m < M; ++m) {
        std::vector<double> s(n_kv);
        double mx = -INFINITY;
        for (int n = 0; n < n_kv; ++n) {
            double acc = 0.0;
            for (int d = 0; d < D; ++d) {
                uint32_t u = (uint32_t) K[(size_t) n * sk + d].bits << 16;
                float kf;
                std::memcpy(&kf, &u, 4);
                acc += (double) bf16_round(q[(size_t) m * sq + d]) * kf;
            }
            double x = acc * scale;
            if (cs.softcap > 0.0f) {
                x = cs.softcap * std::tanh(x);
            }
            if (cs.mask && apply_mask_in_ref) {
                x += ggml_amx_attn::fp16_to_f32(mask[(size_t) m * n_kv + n]);
            }
            s[n] = x;
            mx = x > mx ? x : mx;
        }
        if (cs.sinks) {
            mx = sinks[m] > mx ? sinks[m] : mx;
        }
        double l = 0.0;
        std::vector<double> o(Dv, 0.0);
        for (int n = 0; n < n_kv; ++n) {
            const double e = std::isinf(s[n]) ? 0.0 : std::exp(s[n] - mx);
            l += e;
            for (int c = 0; c < Dv; ++c) {
                uint32_t u = (uint32_t) V[(size_t) n * sv + c].bits << 16;
                float vf;
                std::memcpy(&vf, &u, 4);
                o[c] += e * vf;
            }
        }
        if (cs.sinks) {
            l += std::exp(sinks[m] - mx);
        }
        for (int c = 0; c < Dv; ++c) {
            const double r = l > 0 ? o[c] / l : 0.0;
            const double g = out[(size_t) m * so + c];
            err += (g - r) * (g - r);
            ref2 += r * r;
        }
    }
    *nmse_out = ref2 > 0 ? err / ref2 : err;
    return *nmse_out;
}

int main() {
    // fp16 decode self-check (the mask path depends on it): 1.0, -2.0, -inf, smallest subnormal, 65504
    const struct { uint16_t h; float f; } h16[] = {
        { 0x3c00, 1.0f }, { 0xc000, -2.0f }, { 0x0001, 5.9604645e-08f }, { 0x7bff, 65504.0f }, { 0x0200, 3.0517578e-05f },
    };
    for (const auto & t : h16) {
        if (ggml_amx_attn::fp16_to_f32(t.h) != t.f) {
            std::printf("FAIL fp16_to_f32(0x%04x) = %g, want %g\n", t.h, ggml_amx_attn::fp16_to_f32(t.h), t.f);
            return 1;
        }
    }
    if (!std::isinf(ggml_amx_attn::fp16_to_f32(0xfc00)) || ggml_amx_attn::fp16_to_f32(0xfc00) > 0) {
        std::printf("FAIL fp16_to_f32(-inf)\n");
        return 1;
    }
    std::printf("ok   fp16 decode self-check\n");

#if !defined(GGML_AMX_ATTN_BF16)
    std::printf("SKIP built without -mamx-tile -mamx-bf16\n");
    return 77;
#else
    const char * force = std::getenv("GGML_AMX_TEST_FORCE");
    if (!cpu_has_amx_bf16() && !(force && force[0] == '1')) {
        std::printf("SKIP CPU reports no AMX-TILE/AMX-BF16 (run under `sde -spr --`, or GGML_AMX_TEST_FORCE=1 for the negative control)\n");
        return 77;
    }
    if (!os_allows_amx()) {
        std::printf("SKIP the OS refused the AMX tile-data permission\n");
        return 77;
    }
    std::printf("cpu AMX-BF16: %s%s\n", cpu_has_amx_bf16() ? "yes" : "NO", force && force[0] == '1' ? " (forced)" : "");

    const Case cases[] = {
        { 1, 64, 64, 32, false, 0.0f, false, "decode M=1 D=64 one block" },
        { 1, 128, 128, 256, false, 0.0f, false, "decode M=1 D=128 n_kv=256" },
        { 4, 128, 128, 96, true, 0.0f, false, "GQA M=4 masked, block 32..63 fully masked" },
        { 16, 64, 64, 128, true, 0.0f, true, "prefill M=16 masked + sinks" },
        { 7, 256, 128, 64, false, 30.0f, false, "M=7 D=256 Dv=128 softcap 30" },
        { 16, 128, 64, 512, true, 50.0f, true, "M=16 Dv<D masked softcap sinks n_kv=512" },
        { 3, 96, 80, 160, true, 0.0f, true, "odd sizes D=96 Dv=80" },
    };
    std::mt19937 rng(1234);
    int fails = 0;
    double worst = 0.0;
    for (const auto & cs : cases) {
        double nmse = 0.0;
        run_case(cs, rng, true, &nmse);
        const bool ok = nmse < 1e-3 && std::isfinite(nmse);
        worst = nmse > worst ? nmse : worst;
        std::printf("%s %-48s NMSE %.3e\n", ok ? "ok  " : "FAIL", cs.name, nmse);
        fails += ok ? 0 : 1;
    }
    // control: the same masked case against a reference that ignores the mask must fail the bound
    {
        std::mt19937 rng2(99);
        double nmse = 0.0;
        run_case(cases[2], rng2, false, &nmse);
        const bool discriminates = nmse > 1e-2;
        std::printf("%s control: unmasked reference vs masked kernel     NMSE %.3e (must be > 1e-2)\n",
                    discriminates ? "ok  " : "FAIL", nmse);
        fails += discriminates ? 0 : 1;
    }
    // can_run gate
    const bool gate = ggml_amx_attn::can_run(1, 128, 128, 32) && !ggml_amx_attn::can_run(17, 128, 128, 32) &&
                      !ggml_amx_attn::can_run(1, 80, 128, 32) && !ggml_amx_attn::can_run(1, 128, 128, 40) &&
                      !ggml_amx_attn::can_run(1, 128, 72, 32);
    std::printf("%s can_run gate\n", gate ? "ok  " : "FAIL");
    fails += gate ? 0 : 1;
    std::printf("%s: %d failure(s), worst NMSE %.3e\n", fails ? "FAILED" : "PASSED", fails, worst);
    return fails ? 1 : 0;
#endif
}

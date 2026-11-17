// GGML_OP_FMA (ggml_fma / ggml_fma_inplace, ik_llama #2794): dst = c + a*b, a and b broadcast to c.
//
// WHAT IT CHECKS. Builds c + a*b on the CPU backend for several broadcast layouts and compares every element against a scalar host
// reference computed from the same input arrays. test-backend-ops compares a backend against the CPU backend, so on a CPU-only build
// it would compare the kernel with itself; this is the independent check of the CPU kernel's numbers. Also: supports_op answers TRUE
// for F32 and FALSE for F16 on the CPU backend (the kernel is F32 only), and the in-place variant writes into c.
//
// CONTROLS. (a) the reference must differ from c alone (a kernel that returned c unchanged would not pass); (b) the mismatch counter
// is run on a deliberately corrupted copy and must report it (a comparison that cannot see an error would pass vacuously).
// Exit 0 pass, 1 mismatch, 2 setup/control failure.
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

#include <cmath>
#include <cstdio>
#include <cstdint>
#include <vector>

struct shape { int64_t c[4]; int64_t a[4]; int64_t b[4]; bool inplace; const char * name; };

static int count_mismatch(const std::vector<float> & got, const std::vector<float> & ref) {
    int bad = 0;
    for (size_t i = 0; i < ref.size(); ++i) {
        if (!(std::fabs(got[i] - ref[i]) <= 1e-5f * (1.0f + std::fabs(ref[i])))) {
            ++bad;
        }
    }
    return bad;
}

static int run_case(ggml_backend_t be, const shape & s) {
    ggml_init_params ip = { 16u * 1024 * 1024, nullptr, true };
    ggml_context * ctx = ggml_init(ip);
    ggml_tensor * c = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, s.c[0], s.c[1], s.c[2], s.c[3]);
    ggml_tensor * a = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, s.a[0], s.a[1], s.a[2], s.a[3]);
    ggml_tensor * b = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, s.b[0], s.b[1], s.b[2], s.b[3]);
    ggml_tensor * out = s.inplace ? ggml_fma_inplace(ctx, c, a, b) : ggml_fma(ctx, c, a, b);
    if (out->op != GGML_OP_FMA) {
        printf("[%s] SETUP FAILED: result op is not GGML_OP_FMA\n", s.name);
        ggml_free(ctx);
        return 2;
    }
    if (!ggml_backend_supports_op(be, out)) {
        printf("[%s] FAIL: CPU backend refuses an F32 FMA\n", s.name);
        ggml_free(ctx);
        return 1;
    }
    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, out);
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, be);
    if (!buf) {
        printf("[%s] SETUP FAILED: alloc\n", s.name);
        ggml_free(ctx);
        return 2;
    }

    const int64_t nc = ggml_nelements(c), na = ggml_nelements(a), nb = ggml_nelements(b);
    std::vector<float> hc(nc), ha(na), hb(nb);
    for (int64_t i = 0; i < nc; ++i) hc[i] = std::sin(0.37f * i) * 3.0f;
    for (int64_t i = 0; i < na; ++i) ha[i] = std::cos(0.11f * i) * 2.0f + 0.5f;
    for (int64_t i = 0; i < nb; ++i) hb[i] = std::sin(0.23f * i + 1.0f) - 0.25f;
    ggml_backend_tensor_set(c, hc.data(), 0, nc * sizeof(float));
    ggml_backend_tensor_set(a, ha.data(), 0, na * sizeof(float));
    ggml_backend_tensor_set(b, hb.data(), 0, nb * sizeof(float));

    if (ggml_backend_graph_compute(be, gf) != GGML_STATUS_SUCCESS) {
        printf("[%s] FAIL: graph compute\n", s.name);
        ggml_backend_buffer_free(buf);
        ggml_free(ctx);
        return 1;
    }

    std::vector<float> got(nc), ref(nc);
    ggml_backend_tensor_get(out, got.data(), 0, nc * sizeof(float));
    for (int64_t i3 = 0; i3 < s.c[3]; ++i3)
    for (int64_t i2 = 0; i2 < s.c[2]; ++i2)
    for (int64_t i1 = 0; i1 < s.c[1]; ++i1)
    for (int64_t i0 = 0; i0 < s.c[0]; ++i0) {
        const int64_t ic = i0 + s.c[0]*(i1 + s.c[1]*(i2 + s.c[2]*i3));
        const int64_t ia = (i0 % s.a[0]) + s.a[0]*((i1 % s.a[1]) + s.a[1]*((i2 % s.a[2]) + s.a[2]*(i3 % s.a[3])));
        const int64_t ib = (i0 % s.b[0]) + s.b[0]*((i1 % s.b[1]) + s.b[1]*((i2 % s.b[2]) + s.b[2]*(i3 % s.b[3])));
        ref[ic] = hc[ic] + ha[ia] * hb[ib];
    }

    // control (a): the reference is not just c
    if (count_mismatch(hc, ref) == 0) {
        printf("[%s] CONTROL FAILED: reference equals c, the case cannot see a no-op kernel\n", s.name);
        ggml_backend_buffer_free(buf);
        ggml_free(ctx);
        return 2;
    }
    // control (b): the comparison sees a corrupted result
    std::vector<float> bad = got;
    bad[nc/2] += 1.0f;
    if (count_mismatch(bad, ref) != 1) {
        printf("[%s] CONTROL FAILED: the mismatch counter did not flag a corrupted element\n", s.name);
        ggml_backend_buffer_free(buf);
        ggml_free(ctx);
        return 2;
    }

    int rc = 0;
    const int mm = count_mismatch(got, ref);
    if (mm != 0) {
        printf("[%s] FAIL: %d of %lld elements differ from the scalar reference\n", s.name, mm, (long long) nc);
        rc = 1;
    }
    if (s.inplace) {
        // the in-place variant is a view of c: its data pointer is c's
        if (out->data != c->data) {
            printf("[%s] FAIL: ggml_fma_inplace did not write into c\n", s.name);
            rc = 1;
        }
    }
    if (rc == 0) {
        printf("[%s] OK (%lld elements)\n", s.name, (long long) nc);
    }
    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
    return rc;
}

int main() {
    ggml_backend_t be = ggml_backend_cpu_init();
    if (!be) {
        printf("REFUSED: no CPU backend\n");
        return 2;
    }
    ggml_backend_cpu_set_n_threads(be, 3);

    const shape shapes[] = {
        // c                a                b                 inplace name
        {{64, 5, 3, 2},   {64, 5, 3, 2},   {64, 5, 3, 2},    false, "full-shape"},
        {{64, 5, 3, 2},   {64, 5, 3, 2},   {1, 5, 3, 2},     false, "b-per-row-scalar"},
        {{64, 5, 3, 2},   {64, 1, 1, 1},   {64, 1, 1, 1},    false, "a-b-row-broadcast"},
        {{64, 5, 3, 2},   {64, 5, 1, 1},   {1, 1, 3, 1},     false, "mixed-broadcast"},
        {{32, 4, 6, 1},   {32, 4, 6, 1},   {1, 1, 1, 1},     false, "b-scalar"},
        {{129, 7, 1, 1},  {129, 7, 1, 1},  {129, 7, 1, 1},   true,  "inplace-odd-width"},
        {{256, 1, 8, 1},  {256, 1, 8, 1},  {1, 1, 8, 1},     false, "hc-combine-like"},
    };
    int rc = 0;
    for (const shape & s : shapes) {
        const int r = run_case(be, s);
        if (r > rc) rc = r;
    }

    // supports_op: F16 is not implemented on the CPU backend and must be refused, not claimed
    {
        ggml_init_params ip = { 4u * 1024 * 1024, nullptr, true };
        ggml_context * ctx = ggml_init(ip);
        ggml_tensor * c = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, 32, 4);
        ggml_tensor * a = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, 32, 4);
        ggml_tensor * b = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, 1, 4);
        ggml_tensor * out = ggml_fma(ctx, c, a, b);
        if (ggml_backend_supports_op(be, out)) {
            printf("[f16-refused] FAIL: CPU backend claims an F16 FMA it cannot compute\n");
            rc = 1;
        } else {
            printf("[f16-refused] OK\n");
        }
        ggml_free(ctx);
    }

    ggml_backend_free(be);
    printf(rc == 0 ? "ALL OK\n" : "FAILED rc=%d\n", rc);
    return rc;
}

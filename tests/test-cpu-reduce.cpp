// GATE-CLAIM-OK: comment only cites the CUDA reduce kernel as the semantic reference, no resource is declared unavailable.
// GGML_OP_REDUCE on the CPU backend (row S69-ENGREV-DLLAMA-CPU-REDUCE-OP-Q80-PARTIALS-2026-11-06-01).
//
// ggml_reduce(srcs, n, ADD) sums n same-shape tensors (some may be NULL = a device that holds no partial) and leaves the sum
// in every non-null source; the result is a view of the last source. ggml-cuda/reduce.cu is the semantic reference (f16/f32/bf16/q8_0).
// Before this change the CPU backend declined the op (supports_op false) and ggml_compute_forward aborted, which made a
// graph-split model over CPU rpc-servers abort the whole server.
//
// WHAT IT CHECKS. For f32/f16/bf16/q8_0, n = 2..4 sources and NULL-source cases, builds a REDUCE graph on the CPU backend and
// compares the result and every non-null source against a scalar host sum of the dequantised inputs (tolerance per type: f32 1e-5
// relative, f16/bf16 one rounding of the output, q8_0 one quantisation step amax/127 of the block).
//
// CONTROLS. (a) the reference differs from the first source (a no-op kernel would not pass); (b) the mismatch counter flags a
// deliberately corrupted copy (a comparison that cannot see an error would pass vacuously).
// Exit 0 pass, 1 mismatch, 2 setup/control failure.
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

#include <cmath>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <vector>

struct rcase { ggml_type type; int n; int null_mask; int64_t ne0; int64_t ne1; const char * name; };

static float tol_for(ggml_type t, float amax) {
    switch (t) {
        case GGML_TYPE_F32:  return 1e-5f * (1.0f + amax);
        case GGML_TYPE_F16:  return 2e-3f * (1.0f + amax);
        case GGML_TYPE_BF16: return 1.6e-2f * (1.0f + amax);
        case GGML_TYPE_Q8_0: return amax / 127.0f * 1.01f + 1e-6f;
        default:             return 1e-5f;
    }
}

static int count_mismatch(ggml_type t, int64_t ne0, const std::vector<float> & got, const std::vector<float> & ref) {
    int bad = 0;
    const int64_t blk = t == GGML_TYPE_Q8_0 ? 32 : ne0;
    for (size_t i0 = 0; i0 < ref.size(); i0 += blk) {
        float amax = 0.0f;
        for (int64_t k = 0; k < blk; ++k) amax = std::fmax(amax, std::fabs(ref[i0 + k]));
        for (int64_t k = 0; k < blk; ++k) {
            if (!(std::fabs(got[i0 + k] - ref[i0 + k]) <= tol_for(t, amax))) ++bad;
        }
    }
    return bad;
}

static void read_f32(ggml_tensor * t, std::vector<float> & out) {
    const auto * tr = ggml_get_type_traits(t->type);
    std::vector<uint8_t> raw(ggml_nbytes(t));
    ggml_backend_tensor_get(t, raw.data(), 0, raw.size());
    out.resize(ggml_nelements(t));
    if (t->type == GGML_TYPE_F32) {
        memcpy(out.data(), raw.data(), raw.size());
    } else {
        tr->to_float(raw.data(), out.data(), (int64_t) out.size());
    }
}

static int run_case(ggml_backend_t be, const rcase & c) {
    ggml_init_params ip = { 16u * 1024 * 1024, nullptr, true };
    ggml_context * ctx = ggml_init(ip);
    ggml_tensor * src[GGML_MAX_SRC] = {};
    int nhave = 0;
    ggml_tensor * last = nullptr;
    for (int j = 0; j < c.n; ++j) {
        if (c.null_mask & (1 << j)) continue;
        src[j] = ggml_new_tensor_2d(ctx, c.type, c.ne0, c.ne1);
        ++nhave;
        last = src[j];
    }
    ggml_tensor * out = ggml_reduce(ctx, src, c.n, GGML_OP_ADD);
    if (out->op != GGML_OP_REDUCE || out->view_src != last) {
        printf("[%s] SETUP FAILED: result is not a REDUCE view of the last source\n", c.name);
        ggml_free(ctx);
        return 2;
    }
    if (!ggml_backend_supports_op(be, out)) {
        printf("[%s] FAIL: CPU backend refuses a REDUCE it must compute\n", c.name);
        ggml_free(ctx);
        return 1;
    }
    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, out);
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, be);
    if (!buf) {
        printf("[%s] SETUP FAILED: alloc\n", c.name);
        ggml_free(ctx);
        return 2;
    }

    const int64_t ne = c.ne0 * c.ne1;
    const auto * tr = ggml_get_type_traits(c.type);
    std::vector<std::vector<float>> in(c.n);
    for (int j = 0; j < c.n; ++j) {
        if (!src[j]) continue;
        std::vector<float> f(ne);
        for (int64_t i = 0; i < ne; ++i) f[i] = std::sin(0.31f * i + 1.7f * j) * (1.0f + j) + 0.1f * j;
        std::vector<uint8_t> raw(ggml_nbytes(src[j]));
        if (c.type == GGML_TYPE_F32) memcpy(raw.data(), f.data(), raw.size());
        else tr->from_float_ref(f.data(), raw.data(), ne);
        ggml_backend_tensor_set(src[j], raw.data(), 0, raw.size());
        read_f32(src[j], in[j]);
    }

    if (ggml_backend_graph_compute(be, gf) != GGML_STATUS_SUCCESS) {
        printf("[%s] FAIL: graph compute\n", c.name);
        ggml_backend_buffer_free(buf);
        ggml_free(ctx);
        return 1;
    }

    std::vector<float> ref(ne, 0.0f), got;
    for (int j = 0; j < c.n; ++j) {
        if (!src[j]) continue;
        for (int64_t i = 0; i < ne; ++i) ref[i] += in[j][i];
    }
    int first = 0;
    while (!src[first]) ++first;
    if (count_mismatch(c.type, c.ne0, in[first], ref) == 0) {
        printf("[%s] CONTROL FAILED: reference equals the first source, a no-op kernel would pass\n", c.name);
        ggml_backend_buffer_free(buf);
        ggml_free(ctx);
        return 2;
    }

    read_f32(out, got);
    std::vector<float> bad = got;
    bad[ne / 2] += 1.0f;
    if (count_mismatch(c.type, c.ne0, bad, ref) < 1) {
        printf("[%s] CONTROL FAILED: the mismatch counter did not flag a corrupted element\n", c.name);
        ggml_backend_buffer_free(buf);
        ggml_free(ctx);
        return 2;
    }

    int rc = 0;
    int mm = count_mismatch(c.type, c.ne0, got, ref);
    if (mm) {
        printf("[%s] FAIL: result has %d of %lld elements off the scalar sum\n", c.name, mm, (long long) ne);
        rc = 1;
    }
    for (int j = 0; j < c.n; ++j) {
        if (!src[j]) continue;
        std::vector<float> s;
        read_f32(src[j], s);
        mm = count_mismatch(c.type, c.ne0, s, ref);
        if (mm) {
            printf("[%s] FAIL: source %d has %d elements off the sum\n", c.name, j, mm);
            rc = 1;
        }
    }
    if (rc == 0) printf("[%s] OK (%lld elements, %d sources)\n", c.name, (long long) ne, nhave);
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

    const rcase cases[] = {
        {GGML_TYPE_F32,  2, 0,    96, 5, "f32-n2"},
        {GGML_TYPE_F32,  3, 0,    96, 5, "f32-n3"},
        {GGML_TYPE_F32,  4, 0,    96, 7, "f32-n4"},
        {GGML_TYPE_F32,  4, 0x2,  96, 7, "f32-n4-null1"},
        {GGML_TYPE_F32,  3, 0x1,  64, 3, "f32-n3-null0"},
        {GGML_TYPE_F16,  3, 0,    96, 5, "f16-n3"},
        {GGML_TYPE_BF16, 3, 0,    96, 5, "bf16-n3"},
        {GGML_TYPE_Q8_0, 2, 0,    64, 5, "q8_0-n2"},
        {GGML_TYPE_Q8_0, 4, 0,    128, 6, "q8_0-n4"},
        {GGML_TYPE_Q8_0, 3, 0x2,  64, 5, "q8_0-n3-null1"},
    };
    int rc = 0;
    for (const rcase & c : cases) {
        const int r = run_case(be, c);
        if (r > rc) rc = r;
    }
    ggml_backend_free(be);
    printf(rc == 0 ? "ALL OK\n" : "FAILED rc=%d\n", rc);
    return rc;
}

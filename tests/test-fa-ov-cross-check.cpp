// test-fa-ov-cross-check -- the fork's CPU flash attention (ggml_flash_attn_ext) against OpenVINO's CPU plugin
// ScaledDotProductAttention (opset13) on the same random Q/K/V: single-token decode and short causal prefill, MHA and GQA.
// Row: S69-A-PICK-INTEL-NPU-XPU-OPENVINO-PLUGIN-FEATURES-IPEXLLM-OPENVINO-CORE-2026-10-05-01 (owed item: "single-token MHA
// numerical comparison OpenVINO CPU vs fork FA on random Q/K/V, NMSE").
//
// OpenVINO is reached through its C API, loaded at RUNTIME (the same discipline as tools/mtmd/providers/openvino): the
// library is MSVC-built and this test is built by MinGW/clang/MSVC alike, so nothing links against the OpenVINO C++ ABI.
// Library: $OV_C_LIB (full path to openvino_c.dll / libopenvino_c.so), else $MTMD_OPENVINO_C_LIB. Neither set or the load
// fails -> exit 77 (ctest SKIP, never a PASS). Headers come from LLAMA_TEST_OPENVINO_INCLUDE (tests/CMakeLists.txt).
//
// The model is a one-op IR written in memory: five Parameters (q, k, v, mask, scale) -> ScaledDotProductAttention -> Result.
// No constants, so no weights blob. Compiled for "CPU" with INFERENCE_PRECISION_HINT=f32 (the CPU plugin may otherwise run
// bf16 on hosts that have it).
//
// Both sides see the SAME numbers: K and V are rounded to f16 once (ggml stores them f16) and handed to OpenVINO as f32
// holding those values. GQA: OpenVINO gets K/V repeated per query head with the ggml broadcast rule (head h reads kv head
// h / (n_head / n_head_kv)). Mask: additive f32 for OpenVINO, the same values as f16 for ggml.
// PASS: NMSE < 1e-5 on every case. CONTROL: the same comparison with OpenVINO given scale x1.25 must FAIL (NMSE > 1e-3),
// so the check is shown able to see a real difference.
#include "ggml.h"
#include "ggml-cpu.h"

#define BOOLEAN OV_BOOLEAN
#include "openvino/c/ov_common.h"
#include "openvino/c/ov_shape.h"
#include "openvino/c/ov_tensor.h"
#include "openvino/c/ov_core.h"
#include "openvino/c/ov_model.h"
#include "openvino/c/ov_compiled_model.h"
#include "openvino/c/ov_infer_request.h"
#undef BOOLEAN

#if defined(_WIN32)
#  define WIN32_LEAN_AND_MEAN
#  include <windows.h>
#else
#  include <dlfcn.h>
#endif

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <string>
#include <vector>

namespace {

#define OV_FN(name) decltype(&name) name##_fn = nullptr
struct ov_api {
    OV_FN(ov_core_create);
    OV_FN(ov_core_free);
    OV_FN(ov_core_read_model_from_memory_buffer);
    OV_FN(ov_core_compile_model);
    OV_FN(ov_model_free);
    OV_FN(ov_compiled_model_create_infer_request);
    OV_FN(ov_compiled_model_free);
    OV_FN(ov_infer_request_set_input_tensor_by_index);
    OV_FN(ov_infer_request_infer);
    OV_FN(ov_infer_request_get_output_tensor_by_index);
    OV_FN(ov_infer_request_free);
    OV_FN(ov_tensor_create_from_host_ptr);
    OV_FN(ov_tensor_get_size);
    OV_FN(ov_tensor_data);
    OV_FN(ov_tensor_free);
    OV_FN(ov_shape_create);
    OV_FN(ov_shape_free);
    OV_FN(ov_get_last_err_msg);
};
#undef OV_FN

bool load_api(ov_api & a) {
    const char * p = std::getenv("OV_C_LIB");
    if (!p || !*p) p = std::getenv("MTMD_OPENVINO_C_LIB");
    if (!p || !*p) {
        std::printf("SKIP: set OV_C_LIB to the openvino_c library\n");
        return false;
    }
#if defined(_WIN32)
    // ALTERED_SEARCH_PATH: openvino_c.dll's own dependencies (openvino.dll, the CPU plugin) resolve from its directory
    void * lib = (void *) LoadLibraryExA(p, nullptr, LOAD_WITH_ALTERED_SEARCH_PATH);
#  define SYM(n) (void *) GetProcAddress((HMODULE) lib, n)
#else
    void * lib = dlopen(p, RTLD_NOW | RTLD_LOCAL);
#  define SYM(n) dlsym(lib, n)
#endif
    if (!lib) {
        std::printf("SKIP: could not load %s\n", p);
        return false;
    }
#define OV_LOAD(name) \
    a.name##_fn = (decltype(a.name##_fn)) SYM(#name); \
    if (!a.name##_fn) { std::printf("SKIP: %s lacks %s\n", p, #name); return false; }
    OV_LOAD(ov_core_create)
    OV_LOAD(ov_core_free)
    OV_LOAD(ov_core_read_model_from_memory_buffer)
    OV_LOAD(ov_core_compile_model)
    OV_LOAD(ov_model_free)
    OV_LOAD(ov_compiled_model_create_infer_request)
    OV_LOAD(ov_compiled_model_free)
    OV_LOAD(ov_infer_request_set_input_tensor_by_index)
    OV_LOAD(ov_infer_request_infer)
    OV_LOAD(ov_infer_request_get_output_tensor_by_index)
    OV_LOAD(ov_infer_request_free)
    OV_LOAD(ov_tensor_create_from_host_ptr)
    OV_LOAD(ov_tensor_get_size)
    OV_LOAD(ov_tensor_data)
    OV_LOAD(ov_tensor_free)
    OV_LOAD(ov_shape_create)
    OV_LOAD(ov_shape_free)
    OV_LOAD(ov_get_last_err_msg)
#undef OV_LOAD
#undef SYM
    return true;
}

struct fa_case { int hd, n_q, n_kv, n_head, n_head_kv; const char * name; };

float f16_round(float x) { return ggml_fp16_to_fp32(ggml_fp32_to_fp16(x)); }

std::string dims_xml(const std::vector<int64_t> & d) {
    std::string s;
    for (int64_t v : d) s += "<dim>" + std::to_string(v) + "</dim>";
    return s;
}

std::string shape_attr(const std::vector<int64_t> & d) {
    std::string s;
    for (size_t i = 0; i < d.size(); ++i) s += (i ? "," : "") + std::to_string(d[i]);
    return s;
}

// IR v11: Parameter x5 -> ScaledDotProductAttention(opset13, causal=false) -> Result
std::string sdpa_ir(const fa_case & c) {
    const std::vector<int64_t> qs = {1, c.n_head, c.n_q, c.hd}, ks = {1, c.n_head, c.n_kv, c.hd}, ms = {c.n_q, c.n_kv}, ss = {1};
    const std::vector<int64_t> shp[5] = {qs, ks, ks, ms, ss};
    const char * nm[5] = {"q", "k", "v", "mask", "scale"};
    std::string x = "<?xml version=\"1.0\"?>\n<net name=\"sdpa\" version=\"11\">\n<layers>\n";
    for (int i = 0; i < 5; ++i) {
        x += "<layer id=\"" + std::to_string(i) + "\" name=\"" + nm[i] + "\" type=\"Parameter\" version=\"opset1\">"
             "<data shape=\"" + shape_attr(shp[i]) + "\" element_type=\"f32\"/>"
             "<output><port id=\"0\" precision=\"FP32\">" + dims_xml(shp[i]) + "</port></output></layer>\n";
    }
    x += "<layer id=\"5\" name=\"sdpa\" type=\"ScaledDotProductAttention\" version=\"opset13\"><data causal=\"false\"/><input>";
    for (int i = 0; i < 5; ++i) x += "<port id=\"" + std::to_string(i) + "\" precision=\"FP32\">" + dims_xml(shp[i]) + "</port>";
    x += "</input><output><port id=\"5\" precision=\"FP32\">" + dims_xml(qs) + "</port></output></layer>\n";
    x += "<layer id=\"6\" name=\"out\" type=\"Result\" version=\"opset1\"><input><port id=\"0\" precision=\"FP32\">" + dims_xml(qs) +
         "</port></input></layer>\n</layers>\n<edges>\n";
    for (int i = 0; i < 5; ++i) {
        x += "<edge from-layer=\"" + std::to_string(i) + "\" from-port=\"0\" to-layer=\"5\" to-port=\"" + std::to_string(i) + "\"/>\n";
    }
    x += "<edge from-layer=\"5\" from-port=\"5\" to-layer=\"6\" to-port=\"0\"/>\n</edges>\n</net>\n";
    return x;
}

// returns out[(t * n_head + h) * hd + d]
std::vector<float> run_ggml(const fa_case & c, const std::vector<float> & Q, const std::vector<float> & K, const std::vector<float> & V,
                            const std::vector<float> & mask, float scale) {
    ggml_init_params ip = { (size_t) 256 * 1024 * 1024, nullptr, false };
    ggml_context * ctx = ggml_init(ip);
    ggml_tensor * q = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, c.hd, c.n_q, c.n_head);
    ggml_tensor * k = ggml_new_tensor_3d(ctx, GGML_TYPE_F16, c.hd, c.n_kv, c.n_head_kv);
    ggml_tensor * v = ggml_new_tensor_3d(ctx, GGML_TYPE_F16, c.hd, c.n_kv, c.n_head_kv);
    const int n_q_pad = GGML_PAD(c.n_q, GGML_KQ_MASK_PAD);
    ggml_tensor * m = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, c.n_kv, n_q_pad);
    for (size_t i = 0; i < Q.size(); ++i) ((float *) q->data)[i] = Q[i];              // host layout [h][t][d] == ggml [d, t, h]
    for (size_t i = 0; i < K.size(); ++i) {
        ((ggml_fp16_t *) k->data)[i] = ggml_fp32_to_fp16(K[i]);
        ((ggml_fp16_t *) v->data)[i] = ggml_fp32_to_fp16(V[i]);
    }
    for (int t = 0; t < n_q_pad; ++t)
        for (int s = 0; s < c.n_kv; ++s)
            ((ggml_fp16_t *) m->data)[(size_t) t * c.n_kv + s] = ggml_fp32_to_fp16(t < c.n_q ? mask[(size_t) t * c.n_kv + s] : -INFINITY);
    ggml_tensor * out = ggml_flash_attn_ext(ctx, q, k, v, m, scale, 0.0f, 0.0f);
    ggml_flash_attn_ext_set_prec(out, GGML_PREC_F32);
    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, out);
    ggml_graph_compute_with_ctx(ctx, gf, 4);
    std::vector<float> r((size_t) c.n_q * c.n_head * c.hd);                         // out [hd, n_head, n_q]
    for (size_t i = 0; i < r.size(); ++i) r[i] = ((float *) out->data)[i];
    ggml_free(ctx);
    return r;
}

struct ov_run {
    const ov_api & a;
    ov_core_t * core = nullptr;
    bool fail(const char * what, int st) {
        const char * m = a.ov_get_last_err_msg_fn();
        std::printf("OpenVINO %s failed, status %d: %s\n", what, st, m ? m : "");
        return false;
    }
    // returns false on any OpenVINO error (the test then FAILS: the library loaded, so an error is a real failure)
    bool run(const fa_case & c, const std::vector<float> & Q, const std::vector<float> & K, const std::vector<float> & V,
             const std::vector<float> & mask, float scale, std::vector<float> & r) {
        const std::string ir = sdpa_ir(c);
        ov_model_t * model = nullptr;
        int st = a.ov_core_read_model_from_memory_buffer_fn(core, ir.data(), ir.size(), nullptr, &model);
        if (st != 0) return fail("read_model", st);
        ov_compiled_model_t * cm = nullptr;
        st = a.ov_core_compile_model_fn(core, model, "CPU", 2, &cm, "INFERENCE_PRECISION_HINT", "f32");
        a.ov_model_free_fn(model);
        if (st != 0) return fail("compile_model", st);
        ov_infer_request_t * req = nullptr;
        st = a.ov_compiled_model_create_infer_request_fn(cm, &req);
        if (st != 0) { a.ov_compiled_model_free_fn(cm); return fail("create_infer_request", st); }
        const int gqa = c.n_head / c.n_head_kv;
        std::vector<float> kx((size_t) c.n_head * c.n_kv * c.hd), vx(kx.size());
        for (int h = 0; h < c.n_head; ++h)
            for (int s = 0; s < c.n_kv; ++s)
                for (int d = 0; d < c.hd; ++d) {
                    const size_t src = ((size_t) (h / gqa) * c.n_kv + s) * c.hd + d;
                    kx[((size_t) h * c.n_kv + s) * c.hd + d] = f16_round(K[src]);
                    vx[((size_t) h * c.n_kv + s) * c.hd + d] = f16_round(V[src]);
                }
        std::vector<float> qx(Q), mx(mask), sx(1, scale);
        float * bufs[5] = {qx.data(), kx.data(), vx.data(), mx.data(), sx.data()};
        const std::vector<int64_t> shp[5] = {{1, c.n_head, c.n_q, c.hd}, {1, c.n_head, c.n_kv, c.hd}, {1, c.n_head, c.n_kv, c.hd},
                                             {c.n_q, c.n_kv}, {1}};
        ov_tensor_t * tin[5] = {};
        bool ok = true;
        for (int i = 0; i < 5 && ok; ++i) {
            ov_shape_t sh;
            st = a.ov_shape_create_fn((int64_t) shp[i].size(), shp[i].data(), &sh);
            if (st != 0) { ok = fail("shape_create", st); break; }
            st = a.ov_tensor_create_from_host_ptr_fn(F32, sh, bufs[i], &tin[i]);
            a.ov_shape_free_fn(&sh);
            if (st != 0) { ok = fail("tensor_create", st); break; }
            st = a.ov_infer_request_set_input_tensor_by_index_fn(req, (size_t) i, tin[i]);
            if (st != 0) ok = fail("set_input", st);
        }
        if (ok) {
            st = a.ov_infer_request_infer_fn(req);
            if (st != 0) ok = fail("infer", st);
        }
        if (ok) {
            ov_tensor_t * to = nullptr;
            st = a.ov_infer_request_get_output_tensor_by_index_fn(req, 0, &to);
            if (st != 0) {
                ok = fail("get_output", st);
            } else {
                size_t n = 0;
                void * od = nullptr;
                a.ov_tensor_get_size_fn(to, &n);
                a.ov_tensor_data_fn(to, &od);
                r.assign((size_t) c.n_q * c.n_head * c.hd, 0.0f);
                if (n != r.size()) {
                    std::printf("OpenVINO output has %zu elements, expected %zu\n", n, r.size());
                    ok = false;
                } else {
                    const float * o = (const float *) od;                            // [1, n_head, n_q, hd] -> [t][h][d]
                    for (int h = 0; h < c.n_head; ++h)
                        for (int t = 0; t < c.n_q; ++t)
                            for (int d = 0; d < c.hd; ++d)
                                r[((size_t) t * c.n_head + h) * c.hd + d] = o[((size_t) h * c.n_q + t) * c.hd + d];
                }
                a.ov_tensor_free_fn(to);
            }
        }
        for (ov_tensor_t * t : tin) if (t) a.ov_tensor_free_fn(t);
        a.ov_infer_request_free_fn(req);
        a.ov_compiled_model_free_fn(cm);
        return ok;
    }
};

double nmse(const std::vector<float> & a, const std::vector<float> & ref) {
    double num = 0.0, den = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
        const double e = (double) a[i] - ref[i];
        num += e * e;
        den += (double) ref[i] * ref[i];
    }
    return den > 0.0 ? num / den : (num > 0.0 ? INFINITY : 0.0);
}

} // namespace

int main() {
    ov_api api;
    if (!load_api(api)) return 77;
    ov_run ov{api};
    int st = api.ov_core_create_fn(&ov.core);
    if (st != 0) { ov.fail("core_create", st); return 1; }

    const fa_case cases[] = {
        { 64,  1,  257, 8, 8, "decode MHA hd64 nkv257" },
        { 128, 1, 1024, 8, 2, "decode GQA4 hd128 nkv1024" },
        { 64,  1,   33, 9, 3, "decode GQA3 hd64 nkv33" },
        { 64, 16,  300, 8, 4, "prefill16 GQA2 hd64 nkv300 causal" },
        { 128, 7,  512, 4, 4, "prefill7 MHA hd128 nkv512 causal" },
    };
    std::mt19937 rng(20261019);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    int fails = 0;
    bool control_ok = true;
    for (const fa_case & c : cases) {
        std::vector<float> Q((size_t) c.n_head * c.n_q * c.hd), K((size_t) c.n_head_kv * c.n_kv * c.hd), V(K.size());
        for (float & x : Q) x = nd(rng);
        for (float & x : K) x = nd(rng);
        for (float & x : V) x = nd(rng);
        std::vector<float> mask((size_t) c.n_q * c.n_kv, 0.0f);
        for (int t = 0; t < c.n_q; ++t)                          // query t sits at position n_kv - n_q + t
            for (int s = 0; s < c.n_kv; ++s)
                if (s > c.n_kv - c.n_q + t) mask[(size_t) t * c.n_kv + s] = -INFINITY;
        const float scale = 1.0f / std::sqrt((float) c.hd);
        const std::vector<float> g = run_ggml(c, Q, K, V, mask, scale);
        std::vector<float> o, o_bad;
        if (!ov.run(c, Q, K, V, mask, scale, o) || !ov.run(c, Q, K, V, mask, scale * 1.25f, o_bad)) {
            std::printf("%-36s OpenVINO run FAILED\n", c.name);
            ++fails;
            control_ok = false;
            continue;
        }
        const double e = nmse(g, o);
        const bool ok = e < 1e-5;
        fails += !ok;
        const double eb = nmse(g, o_bad);
        control_ok = control_ok && eb > 1e-3;
        std::printf("%-36s NMSE %.3e %s | control (OV scale x1.25) NMSE %.3e %s\n", c.name, e, ok ? "ok" : "FAIL", eb,
                    eb > 1e-3 ? "detected" : "MISSED");
    }
    api.ov_core_free_fn(ov.core);
    std::printf("%s (%d case failure(s), control %s)\n", (fails == 0 && control_ok) ? "PASS" : "FAIL", fails, control_ok ? "ok" : "FAILED");
    return (fails == 0 && control_ok) ? 0 : 1;
}

// mtmd encoder provider backed by the OpenVINO C API (CPU / GPU / NPU plugins of OpenVINO).
// Contract: tools/mtmd/mtmd-encoder-provider.h.
//
// SOURCE-READ (2026-10-05): whisper.cpp @ 6e4ab854f67f743900934a703d5603419384c961 (MIT)
//   src/openvino/whisper-openvino-encoder.cpp, 108 lines: init(path, device, cache) / encode(mel -> embd) / free,
//   ov::cache_dir so compiled device blobs persist, ggml tensors wrapped as ov::Tensor without a copy.
// DIFFERENCES from that shim (own design, no code copied):
//   - the C API (openvino_c.dll / libopenvino_c.so) is loaded at RUNTIME, so building this module needs only the headers
//     and a MinGW/clang/MSVC toolchain without the OpenVINO C++ ABI; a machine without OpenVINO simply opts out;
//   - every status code is checked and turned into a NULL init / non-zero encode, nothing aborts the process;
//   - the input is rank 3 [1][n_mel][n_step] (what an exported whisper encoder takes), the output element count
//     is checked against the caller's n_out_floats BEFORE the copy.
//
// Model: an OpenVINO IR (.xml + .bin) whose single input is f32 [1][n_mel][n_step] and whose first output holds
// n_out_floats f32 values. scripts/mtmd-export-whisper-openvino.py writes such an IR from a whisper-style mmproj GGUF.
//
// Library discovery order: $MTMD_OPENVINO_C_LIB (full path), then $OPENVINO_ROOT/libs/openvino_c.dll (the layout of
// the `openvino` PyPI wheel; $OPENVINO_ROOT is then the directory holding `libs`), then the loader's default search.

#include "mtmd-encoder-provider.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

// ov_element_type_e has an enumerator named BOOLEAN, which collides with the BOOLEAN typedef of <windows.h>. The
// vendor's own umbrella header (openvino/c/openvino.h) handles it with `#define BOOLEAN OV_BOOLEAN`; the individual headers
// do not, so this file does the same and includes the vendor headers BEFORE windows.h.
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

namespace {

// function pointer types come from the vendor headers via decltype, which does not reference the (dllimport) symbols
#define OV_FN(name) decltype(&name) name##_fn = nullptr
struct ov_api {
    void * lib = nullptr;
    OV_FN(ov_core_create);
    OV_FN(ov_core_free);
    OV_FN(ov_core_read_model);
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
    decltype(&ov_get_last_err_msg) ov_get_last_err_msg_fn = nullptr;
};
#undef OV_FN

void * lib_open(const std::string & path) {
#if defined(_WIN32)
    return (void *) LoadLibraryA(path.c_str());
#else
    return dlopen(path.c_str(), RTLD_NOW | RTLD_LOCAL);
#endif
}

void * lib_sym(void * lib, const char * name) {
#if defined(_WIN32)
    return (void *) GetProcAddress((HMODULE) lib, name);
#else
    return dlsym(lib, name);
#endif
}

std::string env_or(const char * k) {
    const char * v = std::getenv(k);
    return v ? std::string(v) : std::string();
}

bool load_api(ov_api & a) {
    std::string cands[3];
    cands[0] = env_or("MTMD_OPENVINO_C_LIB");
    const std::string root = env_or("OPENVINO_ROOT");
#if defined(_WIN32)
    if (!root.empty()) {
        cands[1] = root + "/libs/openvino_c.dll";
    }
    cands[2] = "openvino_c.dll";
#else
    if (!root.empty()) {
        cands[1] = root + "/libs/libopenvino_c.so";
    }
    cands[2] = "libopenvino_c.so";
#endif
    for (const auto & c : cands) {
        if (c.empty()) {
            continue;
        }
        a.lib = lib_open(c);
        if (a.lib) {
            break;
        }
    }
    if (!a.lib) {
        std::fprintf(stderr, "mtmd-encoder-openvino: openvino_c library not found (set MTMD_OPENVINO_C_LIB or OPENVINO_ROOT)\n");
        return false;
    }
#define OV_LOAD(name) \
    a.name##_fn = (decltype(a.name##_fn)) lib_sym(a.lib, #name); \
    if (!a.name##_fn) { std::fprintf(stderr, "mtmd-encoder-openvino: missing symbol %s\n", #name); return false; }
    OV_LOAD(ov_core_create)
    OV_LOAD(ov_core_free)
    OV_LOAD(ov_core_read_model)
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
    return true;
}

struct ov_session {
    ov_api api;
    ov_core_t * core = nullptr;
    ov_model_t * model = nullptr;
    ov_compiled_model_t * compiled = nullptr;
    ov_infer_request_t * req = nullptr;
};

void ov_session_free(ov_session * s) {
    if (!s) {
        return;
    }
    if (s->req)      s->api.ov_infer_request_free_fn(s->req);
    if (s->compiled) s->api.ov_compiled_model_free_fn(s->compiled);
    if (s->model)    s->api.ov_model_free_fn(s->model);
    if (s->core)     s->api.ov_core_free_fn(s->core);
    // the library stays loaded for the life of the process: OpenVINO's plugins and TBB threads must not be unloaded
    delete s;
}

void report(const ov_session * s, const char * what, int st) {
    const char * m = s->api.ov_get_last_err_msg_fn ? s->api.ov_get_last_err_msg_fn() : nullptr;
    std::fprintf(stderr, "mtmd-encoder-openvino: %s failed, status %d%s%s\n", what, st, m ? ": " : "", m ? m : "");
}

void * ov_init(const char * model_path, const char * device, const char * cache_dir) {
    if (model_path == nullptr || !*model_path) {
        std::fprintf(stderr, "mtmd-encoder-openvino: MTMD_ENCODER_PROVIDER_MODEL (an IR .xml) is required\n");
        return nullptr;
    }
    ov_session * s = new ov_session();
    if (!load_api(s->api)) {
        delete s;
        return nullptr;
    }
    int st = s->api.ov_core_create_fn(&s->core);
    if (st != 0) { report(s, "ov_core_create", st); ov_session_free(s); return nullptr; }
    st = s->api.ov_core_read_model_fn(s->core, model_path, nullptr, &s->model);
    if (st != 0) { report(s, "ov_core_read_model", st); ov_session_free(s); return nullptr; }
    const char * dev = (device && *device) ? device : "CPU";
    if (cache_dir && *cache_dir) {
        // property pair CACHE_DIR=<dir> (ov_property_key_cache_dir): compiled blobs persist between runs
        st = s->api.ov_core_compile_model_fn(s->core, s->model, dev, 2, &s->compiled, "CACHE_DIR", cache_dir);
    } else {
        st = s->api.ov_core_compile_model_fn(s->core, s->model, dev, 0, &s->compiled);
    }
    if (st != 0) { report(s, "ov_core_compile_model", st); ov_session_free(s); return nullptr; }
    st = s->api.ov_compiled_model_create_infer_request_fn(s->compiled, &s->req);
    if (st != 0) { report(s, "ov_compiled_model_create_infer_request", st); ov_session_free(s); return nullptr; }
    return s;
}

int ov_encode(void * handle, const float * mel, int32_t n_step, int32_t n_mel, float * out, int64_t n_out_floats) {
    ov_session * s = (ov_session *) handle;
    int64_t dims[3] = { 1, (int64_t) n_mel, (int64_t) n_step };
    ov_shape_t shape = {};
    int st = s->api.ov_shape_create_fn(3, dims, &shape);
    if (st != 0) { report(s, "ov_shape_create", st); return 1; }
    ov_tensor_t * in = nullptr;
    // the host buffer is read-only for us; OpenVINO does not write to an input tensor
    st = s->api.ov_tensor_create_from_host_ptr_fn(F32, shape, const_cast<float *>(mel), &in);
    s->api.ov_shape_free_fn(&shape);
    if (st != 0) { report(s, "ov_tensor_create_from_host_ptr", st); return 2; }
    int rc = 0;
    ov_tensor_t * res = nullptr;
    do {
        st = s->api.ov_infer_request_set_input_tensor_by_index_fn(s->req, 0, in);
        if (st != 0) { report(s, "set_input_tensor", st); rc = 3; break; }
        st = s->api.ov_infer_request_infer_fn(s->req);
        if (st != 0) { report(s, "infer", st); rc = 4; break; }
        st = s->api.ov_infer_request_get_output_tensor_by_index_fn(s->req, 0, &res);
        if (st != 0) { report(s, "get_output_tensor", st); rc = 5; break; }
        size_t n = 0;
        void * data = nullptr;
        st = s->api.ov_tensor_get_size_fn(res, &n);
        if (st != 0) { report(s, "ov_tensor_get_size", st); rc = 6; break; }
        if ((int64_t) n != n_out_floats) {
            std::fprintf(stderr, "mtmd-encoder-openvino: model output holds %zu floats, the projector expects %lld\n",
                         n, (long long) n_out_floats);
            rc = 7;
            break;
        }
        st = s->api.ov_tensor_data_fn(res, &data);
        if (st != 0 || data == nullptr) { report(s, "ov_tensor_data", st); rc = 8; break; }
        std::memcpy(out, data, (size_t) n_out_floats * sizeof(float));
    } while (false);
    if (res) s->api.ov_tensor_free_fn(res);
    s->api.ov_tensor_free_fn(in);
    return rc;
}

void ov_free(void * handle) {
    ov_session_free((ov_session *) handle);
}

const mtmd_encoder_provider_api g_api = { MTMD_ENCODER_PROVIDER_ABI, "openvino", ov_init, ov_encode, ov_free };

} // namespace

// declared first with the export attribute so -Wmissing-declarations has a prior declaration
extern "C"
#if defined(_WIN32)
__declspec(dllexport)
#else
__attribute__((visibility("default")))
#endif
const mtmd_encoder_provider_api * mtmd_encoder_provider_entry(void);

extern "C"
#if defined(_WIN32)
__declspec(dllexport)
#else
__attribute__((visibility("default")))
#endif
const mtmd_encoder_provider_api * mtmd_encoder_provider_entry(void) {
    return &g_api;
}

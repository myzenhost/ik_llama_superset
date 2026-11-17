// Pluggable external-runtime encoder provider for the mtmd audio projectors. See mtmd-encoder-provider.h.
//
// SOURCE-READ (2026-10-05): whisper.cpp @ 6e4ab854f67f743900934a703d5603419384c961 (MIT)
//   src/openvino/whisper-openvino-encoder.cpp (init/encode/free contract, ov::cache_dir, mel/out layout),
//   src/coreml/whisper-encoder.mm (copy-out), src/aneforge/whisper-aneforge.cpp (stub that returns nullptr off Apple).
// This file is an independent implementation of the common contract: no code copied.

#include "mtmd-encoder-provider.h"

#include <cmath>
#include <cstdlib>
#include <cstring>

#if defined(_WIN32)
#  define WIN32_LEAN_AND_MEAN
#  include <windows.h>
#else
#  include <dlfcn.h>
#endif

static std::string env_str(const char * name, const char * dflt = "") {
    const char * v = std::getenv(name);
    return (v && *v) ? std::string(v) : std::string(dflt);
}

mtmd_encoder_provider_diff mtmd_encoder_provider_compare(const float * a, const float * ref, size_t n) {
    mtmd_encoder_provider_diff d;
    double num = 0.0, den = 0.0, dot = 0.0, na = 0.0, nb = 0.0;
    for (size_t i = 0; i < n; i++) {
        const double x = a[i];
        const double y = ref[i];
        if (!std::isfinite(x)) {
            d.n_nonfinite++;
            continue;
        }
        const double e = x - y;
        num += e * e;
        den += y * y;
        dot += x * y;
        na  += x * x;
        nb  += y * y;
        const double ae = std::fabs(e);
        if (ae > d.max_abs) {
            d.max_abs = ae;
        }
    }
    // an all-zero reference makes the ratio meaningless: report the absolute energy of the difference instead of NaN
    d.nmse   = den > 0.0 ? num / den : num;
    d.cosine = (na > 0.0 && nb > 0.0) ? dot / std::sqrt(na * nb) : (na == nb ? 1.0 : 0.0);
    if (d.n_nonfinite > 0) {
        d.nmse = INFINITY;
    }
    return d;
}

mtmd_encoder_provider_session::~mtmd_encoder_provider_session() {
    if (api_ && handle_ && api_->free_handle) {
        api_->free_handle(handle_);
    }
    handle_ = nullptr;
    if (lib_) {
#if defined(_WIN32)
        FreeLibrary((HMODULE) lib_);
#else
        dlclose(lib_);
#endif
        lib_ = nullptr;
    }
}

const char * mtmd_encoder_provider_session::name() const {
    return (api_ && api_->name) ? api_->name : "?";
}

std::unique_ptr<mtmd_encoder_provider_session> mtmd_encoder_provider_session::open_api(
        const mtmd_encoder_provider_api * api, const std::string & model, const std::string & device,
        const std::string & cache, std::string & err) {
    if (api == nullptr) {
        err = "provider api table is null";
        return nullptr;
    }
    if (api->abi_version != MTMD_ENCODER_PROVIDER_ABI) {
        err = "provider abi_version " + std::to_string(api->abi_version) + " != " + std::to_string(MTMD_ENCODER_PROVIDER_ABI);
        return nullptr;
    }
    if (api->init == nullptr || api->encode == nullptr || api->free_handle == nullptr) {
        err = "provider api table has a null entry (init/encode/free_handle)";
        return nullptr;
    }
    void * h = api->init(model.c_str(), device.empty() ? "CPU" : device.c_str(), cache.empty() ? nullptr : cache.c_str());
    if (h == nullptr) {
        err = std::string("provider '") + (api->name ? api->name : "?") + "' opted out (init returned NULL) for model '" + model +
              "' on device '" + device + "'";
        return nullptr;
    }
    std::unique_ptr<mtmd_encoder_provider_session> s(new mtmd_encoder_provider_session());
    s->api_    = api;
    s->handle_ = h;
    return s;
}

std::unique_ptr<mtmd_encoder_provider_session> mtmd_encoder_provider_session::open_library(
        const std::string & lib_path, const std::string & model, const std::string & device,
        const std::string & cache, std::string & err) {
    void * lib = nullptr;
    void * sym = nullptr;
#if defined(_WIN32)
    lib = (void *) LoadLibraryA(lib_path.c_str());
    if (lib) {
        sym = (void *) GetProcAddress((HMODULE) lib, "mtmd_encoder_provider_entry");
    }
#else
    lib = dlopen(lib_path.c_str(), RTLD_NOW | RTLD_LOCAL);
    if (lib) {
        sym = dlsym(lib, "mtmd_encoder_provider_entry");
    }
#endif
    if (lib == nullptr) {
        err = "cannot load provider library '" + lib_path + "'";
        return nullptr;
    }
    auto close_lib = [&]() {
#if defined(_WIN32)
        FreeLibrary((HMODULE) lib);
#else
        dlclose(lib);
#endif
    };
    if (sym == nullptr) {
        err = "provider library '" + lib_path + "' does not export mtmd_encoder_provider_entry";
        close_lib();
        return nullptr;
    }
    const mtmd_encoder_provider_api * api = ((mtmd_encoder_provider_entry_fn) sym)();
    auto s = open_api(api, model, device, cache, err);
    if (!s) {
        close_lib();
        return nullptr;
    }
    s->lib_ = lib;
    return s;
}

std::unique_ptr<mtmd_encoder_provider_session> mtmd_encoder_provider_session::open_from_env(std::string & err) {
    const std::string lib = env_str("MTMD_ENCODER_PROVIDER");
    if (lib.empty()) {
        return nullptr; // feature off
    }
    const std::string model  = env_str("MTMD_ENCODER_PROVIDER_MODEL");
    const std::string device = env_str("MTMD_ENCODER_PROVIDER_DEVICE", "CPU");
    const std::string cache  = env_str("MTMD_ENCODER_PROVIDER_CACHE");
    auto s = open_library(lib, model, device, cache, err);
    if (!s) {
        return nullptr;
    }
    const std::string v = env_str("MTMD_ENCODER_PROVIDER_VERIFY");
    double tol = 1e-3;
    const std::string t = env_str("MTMD_ENCODER_PROVIDER_TOL");
    if (!t.empty()) {
        char * end = nullptr;
        const double x = std::strtod(t.c_str(), &end);
        if (end != t.c_str() && x > 0.0) {
            tol = x;
        }
    }
    s->set_verify(v == "1", tol);
    return s;
}

bool mtmd_encoder_provider_session::encode(const float * mel, int32_t n_step, int32_t n_mel, float * out,
                                           int64_t n_out_floats, std::string & err) {
    n_calls++;
    if (mel == nullptr || out == nullptr || n_step <= 0 || n_mel <= 0 || n_out_floats <= 0) {
        n_failed++;
        err = "bad arguments to provider encode";
        return false;
    }
    const int rc = api_->encode(handle_, mel, n_step, n_mel, out, n_out_floats);
    if (rc != 0) {
        n_failed++;
        err = std::string("provider '") + name() + "' encode returned " + std::to_string(rc);
        return false;
    }
    // a provider that leaves its output untouched or half written would feed garbage to the LLM: refuse non-finite
    for (int64_t i = 0; i < n_out_floats; i++) {
        if (!std::isfinite(out[i])) {
            n_failed++;
            err = std::string("provider '") + name() + "' produced a non-finite value at index " + std::to_string(i);
            return false;
        }
    }
    return true;
}

bool mtmd_encoder_provider_projector_ok(const char * proj_name) {
    if (proj_name == nullptr) {
        return false;
    }
    static const char * const ok[] = { "ultravox", "voxtral", "qwen2a", "glma", "meralion", "musicflamingo" };
    for (const char * n : ok) {
        if (std::strcmp(n, proj_name) == 0) {
            return true;
        }
    }
    return false;
}

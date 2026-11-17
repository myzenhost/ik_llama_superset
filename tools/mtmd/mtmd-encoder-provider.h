#pragma once

// Pluggable external-runtime ENCODER provider for the mtmd audio projectors.
//
// WHY: the whisper-style audio encoders (ULTRAVOX, VOXTRAL, QWEN2A, GLMA, MERALION, MUSIC_FLAMINGO) are run as a
// ggml graph (models/whisper-enc.cpp). Some runtimes (OpenVINO CPU/GPU/NPU, CoreML/ANE, VitisAI) run exactly this
// shape of network on devices outside the ggml backend list. BENCH-PENDING: no speed number exists for any provider
// yet; the reason for the hook is device reach, and every provider is verified for numerical equality before use.
// whisper.cpp ships three such shims
// (src/openvino/whisper-openvino-encoder.cpp, src/coreml/whisper-encoder.mm, src/aneforge/whisper-aneforge.cpp, all MIT)
// that share one three-call contract, init(model, device, cache) / encode(mel -> embd) / free. This header is that
// contract, with two additions a shim needs to be safe: a version word, and an opt-out path (init returns NULL and
// the caller keeps the ggml graph, the same stub discipline whisper-aneforge.cpp uses on non-Apple builds).
//
// A provider is a shared library exporting `mtmd_encoder_provider_entry()`. It is selected with environment
// variables (read once per audio clip_ctx):
//   MTMD_ENCODER_PROVIDER         path of the provider library (unset: feature off, zero overhead)
//   MTMD_ENCODER_PROVIDER_MODEL   what the provider loads (an OpenVINO IR .xml, a .mlmodelc dir, ...)
//   MTMD_ENCODER_PROVIDER_DEVICE  provider device string, default "CPU"
//   MTMD_ENCODER_PROVIDER_CACHE   compiled-blob cache directory (optional)
//   MTMD_ENCODER_PROVIDER_VERIFY  "1": run BOTH the provider and the ggml graph, return the ggml result, log the
//                                 error between them, and stop using the provider for this ctx when it exceeds
//                                 MTMD_ENCODER_PROVIDER_TOL (default 1e-3 normalised MSE)
//
// What the provider computes is whatever the model it was exported from computes: the contract is only that
// encode() returns the SAME tensor the ggml graph would have produced for that mel, i.e. [n_out_tokens][n_embd_out]
// row-major floats, n_out_tokens * n_embd_out == the n_out_floats argument. The caller sizes that from
// clip_n_output_tokens() * clip_n_mmproj_embd(), so a provider exported for the wrong projector fails the size
// check instead of silently feeding the wrong shape to the LLM.

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>

#define MTMD_ENCODER_PROVIDER_ABI 1

extern "C" {

struct mtmd_encoder_provider_api {
    uint32_t abi_version;   // must equal MTMD_ENCODER_PROVIDER_ABI
    const char * name;      // for logs

    // returns an opaque handle, or NULL to opt out (library missing at runtime, no such device, bad model):
    // the caller then keeps the ggml graph. Must not abort the process.
    void * (*init)(const char * model_path, const char * device, const char * cache_dir);

    // mel: row-major [n_mel][n_step] float32, the exact buffer clip_encode() feeds to the ggml input "inp_raw"
    // (ggml ne0 = n_step, ne1 = n_mel). out: row-major floats, n_out_floats of them.
    // returns 0 on success, anything else is a failure and the caller falls back to the ggml graph.
    int (*encode)(void * handle, const float * mel, int32_t n_step, int32_t n_mel, float * out, int64_t n_out_floats);

    void (*free_handle)(void * handle);
};

typedef const mtmd_encoder_provider_api * (*mtmd_encoder_provider_entry_fn)(void);

} // extern "C"

// The result of comparing a provider output with the reference (ggml graph) output.
struct mtmd_encoder_provider_diff {
    double nmse   = 0.0; // sum((a-b)^2) / sum(b^2)
    double cosine = 1.0; // a.b / (|a| |b|)
    double max_abs = 0.0;
    size_t n_nonfinite = 0; // non-finite values in `a` (the provider output)
};

mtmd_encoder_provider_diff mtmd_encoder_provider_compare(const float * a, const float * ref, size_t n);

// One opened provider (library + handle). Not copyable.
class mtmd_encoder_provider_session {
public:
    ~mtmd_encoder_provider_session();
    mtmd_encoder_provider_session(const mtmd_encoder_provider_session &) = delete;
    mtmd_encoder_provider_session & operator=(const mtmd_encoder_provider_session &) = delete;

    // Opens a session around an api table that is already in the process (tests, statically linked providers).
    // Returns null when the table is invalid or init() opts out; `err` says why.
    static std::unique_ptr<mtmd_encoder_provider_session> open_api(
            const mtmd_encoder_provider_api * api, const std::string & model, const std::string & device,
            const std::string & cache, std::string & err);

    // Loads `lib_path` (LoadLibrary / dlopen), looks up mtmd_encoder_provider_entry and opens the session.
    static std::unique_ptr<mtmd_encoder_provider_session> open_library(
            const std::string & lib_path, const std::string & model, const std::string & device,
            const std::string & cache, std::string & err);

    // Reads the MTMD_ENCODER_PROVIDER* environment. Returns null when the feature is off (no env) or the provider
    // opted out (`err` is filled when a provider was requested).
    static std::unique_ptr<mtmd_encoder_provider_session> open_from_env(std::string & err);

    // Runs the provider. Returns false (and leaves `out` unspecified) when the provider fails or when the output
    // holds a non-finite value; the caller must then use the ggml graph.
    bool encode(const float * mel, int32_t n_step, int32_t n_mel, float * out, int64_t n_out_floats, std::string & err);

    const char * name() const;
    bool   verify() const { return verify_; }
    double tol()    const { return tol_; }
    void   set_verify(bool v, double tol) { verify_ = v; tol_ = tol; }

    // counters, for logs and tests
    uint64_t n_calls = 0;
    uint64_t n_failed = 0;

private:
    mtmd_encoder_provider_session() = default;
    const mtmd_encoder_provider_api * api_ = nullptr;
    void * handle_ = nullptr;
    void * lib_ = nullptr;
    bool   verify_ = false;
    double tol_ = 1e-3;
};

// projector types whose graph is models/whisper-enc.cpp, by GGUF projector name; keeps the eligibility test free of
// clip-impl.h so tests can call it.
bool mtmd_encoder_provider_projector_ok(const char * proj_name);

// audio-capture.cpp -- see audio-capture.h
//
// The mtmd library already compiles miniaudio's implementation (tools/mtmd/mtmd-helper.cpp) with MA_NO_DEVICE_IO,
// and both end up in llama-audio. This translation unit compiles a second, device-capable copy with every miniaudio
// function given internal linkage (MA_API static), so the two copies cannot collide at link time.

#include "audio-capture.h"

#define MINIAUDIO_IMPLEMENTATION
#define MA_API static
// the one global miniaudio defines without MA_API (miniaudio.h:14576 `ma_atomic_spinlock ma_atomic_global_lock;`):
// renamed so this copy's lock does not collide with the mtmd copy's (measured: ld "multiple definition", build b14)
#define ma_atomic_global_lock audio_capture_ma_atomic_global_lock
#define MA_NO_DECODING
#define MA_NO_ENCODING
#define MA_NO_RESOURCE_MANAGER
#define MA_NO_NODE_GRAPH
#define MA_NO_ENGINE
#define MA_NO_GENERATION
#if defined(__GNUC__)
#    pragma GCC diagnostic push
#    pragma GCC diagnostic ignored "-Wunused-function"
#    pragma GCC diagnostic ignored "-Wunused-variable"
#    pragma GCC diagnostic ignored "-Wunused-parameter"
#    pragma GCC diagnostic ignored "-Wmissing-declarations"
#    pragma GCC diagnostic ignored "-Wmissing-prototypes"
#endif
#include "miniaudio/miniaudio.h"
#if defined(__GNUC__)
#    pragma GCC diagnostic pop
#endif

#include <cstring>

struct audio_capture {
    ma_context  ctx;
    ma_device   dev;
    bool        ctx_ok = false;
    bool        dev_ok = false;
    audio_ring * ring  = nullptr;
};

static void audio_capture_cb(ma_device * d, void * /*out*/, const void * in, ma_uint32 n_frames) {
    auto * c = (audio_capture *) d->pUserData;
    if (c && c->ring && in) {
        c->ring->push((const float *) in, (size_t) n_frames);   // mono float32 (configured below)
    }
}

static bool audio_capture_ctx(const std::string & backend, ma_context & ctx, std::string & err) {
    ma_result r;
    if (backend.empty() || backend == "default") {
        r = ma_context_init(nullptr, 0, nullptr, &ctx);
    } else if (backend == "null") {
        const ma_backend b[] = { ma_backend_null };
        r = ma_context_init(b, 1, nullptr, &ctx);
    } else {
        err = "unknown capture backend '" + backend + "' (want default|null)";
        return false;
    }
    if (r != MA_SUCCESS) {
        err = std::string("ma_context_init failed: ") + ma_result_description(r);
        return false;
    }
    return true;
}

bool audio_capture_list(const std::string & backend, std::vector<std::string> & names, std::string & err) {
    names.clear();
    ma_context ctx;
    if (!audio_capture_ctx(backend, ctx, err)) {
        return false;
    }
    ma_device_info * play = nullptr, * cap = nullptr;
    ma_uint32 n_play = 0, n_cap = 0;
    const ma_result r = ma_context_get_devices(&ctx, &play, &n_play, &cap, &n_cap);
    if (r != MA_SUCCESS) {
        err = std::string("ma_context_get_devices failed: ") + ma_result_description(r);
        ma_context_uninit(&ctx);
        return false;
    }
    for (ma_uint32 i = 0; i < n_cap; ++i) {
        names.emplace_back(cap[i].name);
    }
    ma_context_uninit(&ctx);
    return true;
}

audio_capture * audio_capture_open(const std::string & backend, int device_index, int sample_rate, audio_ring & ring,
                                   std::string & err) {
    auto * c = new audio_capture();
    c->ring = &ring;
    if (!audio_capture_ctx(backend, c->ctx, err)) {
        delete c;
        return nullptr;
    }
    c->ctx_ok = true;

    ma_device_id id;
    bool have_id = false;
    if (device_index >= 0) {
        ma_device_info * play = nullptr, * cap = nullptr;
        ma_uint32 n_play = 0, n_cap = 0;
        if (ma_context_get_devices(&c->ctx, &play, &n_play, &cap, &n_cap) != MA_SUCCESS || (ma_uint32) device_index >= n_cap) {
            err = "capture device " + std::to_string(device_index) + " not found (" + std::to_string(n_cap) + " present)";
            audio_capture_close(c);
            return nullptr;
        }
        id = cap[device_index].id;
        have_id = true;
    }

    ma_device_config cfg = ma_device_config_init(ma_device_type_capture);
    cfg.capture.format   = ma_format_f32;
    cfg.capture.channels = 1;
    cfg.capture.pDeviceID = have_id ? &id : nullptr;
    cfg.sampleRate       = (ma_uint32) sample_rate;
    cfg.dataCallback     = audio_capture_cb;
    cfg.pUserData        = c;
    const ma_result r = ma_device_init(&c->ctx, &cfg, &c->dev);
    if (r != MA_SUCCESS) {
        err = std::string("ma_device_init (capture) failed: ") + ma_result_description(r);
        audio_capture_close(c);
        return nullptr;
    }
    c->dev_ok = true;
    return c;
}

bool audio_capture_start(audio_capture * c, std::string & err) {
    const ma_result r = ma_device_start(&c->dev);
    if (r != MA_SUCCESS) {
        err = std::string("ma_device_start failed: ") + ma_result_description(r);
        return false;
    }
    return true;
}

void audio_capture_stop(audio_capture * c) {
    if (c && c->dev_ok) {
        ma_device_stop(&c->dev);
    }
}

void audio_capture_close(audio_capture * c) {
    if (!c) {
        return;
    }
    if (c->dev_ok) {
        ma_device_uninit(&c->dev);
    }
    if (c->ctx_ok) {
        ma_context_uninit(&c->ctx);
    }
    delete c;
}

std::string audio_capture_device_name(const audio_capture * c) {
    return c && c->dev_ok ? std::string(c->dev.capture.name) : std::string();
}

std::string audio_capture_backend_name(const audio_capture * c) {
    return c && c->ctx_ok ? std::string(ma_get_backend_name(c->ctx.backend)) : std::string();
}

// ggml-backend-dev-compat.cpp -- implementation of the backend-DEVICE shim.
//
// See ggml/include/ggml-backend-dev-compat.h for why this exists. In short: upstream ggml moved
// its backend registry from integer indices to opaque handles; our base predates that. Upstream's
// mtmd uses the handle spelling in 14 auxiliary places (device selection and memory reporting),
// so translating between the two spellings is enough to compile upstream's mtmd unmodified.
//
// A handle is an INDEX carried in a pointer, biased by +1 so NULL keeps meaning "unspecified".
// Nothing here dereferences a handle.

#include "ggml-backend-dev-compat.h"

#include <cstddef>
#include <cstdint>
#include <cstring>

// Bias so that a zero-initialised handle (NULL) is distinguishable from registry slot 0.
static inline size_t dev_to_index(ggml_backend_dev_t dev) {
    return (size_t) ((uintptr_t) dev) - 1;
}

static inline ggml_backend_dev_t index_to_dev(size_t i) {
    return (ggml_backend_dev_t) (uintptr_t) (i + 1);
}

size_t ggml_backend_dev_count(void) {
    return ggml_backend_reg_get_count();
}

ggml_backend_dev_t ggml_backend_dev_get(size_t i) {
    // Out of range yields NULL rather than a handle that would index past the registry. Upstream
    // callers already treat NULL as "unspecified", so this degrades to the default path instead of
    // producing an invalid lookup.
    if (i >= ggml_backend_reg_get_count()) {
        return NULL;
    }
    return index_to_dev(i);
}

const char * ggml_backend_dev_name(ggml_backend_dev_t dev) {
    // clip.cpp:193 logs this inside an error path where a null device is a legitimate state, so a
    // name is returned rather than crashing on the diagnostic itself.
    if (dev == NULL) {
        return "default";
    }
    const size_t i = dev_to_index(dev);
    if (i >= ggml_backend_reg_get_count()) {
        return "invalid";
    }
    return ggml_backend_reg_get_name(i);
}

ggml_backend_t ggml_backend_dev_init(ggml_backend_dev_t dev, const char * params) {
    if (dev == NULL) {
        // "Unspecified" -- upstream's clip_ctx_params.device defaults to nothing and expects the
        // implementation to choose. Our base exposes no explicit default-backend call, so slot 0
        // is used when the registry is non-empty. If it is empty there is nothing to initialise
        // and NULL is the honest answer, not a fabricated backend.
        if (ggml_backend_reg_get_count() == 0) {
            return NULL;
        }
        return ggml_backend_reg_init_backend(0, params);
    }
    const size_t i = dev_to_index(dev);
    if (i >= ggml_backend_reg_get_count()) {
        return NULL;
    }
    return ggml_backend_reg_init_backend(i, params);
}

ggml_backend_dev_t ggml_backend_get_device(ggml_backend_t backend) {
    // Recover the registry slot by NAME. ggml_backend_name() and ggml_backend_reg_get_name()
    // draw from the same registration, so the match is exact rather than heuristic.
    if (backend == NULL) {
        return NULL;
    }
    const char * want = ggml_backend_name(backend);
    if (want == NULL) {
        return NULL;
    }
    const size_t n = ggml_backend_reg_get_count();
    for (size_t i = 0; i < n; i++) {
        const char * have = ggml_backend_reg_get_name(i);
        if (have != NULL && strcmp(have, want) == 0) {
            return index_to_dev(i);
        }
    }
    return NULL;
}

ggml_backend_reg_t ggml_backend_dev_backend_reg(ggml_backend_dev_t dev) {
    // Our base exposes no registry OBJECT, only free functions over indices. Upstream callers
    // treat the result as an opaque token and hand it straight back to
    // ggml_backend_reg_get_proc_address, so the device handle itself serves as that token.
    return (ggml_backend_reg_t) dev;
}

enum ggml_backend_dev_type ggml_backend_dev_type(ggml_backend_dev_t dev) {
    // Same name rule as ggml_backend_init_by_type, read in the opposite direction, so the two
    // cannot disagree about which registry slot is the CPU.
    const char * name = ggml_backend_dev_name(dev);
    if (name == NULL || strcmp(name, "CPU") == 0 || strcmp(name, "default") == 0) {
        // "default" is what ggml_backend_dev_name returns for a NULL handle, and a NULL handle
        // resolves to registry slot 0 in ggml_backend_dev_init. Reporting CPU here would be a
        // guess if slot 0 were a GPU, so the name of slot 0 is what actually decides it.
        if (name != NULL && strcmp(name, "default") == 0 && ggml_backend_reg_get_count() > 0) {
            const char * slot0 = ggml_backend_reg_get_name(0);
            if (slot0 != NULL && strcmp(slot0, "CPU") != 0) {
                return GGML_BACKEND_DEVICE_TYPE_GPU;
            }
        }
        return GGML_BACKEND_DEVICE_TYPE_CPU;
    }
    return GGML_BACKEND_DEVICE_TYPE_GPU;
}

const char * ggml_backend_dev_description(ggml_backend_dev_t dev) {
    // Our base holds no description separate from the registry name; the name IS the description
    // we have. Callers use it for logging only.
    return ggml_backend_dev_name(dev);
}

ggml_backend_t ggml_backend_init_by_type(enum ggml_backend_dev_type type, const char * params) {
    const size_t n = ggml_backend_reg_get_count();

    switch (type) {
        case GGML_BACKEND_DEVICE_TYPE_CPU:
            // ggml_backend_cpu_init() is the direct constructor and does not depend on the CPU
            // having been registered, which makes it the reliable answer here.
            return ggml_backend_cpu_init();

        case GGML_BACKEND_DEVICE_TYPE_GPU:
        case GGML_BACKEND_DEVICE_TYPE_ACCEL:
            for (size_t i = 0; i < n; i++) {
                const char * name = ggml_backend_reg_get_name(i);
                if (name != NULL && strcmp(name, "CPU") != 0) {
                    return ggml_backend_reg_init_backend(i, params);
                }
            }
            return NULL;

        case GGML_BACKEND_DEVICE_TYPE_IGPU:
            // Our registry does not distinguish integrated from discrete GPUs, and guessing
            // would hand back the discrete card under an integrated label. clip.cpp asks for
            // IGPU only as a fallback AFTER GPU has already failed, and treats NULL as "none
            // available", so declining is both honest and the behaviour that path expects.
            return NULL;
    }

    return NULL;
}

void ggml_backend_load_all(void) {
    // No-op by construction: our backends are compiled in and registered during static init,
    // so by the time anything could call this they are already present.
}

void * ggml_backend_reg_get_proc_address(ggml_backend_reg_t reg, const char * name) {
    if (reg == NULL || name == NULL) {
        return NULL;
    }
    if (strcmp(name, "ggml_backend_set_n_threads") == 0) {
        const size_t i = dev_to_index((ggml_backend_dev_t) reg);
        if (i >= ggml_backend_reg_get_count()) {
            return NULL;
        }
        const char * bname = ggml_backend_reg_get_name(i);
        // Only the CPU backend takes a thread count in our base.
        if (bname != NULL && strcmp(bname, "CPU") == 0) {
            return (void *) ggml_backend_cpu_set_n_threads;
        }
        return NULL;
    }
    return NULL;
}

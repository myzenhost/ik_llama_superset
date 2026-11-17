#pragma once

// ggml-backend-dev-compat.h -- backend-DEVICE API shim for the JusticianAI ik_llama fork.
//
// WHY THIS EXISTS
//
// Upstream ggml replaced its INDEX-based backend registry with an opaque HANDLE-based device API.
// Our base predates that change. Both express the same idea -- an enumerable set of backends -- in
// two spellings:
//
//     ours (index-based, ggml-backend.h)          upstream (handle-based)
//     ggml_backend_reg_get_count()                ggml_backend_dev_count()
//     ggml_backend_reg_get_name(size_t i)         ggml_backend_dev_name(dev)
//     ggml_backend_reg_init_backend(i, params)    ggml_backend_dev_init(dev, params)
//
// Upstream's mtmd sources use the handle spelling in 14 places across clip.h, clip.cpp, mtmd.h and
// mtmd.cpp -- MEASURED, and cross-checked by MSVC independently flagging clip.h(51) and mtmd.h(92),
// the same declarations. Every one of those uses is AUXILIARY: a device SELECTOR field on
// clip_ctx_params, and the memory-REPORTING functions clip_get_mem_usage() /
// mtmd_get_memory_usage(), which return std::map keyed by device. None of it participates in graph
// construction or tensor math, so a faithful translation is enough -- no inference behaviour rides
// on it.
//
// THE ENCODING, and why it is safe. A ggml_backend_dev_t here is not a real pointer: it carries a
// registry INDEX, stored as (index + 1) so that nullptr keeps its natural meaning of "unspecified,
// use the default". Nothing dereferences it -- upstream treats the handle as opaque and only ever
// passes it back to the API or uses it as a std::map key, and an integer-in-a-pointer works
// correctly for both. The +1 bias is what makes a zero-initialised clip_ctx_params.device mean
// "default" rather than "device 0".
//
// WHY A SEPARATE HEADER RATHER THAN EDITING UPSTREAM'S FILES. Upstream mtmd includes
// "ggml-backend.h" and expects these names from it. Patching the upstream sources instead would
// have to be redone on every future sync; this way a sync stays a file copy. Our ggml-backend.h
// gains one #include at the end and nothing else.
//
// WHEN TO DELETE THIS. If our ggml is ever uplifted to a base that has the real device API, remove
// the #include from ggml-backend.h and delete this file plus its .cpp. The real symbols then
// resolve natively and no call site changes.

#include "ggml-backend.h"

#ifdef __cplusplus
extern "C" {
#endif

// Opaque handles. Never dereferenced -- see the encoding note above.
typedef struct ggml_backend_device * ggml_backend_dev_t;
typedef struct ggml_backend_registry * ggml_backend_reg_t;

// Number of registered backends. Direct rename of our index-based call.
GGML_API size_t ggml_backend_dev_count(void);

// Handle for registry slot i, or NULL when i is out of range.
GGML_API ggml_backend_dev_t ggml_backend_dev_get(size_t i);

// Name of the backend behind a handle. Returns "default" for NULL rather than crashing: upstream
// logs this in an error path (clip.cpp:193) where a null device is a legitimate state.
GGML_API const char * ggml_backend_dev_name(ggml_backend_dev_t dev);

// Initialise the backend behind a handle. NULL means "the default backend", matching upstream's
// treatment of an unset clip_ctx_params.device.
GGML_API ggml_backend_t ggml_backend_dev_init(ggml_backend_dev_t dev, const char * params);

// Upstream returns the device a backend belongs to, and the registry a device belongs to.
//
// Our base keeps no backend -> index mapping, but the registry is enumerable and every entry is
// named, so the mapping is recoverable by matching ggml_backend_name() against
// ggml_backend_reg_get_name(). That matters for more than diagnostics: clip.cpp:5641-5647 uses
// this pair to reach ggml_backend_set_n_threads, and an earlier revision of this shim returned
// NULL here, which made that whole block dead and left CLIP running on the backend's default
// thread count with nothing reporting it.
GGML_API ggml_backend_dev_t ggml_backend_get_device(ggml_backend_t backend);
GGML_API ggml_backend_reg_t ggml_backend_dev_backend_reg(ggml_backend_dev_t dev);

// Device CLASSES. Our registry has no notion of device type, so these are resolved by name:
// the entry called "CPU" is the CPU, and the first entry that is not is treated as the GPU.
// IGPU has no counterpart in our base at all -- see ggml_backend_init_by_type.
enum ggml_backend_dev_type {
    GGML_BACKEND_DEVICE_TYPE_CPU,
    GGML_BACKEND_DEVICE_TYPE_GPU,
    GGML_BACKEND_DEVICE_TYPE_ACCEL,
    GGML_BACKEND_DEVICE_TYPE_IGPU,
};

// Class of the backend behind a handle, resolved by the same name rule as
// ggml_backend_init_by_type: the registry entry called "CPU" is the CPU, anything else is a GPU.
// A NULL handle means "unspecified", and the default our ggml_backend_dev_init picks for NULL is
// registry slot 0, so CPU is the honest answer there rather than a guess.
//
// NOTE ON THE NAME. This is deliberately spelled the same as `enum ggml_backend_dev_type`, exactly
// as upstream ggml-backend.h:184 spells it. In C the two live in different namespaces; in C++ the
// function hides the enum name, which is why every declaration in this header uses the ELABORATED
// form `enum ggml_backend_dev_type`. Dropping the `enum` keyword anywhere here stops compiling.
GGML_API enum ggml_backend_dev_type ggml_backend_dev_type(ggml_backend_dev_t dev);

// Human-readable description of a device. Our base records only a registry NAME per backend and
// no separate description string, so the name is returned. This is the whole truth we hold, not a
// placeholder: upstream uses this purely for logging and test banners.
GGML_API const char * ggml_backend_dev_description(ggml_backend_dev_t dev);

// Initialise the first backend of the given class, or NULL when there is none.
GGML_API ggml_backend_t ggml_backend_init_by_type(enum ggml_backend_dev_type type, const char * params);

// Upstream's dynamic-backend loader. Our backends are linked in statically and registered at
// startup, so there is nothing to load and this is a no-op -- not a stub standing in for missing
// behaviour, the behaviour simply happens earlier here.
GGML_API void ggml_backend_load_all(void);

typedef void (*ggml_backend_set_n_threads_t)(ggml_backend_t backend, int n_threads);

// Upstream's escape hatch for backend-specific entry points, looked up by name. We support the
// one name mtmd asks for, "ggml_backend_set_n_threads"; anything else returns NULL, which is
// exactly how upstream reports "this backend does not offer that".
GGML_API void * ggml_backend_reg_get_proc_address(ggml_backend_reg_t reg, const char * name);

#ifdef __cplusplus
}

// Upstream grew a sixth ggml_backend_sched_new parameter, `op_offload`: whether the scheduler
// may run an op on an accelerator when that op's weights live in host memory. Our scheduler
// predates the switch and always applies its own built-in policy, so there is no setting to
// forward the flag to -- it is accepted and ignored, which matches what our scheduler does
// either way. This is an ARITY overload, so the original 5-argument C function is untouched
// and every existing caller keeps binding to it.
static inline ggml_backend_sched_t ggml_backend_sched_new(
        ggml_backend_t * backends,
        ggml_backend_buffer_type_t * bufts,
        int n_backends,
        size_t graph_size,
        bool parallel,
        bool op_offload) {
    (void) op_offload;
    return ggml_backend_sched_new(backends, bufts, n_backends, graph_size, parallel);
}
#endif

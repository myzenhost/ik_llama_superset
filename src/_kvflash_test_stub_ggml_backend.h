#pragma once
// Minimal ggml-backend stub for the KVFlash host-side selftest: data ptr = "device" memory (memcpy).
#include "ggml.h"
#include <cstring>
static inline void ggml_backend_tensor_set(struct ggml_tensor * t, const void * data, size_t offset, size_t size) {
    std::memcpy((char *) t->data + offset, data, size);
}
static inline void ggml_backend_tensor_get(const struct ggml_tensor * t, void * data, size_t offset, size_t size) {
    std::memcpy(data, (const char *) t->data + offset, size);
}

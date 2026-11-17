// VRAM-spill placement policy (default OFF).
//
// Decides, per large device allocation, whether it stays in VRAM or goes to a spill tier that lives in system RAM.
// This header and ggml/src/ggml-vram-spill.cpp are plain CPU code: they pick a tier, they allocate nothing.
// The CUDA backend (ggml/src/ggml-cuda.cu, ggml_cuda_device_malloc) asks ggml_vram_spill_choose() and performs the
// allocation. Design and cost notes: docs/agent-reports/link-vram-spill-1014.md.
#pragma once

#include "ggml.h"

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

enum ggml_vram_spill_mode {
    GGML_VRAM_SPILL_OFF     = 0, // never spill (default; placement identical to a build without the feature)
    GGML_VRAM_SPILL_MANAGED = 1, // every allocation >= threshold goes to CUDA managed (unified) memory
    GGML_VRAM_SPILL_HOSTMAP = 2, // every allocation >= threshold goes to mapped pinned host memory (zero-copy)
    GGML_VRAM_SPILL_AUTO    = 3, // allocation >= threshold that does not fit in (free VRAM - reserve) goes to host-mapped
};

enum ggml_vram_spill_place {
    GGML_VRAM_SPILL_PLACE_DEVICE  = 0,
    GGML_VRAM_SPILL_PLACE_MANAGED = 1,
    GGML_VRAM_SPILL_PLACE_HOSTMAP = 2,
};

struct ggml_vram_spill_policy {
    enum ggml_vram_spill_mode mode;
    uint64_t threshold_bytes; // allocations smaller than this never spill
    uint64_t reserve_bytes;   // auto mode: VRAM kept free for compute buffers / the driver
};

struct ggml_vram_spill_stats {
    uint64_t n_device,  bytes_device;
    uint64_t n_managed, bytes_managed;
    uint64_t n_hostmap, bytes_hostmap;
};

// "off" | "managed" | "hostmap" | "auto". Returns false (and leaves *out untouched) for anything else.
GGML_API bool        ggml_vram_spill_parse_mode(const char * name, enum ggml_vram_spill_mode * out);
GGML_API const char* ggml_vram_spill_mode_name (enum ggml_vram_spill_mode mode);

// Process-wide policy. Default: { OFF, 256 MiB, 1 GiB }.
GGML_API void                          ggml_vram_spill_configure(const struct ggml_vram_spill_policy * policy);
GGML_API struct ggml_vram_spill_policy ggml_vram_spill_get_policy(void);

// Pure function. free_vram_bytes is only read in AUTO mode.
GGML_API enum ggml_vram_spill_place ggml_vram_spill_choose(const struct ggml_vram_spill_policy * policy,
                                                           uint64_t size_bytes, uint64_t free_vram_bytes);

// Counters of what the allocator actually did (called by the backend after each allocation).
GGML_API void                        ggml_vram_spill_record(enum ggml_vram_spill_place place, uint64_t size_bytes);
GGML_API struct ggml_vram_spill_stats ggml_vram_spill_get_stats(void);
GGML_API void                        ggml_vram_spill_stats_reset(void);

#ifdef __cplusplus
}
#endif

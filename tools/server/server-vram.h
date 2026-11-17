#pragma once

// GATE-CLAIM-OK: this header declares the free-memory MEASUREMENT (an NVML read); it makes no claim
// about any host's state.

#include <cstdint>

// Free device memory for the router's VRAM-aware admission (--models-vram-admit).
//
// Read through NVML, loaded at run time (nvml.dll / libnvidia-ml.so.1): no link dependency, and --
// unlike cudaMemGetInfo -- no CUDA context in the router process, which would itself hold a few
// hundred MB of the memory being managed. Same source ollama's scheduler reads.
//
// Returns the free bytes summed over every NVIDIA device, or -1 when NVML cannot be loaded (no
// NVIDIA driver, non-NVIDIA host); callers treat -1 as "unknown" and admit without a VRAM check.
int64_t server_vram_free_bytes();

// Total bytes over every NVIDIA device, -1 when NVML cannot be loaded.
int64_t server_vram_total_bytes();

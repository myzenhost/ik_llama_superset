#pragma once

#include <cstddef>

// Free device memory as NVML reports it (the device-global figure nvidia-smi shows), for the device with the given
// PCI bus id ("0000:03:00.0" form, as cudaDeviceGetPCIBusId writes it). Returns false when NVML is unavailable, the
// device is not found, or the query fails -- the caller then keeps its own figure.
//
// WHY: on Windows/WDDM cudaMemGetInfo reports what the WDDM video-memory manager offers the process, which includes
// oversubscription, so it can exceed the physically free memory (NVIDIA forum thread 320791: "the report is a
// function of what WDDM is telling the CUDA subsystem ... It can oversubscribe that memory"; the same thread measures
// nvmlDeviceGetMemoryInfo tracking nvidia-smi). Measured here 2026-09-27: cudaMemGetInfo 30682 MiB free vs
// nvidia-smi 26302 MiB free, the ~4.4 GB gap being another process's allocation. fit sized layers against the
// larger figure and the overflow landed in the driver's system-memory fallback.
// Row S69-FIT-NGL-AUTO-OVERCOMMITS-VRAM-ON-HYBRID-27B-F16-2026-09-27-01.
bool ggml_cuda_nvml_free_for_pci(const char * pci_bus_id, size_t * free_out);

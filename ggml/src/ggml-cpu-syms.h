#pragma once
// ggml-base <-> CPU-backend function binding for GGML_BACKEND_DL builds.
//
// WHY. This fork's ggml-base (ggml.c type_traits[] and ggml_quantize_chunk, ggml-quants.c, iqk_quantize_base.cpp)
// names ~160 functions that are DEFINED in the CPU backend (iqk/iqk_quantize.cpp, iqk/iqk_mul_mat.cpp,
// ggml-cpu/arch/*/repack.cpp). A static build links both halves into one image, so that is fine. With
// GGML_BACKEND_DL the CPU backend is a separate, per-ISA DLL (ggml-cpu-haswell, -zen4, ...) loaded at run time,
// and ggml-base.dll cannot import from a DLL that does not exist at its own link time: the link failed with 173
// undefined references (row S69-FORK-BACKEND-DL-CPU-ALL-VARIANTS-BASE-CPU-SPLIT-2026-10-20-01).
//
// HOW. In a GGML_BACKEND_DL build only, ggml-base defines a forwarding stub with the exact name and signature of
// every function listed in ggml-cpu-syms.inc. The loaded CPU variant hands ggml-base the addresses of its own
// definitions from ggml_backend_cpu_reg() -- one CPU variant per process, so one binding is sound. The table
// itself, ggml_quantize_chunk and every caller stay byte-for-byte as they are, and a static build compiles none of
// this (the .cpp body and every call below are under #ifdef GGML_BACKEND_DL).
#include "ggml.h"
#include "ggml-backend.h"

#ifdef __cplusplus
extern "C" {
#endif

// CPU-backend side. names[i] / fns[i] for i < n, in ggml-cpu-syms.inc order. Returns the number bound, or a
// negative value when the lists disagree with ggml-base's (a CPU DLL from another build), in which case
// nothing is bound.
GGML_API int ggml_internal_cpu_syms_bind(const char * const * names, void * const * fns, int n);

// ggml (ggml-backend-reg.cpp) side, installed when ggml.dll loads.
//   load_cpu : loads the best CPU variant; called once, the first time a stub runs with nothing bound
//              (llama-quantize, for one, never loads a backend before ggml_quantize_chunk)
//   cpu_dev  : returns the CPU device; ggml-backend.cpp's CPU buffer types ask for it, and ggml-base cannot
//              link ggml_backend_dev_by_type itself (it lives in ggml.dll, which links ggml-base)
typedef void               (*ggml_cpu_syms_load_fn)(void);
typedef ggml_backend_dev_t (*ggml_cpu_syms_dev_fn)(void);
GGML_API void ggml_internal_cpu_syms_set_hooks(ggml_cpu_syms_load_fn load_cpu, ggml_cpu_syms_dev_fn cpu_dev);

// ggml-base side: the CPU device through the hook above, NULL while no hook is installed.
ggml_backend_dev_t ggml_internal_cpu_syms_cpu_dev(void);

#ifdef __cplusplus
}
#endif

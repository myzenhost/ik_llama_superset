// CPU-backend half of the GGML_BACKEND_DL function binding; see ../ggml-cpu-syms.h.
// Compiled into every ggml-cpu variant only when GGML_BACKEND_DL is set (ggml-cpu/CMakeLists.txt). Hands
// ggml-base the addresses of this variant's own definitions of every function in ggml-cpu-syms.inc.
#include "ggml-cpu-syms.h"

#include "ggml-impl.h"
#include "ggml-quants.h"
#include "ggml-aarch64.h"
#include "iqk/iqk_quantize.h"
#include "iqk/iqk_mul_mat.h"
#include "iqk/iqk_cpu_ops.h"
#include "ggml-cpu.h"

#include "cpu-syms-bind.h"

#ifdef GGML_BASE_CPU_SYMS

static const char * const k_names[] = {
#define GGML_CPU_SYM(ret, name, params, args) #name,
#include "ggml-cpu-syms.inc"
#undef GGML_CPU_SYM
};

// a pointer to function converts to void * here (conditionally-supported, accepted by GCC, Clang and MSVC);
// ggml-base converts it back to the declared type before every call
static void * const k_fns[] = {
#define GGML_CPU_SYM(ret, name, params, args) reinterpret_cast<void *>(&name),
#include "ggml-cpu-syms.inc"
#undef GGML_CPU_SYM
};

static_assert(sizeof(k_names) / sizeof(k_names[0]) == sizeof(k_fns) / sizeof(k_fns[0]), "name/function lists diverge");

extern "C" int ggml_cpu_syms_bind_to_base(void) {
    const int n   = (int) (sizeof(k_fns) / sizeof(k_fns[0]));
    const int got = ggml_internal_cpu_syms_bind(k_names, k_fns, n);
    if (got != n) {
        GGML_ABORT("ggml-cpu: binding %d CPU functions to ggml-base failed (%d): ggml-base and this CPU backend come from different builds", n, got);
    }
    return got;
}

#else

extern "C" int ggml_cpu_syms_bind_to_base(void) { return 0; }

#endif // GGML_BASE_CPU_SYMS

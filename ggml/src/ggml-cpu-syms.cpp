// ggml-base half of the GGML_BACKEND_DL function binding; see ggml-cpu-syms.h for why it exists.
// Compiled into ggml-base only when GGML_BACKEND_DL is set (ggml/src/CMakeLists.txt). Static builds do not
// compile this file at all, so they keep calling the CPU definitions directly, exactly as before.
#include "ggml-cpu-syms.h"

// the headers that declare every function in ggml-cpu-syms.inc -- the same ones ggml.c and ggml-quants.c use,
// so a stub below whose signature drifts from its declaration is a compile error ("conflicting declaration of
// C function"), not a silent mismatch
#include "ggml-impl.h"
#include "ggml-quants.h"
#include "ggml-aarch64.h"
#include "iqk/iqk_quantize.h"
#include "iqk/iqk_mul_mat.h"
#include "iqk/iqk_cpu_ops.h"
#include "ggml-cpu.h"

#include <atomic>
#include <cstring>
#include <mutex>

#ifdef GGML_BASE_CPU_SYMS

namespace {

enum ggml_cpu_sym_id {
#define GGML_CPU_SYM(ret, name, params, args) GGML_CPU_SYM_ID_##name,
#include "ggml-cpu-syms.inc"
#undef GGML_CPU_SYM
    GGML_CPU_SYM_COUNT
};

const char * const k_cpu_sym_names[GGML_CPU_SYM_COUNT] = {
#define GGML_CPU_SYM(ret, name, params, args) #name,
#include "ggml-cpu-syms.inc"
#undef GGML_CPU_SYM
};

std::atomic<void *>                g_cpu_fn[GGML_CPU_SYM_COUNT];
std::atomic<ggml_cpu_syms_load_fn> g_load_cpu{nullptr};
std::atomic<ggml_cpu_syms_dev_fn>  g_cpu_dev{nullptr};
std::once_flag                     g_load_once;

[[noreturn]] void ggml_cpu_sym_missing(int id) {
    GGML_ABORT("GGML_BACKEND_DL: %s is defined in the CPU backend, and no CPU backend is loaded "
               "(no ggml-cpu-*.dll found, or none this CPU can run; ggml_backend_load_all() loads one)",
               k_cpu_sym_names[id]);
}

void * ggml_cpu_sym_slow(int id) {
    // First call with nothing bound: ask ggml.dll to load the best CPU variant, once per process. The variant
    // binds from ggml_backend_cpu_reg() BEFORE anything else runs in it, so the load cannot re-enter here.
    ggml_cpu_syms_load_fn load = g_load_cpu.load(std::memory_order_acquire);
    if (load) {
        std::call_once(g_load_once, load);
    }
    void * p = g_cpu_fn[id].load(std::memory_order_acquire);
    if (!p) {
        ggml_cpu_sym_missing(id);
    }
    return p;
}

inline void * ggml_cpu_sym(int id) {
    void * p = g_cpu_fn[id].load(std::memory_order_acquire);
    return p ? p : ggml_cpu_sym_slow(id);
}

} // namespace

// One forwarding stub per entry: same name and signature as the CPU definition, so ggml.c's type_traits[] and
// ggml_quantize_chunk link against these unchanged. The CPU variant DLL is linked before ggml-base's import
// library is consulted, so its own calls bind to its own definitions, never to these.
#define GGML_CPU_SYM(ret, name, params, args)                                                    \
    extern "C" ret name params {                                                                 \
        return reinterpret_cast<decltype(&name)>(ggml_cpu_sym(GGML_CPU_SYM_ID_##name)) args;     \
    }
#include "ggml-cpu-syms.inc"
#undef GGML_CPU_SYM

extern "C" int ggml_internal_cpu_syms_bind(const char * const * names, void * const * fns, int n) {
    if (n != GGML_CPU_SYM_COUNT) {
        GGML_LOG_ERROR("%s: CPU backend lists %d functions, ggml-base expects %d -- mismatched builds, nothing bound\n",
                       __func__, n, (int) GGML_CPU_SYM_COUNT);
        return -1;
    }
    for (int i = 0; i < n; ++i) {
        if (strcmp(names[i], k_cpu_sym_names[i]) != 0 || fns[i] == nullptr) {
            GGML_LOG_ERROR("%s: entry %d is %s, ggml-base expects %s -- mismatched builds, nothing bound\n",
                           __func__, i, names[i], k_cpu_sym_names[i]);
            return -2;
        }
    }
    for (int i = 0; i < n; ++i) {
        g_cpu_fn[i].store(fns[i], std::memory_order_release);
    }
    return n;
}

extern "C" void ggml_internal_cpu_syms_set_hooks(ggml_cpu_syms_load_fn load_cpu, ggml_cpu_syms_dev_fn cpu_dev) {
    g_load_cpu.store(load_cpu, std::memory_order_release);
    g_cpu_dev.store(cpu_dev, std::memory_order_release);
}

extern "C" ggml_backend_dev_t ggml_internal_cpu_syms_cpu_dev(void) {
    ggml_cpu_syms_dev_fn f = g_cpu_dev.load(std::memory_order_acquire);
    return f ? f() : nullptr;
}

#endif // GGML_BASE_CPU_SYMS

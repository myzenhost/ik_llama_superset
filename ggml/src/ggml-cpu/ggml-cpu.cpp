#include "ggml-backend.h"
#include "ggml-backend-impl.h"
#include "ggml-cpu.h"
#include "repack.h"
#include "traits.h"
#include "ggml-impl.h"
#include "amx/amx.h"
#ifdef GGML_BASE_CPU_SYMS
#include "cpu-syms-bind.h"
#endif

#include <cctype>
#include <cstdlib>
#include <string>
#include <vector>

#ifdef GGML_USE_CPU_HBM
#    include "hbm.h"
#endif

#ifdef GGML_USE_CPU_KLEIDIAI
#    include "kleidiai/kleidiai.h"
#endif

#ifdef GGML_USE_CPU_RISCV64_SPACEMIT
#    include "spacemit/ime.h"
#endif

#if defined(_WIN32)
#    define WIN32_LEAN_AND_MEAN
#    ifndef NOMINMAX
#        define NOMINMAX
#    endif
#    include <windows.h>
#else
#    include <unistd.h>
#endif

#if defined(__APPLE__)
#    include <sys/sysctl.h>
#    include <sys/types.h>
#endif

#if defined(__linux__)
#    include <algorithm>
#    include "ggml-host-memory.h"
#endif

// ggml-backend interface

std::vector<ggml_backend_buffer_type_t> & ggml_backend_cpu_get_extra_buffer_types() {
    static std::vector<ggml_backend_buffer_type_t> bufts = []() {
        std::vector<ggml_backend_buffer_type_t> bufts;

#if defined(__AMX_INT8__) && defined(__AVX512VNNI__)
        if (ggml_backend_amx_buffer_type()) {
            bufts.push_back(ggml_backend_amx_buffer_type());
        }
#endif

#ifdef GGML_USE_CPU_RISCV64_SPACEMIT
        if (ggml_backend_cpu_riscv64_spacemit_buffer_type()) {
            bufts.push_back(ggml_backend_cpu_riscv64_spacemit_buffer_type());
        }
#endif

#ifdef GGML_USE_CPU_KLEIDIAI
        if (ggml_backend_cpu_kleidiai_buffer_type()) {
            bufts.push_back(ggml_backend_cpu_kleidiai_buffer_type());
        }
#endif

#ifdef GGML_USE_CPU_REPACK
        if (ggml_backend_cpu_repack_buffer_type()) {
            bufts.push_back(ggml_backend_cpu_repack_buffer_type());
        }
#endif

        return bufts;
    }();

    return bufts;
}

static ggml_backend_buffer_type_t * ggml_backend_cpu_device_get_extra_buffers_type(ggml_backend_dev_t device) {
    static std::vector<ggml_backend_buffer_type_t> extra_bufts = [] {
        std::vector<ggml_backend_buffer_type_t> bufts = ggml_backend_cpu_get_extra_buffer_types();
        bufts.push_back(nullptr);
        return bufts;
    }();

    return extra_bufts.data();

    GGML_UNUSED(device);
}

static bool ggml_backend_cpu_is_extra_buffer_type(ggml_backend_buffer_type_t buft) {
    for (auto * extra : ggml_backend_cpu_get_extra_buffer_types()) {
        if (extra == buft) {
            return true;
        }
    }
    return false;
}

// CPU backend - backend (stream)

struct ggml_backend_cpu_context {
    int                 n_threads;
    ggml_threadpool_t   threadpool;

    uint8_t *           work_data;
    size_t              work_size;

    ggml_abort_callback abort_callback;
    void *              abort_callback_data;

    bool                use_ref;  // use reference implementation
    // ik_llama: MoE expert prefetch. The consumers live in ggml.c (18676, 19017) and the
    // implementation in ggml-moe-prefetch.cpp; this is the CPU backend's half of it.
    bool                moe_expert_prefetch;

    // ik afdbc5e13 (#2565): optional per-thread CPU affinity
    int32_t             cpu_affinity[GGML_MAX_CPU_AFFINITY];
    int32_t             n_cpu_affinity;

    // Backend-owned worker pool for builds WITHOUT OpenMP (MinGW GCC defaults to GGML_OPENMP=OFF: libgomp's mingw
    // barrier sleeps in the kernel per node). With no pool attached, ggml_graph_compute (ggml-cpu.c) builds a
    // disposable threadpool per call: n_threads-1 CreateThread + join for EVERY graph. OpenMP builds keep a persistent
    // team and never pay this; llama_context attaches its own pools (common.cpp), so only direct backend users hit it
    // (test-backend-ops, mtmd/clip encoders, ggml-only callers). Lazily created on the first multi-thread compute,
    // grown when n_threads rises, freed with the backend or when a caller attaches its own pool.
    // GGML_CPU_BACKEND_POOL=0 restores the per-call pool (A/B switch).
    ggml_threadpool_t   owned_threadpool;
};

static const char * ggml_backend_cpu_get_name(ggml_backend_t backend) {
    return "CPU";

    GGML_UNUSED(backend);
}

static void ggml_backend_cpu_free_owned_pool(struct ggml_backend_cpu_context * ctx) {
    if (ctx->owned_threadpool) {
        ggml_threadpool_free(ctx->owned_threadpool);
        ctx->owned_threadpool = NULL;
    }
}

// -> the pool graph_compute should use: the attached one, else (no OpenMP, > 1 thread) the backend-owned one, else NULL
static ggml_threadpool_t ggml_backend_cpu_compute_pool(struct ggml_backend_cpu_context * ctx) {
    if (ctx->threadpool) {
        return ctx->threadpool;
    }
#ifndef GGML_USE_OPENMP
    static const bool enabled = [] {
        const char * s = getenv("GGML_CPU_BACKEND_POOL");
        return !(s && s[0] == '0');
    }();
    if (enabled && ctx->n_threads > 1) {
        if (ctx->owned_threadpool && ggml_threadpool_get_n_threads(ctx->owned_threadpool) < ctx->n_threads) {
            ggml_backend_cpu_free_owned_pool(ctx);  // a pool may run FEWER threads than it has, never more
        }
        if (!ctx->owned_threadpool) {
            struct ggml_threadpool_params tpp = ggml_threadpool_params_default(ctx->n_threads);
            ctx->owned_threadpool = ggml_threadpool_new(&tpp);
        }
        return ctx->owned_threadpool;
    }
#endif
    return NULL;
}

static void ggml_backend_cpu_free(ggml_backend_t backend) {
    struct ggml_backend_cpu_context * cpu_ctx = (struct ggml_backend_cpu_context *)backend->context;
    ggml_backend_cpu_free_owned_pool(cpu_ctx);
    delete[] cpu_ctx->work_data;
    delete cpu_ctx;
    delete backend;
}

struct ggml_backend_plan_cpu {
    struct ggml_cplan cplan;
    struct ggml_cgraph cgraph;
};

static ggml_backend_graph_plan_t ggml_backend_cpu_graph_plan_create(ggml_backend_t backend, const struct ggml_cgraph * cgraph) {
    struct ggml_backend_cpu_context * cpu_ctx = (struct ggml_backend_cpu_context *)backend->context;

    struct ggml_backend_plan_cpu * cpu_plan = new ggml_backend_plan_cpu;

    cpu_plan->cplan = ggml_graph_plan(cgraph, cpu_ctx->n_threads, cpu_ctx->threadpool);
    cpu_plan->cgraph = *cgraph; // FIXME: deep copy

    if (cpu_plan->cplan.work_size > 0) {
        cpu_plan->cplan.work_data = new uint8_t[cpu_plan->cplan.work_size];
        if (cpu_plan->cplan.work_data == NULL) {
            delete cpu_plan;
            return NULL;
        }
    }

    cpu_plan->cplan.abort_callback      = cpu_ctx->abort_callback;
    cpu_plan->cplan.abort_callback_data = cpu_ctx->abort_callback_data;
    cpu_plan->cplan.use_ref             = cpu_ctx->use_ref;
    cpu_plan->cplan.moe_expert_prefetch = cpu_ctx->moe_expert_prefetch;
    cpu_plan->cplan.cpu_affinity        = cpu_ctx->n_cpu_affinity > 0 ? cpu_ctx->cpu_affinity : NULL;
    cpu_plan->cplan.n_cpu_affinity      = cpu_ctx->n_cpu_affinity;

    return cpu_plan;
}

static void ggml_backend_cpu_graph_plan_free(ggml_backend_t backend, ggml_backend_graph_plan_t plan) {
    struct ggml_backend_plan_cpu * cpu_plan = (struct ggml_backend_plan_cpu *)plan;

    delete[] cpu_plan->cplan.work_data;
    delete cpu_plan;

    GGML_UNUSED(backend);
}

static enum ggml_status ggml_backend_cpu_graph_plan_compute(ggml_backend_t backend, ggml_backend_graph_plan_t plan) {
    struct ggml_backend_plan_cpu * cpu_plan = (struct ggml_backend_plan_cpu *)plan;

    return ggml_graph_compute(&cpu_plan->cgraph, &cpu_plan->cplan);

    GGML_UNUSED(backend);
}

static enum ggml_status ggml_backend_cpu_graph_compute(ggml_backend_t backend, struct ggml_cgraph * cgraph) {
    struct ggml_backend_cpu_context * cpu_ctx = (struct ggml_backend_cpu_context *)backend->context;

    // (graph_plan_create keeps cpu_ctx->threadpool: a stored plan must not hold the owned pool, which can be replaced)
    struct ggml_cplan cplan = ggml_graph_plan(cgraph, cpu_ctx->n_threads, ggml_backend_cpu_compute_pool(cpu_ctx));

    if (cpu_ctx->work_size < cplan.work_size) {
        delete[] cpu_ctx->work_data;
        cpu_ctx->work_data = new uint8_t[cplan.work_size];
        if (cpu_ctx->work_data == NULL) {
            cpu_ctx->work_size = 0;
            return GGML_STATUS_ALLOC_FAILED;
        }
        cpu_ctx->work_size = cplan.work_size;
    }
    cplan.work_data = (uint8_t *)cpu_ctx->work_data;

    cplan.abort_callback      = cpu_ctx->abort_callback;
    cplan.abort_callback_data = cpu_ctx->abort_callback_data;
    cplan.use_ref             = cpu_ctx->use_ref;
    cplan.moe_expert_prefetch = cpu_ctx->moe_expert_prefetch;
    cplan.cpu_affinity        = cpu_ctx->n_cpu_affinity > 0 ? cpu_ctx->cpu_affinity : NULL;
    cplan.n_cpu_affinity      = cpu_ctx->n_cpu_affinity;

    return ggml_graph_compute(cgraph, &cplan);
}

static const struct ggml_backend_i ggml_backend_cpu_i = {
    /* .get_name                = */ ggml_backend_cpu_get_name,
    /* .free                    = */ ggml_backend_cpu_free,
    /* .set_tensor_async        = */ NULL,
    /* .get_tensor_async        = */ NULL,
    /* .set_tensor_2d_async     = */ NULL,
    /* .get_tensor_2d_async     = */ NULL,
    /* .cpy_tensor_async        = */ NULL,
    /* .synchronize             = */ NULL,
    /* .graph_plan_create       = */ ggml_backend_cpu_graph_plan_create,
    /* .graph_plan_free         = */ ggml_backend_cpu_graph_plan_free,
    /* .graph_plan_update       = */ NULL,
    /* .graph_plan_compute      = */ ggml_backend_cpu_graph_plan_compute,
    /* .graph_compute           = */ ggml_backend_cpu_graph_compute,
    /* .event_record            = */ NULL,
    /* .event_wait              = */ NULL,
    /* .graph_optimize          = */ NULL,
};

static ggml_guid_t ggml_backend_cpu_guid(void) {
    static ggml_guid guid = { 0xaa, 0x67, 0xc7, 0x43, 0x96, 0xe6, 0xa3, 0x8a, 0xe3, 0xaf, 0xea, 0x92, 0x36, 0xbc, 0xfc, 0x89 };
    return &guid;
}

ggml_backend_t ggml_backend_cpu_init(void) {
    // initialize CPU backend now to avoid slowing the first graph computation
    ggml_cpu_init();

    struct ggml_backend_cpu_context * ctx = new ggml_backend_cpu_context;
    if (ctx == NULL) {
        return NULL;
    }

    ctx->n_threads           = GGML_DEFAULT_N_THREADS;
    ctx->threadpool          = NULL;
    ctx->work_data           = NULL;
    ctx->work_size           = 0;
    ctx->abort_callback      = NULL;
    ctx->abort_callback_data = NULL;
    ctx->use_ref             = false;
    ctx->moe_expert_prefetch = false;
    ctx->n_cpu_affinity      = 0;
    ctx->owned_threadpool    = NULL;

    ggml_backend_t cpu_backend = new ggml_backend {
        /* .guid    = */ ggml_backend_cpu_guid(),
        /* .iface   = */ ggml_backend_cpu_i,
        /* .device  = */ ggml_backend_reg_dev_get(ggml_backend_cpu_reg(), 0),
        /* .context = */ ctx,
    };

    if (cpu_backend == NULL) {
        delete ctx;
        return NULL;
    }

    return cpu_backend;
}

bool ggml_backend_is_cpu(ggml_backend_t backend) {
    return backend != NULL && ggml_guid_matches(backend->guid, ggml_backend_cpu_guid());
}

void ggml_backend_cpu_set_n_threads(ggml_backend_t backend_cpu, int n_threads) {
    GGML_ASSERT(ggml_backend_is_cpu(backend_cpu));

    struct ggml_backend_cpu_context * ctx = (struct ggml_backend_cpu_context *)backend_cpu->context;
    ctx->n_threads = n_threads;
}

// ik_llama: declared in ggml-backend.h. It lived in the fork's ggml-backend.cpp, which
// upstream replaced; the CPU backend is here now, so the definition follows it.
void ggml_backend_cpu_set_moe_expert_prefetch(ggml_backend_t backend_cpu, bool enable) {
    GGML_ASSERT(ggml_backend_is_cpu(backend_cpu));

    struct ggml_backend_cpu_context * ctx = (struct ggml_backend_cpu_context *)backend_cpu->context;
    ctx->moe_expert_prefetch = enable;
}

// ik afdbc5e13 (#2565), defined with the CPU backend (ik defines it in ggml-backend.cpp).
void ggml_backend_cpu_set_cpu_affinity(ggml_backend_t backend_cpu, const int32_t * cpus, int n_cpus) {
    GGML_ASSERT(ggml_backend_is_cpu(backend_cpu));
    GGML_ASSERT(n_cpus >= 0 && n_cpus <= GGML_MAX_CPU_AFFINITY);
    GGML_ASSERT(n_cpus == 0 || cpus != NULL);

    struct ggml_backend_cpu_context * ctx = (struct ggml_backend_cpu_context *)backend_cpu->context;

    ctx->n_cpu_affinity = 0;
    if (n_cpus > 0) {
        memcpy(ctx->cpu_affinity, cpus, (size_t)n_cpus * sizeof(int32_t));
        ctx->n_cpu_affinity = n_cpus;
    }
}

void ggml_backend_cpu_set_threadpool(ggml_backend_t backend_cpu, ggml_threadpool_t threadpool) {
    GGML_ASSERT(ggml_backend_is_cpu(backend_cpu));

    struct ggml_backend_cpu_context * ctx = (struct ggml_backend_cpu_context *)backend_cpu->context;

    if (ctx->threadpool && ctx->threadpool != threadpool) {
        // already had a different threadpool, pause/suspend it before switching
        ggml_threadpool_pause(ctx->threadpool);
    }
    if (threadpool) {
        ggml_backend_cpu_free_owned_pool(ctx);  // the caller's pool replaces ours; no idle second set of workers
    }
    ctx->threadpool = threadpool;
}

void ggml_backend_cpu_set_abort_callback(ggml_backend_t backend_cpu, ggml_abort_callback abort_callback, void * abort_callback_data) {
    GGML_ASSERT(ggml_backend_is_cpu(backend_cpu));

    struct ggml_backend_cpu_context * ctx = (struct ggml_backend_cpu_context *)backend_cpu->context;
    ctx->abort_callback = abort_callback;
    ctx->abort_callback_data = abort_callback_data;
}

void ggml_backend_cpu_set_use_ref(ggml_backend_t backend_cpu, bool use_ref) {
    GGML_ASSERT(ggml_backend_is_cpu(backend_cpu));

    struct ggml_backend_cpu_context * ctx = (struct ggml_backend_cpu_context *)backend_cpu->context;
    ctx->use_ref = use_ref;
}

// CPU backend - device

struct ggml_backend_cpu_device_context {
    std::string description = "CPU";

    ggml_backend_cpu_device_context() {
#ifdef __APPLE__
        size_t len = 0;
        if (!sysctlbyname("machdep.cpu.brand_string", NULL, &len, NULL, 0)) {
            description.resize(len);
            sysctlbyname("machdep.cpu.brand_string", &description[0], &len, NULL, 0); // NOLINT
        }
#elif defined(__linux__)
        FILE * f = fopen("/proc/cpuinfo", "r");
        if (f) {
            char buf[1024];
            while (fgets(buf, sizeof(buf), f)) {
                if (strncmp(buf, "model name", 10) == 0) {
                    char * p = strchr(buf, ':');
                    if (p) {
                        p++;
                        while (std::isspace(*p)) {
                            p++;
                        }
                        while (std::isspace(p[strlen(p) - 1])) {
                            p[strlen(p) - 1] = '\0';
                        }
                        description = p;
                        break;
                    }
                }
            }
            fclose(f);
        }
#elif defined(_WIN32)
        HKEY hKey;
        if (RegOpenKeyEx(HKEY_LOCAL_MACHINE,
                        TEXT("HARDWARE\\DESCRIPTION\\System\\CentralProcessor\\0"),
                        0,
                        KEY_READ,
                        &hKey) == ERROR_SUCCESS) {
            DWORD cpu_brand_size = 0;
            if (RegQueryValueExA(hKey,
                                "ProcessorNameString",
                                NULL,
                                NULL,
                                NULL,
                                &cpu_brand_size) == ERROR_SUCCESS) {
                description.resize(cpu_brand_size);
                if (RegQueryValueExA(hKey,
                                    "ProcessorNameString",
                                    NULL,
                                    NULL,
                                    (LPBYTE)&description[0], // NOLINT
                                    &cpu_brand_size) == ERROR_SUCCESS) {
                    if (description.find('\0') != std::string::npos) {
                        description.resize(description.find('\0'));
                    }
                }
            }
            RegCloseKey(hKey);
        }
#endif
    }
};

static const char * ggml_backend_cpu_device_get_name(ggml_backend_dev_t dev) {
    return "CPU";

    GGML_UNUSED(dev);
}

static const char * ggml_backend_cpu_device_get_description(ggml_backend_dev_t dev) {
    struct ggml_backend_cpu_device_context * ctx = (struct ggml_backend_cpu_device_context *)dev->context;

    return ctx->description.c_str();
}

static void ggml_backend_cpu_device_get_memory(ggml_backend_dev_t dev, size_t * free, size_t * total) {
#ifdef _WIN32
    MEMORYSTATUSEX status;
    status.dwLength = sizeof(status);
    GlobalMemoryStatusEx(&status);
    *total = status.ullTotalPhys;
    *free = status.ullAvailPhys;
#elif defined(__linux__)
    // Linux: MemTotal / MemAvailable, each tightened by every cgroup v2 memory limit from this process's cgroup up to the
    // root (ggml-host-memory.h). The old branch below reported the whole host as both total and free, so inside a
    // container with a memory limit (docker --memory, a Kubernetes pod limit, a systemd slice) the CPU device claimed RAM
    // the process can never get, and --fit / model placement planned against it. Same behaviour as ollama
    // discover/cpu_linux.go (MIT, @a8aaf9fc), extended to nested limits and "max"; no code copied.
    // HARNESS-CONSIDERED: engine memory reporting only; no harness capability involved.
    const long pages     = sysconf(_SC_PHYS_PAGES);
    const long page_size = sysconf(_SC_PAGE_SIZE);
    const size_t phys    = pages > 0 && page_size > 0 ? (size_t) pages * (size_t) page_size : 0;
    const ggml_host_mem::host_memory m = ggml_host_mem::linux_host_memory(ggml_host_mem::read_file);
    size_t t = phys;
    if (m.total && (t == 0 || *m.total < t)) {
        t = (size_t) *m.total;
    }
    *total = t;
    *free  = m.free ? std::min((size_t) *m.free, t) : t;
#else
    long pages = sysconf(_SC_PHYS_PAGES);
    long page_size = sysconf(_SC_PAGE_SIZE);
    *total = pages * page_size;

    // "free" system memory is ill-defined, for practical purposes assume that all of it is free:
    *free = *total;
#endif // _WIN32

    GGML_UNUSED(dev);
}

static enum ggml_backend_dev_type ggml_backend_cpu_device_get_type(ggml_backend_dev_t dev) {
    return GGML_BACKEND_DEVICE_TYPE_CPU;

    GGML_UNUSED(dev);
}

static void ggml_backend_cpu_device_get_props(ggml_backend_dev_t dev, struct ggml_backend_dev_props * props) {
    props->name        = ggml_backend_cpu_device_get_name(dev);
    props->description = ggml_backend_cpu_device_get_description(dev);
    props->type        = ggml_backend_cpu_device_get_type(dev);
    ggml_backend_cpu_device_get_memory(dev, &props->memory_free, &props->memory_total);
    props->caps = {
        /* .async                 = */ false,
        /* .host_buffer           = */ false,
        /* .buffer_from_host_ptr  = */ true,
        /* .events                = */ false,
        /* .mmap_support          = */ true,
    };
}

static ggml_backend_t ggml_backend_cpu_device_init_backend(ggml_backend_dev_t dev, const char * params) {
    return ggml_backend_cpu_init();

    GGML_UNUSED(dev);
    GGML_UNUSED(params);
}

static ggml_backend_buffer_type_t ggml_backend_cpu_device_get_buffer_type(ggml_backend_dev_t dev) {
    return ggml_backend_cpu_buffer_type();

    GGML_UNUSED(dev);
}

static ggml_backend_buffer_t ggml_backend_cpu_device_buffer_from_host_ptr(ggml_backend_dev_t dev, void * ptr, size_t size, size_t max_tensor_size) {
    return ggml_backend_cpu_buffer_from_ptr(ptr, size);

    GGML_UNUSED(dev);
    GGML_UNUSED(max_tensor_size);
}

// Types this backend has NO compute row for: ggml-cpu.c type_traits_cpu carries no entry, so there is no from_float / vec_dot to call.
// MEASURED 2026-09-30 (temp/_ikmerge_0930_traitsrun.cpp, CPU backend, one op per process): before this list, supports_op answered
// TRUE for MUL_MAT / MUL_MAT_ID / CPY / GET_ROWS on all five, and executing them died -- CPY aborted (exit 3), MUL_MAT segfaulted
// (0xC0000005) -- while the control Q4_0 ran. Row S69-IKMERGE-CPU-TRAITS-PARITY-LINT-MISSING-2026-09-29-01.
//   (TURBO2/3/4_0 were listed here until 2026-10-07: they now have a row -- from_float + an f32 vec_dot -- and GGML_OP_TURBO_WHT has a CPU kernel,
//   so a turbo KV cache can be attended on the CPU; see src/llama-turbo-graph.h.)
//   Q8_1_X4, Q8_KR8 : internal activation layouts (vec_dot_type of some weight types, used inside iqk_mul_mat); no graph in this tree
//                  builds a tensor of either type (grep of src/ and ggml/src/ finds only the traits table and iqk internals).
// This only makes the answer truthful; it removes no capability the CPU backend had.
// cortex/scripts/lint_cpu_traits_parity.py (pre-commit step cpu-traits-parity) reads THIS list: a type may lack a type_traits_cpu row only
// while it is named here, so removing a case below fails that lint for any type ggml.c gives a vec_dot/from_float.
// tests/test-cpu-supports-op-rows.cpp checks the answers themselves for every type (claimed op => the row it needs exists).
static bool ggml_backend_cpu_type_has_no_row(enum ggml_type type) {
    switch (type) {
        case GGML_TYPE_Q8_1_X4:
        case GGML_TYPE_Q8_KR8:
            return true;
        default:
            return false;
    }
}

static bool ggml_backend_cpu_device_supports_op(ggml_backend_dev_t dev, const struct ggml_tensor * op) {
    const struct ggml_tensor * src0 = op->src[0];
    const struct ggml_tensor * src1 = op->src[1];

    if (op->op == GGML_OP_NONE || op->op == GGML_OP_RESHAPE || op->op == GGML_OP_VIEW || op->op == GGML_OP_PERMUTE || op->op == GGML_OP_TRANSPOSE) {
        return true;
    }

    switch (op->op) {
        case GGML_OP_CPY:
        case GGML_OP_SET_ROWS:
        case GGML_OP_GET_ROWS:
        case GGML_OP_MUL_MAT:
        case GGML_OP_MUL_MAT_ID:
            if (ggml_backend_cpu_type_has_no_row(op->type) || (src0 && ggml_backend_cpu_type_has_no_row(src0->type))) {
                return false;
            }
            // A matmul weight needs a vec_dot: ggml_compute_forward_mul_mat calls it for every row pair the iqk path does not take.
            // MEASURED 2026-09-30 by executing MUL_MAT on the CPU backend, one process per type: all 22 types that answered TRUE here
            // while type_traits_cpu[].vec_dot was NULL died with a segfault or abort (the 5 above, plus q8_1, q8_K, q8_K16/32/64/128,
            // q8_0_x4, q8_2_x4 = activation layouts; i8, i16, i32, i64, f64 = plain integers/doubles; q4_0_4x4/4x8/8x8 = removed repack
            // layouts; i2_s), and the controls with a vec_dot (q4_0, iq4_ks_r16, q1_0_g128_r8) ran. So refusing loses no working case.
            if ((op->op == GGML_OP_MUL_MAT || op->op == GGML_OP_MUL_MAT_ID) && src0 &&
                ggml_get_type_traits_cpu(src0->type)->vec_dot == NULL) {
                return false;
            }
            // The copy/store mirror of the same rule: converting F32 into another type calls that type's from_float. Same measurement:
            // CPY F32 -> i8, i16, i64, f64, q4_0_4x4/4x8/8x8 and i2_s (from_float NULL) aborted (exit 3); q4_0, iq4_ks_r16, q1_0_g128_r8,
            // i32 and the q8_* activation layouts (from_float present) ran.
            if ((op->op == GGML_OP_CPY || op->op == GGML_OP_SET_ROWS) && src0 && src0->type == GGML_TYPE_F32 &&
                op->type != GGML_TYPE_F32 && ggml_get_type_traits_cpu(op->type)->from_float == NULL) {
                return false;
            }
            break;
        case GGML_OP_FLASH_ATTN_EXT:
            // K and V are src[1] and src[2]
            if ((op->src[1] && ggml_backend_cpu_type_has_no_row(op->src[1]->type)) ||
                (op->src[2] && ggml_backend_cpu_type_has_no_row(op->src[2]->type))) {
                return false;
            }
            break;
        default:
            break;
    }

    // check extra buffer types
    // note: only the first sources are checked for extra buffer types to reduce overhead, increase if necessary
    for (int i = 0; i < 4; i++) {
        if (op->src[i] && op->src[i]->buffer &&
            ggml_backend_cpu_is_extra_buffer_type(op->src[i]->buffer->buft)) {
            auto * buf_extra = (ggml::cpu::extra_buffer_type *) op->src[i]->buffer->buft->context;
            return buf_extra->supports_op(dev, op);
        }
    }

    switch (op->op) {
        case GGML_OP_CPY:
        case GGML_OP_SET_ROWS:
            return
                op->type != GGML_TYPE_IQ3_XXS &&
                op->type != GGML_TYPE_IQ3_S   &&
                op->type != GGML_TYPE_IQ2_XXS &&
                op->type != GGML_TYPE_IQ2_XS  &&
                op->type != GGML_TYPE_IQ2_S   &&
                op->type != GGML_TYPE_IQ1_S   &&
                op->type != GGML_TYPE_IQ1_XS   &&
                op->type != GGML_TYPE_IQ1_XXS  &&
                op->type != GGML_TYPE_IQ1_XXXS &&
                op->type != GGML_TYPE_IQ1_M; // missing type_traits.from_float
        case GGML_OP_MUL_MAT: {
            // upstream: the Hadamard-hint MUL_MAT is a fast Walsh-Hadamard transform (ggml_compute_forward_fwht, F32 / F16 src1)
                if (ggml_get_op_params_i32(op, 1) == GGML_HINT_SRC0_IS_HADAMARD &&
                    src0->type == GGML_TYPE_F32 && op->type == GGML_TYPE_F32) {
                    return src1->type == GGML_TYPE_F32 || src1->type == GGML_TYPE_F16;
                }
            const enum ggml_type vec_dot_type = ggml_get_type_traits_cpu(src0->type)->vec_dot_type;
            if (src1->type == GGML_TYPE_F32 || src1->type == vec_dot_type) {
                return true;
            }
#if GGML_USE_IQK_MULMAT
            // any other src1 with a to_float is converted by ggml_compute_forward_mul_mat (iqk_quantize_any),
            // as the pre-split ggml.c did -- the MLA prompt path multiplies a quantized wkv_b by an F16 cache view
            return ggml_get_type_traits(src1->type)->to_float != NULL &&
                   ggml_get_type_traits_cpu(vec_dot_type)->from_float != NULL &&
                   src1->nb[0] == ggml_type_size(src1->type);
#else
            return false;
#endif
        }
        case GGML_OP_SOFT_MAX_BACK: {
            if (op->src[0]->type != GGML_TYPE_F32 || op->src[1]->type != GGML_TYPE_F32) {
                return false;
            }
            float max_bias = 0.0f;

            memcpy(&max_bias, (const float *) op->op_params + 1, sizeof(float));

            return max_bias == 0.0f;
        }
        case GGML_OP_IM2COL_BACK:
            return src0->type == GGML_TYPE_F32 && (src1->type == GGML_TYPE_F32 || src1->type == GGML_TYPE_F16);
        case GGML_OP_GET_ROWS_BACK:
            return src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_F16;
        case GGML_OP_OUT_PROD:
            return (src0->type == GGML_TYPE_F32 ||
                    ((src0->type == GGML_TYPE_F16 || ggml_is_quantized(src0->type)) && src0->ne[2] == src1->ne[2] && src0->ne[3] == src1->ne[3])) &&
                src1->type == GGML_TYPE_F32 && op->type == GGML_TYPE_F32;
        case GGML_OP_CONV_2D:
            return ggml_is_contiguous(op->src[0]);
        case GGML_OP_SSM_SCAN:
            return ggml_get_op_params_i32(op, 0) == 1 || op->src[3]->ne[0] == 1;
        // NO CPU KERNEL YET -- report it, do not claim it. These ops were dispatched by the pre-split
        // ggml.c and have no case in this backend's compute switch (ggml-cpu.c), so claiming them
        // (the `default: return true` below) sent the scheduler here and aborted in
        // ggml_get_n_tasks ("op not implemented"). MEASURED 2026-09-26 by
        // temp/_scope_cpu_op_gap.py; row S69-THIRTY-OPS-HAVE-NO-CASE-IN-THE-SPLIT-CPU-BACKEND-2026-09-26-01.
        // Remove an op from this list IN THE SAME CHANGE that ports its kernel.
        // (MULTI_ADD, FUSED_UP_GATE, MOE_FUSED_UP_GATE, ROPE_CACHE, ROPE_FAST, ARGSORT_THRESH,
        //  GROUPED_TOPK, FUSED_NORM, SINKHORN, HC_PRE, HC_POST, LATENT_ATTN, DS4_COMP, INDEXER_TOPK
        //  and the five MAP_* ops were ported 2026-09-26 and removed from this list.)
        // What remains has NO CPU kernel to port -- the pre-split ggml.c aborted on FAKE_CPY
        // ("not implemented"): it is a GPU-only op, so declining it here is the correct answer.
        // (TURBO_WHT was in this list until 2026-10-07 and REDUCE until the row
        // S69-ENGREV-DLLAMA-CPU-REDUCE-OP-Q80-PARTIALS-2026-11-06-01; both have CPU kernels now.)
        case GGML_OP_FAKE_CPY:
            return false;
        case GGML_OP_REDUCE:
            // ADD all-reduce over F32/F16/BF16/Q8_0 contiguous partials (ops.cpp ggml_compute_forward_reduce)
            return (op->type == GGML_TYPE_F32 || op->type == GGML_TYPE_F16 || op->type == GGML_TYPE_BF16 ||
                    op->type == GGML_TYPE_Q8_0) && ggml_is_contiguous(op);
        case GGML_OP_FMA:
            // the CPU kernel is F32 only (ggml_compute_forward_fma_f32): rows contiguous along dim 0
            return op->type == GGML_TYPE_F32 && op->src[0]->type == GGML_TYPE_F32 && op->src[1]->type == GGML_TYPE_F32 &&
                   op->src[2]->type == GGML_TYPE_F32 && op->src[0]->nb[0] == sizeof(float) &&
                   op->src[1]->nb[0] == sizeof(float) && op->src[2]->nb[0] == sizeof(float) && op->nb[0] == sizeof(float);
        case GGML_OP_TURBO_WHT:
            // the CPU kernel: F32 contiguous, 64/128-wide groups along dim 0, no InnerQ scale tensor
            return op->src[0]->type == GGML_TYPE_F32 && ggml_is_contiguous(op->src[0]) && op->src[1] == nullptr &&
                   (ggml_get_op_params_i32(op, 1) == 64 || ggml_get_op_params_i32(op, 1) == 128) &&
                   op->src[0]->ne[0] % ggml_get_op_params_i32(op, 1) == 0;
        default:
            return true;
    }
}

static bool ggml_backend_cpu_device_supports_buft(ggml_backend_dev_t dev, ggml_backend_buffer_type_t buft) {
    return ggml_backend_buft_is_host(buft) || ggml_backend_cpu_is_extra_buffer_type(buft);
    GGML_UNUSED(dev);
}

static const struct ggml_backend_device_i ggml_backend_cpu_device_i = {
    /* .get_name             = */ ggml_backend_cpu_device_get_name,
    /* .get_description      = */ ggml_backend_cpu_device_get_description,
    /* .get_memory           = */ ggml_backend_cpu_device_get_memory,
    /* .get_type             = */ ggml_backend_cpu_device_get_type,
    /* .get_props            = */ ggml_backend_cpu_device_get_props,
    /* .init_backend         = */ ggml_backend_cpu_device_init_backend,
    /* .get_buffer_type      = */ ggml_backend_cpu_device_get_buffer_type,
    /* .get_host_buffer_type = */ NULL,
    /* .buffer_from_host_ptr = */ ggml_backend_cpu_device_buffer_from_host_ptr,
    /* .supports_op          = */ ggml_backend_cpu_device_supports_op,
    /* .supports_buft        = */ ggml_backend_cpu_device_supports_buft,
    /* .offload_op           = */ NULL,
    /* .event_new            = */ NULL,
    /* .event_free           = */ NULL,
    /* .event_synchronize    = */ NULL,
};

// CPU backend - backend (reg)

static const char * ggml_backend_cpu_reg_get_name(ggml_backend_reg_t reg) {
    return "CPU";

    GGML_UNUSED(reg);
}

static size_t ggml_backend_cpu_reg_get_device_count(ggml_backend_reg_t reg) {
    return 1;

    GGML_UNUSED(reg);
}

static ggml_backend_dev_t ggml_backend_cpu_reg_get_device(ggml_backend_reg_t reg, size_t index) {
    GGML_ASSERT(index == 0);

    static ggml_backend_cpu_device_context ctx;
    static ggml_backend_device ggml_backend_cpu_device = {
        /* .iface   = */ ggml_backend_cpu_device_i,
        /* .reg     = */ reg,
        /* .context = */ &ctx,
    };

    return &ggml_backend_cpu_device;
}

// This is intended to replace the the ggml_cpu_has_* functions when loading the CPU backend dynamically,
// and additionally to allow other backends to expose their own list of features that applications can query using the same API
static ggml_backend_feature * ggml_backend_cpu_get_features(ggml_backend_reg_t reg) {
    static std::vector<ggml_backend_feature> features = []() {
        ggml_cpu_init();

        std::vector<ggml_backend_feature> features;
        if (ggml_cpu_has_sse3()) {
            features.push_back({ "SSE3", "1" });
        }
        if (ggml_cpu_has_ssse3()) {
            features.push_back({ "SSSE3", "1" });
        }
        if (ggml_cpu_has_avx()) {
            features.push_back({ "AVX", "1" });
        }
        if (ggml_cpu_has_avx_vnni()) {
            features.push_back({ "AVX_VNNI", "1" });
        }
        if (ggml_cpu_has_avx2()) {
            features.push_back({ "AVX2", "1" });
        }
        if (ggml_cpu_has_f16c()) {
            features.push_back({ "F16C", "1" });
        }
        if (ggml_cpu_has_fma()) {
            features.push_back({ "FMA", "1" });
        }
        if (ggml_cpu_has_bmi2()) {
            features.push_back({ "BMI2", "1" });
        }
        if (ggml_cpu_has_avx512()) {
            features.push_back({ "AVX512", "1" });
        }
        if (ggml_cpu_has_avx512_vbmi()) {
            features.push_back({ "AVX512_VBMI", "1" });
        }
        if (ggml_cpu_has_avx512_vnni()) {
            features.push_back({ "AVX512_VNNI", "1" });
        }
        if (ggml_cpu_has_avx512_bf16()) {
            features.push_back({ "AVX512_BF16", "1" });
        }
        if (ggml_cpu_has_amx_int8()) {
            features.push_back({ "AMX_INT8", "1" });
        }
        if (ggml_cpu_has_neon()) {
            features.push_back({ "NEON", "1" });
        }
        if (ggml_cpu_has_arm_fma()) {
            features.push_back({ "ARM_FMA", "1" });
        }
        if (ggml_cpu_has_fp16_va()) {
            features.push_back({ "FP16_VA", "1" });
        }
        if (ggml_cpu_has_matmul_int8()) {
            features.push_back({ "MATMUL_INT8", "1" });
        }
        if (ggml_cpu_has_sve()) {
            features.push_back({ "SVE", "1" });
        }
        if (ggml_cpu_has_dotprod()) {
            features.push_back({ "DOTPROD", "1" });
        }
        if (ggml_cpu_get_sve_cnt() > 0) {
            static std::string sve_cnt = std::to_string(ggml_cpu_get_sve_cnt());
            features.push_back({ "SVE_CNT", sve_cnt.c_str() });
        }
        if (ggml_cpu_has_sme()) {
            features.push_back({ "SME", "1" });
        }
        if (ggml_cpu_has_sme2()) {
            features.push_back({ "SME2", "1" });
        }
        if (ggml_cpu_has_riscv_v()) {
            features.push_back({ "RISCV_V", "1" });
        }
        if (ggml_cpu_get_rvv_vlen() > 0) {
            static std::string rvv_vlen = std::to_string(ggml_cpu_get_rvv_vlen());
            features.push_back({ "RVV_VLEN", rvv_vlen.c_str() });
        }
        if (ggml_cpu_has_vsx()) {
            features.push_back({ "VSX", "1" });
        }
        if (ggml_cpu_has_vxe()) {
            features.push_back({ "VXE", "1" });
        }
        if (ggml_cpu_has_wasm_simd()) {
            features.push_back({ "WASM_SIMD", "1" });
        }
        if (ggml_cpu_has_llamafile()) {
            features.push_back({ "LLAMAFILE", "1" });
        }
    #ifdef GGML_USE_ACCELERATE
        features.push_back({ "ACCELERATE", "1" });
    #endif
    #ifdef GGML_USE_CPU_HBM
        features.push_back({ "CPU_HBM", "1" });
    #endif
    #ifdef GGML_USE_OPENMP
        features.push_back({ "OPENMP", "1" });
    #endif
    #ifdef GGML_USE_CPU_KLEIDIAI
        features.push_back({ "KLEIDIAI", "1" });
    #endif
    #ifdef GGML_USE_CPU_REPACK
        features.push_back({ "REPACK", "1" });
    #endif

        features.push_back({ nullptr, nullptr });

        return features;
    }();

    return features.data();

    GGML_UNUSED(reg);
}

static void * ggml_backend_cpu_get_proc_address(ggml_backend_reg_t reg, const char * name) {
    if (strcmp(name, "ggml_backend_set_n_threads") == 0) {
        ggml_backend_set_n_threads_t fct = ggml_backend_cpu_set_n_threads;
        return (void *)fct;
    }
    if (strcmp(name, "ggml_backend_dev_get_extra_bufts") == 0) {
        ggml_backend_dev_get_extra_bufts_t fct = ggml_backend_cpu_device_get_extra_buffers_type;
        return (void *)fct;
    }
    if (strcmp(name, "ggml_backend_get_features") == 0) {
        return (void *)ggml_backend_cpu_get_features;
    }
    if (strcmp(name, "ggml_backend_set_abort_callback") == 0) {
        return (void *)ggml_backend_cpu_set_abort_callback;
    }
    if (strcmp(name, "ggml_backend_cpu_numa_init") == 0) {
        return (void *)ggml_numa_init;
    }
    if (strcmp(name, "ggml_backend_cpu_is_numa") == 0) {
        return (void *)ggml_is_numa;
    }
    if (strcmp(name, "ggml_backend_cpu_set_use_ref") == 0) {
        return (void *)ggml_backend_cpu_set_use_ref;
    }

    // threadpool - TODO:  move to ggml-base
    if (strcmp(name, "ggml_threadpool_new") == 0) {
        return (void *)ggml_threadpool_new;
    }
    if (strcmp(name, "ggml_threadpool_free") == 0) {
        return (void *)ggml_threadpool_free;
    }
    if (strcmp(name, "ggml_backend_cpu_set_threadpool") == 0) {
        return (void *)ggml_backend_cpu_set_threadpool;
    }

    return NULL;

    GGML_UNUSED(reg);
}

static const struct ggml_backend_reg_i ggml_backend_cpu_reg_i = {
    /* .get_name         = */ ggml_backend_cpu_reg_get_name,
    /* .get_device_count = */ ggml_backend_cpu_reg_get_device_count,
    /* .get_device       = */ ggml_backend_cpu_reg_get_device,
    /* .get_proc_address = */ ggml_backend_cpu_get_proc_address,
};

ggml_backend_reg_t ggml_backend_cpu_reg(void) {
#ifdef GGML_BASE_CPU_SYMS
    // FIRST, before ggml_cpu_init or anything else here can reach ggml-base code that calls into this backend:
    // give ggml-base this variant's definitions of the functions it names but cannot link (cpu-syms-bind.h).
    // Only the variant the loader chose gets here; the others were loaded just to read their score.
    static const int n_bound = ggml_cpu_syms_bind_to_base();
    GGML_UNUSED(n_bound);
#endif
    // init CPU feature detection
    ggml_cpu_init();

    static struct ggml_backend_reg ggml_backend_cpu_reg = {
        /* .api_version = */ GGML_BACKEND_API_VERSION,
        /* .iface       = */ ggml_backend_cpu_reg_i,
        /* .context     = */ NULL,
    };

    return &ggml_backend_cpu_reg;
}

GGML_BACKEND_DL_IMPL(ggml_backend_cpu_reg)

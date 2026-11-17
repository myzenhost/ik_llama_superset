// Windows feature gates that a MinGW build compiled to nothing.
//
// mingw-w64 defaults _WIN32_WINNT to 0x0601 (x86_64-w64-mingw32/include/_mingw.h), so every
// `#if _WIN32_WINNT >= 0x0602` branch is dropped unless the build defines the version. MSVC's SDK
// defaults to the newest version, which is why the MSVC/CUDA builds never showed it. Two such
// branches sit on hot paths:
//
//  part A  src/llama-mmap.cpp, llama_mmap constructor: the model-load PrefetchVirtualMemory call.
//          Observable through llama_mmap_load_prefetch_os_calls().
//  part B  ggml/src/ggml-cpu/ggml-cpu.c, ggml_thread_apply_priority: every CPU worker opts out of
//          Windows 11 execution-speed power throttling (EcoQoS) with SetThreadInformation. Observable
//          from inside a custom op running on worker thread 1, with GetThreadInformation.
//
// ik_llama.cpp carries `if (MINGW) add_compile_definitions(_WIN32_WINNT=${GGML_WIN_VER})` in
// ggml/src/CMakeLists.txt (origin/main), and ik_llama #2563 (Thireus fork commit 6a1a8f57b) adds the
// same definition to the llama target; this fork had neither.
// Row: S69-DECL-OVERTURN-THIREUS-DEFER-PLE-WINVER-2026-10-17-01.
// SOURCE-READ: git show 6a1a8f57b -- src/CMakeLists.txt (ik_llama #2563, MIT);
//              git show origin/main:ggml/src/CMakeLists.txt:1272-1275 (ik_llama, MIT).

#ifdef _WIN32
#    ifndef _WIN32_WINNT
#        define _WIN32_WINNT 0x0A00   // this TU needs the 0x0602+ declarations to QUERY the state
#    endif
#    define WIN32_LEAN_AND_MEAN
#    include <windows.h>
#endif

#include "llama.h"
#include "ggml.h"
#include "ggml-cpu.h"

#include "../src/llama-mmap.h"

#include <atomic>
#include <cstdio>
#include <cstdint>
#include <vector>

static int g_fail = 0;
#define CHECK(cond, ...) do { if (!(cond)) { g_fail++; printf("FAIL  %s:%d  %s  --  ", __FILE__, __LINE__, #cond); printf(__VA_ARGS__); printf("\n"); } else { printf("ok    %s\n", #cond); } } while (0)

static void part_a() {
    printf("--- part A: model-load prefetch in the llama_mmap constructor\n");
    const char * path = "test-win-version-gates.tmp";
    {
        FILE * f = fopen(path, "wb");
        std::vector<char> z(1 << 20, 1);
        fwrite(z.data(), 1, z.size(), f);
        fclose(f);
    }
    const uint64_t before = llama_mmap_load_prefetch_os_calls();
    {
        llama_file file(path, "rb");
        llama_mmap map(&file, 0, false, false);   // CONTROL: prefetch == 0 must issue nothing
    }
    const uint64_t after_control = llama_mmap_load_prefetch_os_calls();
    {
        llama_file file(path, "rb");
        llama_mmap map(&file, (size_t) -1, false, false);   // what a normal mmap model load does
    }
    const uint64_t after = llama_mmap_load_prefetch_os_calls();
    remove(path);

    CHECK(after_control == before, "prefetch == 0 issued %llu load prefetch call(s)", (unsigned long long) (after_control - before));
    CHECK(after == after_control + 1, "a prefetching mmap issued %llu load prefetch call(s), expected 1 "
          "(0 on Windows = the _WIN32_WINNT >= 0x602 branch was compiled out)", (unsigned long long) (after - after_control));
}

#ifdef _WIN32
struct throttle_probe {
    std::atomic<int> worker_ran{0};
    BOOL  ok      = FALSE;
    DWORD err     = 0;
    ULONG control = 0;
    ULONG state   = 0;
};

static void probe_op(ggml_tensor * dst, const ggml_tensor * a, int ith, int nth, void * userdata) {
    (void) dst; (void) a; (void) nth;
    if (ith != 1) {
        return;   // thread 0 is the caller's thread; 1 is a pool worker that applied the pool priority
    }
    throttle_probe * p = (throttle_probe *) userdata;
    THREAD_POWER_THROTTLING_STATE t;
    ZeroMemory(&t, sizeof(t));
    t.Version = THREAD_POWER_THROTTLING_CURRENT_VERSION;
    p->ok      = GetThreadInformation(GetCurrentThread(), ThreadPowerThrottling, &t, sizeof(t));
    p->err     = p->ok ? 0 : GetLastError();
    p->control = t.ControlMask;
    p->state   = t.StateMask;
    p->worker_ran.store(1);
}

static void part_b() {
    printf("--- part B: CPU worker threads opt out of execution-speed power throttling\n");
    ggml_init_params ip = { 16u * 1024 * 1024, nullptr, false };
    ggml_context * ctx = ggml_init(ip);
    ggml_tensor * a = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 64);
    throttle_probe p;
    ggml_tensor * out = ggml_map_custom1(ctx, a, probe_op, 2, &p);
    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, out);
    const ggml_status st = ggml_graph_compute_with_ctx(ctx, gf, 2);
    ggml_free(ctx);

    CHECK(st == GGML_STATUS_SUCCESS, "graph compute status %d", (int) st);
    CHECK(p.worker_ran.load() == 1, "the probe op never ran on worker thread 1 (positive control)");
    CHECK(p.ok, "GetThreadInformation(ThreadPowerThrottling) failed, error %lu", (unsigned long) p.err);
    CHECK((p.control & THREAD_POWER_THROTTLING_EXECUTION_SPEED) != 0,
          "worker ControlMask 0x%lx has no EXECUTION_SPEED bit: ggml never called SetThreadInformation "
          "(the _WIN32_WINNT >= 0x0602 branch of ggml_thread_apply_priority was compiled out)", (unsigned long) p.control);
    CHECK((p.state & THREAD_POWER_THROTTLING_EXECUTION_SPEED) == 0,
          "worker StateMask 0x%lx still requests execution-speed throttling", (unsigned long) p.state);
}
#endif

int main() {
    part_a();
#ifdef _WIN32
    part_b();
#else
    printf("--- part B: skipped (Windows only)\n");
#endif
    printf("%s (%d failure(s))\n", g_fail ? "FAILED" : "PASSED", g_fail);
    return g_fail ? 1 : 0;
}

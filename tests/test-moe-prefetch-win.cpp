// Windows expert-warm arm of ggml_moe_prefetch_kernel_hook: at MoE matmul start thread 0 reads the routed expert ids and hands ONLY the selected
// expert ranges of a registered mmap weight to PrefetchVirtualMemory.
//
// Observable: PrefetchVirtualMemory reads the file into the cache but does NOT make the pages Valid in the process working set (measured:
// QueryWorkingSetEx stays 0), so residency cannot be the check. Each phase writes a fresh 64 MiB file UNBUFFERED (never in the file cache), maps it
// as 16 experts of 4 MiB, runs the hook with ids {3,4,9}, waits, then times a first touch of the selected experts and of 3 unselected experts.
//
//   (thresholds: 6 runs measured warm ratio 0.00-0.35, controls 0.74-8.4; cut at 0.5)
//   NOISE (measured 2026-11-11, affected run of work/nvfp4ks-1111 at -j6): one single-sample phase read
//   "bad ids: selected 11.3 ms, unselected 32.0 ms" -- the cold unselected side took a load spike and a negative control
//   failed with nothing warmed (selected 11.3 ms = the cold cost). So each expert is timed alone, selected and unselected
//   touches are INTERLEAVED (a spike hits both sides), the per-side MEDIAN of the 3 experts is compared, and every
//   phase runs 3 times with the check taking the MAJORITY verdict. A real defect (warming when it must not, or no warm
//   when it must) reproduces on every repetition; a one-off spike does not.
//   warm      registered mapping: touching the selected experts is clearly faster per page than touching cold unselected ones
//   negative  mapping NOT registered: no such speed-up (selected is as cold as the rest)
//   thread    a call with ith != 0 does nothing (same as the unregistered case)
//   sentinel  ids {-1, 16, 1000} are skipped, not read out of range (must not crash, no speed-up on any expert)
//
// Off Windows the hook is the Linux madvise/worker arm or a stub; this test says so and passes.
// Row S69-STRATA-S1-WINDOWS-EXPERT-WARM-ARM-2026-10-07-01. Idea read from Strata (MIT) expert_source.cpp; no code copied.
// GATE-CLAIM-OK: unit test, no resource claims.
#include "../src/llama-mmap.h"
#include "../ggml/src/ggml-moe-prefetch.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <utility>
#include <vector>

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <windows.h>

static const int    N_EXP  = 16;
static const size_t STRIDE = 4u << 20;

static void make_cold_file(char * name) {
    char dir[MAX_PATH];
    GetTempPathA(MAX_PATH, dir);
    GetTempFileNameA(dir, "mpw", 0, name);
    HANDLE w = CreateFileA(name, GENERIC_WRITE, 0, nullptr, CREATE_ALWAYS, FILE_FLAG_NO_BUFFERING | FILE_FLAG_WRITE_THROUGH, nullptr);
    if (w == INVALID_HANDLE_VALUE) { fprintf(stderr, "cannot create unbuffered %s (%lu)\n", name, (unsigned long) GetLastError()); exit(1); }
    char * buf = (char *) VirtualAlloc(nullptr, 1u << 20, MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE);
    for (size_t i = 0; i < (1u << 20); ++i) buf[i] = (char) (i * 7 + 1);
    for (size_t o = 0; o < STRIDE * N_EXP; o += (1u << 20)) {
        DWORD n;
        if (!WriteFile(w, buf, 1u << 20, &n, nullptr) || n != (1u << 20)) { fprintf(stderr, "write failed\n"); exit(1); }
    }
    VirtualFree(buf, 0, MEM_RELEASE);
    CloseHandle(w);
}

static double touch_ms(const uint8_t * b, size_t n, size_t page) {
    LARGE_INTEGER f, t0, t1;
    QueryPerformanceFrequency(&f);
    QueryPerformanceCounter(&t0);
    volatile uint8_t s = 0;
    for (size_t o = 0; o < n; o += page) s = (uint8_t) (s + b[o]);
    QueryPerformanceCounter(&t1);
    return (double) (t1.QuadPart - t0.QuadPart) * 1000.0 / (double) f.QuadPart;
}

struct phase_result {
    double   sel_ms, other_ms;   // DIAGNOSTIC ONLY (first-touch timing; flips under host load)
    uint64_t calls, calls_ok;    // VERDICT: PrefetchVirtualMemory calls the hook made, observed inside the hook
    std::vector<std::pair<size_t, size_t>> ranges; // VERDICT: [offset, len) handed to the OS by the last call
};

static double median3(double a, double b, double c) {
    if (a > b) { const double t = a; a = b; b = t; }
    if (b > c) { b = c; }
    return a > b ? a : b;
}

// one phase: fresh cold file, optional registration, run the hook with the given ids and thread, then time the selected vs unselected experts
static phase_result run_phase(bool registered, int ith, std::vector<int32_t> ids_in, size_t page) {
    char name[MAX_PATH];
    make_cold_file(name);
    phase_result res = { 0, 0, 0, 0, {} };
    {
        llama_file file(name, "rb");
        llama_mmap map(&file, 0, false, false);
        const uint8_t * base = (const uint8_t *) map.addr();

        ggml_init_params ip = { 1u << 20, nullptr, true };
        ggml_context * ctx = ggml_init(ip);
        ggml_tensor * w = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 1024, 1024, N_EXP); // one expert = 1024*1024*4 B = STRIDE
        w->data = const_cast<uint8_t *>(base);
        ggml_tensor * ids = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, (int64_t) ids_in.size(), 1);
        ids->data = ids_in.data();
        ggml_tensor * node = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1);
        node->op = GGML_OP_MUL_MAT_ID;
        node->src[0] = w;
        node->src[2] = ids;
        if (ggml_nbytes(w) != STRIDE * N_EXP || w->nb[2] != STRIDE) { fprintf(stderr, "fixture tensor has the wrong layout\n"); exit(1); }

        if (registered) ggml_moe_prefetch_register_mapping(base, STRIDE * N_EXP);
        ggml_moe_prefetch_win_reset_stats();
        ggml_moe_prefetch_kernel_hook(node, ith);
        res.calls = ggml_moe_prefetch_win_calls(&res.calls_ok);
        size_t offs[16], lens[16];
        const size_t nr = ggml_moe_prefetch_win_last_ranges(offs, lens, 16);
        for (size_t i = 0; i < nr && i < 16; ++i) res.ranges.emplace_back(offs[i], lens[i]);
        Sleep(1500);
        if (registered) ggml_moe_prefetch_unregister_mapping(base);

        // selected experts 3, 4, 9 vs unselected 12, 13, 14, first touch of each, interleaved; per-side median
        const int sel_e[3] = { 3, 4, 9 }, oth_e[3] = { 12, 13, 14 };
        double s[3], o[3];
        for (int k = 0; k < 3; ++k) {
            o[k] = touch_ms(base + oth_e[k] * STRIDE, STRIDE, page);
            s[k] = touch_ms(base + sel_e[k] * STRIDE, STRIDE, page);
        }
        res.sel_ms   = median3(s[0], s[1], s[2]);
        res.other_ms = median3(o[0], o[1], o[2]);
        ggml_free(ctx);
    }
    remove(name);
    return res;
}

int main() {
    SYSTEM_INFO si;
    GetSystemInfo(&si);
    const size_t page = si.dwPageSize;
    int fails = 0;

    // VERDICT = what the hook handed to PrefetchVirtualMemory, observed inside the hook (call count and the exact
    // [offset, len) ranges). It does not depend on host load. Row S69-MOE-PREFETCH-WIN-TIMING-FLAKY-UNDER-HOST-LOAD-2026-11-16-01:
    // the earlier verdict (first-touch timing ratio, majority of 3) flipped twice in the merge gate under load
    // ("a non-zero thread warmed experts (2 of 3 repetitions warm)", unselected medians 3.7 / 9.2 / 16.1 ms).
    // The timing is still measured and printed as a diagnostic, and never fails the test. QueryWorkingSetEx cannot replace
    // it: PrefetchVirtualMemory fills the file cache, not the process working set (header comment above).
    typedef std::vector<std::pair<size_t, size_t>> ranges_t;
    struct check { const char * label; bool registered; int ith; std::vector<int32_t> ids; uint64_t expect_calls; ranges_t expect_ranges; const char * fail_msg; };
    const check checks[] = {
        // experts 3,4 coalesce into one range, 9 is its own: 2 ranges, 12 MiB in a single call
        { "registered, ith 0   ", true,  0, { 3, 4, 9 },      1, { { 3 * STRIDE, 2 * STRIDE }, { 9 * STRIDE, STRIDE } }, "the selected expert ranges were not handed to PrefetchVirtualMemory" },
        { "not registered      ", false, 0, { 3, 4, 9 },      0, {},                                                     "an unregistered mapping was prefetched" },
        { "registered, ith 1   ", true,  1, { 3, 4, 9 },      0, {},                                                     "a non-zero thread prefetched experts" },
        { "registered, bad ids ", true,  0, { -1, 16, 1000 }, 0, {},                                                     "out-of-range ids were prefetched" },
    };
    const bool have_prefetch_api = GetProcAddress(GetModuleHandleW(L"kernel32.dll"), "PrefetchVirtualMemory") != nullptr;
    for (const check & c : checks) {
        const phase_result p = run_phase(c.registered, c.ith, c.ids, page);
        printf("%s calls %llu (ok %llu), ranges %zu; diagnostic touch: selected %.2f ms, unselected %.2f ms (ratio %.2f)\n", c.label,
               (unsigned long long) p.calls, (unsigned long long) p.calls_ok, p.ranges.size(), p.sel_ms, p.other_ms, p.sel_ms / p.other_ms);
        if (!have_prefetch_api) {
            printf("  PrefetchVirtualMemory missing on this Windows: the hint is skipped by design, nothing to verify\n");
            continue;
        }
        if (p.calls != c.expect_calls || p.ranges != c.expect_ranges || (c.expect_calls && p.calls_ok != p.calls)) {
            fprintf(stderr, "FAIL: %s (calls %llu want %llu, ok %llu, ranges %zu want %zu)\n", c.fail_msg, (unsigned long long) p.calls,
                    (unsigned long long) c.expect_calls, (unsigned long long) p.calls_ok, p.ranges.size(), c.expect_ranges.size());
            fails++;
        }
    }
    if (fails) return 1;
    printf("all moe-prefetch-win checks passed\n");
    return 0;
}
#else
int main() {
    printf("the Windows expert-warm arm is not built off Windows: nothing to test\n");
    return 0;
}
#endif

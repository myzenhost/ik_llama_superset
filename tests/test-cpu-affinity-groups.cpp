// Windows only: thread affinity across processor groups (more than 64 logical CPUs).
// COMPATIBILITY-INVESTIGATED: the old code used one 64-bit mask; the group API is the Windows route past 64 CPUs.
// GATE-CLAIM-OK: documentation; the one-group branch below is decided at run time from GetActiveProcessorGroupCount.
//
// (1) pure mapping: global CPU index -> (group, bit) against fake group tables.
// (2) real pinning: one thread per logical CPU calls ggml_thread_apply_affinity with exactly
//     bit i set, then reads where it runs. It must be (group, number) of global CPU i.
//     A host with one processor group prints SKIP for this part.
#if !defined(_WIN32)
int main() { return 0; }
#else
#include <windows.h>
#include <atomic>
#include <cstdint>
#include <cstdio>
#include <thread>
#include <vector>

#define MAX_THREADS_T 512

extern "C" bool ggml_win_cpu_to_group(const int32_t * sizes, int32_t n_groups, int32_t cpu, int32_t * group, int32_t * bit);
extern "C" bool ggml_thread_apply_affinity(bool * mask);

static int n_fail = 0;
#define CHECK(c) do { if (!(c)) { fprintf(stderr, "FAIL %s:%d  %s\n", __FILE__, __LINE__, #c); n_fail++; } } while (0)

static void test_mapping() {
    int32_t g = -1, b = -1;
    const int32_t t1[] = {64, 64};
    CHECK(ggml_win_cpu_to_group(t1, 2, 0,   &g, &b) && g == 0 && b == 0);
    CHECK(ggml_win_cpu_to_group(t1, 2, 63,  &g, &b) && g == 0 && b == 63);
    CHECK(ggml_win_cpu_to_group(t1, 2, 64,  &g, &b) && g == 1 && b == 0);
    CHECK(ggml_win_cpu_to_group(t1, 2, 127, &g, &b) && g == 1 && b == 63);
    CHECK(!ggml_win_cpu_to_group(t1, 2, 128, &g, &b));
    const int32_t t2[] = {64, 32, 32};
    CHECK(ggml_win_cpu_to_group(t2, 3, 95,  &g, &b) && g == 1 && b == 31);
    CHECK(ggml_win_cpu_to_group(t2, 3, 96,  &g, &b) && g == 2 && b == 0);
    CHECK(!ggml_win_cpu_to_group(t2, 3, 128, &g, &b));
    const int32_t t3[] = {1};
    CHECK(ggml_win_cpu_to_group(t3, 1, 0, &g, &b) && g == 0 && b == 0);
    CHECK(!ggml_win_cpu_to_group(t3, 1, 1, &g, &b));
    CHECK(!ggml_win_cpu_to_group(t3, 0, 0, &g, &b));
}

static void test_real_pinning() {
    const int n_groups = (int) GetActiveProcessorGroupCount();
    if (n_groups < 2) {
        printf("SKIP real pinning: %d processor group(s)\n", n_groups);
        return;
    }
    std::vector<int> gbase(n_groups);
    int total = 0;
    for (int g = 0; g < n_groups; g++) {
        gbase[g] = total;
        total += (int) GetActiveProcessorCount((WORD) g);
    }
    std::atomic<int> wrong(0), ok(0), call_failed(0);
    std::vector<std::thread> ts;
    for (int i = 0; i < total; i++) {
        ts.emplace_back([&, i]() {
            std::vector<char> m(MAX_THREADS_T, 0);
            bool * mask = reinterpret_cast<bool *>(m.data());
            mask[i] = true;
            if (!ggml_thread_apply_affinity(mask)) { call_failed++; return; }
            Sleep(30); // let the scheduler move the thread
            PROCESSOR_NUMBER pn;
            GetCurrentProcessorNumberEx(&pn);
            int want_g = 0;
            while (want_g + 1 < n_groups && i >= gbase[want_g + 1]) want_g++;
            const int want_n = i - gbase[want_g];
            if ((int) pn.Group == want_g && (int) pn.Number == want_n) ok++; else wrong++;
        });
    }
    for (auto & t : ts) t.join();
    printf("real pinning: %d CPUs, landed %d, wrong %d, call returned false %d\n", total, ok.load(), wrong.load(), call_failed.load());
    CHECK(call_failed.load() == 0);
    CHECK(wrong.load() == 0);
    CHECK(ok.load() == total);
}

int main() {
    test_mapping();
    test_real_pinning();
    if (n_fail) { printf("%d check(s) FAILED\n", n_fail); return 1; }
    printf("test-cpu-affinity-groups: all checks passed\n");
    return 0;
}
#endif

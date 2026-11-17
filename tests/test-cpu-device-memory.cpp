// test-cpu-device-memory: what the CPU backend device reports through the public API
// (ggml_backend_dev_memory on the CPU device = ggml_backend_cpu_device_get_memory in ggml-cpu.cpp).
//
// Always: total > 0 and free <= total.
// With GGML_TEST_MEM_LIMIT=<bytes> set (the memory limit of the container the test runs in, e.g. docker --memory 2g
// -> 2147483648): total must not exceed it. Before the Linux branch read cgroup v2 limits, the CPU device reported
// the whole host (sysconf _SC_PHYS_PAGES) as total AND free inside such a container, so this check failed there.
// The fake-file-system cases of the same logic are in test-load-budget (they run on every host); this test is the
// real-kernel check and is meant to be run inside a memory-limited Linux container.
#include "ggml-backend.h"
#include "ggml-cpu.h"

#include <cstdio>
#include <cstdlib>

int main() {
    ggml_backend_dev_t dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
    if (dev == nullptr) {
        std::printf("FAIL: no CPU backend device registered\n");
        return 1;
    }
    size_t free = 0, total = 0;
    ggml_backend_dev_memory(dev, &free, &total);
    std::printf("cpu device: total %zu bytes (%.1f MiB), free %zu bytes (%.1f MiB)\n", total, total / 1048576.0, free,
                free / 1048576.0);
    int fail = 0;
    if (total == 0) {
        std::printf("FAIL: total is 0\n");
        fail = 1;
    }
    if (free > total) {
        std::printf("FAIL: free > total\n");
        fail = 1;
    }
    if (const char * lim_s = std::getenv("GGML_TEST_MEM_LIMIT")) {
        const unsigned long long lim = std::strtoull(lim_s, nullptr, 10);
        std::printf("limit under test: %llu bytes (%.1f MiB)\n", lim, lim / 1048576.0);
        if (lim > 0 && total > lim) {
            std::printf("FAIL: total exceeds the container memory limit (the device reports RAM the process cannot get)\n");
            fail = 1;
        }
        if (lim > 0 && free > lim) {
            std::printf("FAIL: free exceeds the container memory limit\n");
            fail = 1;
        }
    }
    std::printf(fail ? "test-cpu-device-memory: FAILED\n" : "test-cpu-device-memory: OK\n");
    return fail;
}

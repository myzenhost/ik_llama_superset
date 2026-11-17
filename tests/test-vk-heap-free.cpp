// test-vk-heap-free.cpp -- ggml/src/ggml-vulkan/vk-heap-free.h (no Vulkan, no GPU).
// Row S69-ENGINE-REVIEW-1016-VULKAN-HEAP-UNDERFLOW-CLAMP-01.
//
// VK_EXT_memory_budget lets heapUsage exceed heapBudget. ggml_backend_vk_get_device_memory summed
// heapBudget - heapUsage as uint64, so an over-budget heap reported ~1.8e19 free bytes and --fit would put every
// layer on a full card. Fail-before: the old expression (kept below as old_free) wraps on the over-budget cases.
// Pass-after: ggml_vk_heap_free_bytes clamps at 0 and equals budget - usage whenever budget >= usage.

#include "../ggml/src/ggml-vulkan/vk-heap-free.h"

#include <cstdint>
#include <cstdio>

static int g_fail = 0;
#define CHECK(c) do { if (!(c)) { std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #c); ++g_fail; } } while (0)

static uint64_t old_free(uint64_t budget, uint64_t usage) { return budget - usage; }

int main() {
    const uint64_t GiB = 1ull << 30;
    const uint64_t total = 32 * GiB;  // one RTX 5090 device-local heap

    // control: the normal case is unchanged, and the old expression agrees with the new one there
    CHECK(ggml_vk_heap_free_bytes(30 * GiB, 4 * GiB) == 26 * GiB);
    CHECK(old_free(30 * GiB, 4 * GiB) == ggml_vk_heap_free_bytes(30 * GiB, 4 * GiB));
    CHECK(ggml_vk_heap_free_bytes(8 * GiB, 8 * GiB) == 0);
    CHECK(ggml_vk_heap_free_bytes(0, 0) == 0);

    // the defect: usage above budget (another process holds VRAM). The old value exceeds the heap size.
    const uint64_t budget = 31 * GiB, usage = 31 * GiB + 512ull * 1024 * 1024;
    CHECK(old_free(budget, usage) > total);              // proves the old code was wrong (would read 1.8e19)
    CHECK(ggml_vk_heap_free_bytes(budget, usage) == 0);  // the fix
    CHECK(ggml_vk_heap_free_bytes(1, UINT64_MAX) == 0);
    CHECK(ggml_vk_heap_free_bytes(UINT64_MAX, 1) == UINT64_MAX - 1);

    // summed over heaps the way get_device_memory does: free must never exceed total
    struct { uint64_t size, budget, usage; } heaps[] = { { total, 31 * GiB, 33 * GiB }, { 256ull << 20, 200ull << 20, 10ull << 20 } };
    uint64_t sum_total = 0, sum_free = 0;
    for (auto & h : heaps) { sum_total += h.size; sum_free += ggml_vk_heap_free_bytes(h.budget, h.usage); }
    CHECK(sum_free <= sum_total);
    CHECK(sum_free == (190ull << 20));

    if (g_fail) { std::fprintf(stderr, "test-vk-heap-free: %d FAILURES\n", g_fail); return 1; }
    std::printf("test-vk-heap-free: PASS\n");
    return 0;
}

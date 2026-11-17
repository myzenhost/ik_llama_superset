#pragma once

// Free bytes of one Vulkan memory heap from VK_EXT_memory_budget.
// The spec allows heapUsage to exceed heapBudget (the budget is advisory and other processes share the heap),
// and both are VkDeviceSize (uint64), so a plain budget - usage wraps to ~1.8e19 and reports a full card as
// almost infinitely free. Clamp at 0 instead.
// Spec read in llamafile a3ccf098 llama.cpp.patches/patches/ggml_src_ggml-vulkan_ggml-vulkan.cpp.patch:283-287
// (Apache-2.0); no code copied.

#include <cstdint>

static inline uint64_t ggml_vk_heap_free_bytes(uint64_t budget, uint64_t usage) {
    return budget > usage ? budget - usage : 0;
}

#pragma once
// S69-NCNN-PICK-L2-CACHE-DERIVED-FA-TILE-2026-10-05-01: optional L2-derived bound on the CPU flash-attention K tile (k_step).
// Default OFF: nothing here changes a result or a tile unless IQK_FA_KSTEP_L2=1 is set in the environment.
// Idea from Tencent/ncnn src/layer/x86/sdpa_decode.h (BSD-3-Clause) and ONNX Runtime gqa_attention_base.h ApplyAttentionFlash (MIT):
// kv_block = L2 / (4 buffers * bytes_per_element * 2 * head). Written from the formula quoted in the row, no code copied.
// Header-only and free of ggml includes so tests/test-fa-kstep-l2.cpp can include it directly.
#include <cstddef>
#include <cstdlib>
#include <vector>
#if defined(_WIN32)
#  ifndef WIN32_LEAN_AND_MEAN
#    define WIN32_LEAN_AND_MEAN
#  endif
#  ifndef NOMINMAX
#    define NOMINMAX
#  endif
#  include <windows.h>
#else
#  include <unistd.h>
#endif

// Largest compiled tile in {128, 64, 32} that is <= L2 / (4 * bytes_per_elem * 2 * head); 128 when l2_bytes is 0 (unknown) or inputs are
// degenerate (= today's choice); 32 is the floor because it is the smallest instantiated tile.
inline int iqk_fa_kstep_from_l2(size_t l2_bytes, int head, double bytes_per_elem) {
    if (l2_bytes == 0 || head <= 0 || bytes_per_elem <= 0.0) return 128;
    const double blk = (double)l2_bytes / (4.0 * bytes_per_elem * 2.0 * (double)head);
    if (blk >= 128.0) return 128;
    if (blk >= 64.0)  return 64;
    return 32;
}

// Per-core L2 size in bytes, 0 when it cannot be read. IQK_FA_L2_KIB overrides it (A/B runs on a host whose L2 is not the target's).
inline size_t iqk_fa_l2_bytes_detect() {
    if (const char * e = getenv("IQK_FA_L2_KIB")) {
        const long kib = atol(e);
        if (kib > 0) return (size_t)kib * 1024;
    }
#if defined(_WIN32)
    DWORD len = 0;
    GetLogicalProcessorInformationEx(RelationCache, nullptr, &len);
    if (len == 0) return 0;
    std::vector<char> buf(len);
    if (!GetLogicalProcessorInformationEx(RelationCache, (PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX)buf.data(), &len)) return 0;
    size_t best = 0;
    for (DWORD off = 0; off < len; ) {
        auto * p = (PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX)(buf.data() + off);
        if (p->Relationship == RelationCache && p->Cache.Level == 2 && (best == 0 || p->Cache.CacheSize < best)) best = p->Cache.CacheSize;
        if (p->Size == 0) break;
        off += p->Size;
    }
    return best;
#elif defined(_SC_LEVEL2_CACHE_SIZE)
    const long v = sysconf(_SC_LEVEL2_CACHE_SIZE);
    return v > 0 ? (size_t)v : 0;
#else
    return 0;
#endif
}

// 128 when the mode is off (default) or L2 is unknown; otherwise the tile bound for this head size, assuming an f16 K/V cache (2 bytes).
inline int iqk_fa_kstep_l2_bound(int head) {
    static const bool on = []() { const char * e = getenv("IQK_FA_KSTEP_L2"); return e && e[0] == '1'; }();
    if (!on) return 128;
    static const size_t l2 = iqk_fa_l2_bytes_detect();
    return iqk_fa_kstep_from_l2(l2, head, 2.0);
}

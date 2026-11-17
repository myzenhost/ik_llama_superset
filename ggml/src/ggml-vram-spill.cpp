// VRAM-spill placement policy. See ggml/include/ggml-vram-spill.h.
// Self-contained: no dependency on the rest of ggml (tests compile this file directly).
// Written from our own spec; the GPL GreenBoost shim was read for its behaviour only (no code copied).
#include "ggml-vram-spill.h"

#include <atomic>
#include <cstring>

namespace {

constexpr uint64_t k_default_threshold = 256ull * 1024 * 1024;
constexpr uint64_t k_default_reserve   =   1ull * 1024 * 1024 * 1024;

// The policy is three words; a mutex-free snapshot per field is enough because it is written once at startup
// (before any allocation) and only read afterwards.
std::atomic<int>      g_mode      {GGML_VRAM_SPILL_OFF};
std::atomic<uint64_t> g_threshold {k_default_threshold};
std::atomic<uint64_t> g_reserve   {k_default_reserve};

std::atomic<uint64_t> g_n[3];
std::atomic<uint64_t> g_bytes[3];

} // namespace

extern "C" {

bool ggml_vram_spill_parse_mode(const char * name, enum ggml_vram_spill_mode * out) {
    if (name == nullptr || out == nullptr) {
        return false;
    }
    static const struct { const char * n; enum ggml_vram_spill_mode m; } table[] = {
        {"off", GGML_VRAM_SPILL_OFF}, {"managed", GGML_VRAM_SPILL_MANAGED},
        {"hostmap", GGML_VRAM_SPILL_HOSTMAP}, {"auto", GGML_VRAM_SPILL_AUTO},
    };
    for (const auto & e : table) {
        if (std::strcmp(name, e.n) == 0) {
            *out = e.m;
            return true;
        }
    }
    return false;
}

const char * ggml_vram_spill_mode_name(enum ggml_vram_spill_mode mode) {
    switch (mode) {
        case GGML_VRAM_SPILL_OFF:     return "off";
        case GGML_VRAM_SPILL_MANAGED: return "managed";
        case GGML_VRAM_SPILL_HOSTMAP: return "hostmap";
        case GGML_VRAM_SPILL_AUTO:    return "auto";
    }
    return "off";
}

void ggml_vram_spill_configure(const struct ggml_vram_spill_policy * policy) {
    if (policy == nullptr) {
        return;
    }
    g_mode.store((int) policy->mode);
    g_threshold.store(policy->threshold_bytes);
    g_reserve.store(policy->reserve_bytes);
}

struct ggml_vram_spill_policy ggml_vram_spill_get_policy(void) {
    ggml_vram_spill_policy p;
    p.mode            = (enum ggml_vram_spill_mode) g_mode.load();
    p.threshold_bytes = g_threshold.load();
    p.reserve_bytes   = g_reserve.load();
    return p;
}

enum ggml_vram_spill_place ggml_vram_spill_choose(const struct ggml_vram_spill_policy * policy,
                                                  uint64_t size_bytes, uint64_t free_vram_bytes) {
    if (policy == nullptr || policy->mode == GGML_VRAM_SPILL_OFF || size_bytes < policy->threshold_bytes) {
        return GGML_VRAM_SPILL_PLACE_DEVICE;
    }
    switch (policy->mode) {
        case GGML_VRAM_SPILL_MANAGED:
            return GGML_VRAM_SPILL_PLACE_MANAGED;
        case GGML_VRAM_SPILL_HOSTMAP:
            return GGML_VRAM_SPILL_PLACE_HOSTMAP;
        case GGML_VRAM_SPILL_AUTO: {
            // free < reserve means nothing is usable; do not let the subtraction wrap.
            const uint64_t usable = free_vram_bytes > policy->reserve_bytes ? free_vram_bytes - policy->reserve_bytes : 0;
            return size_bytes > usable ? GGML_VRAM_SPILL_PLACE_HOSTMAP : GGML_VRAM_SPILL_PLACE_DEVICE;
        }
        default:
            break;
    }
    return GGML_VRAM_SPILL_PLACE_DEVICE;
}

void ggml_vram_spill_record(enum ggml_vram_spill_place place, uint64_t size_bytes) {
    const int i = (int) place;
    if (i < 0 || i > 2) {
        return;
    }
    g_n[i].fetch_add(1);
    g_bytes[i].fetch_add(size_bytes);
}

struct ggml_vram_spill_stats ggml_vram_spill_get_stats(void) {
    ggml_vram_spill_stats s;
    s.n_device  = g_n[0].load(); s.bytes_device  = g_bytes[0].load();
    s.n_managed = g_n[1].load(); s.bytes_managed = g_bytes[1].load();
    s.n_hostmap = g_n[2].load(); s.bytes_hostmap = g_bytes[2].load();
    return s;
}

void ggml_vram_spill_stats_reset(void) {
    for (int i = 0; i < 3; ++i) {
        g_n[i].store(0);
        g_bytes[i].store(0);
    }
}

} // extern "C"

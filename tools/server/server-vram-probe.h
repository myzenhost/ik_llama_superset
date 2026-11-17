#pragma once

// GATE-CLAIM-OK: this header holds the pure parsing/arithmetic of the free-GPU-memory probes (sysfs, /proc/meminfo, ioreg, PDH instance
// names). It makes no claim about any host's state; the readings come from the callers.
//
// Row S69-A-PICK-VENDOR-NEUTRAL-VRAM-TELEMETRY-FOR-SERVER-VRAM-AND-WATCHDOGS-2026-10-05-01.
// SOURCE-READ: llama-swap internal/perf/{monitor_unix,monitor_windows,monitor_darwin,gpu_parse}.go and internal/hw/{amd_linux,intel_linux}.go
// (MIT) for WHICH facts each vendor offers; no code copied. Where theirs is wrong or a stub (Linux sysfs path = ErrNotImplemented; Windows sums
// the system-memory aperture and uses a D3DKMT struct layout that returned segment 0 for every id on this Windows build) the probes here differ,
// see the row body for the measurements.
//
// Everything in this header is free of OS calls except std::filesystem reads of a ROOT THE CALLER NAMES, so the unit test can feed it fixture
// directories and recorded command output.

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <set>
#include <sstream>
#include <string>
#include <vector>

namespace vram_probe {

struct device {
    std::string id;          // stable key (canonical sysfs path, DXGI luid, ...)
    std::string vendor;      // "0x1002", "apple", ... (informational)
    int64_t     total = 0;   // bytes the device can use
    int64_t     used  = 0;   // bytes in use, system-wide
    bool        unified = false; // shares system memory (APU, Apple silicon, GB10)
    int64_t free_bytes() const { return total > used ? total - used : 0; }
};

// One unsigned decimal in a file (sysfs style, trailing newline allowed). -1 when the file is missing, empty or not a plain number.
inline int64_t read_i64(const std::filesystem::path & p) {
    std::ifstream in(p, std::ios::binary);
    if (!in) {
        return -1;
    }
    std::string s;
    std::getline(in, s);
    while (!s.empty() && std::isspace((unsigned char) s.back())) {
        s.pop_back();
    }
    if (s.empty() || s.size() > 19) {
        return -1;
    }
    int64_t v = 0;
    for (char c : s) {
        if (c < '0' || c > '9') {
            return -1;
        }
        v = v * 10 + (c - '0');
    }
    return v;
}

inline std::string read_word(const std::filesystem::path & p) {
    std::ifstream in(p, std::ios::binary);
    std::string s;
    if (in) {
        std::getline(in, s);
    }
    while (!s.empty() && std::isspace((unsigned char) s.back())) {
        s.pop_back();
    }
    std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) { return (char) std::tolower(c); });
    return s;
}

// "card" followed by digits only. cardN-HDMI-A-1 (connector nodes), renderD128, version, ... are not devices.
inline bool is_card_dir_name(const std::string & n) {
    if (n.size() < 5 || n.compare(0, 4, "card") != 0) {
        return false;
    }
    return std::all_of(n.begin() + 4, n.end(), [](unsigned char c) { return std::isdigit(c) != 0; });
}

// Linux DRM sysfs (root normally /sys/class/drm). A device counts when its driver exposes BOTH mem_info_vram_total and mem_info_vram_used
// (amdgpu does; some other drivers do not, and are then left to their vendor tool). Several cardN names can point at one PCI device:
// they are deduplicated by the canonical path of the `device` link. Vendors in `skip_vendors` are left to a better reader (NVIDIA -> NVML).
//
// APU nuance (UNVERIFIED on an APU host): an amdgpu APU reports only its BIOS carve-out as VRAM (often 512 MiB - 4 GiB) while the GPU may
// use GTT, which is system RAM. When the carve-out is at most 4 GiB and GTT is at least 4x that, the GTT is counted as part of the device
// and the device is flagged unified, otherwise a model that runs fine on the APU would be refused for "no VRAM".
inline std::vector<device> scan_drm(const std::string & root, const std::set<std::string> & skip_vendors = { "0x10de" }) {
    namespace fs = std::filesystem;
    std::vector<device> out;
    std::set<std::string> seen;
    std::error_code ec;
    if (!fs::is_directory(root, ec)) {
        return out;
    }
    std::vector<std::string> names;
    for (const auto & e : fs::directory_iterator(root, ec)) {
        names.push_back(e.path().filename().string());
    }
    std::sort(names.begin(), names.end());
    for (const auto & n : names) {
        if (!is_card_dir_name(n)) {
            continue;
        }
        const fs::path dev = fs::path(root) / n / "device";
        const std::string vendor = read_word(dev / "vendor");
        if (skip_vendors.count(vendor)) {
            continue;
        }
        const int64_t total = read_i64(dev / "mem_info_vram_total");
        const int64_t used  = read_i64(dev / "mem_info_vram_used");
        if (total <= 0 || used < 0) {
            continue;
        }
        std::error_code ec2;
        fs::path canon = fs::canonical(dev, ec2);
        const std::string key = ec2 ? (fs::path(root) / n).string() : canon.string();
        if (!seen.insert(key).second) {
            continue;
        }
        device d;
        d.id     = key;
        d.vendor = vendor;
        d.total  = total;
        d.used   = used;
        const int64_t gtt_total = read_i64(dev / "mem_info_gtt_total");
        const int64_t gtt_used  = read_i64(dev / "mem_info_gtt_used");
        if (total <= (int64_t) 4 << 30 && gtt_total >= 4 * total && gtt_used >= 0) {
            d.total += gtt_total;
            d.used  += gtt_used;
            d.unified = true;
        }
        out.push_back(d);
    }
    return out;
}

// "MemAvailable:   123456 kB" -> bytes; -1 when the key is absent.
inline int64_t meminfo_bytes(const std::string & text, const std::string & key) {
    std::istringstream in(text);
    std::string line;
    const std::string k = key + ":";
    while (std::getline(in, line)) {
        if (line.compare(0, k.size(), k) != 0) {
            continue;
        }
        std::istringstream ls(line.substr(k.size()));
        long long v = -1;
        std::string unit;
        ls >> v >> unit;
        if (v < 0) {
            return -1;
        }
        if (unit == "kB" || unit == "KB" || unit == "kb") {
            return (int64_t) v * 1024;
        }
        return unit.empty() ? (int64_t) v : -1;
    }
    return -1;
}

// NVML returns NVML_ERROR_NOT_SUPPORTED (3) from nvmlDeviceGetMemoryInfo on unified-memory parts (GB10 / DGX Spark class): the GPU has no
// memory of its own, it uses system RAM. The caller then reads /proc/meminfo for that device instead of failing the whole host reading.
inline bool nvml_memory_is_unified(int nvml_return) { return nvml_return == 3; }

// `ioreg -r -c IOGPU -d 1 -f` (macOS): the GPU's in-use bytes live in PerformanceStatistics as  "In use system memory"=NNN  (no spaces
// around '=' inside the nested dictionary, spaces at top level). -1 when the key is absent.
inline int64_t parse_ioreg_in_use(const std::string & text) {
    const std::string key = "\"In use system memory\"=";
    const size_t p = text.find(key);
    if (p == std::string::npos) {
        return -1;
    }
    size_t i = p + key.size();
    int64_t v = 0;
    size_t digits = 0;
    while (i < text.size() && std::isdigit((unsigned char) text[i]) && digits < 18) {
        v = v * 10 + (text[i] - '0');
        ++i;
        ++digits;
    }
    return digits ? v : -1;
}

// Apple silicon: the GPU may wire only part of the unified memory. `iogpu.wired_limit_mb` (sysctl) when set (> 0) is that limit; with 0 the
// OS default applies. UNVERIFIED default: two thirds of RAM up to 36 GiB, three quarters above (the commonly reported macOS behaviour; Metal's
// recommendedMaxWorkingSetSize is the authority and cannot be read without Metal, so LLAMA_SERVER_VRAM_TOTAL_MB overrides it).
inline int64_t apple_gpu_budget(int64_t memsize_bytes, int64_t wired_limit_mb) {
    if (memsize_bytes <= 0) {
        return -1;
    }
    if (wired_limit_mb > 0) {
        return std::min<int64_t>(memsize_bytes, wired_limit_mb * 1024 * 1024);
    }
    const int64_t gib36 = (int64_t) 36 << 30;
    return memsize_bytes <= gib36 ? memsize_bytes / 3 * 2 : memsize_bytes / 4 * 3;
}

// PDH instance names of the "GPU Adapter Memory" counter set look like  luid_0x00000000_0x00014B1D_phys_0 : high part, low part, physical
// adapter index. Returns the 64-bit LUID (high << 32 | low) and the phys index; false when the name has another shape.
inline bool parse_pdh_instance(const std::string & inst, uint64_t & luid, int & phys) {
    unsigned long long hi = 0, lo = 0;
    int ph = 0;
    char tail[8] = {};
    if (std::sscanf(inst.c_str(), "luid_0x%llx_0x%llx_phys_%d%7s", &hi, &lo, &ph, tail) != 3) {
        return false;
    }
    luid = (hi << 32) | (lo & 0xffffffffull);
    phys = ph;
    return true;
}

} // namespace vram_probe

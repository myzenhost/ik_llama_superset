// ggml-host-memory.h -- host RAM as Linux reports it: /proc/meminfo, tightened by every cgroup v2 limit on the path from
// this process's leaf cgroup to the root. Header-only, pure functions over an injectable file reader, so the Linux logic
// is tested on every host (tests/test-load-budget.cpp drives it through a fake file system).
// HARNESS-CONSIDERED: a memory-reporting fix inside the engine; no harness capability is touched or switched off.
// RESEARCH-DONE: https://raw.githubusercontent.com/ollama/ollama/a8aaf9fcfad23dc8d6f07b8109b1ba5ed310d76c/discover/cpu_linux.go
//
// Two consumers:
//   - ggml_backend_cpu_device_get_memory (ggml-cpu.cpp): the CPU device's total/free. Before this header the non-Windows
//     branch reported total = _SC_PHYS_PAGES and free = total ("assume that all of it is free"), so inside a container
//     with a memory limit the CPU device claimed the whole HOST, and everything that sizes host buffers from it (--fit,
//     the router's model placement) planned against RAM the container cannot have.
//   - src/llama-load-budget.h (the loader's parallel-load budget): moved here from that header unchanged, so the cgroup
//     walk has one implementation.
//
// Behaviour, from two sources, both read in full: ollama/ollama @a8aaf9fc (MIT) discover/cpu_linux.go:18-91 (total =
// MemTotal, free = MemAvailable or MemFree+Buffers+Cached on kernels without MemAvailable, then cgroup v2 memory.max as
// total and memory.max - memory.current as free) and ikawrakow/ik_llama.cpp PR #2125 (MIT; per-level walk, page cache
// counted as reclaimable). Written here, no code copied. Differences from Ollama, on purpose:
//   - every level from the leaf to the root is read, not only /sys/fs/cgroup (a nested limit, e.g. a systemd slice or
//     a pod-level limit above the container, is otherwise missed);
//   - memory.max = "max" means no limit at that level (Ollama's ParseUint fails on it and then computes
//     free = MemTotal - memory.current, i.e. host total minus this container's usage);
//   - reclaimable page cache (memory.stat file - shmem) is not counted as used, and a level charged above its limit
//     reports 0 free instead of an unsigned wrap-around.
#pragma once

#include <cstdint>
#include <fstream>
#include <functional>
#include <optional>
#include <sstream>
#include <string>

namespace ggml_host_mem {

// Unsigned decimal at s[pos] after blanks; nullopt when there is no digit or the value overflows.
inline std::optional<uint64_t> parse_u64(const std::string & s, size_t pos = 0) {
    while (pos < s.size() && (s[pos] == ' ' || s[pos] == '\t')) {
        ++pos;
    }
    if (pos >= s.size() || s[pos] < '0' || s[pos] > '9') {
        return std::nullopt;
    }
    uint64_t v = 0;
    for (; pos < s.size() && s[pos] >= '0' && s[pos] <= '9'; ++pos) {
        const uint64_t d = (uint64_t) (s[pos] - '0');
        if (v > (UINT64_MAX - d) / 10) {
            return std::nullopt;
        }
        v = v * 10 + d;
    }
    return v;
}

// The rest of the first line that starts with `key` (key includes its delimiter, e.g. "MemAvailable:" or "file ").
inline std::optional<std::string> line_value(const std::string & text, const std::string & key) {
    size_t start = 0;
    while (start < text.size()) {
        size_t end = text.find('\n', start);
        if (end == std::string::npos) {
            end = text.size();
        }
        if (end - start >= key.size() && text.compare(start, key.size(), key) == 0) {
            std::string v = text.substr(start + key.size(), end - start - key.size());
            while (!v.empty() && (v.back() == '\r' || v.back() == ' ' || v.back() == '\t')) {
                v.pop_back();
            }
            return v;
        }
        start = end + 1;
    }
    return std::nullopt;
}

// One /proc/meminfo field (reported in kB) in bytes.
inline std::optional<uint64_t> meminfo_bytes(const std::string & meminfo, const std::string & key) {
    const auto v = line_value(meminfo, key);
    if (!v) {
        return std::nullopt;
    }
    const auto kb = parse_u64(*v);
    if (!kb || *kb > UINT64_MAX / 1024) {
        return std::nullopt;
    }
    return *kb * 1024;
}

// /proc/meminfo text -> MemAvailable in bytes. Strict: no fallback (the loader budget wants "unknown" over a guess).
inline std::optional<uint64_t> meminfo_available(const std::string & meminfo) {
    return meminfo_bytes(meminfo, "MemAvailable:");
}

// /proc/meminfo text -> MemTotal in bytes.
inline std::optional<uint64_t> meminfo_total(const std::string & meminfo) {
    return meminfo_bytes(meminfo, "MemTotal:");
}

// Free RAM estimate for the CPU device: MemAvailable, or MemFree + Buffers + Cached on a kernel that predates
// MemAvailable (Linux < 3.14; the same fallback as ollama discover/cpu_linux.go:59-63).
inline std::optional<uint64_t> meminfo_free_estimate(const std::string & meminfo) {
    if (const auto a = meminfo_available(meminfo)) {
        return a;
    }
    const auto fr = meminfo_bytes(meminfo, "MemFree:");
    if (!fr) {
        return std::nullopt;
    }
    const auto buffers = meminfo_bytes(meminfo, "Buffers:");
    const auto cached  = meminfo_bytes(meminfo, "Cached:");
    return *fr + (buffers ? *buffers : 0) + (cached ? *cached : 0);
}

// /proc/self/cgroup text -> the cgroup v2 leaf path (the "0::<path>" line).
inline std::optional<std::string> cgroup_v2_path(const std::string & proc_self_cgroup) {
    auto p = line_value(proc_self_cgroup, "0::");
    if (!p || p->empty() || (*p)[0] != '/') {
        return std::nullopt;
    }
    return p;
}

// memory.max text -> the limit in bytes; nullopt for "max" (no limit at this level) or unreadable text.
inline std::optional<uint64_t> cgroup_limit(const std::string & max_txt) {
    std::string mx = max_txt;
    while (!mx.empty() && (mx.back() == '\n' || mx.back() == '\r' || mx.back() == ' ')) {
        mx.pop_back();
    }
    if (mx == "max") {
        return std::nullopt;
    }
    return parse_u64(mx);
}

// One cgroup v2 level: bytes still allocatable under its memory.max. nullopt = no limit here ("max") or unreadable.
inline std::optional<uint64_t> cgroup_available(const std::string & max_txt, const std::string & current_txt,
                                                const std::string & stat_txt) {
    const auto limit   = cgroup_limit(max_txt);
    const auto current = parse_u64(current_txt);
    if (!limit || !current) {
        return std::nullopt;
    }
    uint64_t reclaimable = 0;
    const auto file_v  = line_value(stat_txt, "file ");
    const auto shmem_v = line_value(stat_txt, "shmem ");
    if (file_v && shmem_v) {
        const auto file  = parse_u64(*file_v);
        const auto shmem = parse_u64(*shmem_v);
        if (file && shmem && *file >= *shmem) {
            reclaimable = *file - *shmem;   // page cache the kernel can drop; shmem lives in it but cannot be dropped
        }
    }
    const uint64_t charged = *current > reclaimable ? *current - reclaimable : 0;
    return *limit > charged ? *limit - charged : 0;
}

using read_fn = std::function<std::optional<std::string>(const std::string & path)>;

// Calls fn(memory.max, memory.current, memory.stat) for every cgroup v2 level that has a memory.max, from this process's
// leaf up to AND INCLUDING the root of the visible hierarchy. Missing current/stat are passed as empty text.
// The root matters: inside a container with its own cgroup namespace (Docker's default on cgroup v2) /proc/self/cgroup
// reads "0::/" and the container's limit sits in /sys/fs/cgroup/memory.max itself. On a host the real root has no
// memory.max file, so reading it costs nothing there. MEASURED 2026-10-20 (docker --memory 2g, debian:bookworm-slim):
// /proc/self/cgroup = "0::/", /sys/fs/cgroup/memory.max = 2147483648; a walk that stopped below the root reported the
// whole 88 GiB VM as the CPU device's total (engq_a_1019/olp1_out/run_after_lim2g.log, first build).
inline void for_each_cgroup_level(const read_fn & read,
                                  const std::function<void(const std::string &, const std::string &, const std::string &)> & fn) {
    const auto cg   = read("/proc/self/cgroup");
    const auto leaf = cg ? cgroup_v2_path(*cg) : std::nullopt;
    if (!leaf) {
        return;
    }
    std::string p = *leaf;
    for (int depth = 0; depth < 128 && !p.empty(); ++depth) {
        const std::string base = p == "/" ? std::string("/sys/fs/cgroup") : "/sys/fs/cgroup" + p;
        if (const auto mx = read(base + "/memory.max")) {
            const auto cur = read(base + "/memory.current");
            const auto st  = read(base + "/memory.stat");
            fn(*mx, cur ? *cur : std::string(), st ? *st : std::string());
        }
        if (p == "/") {
            break;
        }
        const size_t slash = p.find_last_of('/');
        if (slash == std::string::npos) {
            break;
        }
        p = slash == 0 ? std::string("/") : p.substr(0, slash);
    }
}

// Linux: MemAvailable, tightened by every cgroup v2 limit from the leaf up. nullopt when nothing could be read.
// (The loader budget: src/llama-load-budget.h.)
inline std::optional<uint64_t> linux_allocatable_bytes(const read_fn & read) {
    std::optional<uint64_t> best;
    if (const auto mi = read("/proc/meminfo")) {
        best = meminfo_available(*mi);
    }
    for_each_cgroup_level(read, [&](const std::string & mx, const std::string & cur, const std::string & st) {
        const auto a = cgroup_available(mx, cur, st);
        if (a && (!best || *a < *best)) {
            best = a;
        }
    });
    return best;
}

struct host_memory {
    std::optional<uint64_t> total;   // MemTotal, capped by the tightest cgroup memory.max
    std::optional<uint64_t> free;    // MemAvailable (or the old-kernel estimate), capped by the tightest cgroup headroom
};

// Linux: what the CPU device should report. free never exceeds total.
inline host_memory linux_host_memory(const read_fn & read) {
    host_memory m;
    if (const auto mi = read("/proc/meminfo")) {
        m.total = meminfo_total(*mi);
        m.free  = meminfo_free_estimate(*mi);
    }
    for_each_cgroup_level(read, [&](const std::string & mx, const std::string & cur, const std::string & st) {
        if (const auto lim = cgroup_limit(mx)) {
            if (!m.total || *lim < *m.total) {
                m.total = lim;
            }
        }
        if (const auto a = cgroup_available(mx, cur, st)) {
            if (!m.free || *a < *m.free) {
                m.free = a;
            }
        }
    });
    if (m.total && m.free && *m.free > *m.total) {
        m.free = m.total;
    }
    return m;
}

// The real file system reader (whole file; nullopt when it cannot be opened).
inline std::optional<std::string> read_file(const std::string & path) {
    std::ifstream f(path);
    if (!f) {
        return std::nullopt;
    }
    std::stringstream ss;
    ss << f.rdbuf();
    return ss.str();
}

} // namespace ggml_host_mem
